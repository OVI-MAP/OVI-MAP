import os, sys, time, argparse, pickle, logging
from os.path import join as pjoin
from tqdm import tqdm

import cv2, copy
from PIL import Image
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import open3d.core as o3c  # type: ignore
from plyfile import PlyData

# self packages
from utils.common_utils import Segment, class_id_to_one_hot, PointCloudProcessor
from utils.data_loaders import ScannetLoader, ReplicaLoader

from vl_models import VLModel
from scripts.visualizations.vis_utils import get_new_pallete

from panoptic_mapping_ import update_view_cov_map, make_res_dirs


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FORMAT = '%(asctime)s.%(msecs)06d %(levelname)-8s: [%(filename)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt='%H:%M:%S')




class RayCastScene:
    """Ray casting scene for 3D instance projection to 2D"""
    def __init__(self, 
            vertex_positions, vertex_colors, 
            vertex_labels, triangles
        ):
        # tmesh = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)

        self.vertex_colors = vertex_colors # [0, 255]
        self.vertex_labels = vertex_labels # uint16
        self.triangles = triangles

        label_set = np.unique(vertex_labels)
        self.inst_to_c = {}
        # find the majority color for each instance
        for inst_id in label_set:
            if inst_id == 0:
                continue
            inst_mask = (vertex_labels == inst_id)
            inst_c = np.median(vertex_colors[inst_mask], axis=0)
            self.inst_to_c[inst_id] = inst_c.astype(np.uint8)

        self.mesh = o3d.t.geometry.TriangleMesh()
        self.mesh.vertex.positions = o3c.Tensor(vertex_positions, 
            dtype=o3c.Dtype.Float32)
        self.mesh.vertex.colors = o3c.Tensor(
            vertex_colors.astype(np.float32) / 255.0, 
            dtype=o3c.Dtype.Float32
        )
        self.mesh.triangle.indices = o3c.Tensor(triangles, 
            dtype=o3c.Dtype.Int32
        )
        self.mesh.compute_vertex_normals()
        self.mesh.compute_triangle_normals()

        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(self.mesh)


    def init_camera(self, camera_cfgs):
        """Initialize the camera with the given camera matrix."""
        fx, fy, cx, cy, H, W = camera_cfgs
        self.H = H
        self.W = W
        # Generate rays for each pixel
        yx_grid = np.mgrid[0:H, 0:W] # (2, H, W)
        dirs = np.stack([(yx_grid[1] - cx) / fx, (yx_grid[0] - cy) / fy, np.ones((H, W))], axis=-1)  # (H, W, 3)
        self.dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)  # Normalize

    def ray_casting(self, pose_c2w):
        # Transform ray directions to world coordinates
        ray_dirs = self.dirs.reshape(-1, 3) @ pose_c2w[:3, :3].T
        # camera origin in World coordinates
        ray_origins = np.broadcast_to(pose_c2w[:3, 3], ray_dirs.shape)
        rays = np.concatenate([ray_origins, ray_dirs], axis=-1)
        rays = o3c.Tensor(rays, dtype=o3c.Dtype.Float32)
        # do ray casting and check the primitive ids where the rays hit
        ans = self.scene.cast_rays(rays)

        return ans['primitive_ids'].numpy()


    def raycast_to_instances2D(self, pose_c2w):
        """
        Cast rays from the camera pose to the scene and return the instance maps.
        Args:
            pose_c2w (np.ndarray): Camera pose in world coordinates, shape (4, 4).
        """

        triangle_ids = self.ray_casting(pose_c2w) # in shape (H*W, )

        # no hit is scene.INVALID_ID
        hit_mask = np.logical_and(
            triangle_ids >= 0, triangle_ids < len(self.triangles))
        triangle_ids[~hit_mask] = 0 # temporarily set to 0, will be set to bg later

        # take the first vertex in each triangle
        vertex_ids = self.triangles[triangle_ids][:, 0]
        inst_color_img = (self.vertex_colors[vertex_ids] * 255.0).astype(np.uint8)
        inst_color_img = inst_color_img.reshape(self.H, self.W, 3)
        inst_map = self.vertex_labels[vertex_ids].reshape(self.H, self.W)
        
        hit_mask = hit_mask.reshape(self.H, self.W)
        inst_map[~hit_mask] = 0  # Set non-hit pixels to bg
        inst_color_img[~hit_mask] = 0 # Set non-hit pixels to black

        return inst_map, inst_color_img
    


def init_view_cov(
    depth_scaled, inst_d_mask, inst_points, 
    glo_inst_id, cur_inst_info, 
    vis_area_thres, sph_grid_size
):
    depth_arr = depth_scaled[inst_d_mask]
    if len(depth_arr) < 0.5 * vis_area_thres:
        logging.warning(f"[Skip] Not enough depth pts to init inst {glo_inst_id}")
        return False, None

    PCLprocessor = PointCloudProcessor(inst_points)
    bbox_res = PCLprocessor.process(voxel_size=0.01)
    bbox_pos, bbox_extent, bbox_rot_matrix = bbox_res

    view_map = np.zeros(sph_grid_size, dtype=np.uint8)
    cur_inst_info['view_map'] = view_map
    cur_inst_info['bbox_c'] = bbox_pos
    cur_inst_info['bbox_s'] = bbox_extent
    cur_inst_info['bbox_rot'] = np.array(bbox_rot_matrix)

    return True, cur_inst_info



def frameToSegmentsGT(depth_img, camera_K, pose_i, frame_i, seg_map):
    points_map = cv2.rgbd.depthTo3d(depth_img, camera_K)
    segments_list = []

    inst_ids = np.unique(seg_map)
    seg_index = 0
    for _, m_id in enumerate(inst_ids):
        inst_mask = (seg_map == m_id)
        if np.count_nonzero(inst_mask) < 100:
            continue
        
        points = points_map[inst_mask].astype(np.float32).reshape(-1,3)
        # for Background
        if m_id == 0:
            is_thing = False
            sem_feat = class_id_to_one_hot(0, num_classes=2)
            segment = Segment(
                points, is_thing, 0, 0, 
                1.0, 1.0, pose_i, seg_index, sem_feat=sem_feat
            )
            segments_list.append(segment)
            seg_index += 1
        else:
            is_thing = True
            sem_feat = class_id_to_one_hot(1, num_classes=2)
            segment = Segment(
                points, is_thing, m_id, 1, 
                1.0, 1.0, pose_i, seg_index, sem_feat=sem_feat
            )
            segments_list.append(segment)
            seg_index += 1

    return segments_list



def parse_args():
    parse = argparse.ArgumentParser(description='Semantic Mapping-Python') 
    # dataset 
    parse.add_argument("--dataset", type=str, default="scannet_nyu", 
        help="which data set to use, scenenn or scannet ")

    # files path
    parse.add_argument("--scene_num", type=str, required=True, 
        help="which scene for mapping ")
    
    parse.add_argument("--result_folder", type=str, 
        default='/home/zilong/Disk_data/semantic_mapping_result', 
        help="folder of mapping results")
    parse.add_argument("--data_folder", type=str, 
        default='/media/zilong/Documents/MasterProject/scannet_v2', 
        help="which scene for mapping ")
    parse.add_argument("--traj_filename", type=str, default="trajectory.log", 
        help="which trajectory_to_use ")

    # log
    parse.add_argument("--log", type=str, default='', 
        help="log info ")
    # mapping configuration
    parse.add_argument("--start", type=int,default=0, 
        help="start of the sequence, 0 means starting from the beginning")
    parse.add_argument("--end", type=int, default=-1, 
        help="end of the sequence, -1 means using all of them")
    parse.add_argument("--step", type=int, default=-1, 
        help="use one frame for integration every n_step frames ")
    
    return parse.parse_args()



def main(args):
    dataset = args.dataset
    scene_num = args.scene_num
    
    data_path = args.data_folder
    scene_folder = pjoin(data_path, scene_num)

    result_folder = args.result_folder
    result_dirs = make_res_dirs(result_folder)

    log_info = args.log

    # DataLoader
    traj_filename = args.traj_filename
    if dataset == "scannet_nyu":
        data_loader = ScannetLoader(scene_folder)
    elif dataset == "replica":
        data_loader = ReplicaLoader(scene_folder)

    H_depth = data_loader.depth_h
    W_depth = data_loader.depth_w
    K_depth = data_loader.getDepthCameraMatrix()

    # configuration
    start = args.start
    assert (start >= data_loader.indexes[0])
    end = args.end
    if end < 0:
        end = data_loader.index_max + 1
    step = args.step
    # for Scannet Dataset
    if step < 0:
        step = int((end-start) // 200)
    logging.info(f"Running scene {scene_num} from frame {start} to frame {end-1} with step {step}.")

    # TODO set flag for complete open-set segmentation
    # ==========================================================
    # Whether to reconstruct from the 2D projection of the gt inst map
    build_from_gt_inst = False 

    VLM_name = 'siglip-l-16-384'
    exp_results = pjoin(result_folder, 'gt_inst_viewcov')
    gt_inst_mesh_f = pjoin(result_folder, 'gt_instance_mesh.ply')
    temp_feats = pjoin(result_folder, 'gt_inst_vis', 'temp_feats')
    use_prev_feat = False

    # they are mutually exclusive
    select_by_viewcov = True
    select_by_vis = False
    select_combine = False
    inst_sem_name = f'inst_sem_{VLM_name}_{int(np.ceil((end-start)/step))}_incre_viewcov.pkl'
    # ==========================================================

    if not os.path.exists(exp_results):
        os.makedirs(exp_results)
        os.makedirs(temp_feats, exist_ok=True)


    # Aggregate the visul feature based on the gt instance map
    gt_inst_m = PlyData.read(gt_inst_mesh_f)
    vertex_pos = np.stack([
        gt_inst_m['vertex']['x'], gt_inst_m['vertex']['y'], gt_inst_m['vertex']['z']
    ], axis=-1)
    vertex_c = np.stack([
        gt_inst_m['vertex']['red'], gt_inst_m['vertex']['green'], gt_inst_m['vertex']['blue']
    ], axis=-1)
    vertex_label = gt_inst_m['vertex']['label']
    triangle_indices = gt_inst_m['face']['vertex_indices']
    if len(triangle_indices[0]) == 4:
        # need to convert to triangles
        quads = np.stack([row.astype(np.int32) for row in triangle_indices])
        triangles = np.zeros((quads.shape[0]*2, 3), dtype=np.int32)
        triangles[0::2] = quads[:, [0,1,2]]
        triangles[1::2] = quads[:, [0,2,3]]
    else:
        triangles = np.stack([row.astype(np.int32) for row in triangle_indices])

    raycast_scene = RayCastScene(vertex_pos, vertex_c, vertex_label, triangles)
    raycast_scene.init_camera((
        K_depth[0][0], K_depth[1][1], K_depth[0][2], K_depth[1][2], H_depth, W_depth))

    inst_to_c = raycast_scene.inst_to_c # a dict to map glo inst_id to color

    # Create the VL model for language-aligned image feature extraction
    vl_model = VLModel(model_name=VLM_name, img_size=(H_depth, W_depth), device=DEVICE)
    logging.info(f"VL model {VLM_name} initialized!")

    inst_dict = {}
    yx_grid = np.mgrid[0:H_depth, 0:W_depth] # (2, H, W)
    sph_grid_size = (int(H_depth/8), int(W_depth/8))
    theta_phi_grid = np.mgrid[0:sph_grid_size[0], 0:sph_grid_size[1]]

    vis_area_thres = 1000 
    if select_by_viewcov:
        view_overlap_ratio_thres = 0.85
    elif select_combine:
        view_overlap_ratio_thres = 0.7

    if build_from_gt_inst:
        import consistent_gsm # type: ignore
        log_file = os.path.abspath(result_dirs['log'])
        inst_association = 4
        data_association = 2
        num_threads = 10
        seg_graph_confidence = 3
        connection_ratio_th = 0.2
        cos_sim_th = 0.8
        gsm_node = consistent_gsm.GlobalSegmentMap_py(
            log_file, 'Nyu40', False, False, 
            inst_association, data_association, 
            num_threads, False, seg_graph_confidence, 
            True, connection_ratio_th, cos_sim_th
        )
        gsm_node.outputLog(log_info)



    # ========================== start mapping ==========================
    time_s = time.time()
    frame_ids = range(start, end, step)
    for f_i in frame_ids:
        logging.info(f"================ Processing frame {f_i} ================")
        # NOTE img is in BGR, depth is in meters, pose is c2w
        # rgb image has been wrapped to depth image
        rgb_img, depth_scaled, pose = data_loader.getDataFromIndex(f_i)
        # check data validity
        if(rgb_img is None or depth_scaled is None or pose is None):
            logging.warning(f"[Skip] frame {f_i} is lack of RGB / Depth / Pose.")
            continue
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
        # get 2D instance map from ray casting
        glo_inst_map, inst_color_map = raycast_scene.raycast_to_instances2D(pose)
        # # gt depth
        # depth_scaled = dist_hit 
        # points_map = cv2.rgbd.depthTo3d(depth_scaled, K_depth)

        if build_from_gt_inst:
            glo_inst_map = glo_inst_map.astype(np.int32)
            # fuse the gt 2d instance map into a global 3D map and reproeject 
            segment_list: list[Segment] = frameToSegmentsGT(
                depth_scaled, K_depth, pose, f_i, glo_inst_map
            )
            if len(segment_list) == 0:
                logging.warning(f"[Skip] No segment found in frame {f_i}")
                continue
            
            for segment in segment_list:
                segment.calculateBBox()
                # all segs except bg are 'thing' and with a 1.0 inst_conf
                gsm_node.insertSegmentsOpen(
                    segment.points, segment.box_points, 
                    segment.instance_label, segment.class_label, 
                    segment.sem_feat, 
                    segment.inst_confidence, segment.overlap_ratio, 
                    pose, segment.is_thing, segment.segment_label
                )
            gsm_node.integrateFrame()
            glo_inst_map = gsm_node.raycastInstancePredictions(
                pose, glo_inst_map, depth_scaled
            )

        # vis_id_map(glo_inst_map, f'/home/zilong/Downloads/{scene_num}_gt_raycast/raycast_{f_i}.png')

        if select_by_viewcov or select_combine:
            points_map = cv2.rgbd.depthTo3d(depth_scaled, K_depth)
            valid_d_mask = (depth_scaled > 0.0) & (depth_scaled < 30.0)

        
        # ################# View Selection #################
        observed_inst_ids = np.unique(glo_inst_map)
        for glo_inst_id in observed_inst_ids:
            if glo_inst_id == 0:
                continue

            glo_inst_mask = (glo_inst_map == glo_inst_id)  # (H, W)
            glo_inst_area = np.count_nonzero(glo_inst_mask)
            # 1. skip too small instance in the global map
            if glo_inst_area < 0.5 * vis_area_thres:
                logging.warning(f"[Skip] inst {glo_inst_id} with inst area {glo_inst_area}.")
                continue

            # we don't need to fuse 2D panoptic map here, so they are the same
            overlap_area = glo_inst_area

            if glo_inst_id not in inst_dict.keys():
                cur_inst_info = {
                    'frame_id': [], 'feat': [], 'pose': [],
                    'vis_area': [], 'box_2d': [], 
                }
            else:
                cur_inst_info = inst_dict[glo_inst_id]

            # 2. check if it's a good new view to select
            if select_by_viewcov:
                # # the novel view also can't be too small in terms of vis_area
                # if overlap_area < 0.6 * min_area:
                #     logging.warning(f"[Skip] Too small pano area {overlap_area} in view_cov compared to {min_area}.")
                #     continue

                inst_d_mask = np.logical_and(glo_inst_mask, valid_d_mask)  # (H, W)
                if np.count_nonzero(inst_d_mask) < 0.1* vis_area_thres:
                    logging.warning(f"[Skip] Not enough valid depth pts in inst {glo_inst_id}.")
                    continue
                inst_points = points_map[inst_d_mask].astype(np.float32).reshape(-1,3)
                inst_points = inst_points @ pose[:3,:3].T + pose[:3,3:4].T

                # 2.1. Init an object center by estimating the BBOX
                # max_vis_area = 0 if glo_inst_id not in inst_dict.keys() \
                #     else np.array(inst_dict[glo_inst_id]['vis_area']).max()
                # if glo_inst_id not in inst_dict.keys() or (overlap_area > 2.0 * max_vis_area):
                if 'view_map' not in cur_inst_info.keys():
                    # this will add the 3d bbox and the view map into the cur_inst_info
                    success, cur_inst_info = init_view_cov(
                        depth_scaled, inst_d_mask, inst_points, 
                        glo_inst_id, cur_inst_info, 
                        vis_area_thres, sph_grid_size
                    )
                    if not success:
                        continue
                
                # 2.2. Map the rays from obj_c to surface to the spherical grid
                view_map = cur_inst_info['view_map']
                success, view_map = update_view_cov_map(
                    inst_points, cur_inst_info['bbox_c'], 
                    view_map, sph_grid_size, view_overlap_ratio_thres
                )
                if not success:
                    continue
                cur_inst_info['view_map'] = view_map
                
            elif select_by_vis:
                past_vis_areas = cur_inst_info['vis_area']
                if len(past_vis_areas) >= 8:
                    top_vis_s = np.sort(np.array(cur_inst_info['vis_area']))[-8:]
                    if overlap_area <= np.min(top_vis_s):
                        # not visible enough
                        continue

            elif select_combine:
                past_vis_areas = cur_inst_info['vis_area']
                if len(past_vis_areas) >= 8:
                    top_vis_s = np.sort(np.array(cur_inst_info['vis_area']))[-8:]
                    min_area = np.min(top_vis_s)
                    # not visible enough, check the view coverage
                    if overlap_area <= min_area:
                        # TODO important!!!!!
                        # the novel view also can't be too small in terms of vis_area
                        if overlap_area < 0.6 * min_area:
                            logging.warning(f"[Skip] Too small pano area {overlap_area} in view_cov compared to {min_area}.")
                            continue

                        inst_d_mask = np.logical_and(glo_inst_mask, valid_d_mask)  # (H, W)
                        if np.count_nonzero(inst_d_mask) < 0.1* vis_area_thres:
                            logging.warning(f"[Skip] Not enough valid depth pts in inst {glo_inst_id}.")
                            continue
                        inst_points = points_map[inst_d_mask].astype(np.float32).reshape(-1,3)
                        inst_points = inst_points @ pose[:3,:3].T + pose[:3,3:4].T

                        if 'view_map' not in cur_inst_info.keys():
                            success, cur_inst_info = init_view_cov(
                                depth_scaled, inst_d_mask, inst_points, 
                                glo_inst_id, cur_inst_info, 
                                vis_area_thres, sph_grid_size
                            )
                            if not success:
                                continue
                        view_map = cur_inst_info['view_map']
                        success, view_map = update_view_cov_map(
                            inst_points, cur_inst_info['bbox_c'], 
                            view_map, sph_grid_size, view_overlap_ratio_thres
                        )
                        if not success:
                            continue
                        cur_inst_info['view_map'] = view_map
                        overlap_area = -1 # to markdown which selection strategy is used
        
            inst_dict[glo_inst_id] = cur_inst_info

            # 3. Get visual feature from the VL model
            # detrermine the cropping area by the global inst mask
            yxs = yx_grid[:, glo_inst_mask] # (2, M)
            y1, x1 = np.min(yxs[0]), np.min(yxs[1])
            y2, x2 = np.max(yxs[0]), np.max(yxs[1])
            
            vis_feat_name = f"{VLM_name}_F_{f_i}_{x1}-{y1}-{x2-x1}-{y2-y1}.npy"
            vis_feat_path = pjoin(temp_feats, vis_feat_name)
            if use_prev_feat and os.path.exists(vis_feat_path):
                # load the feature from the file
                roi_feat = np.load(vis_feat_path)
            else:
                roi_feat = vl_model.encode_image_with_bbox(
                    rgb_img, glo_inst_mask, (x1, y1, x2, y2)
                )
                np.save(vis_feat_path, roi_feat)

            # 4. Add the instance to the buffer
            inst_dict[glo_inst_id]['frame_id'].append(f_i)
            inst_dict[glo_inst_id]['box_2d'].append((x1, y1, x2, y2))
            inst_dict[glo_inst_id]['feat'].append(roi_feat)
            inst_dict[glo_inst_id]['pose'].append(pose)
            inst_dict[glo_inst_id]['vis_area'].append(overlap_area)

        if build_from_gt_inst:
            gsm_node.clearTemporaryMemory()
        
    logging.info(f"Time taken per frame: {(time.time() - time_s)/len(frame_ids):.2f} seconds")

    if build_from_gt_inst:
        gsm_node.LogLabelInformation()
        gsm_node.LogMeshColors(exp_results)
        logging.info("Start mesh generation!")
        gsm_node.generateMesh(
            exp_results, str(int(np.ceil((end-start)/step))), 
            False, False, True
        )

    # average the features
    np.random.seed(0)

    # sum up the instance features and save them
    inst_sem_dict = {}
    total_query = 0
    for glo_inst_id in inst_dict.keys():
        cur_inst_info = inst_dict[glo_inst_id]
        inst_frames = cur_inst_info['frame_id']
        obs_num = len(inst_frames)
        if obs_num == 0:
            continue
        inst_poses = cur_inst_info['pose']
        inst_bbox2ds = cur_inst_info['box_2d']

        all_feats = np.array(cur_inst_info['feat'])
        total_query += all_feats.shape[0]

        if select_by_viewcov:
            view_map = cur_inst_info['view_map']
            occ_cnt = np.count_nonzero(view_map > 0)
            occ_ratio = occ_cnt / (sph_grid_size[0] * sph_grid_size[1])
            logging.info(f"Instance {glo_inst_id} has {occ_ratio*100:.2f}% observed from {obs_num} frames")
        elif select_by_vis:
            # select the top-k with max vis area
            vis_scores = np.array(cur_inst_info['vis_area'])
            top_indices = np.argsort(vis_scores)[-8:] # indices for values sorted in ascending order
            all_feats = all_feats[top_indices]
            inst_frames = [inst_frames[i] for i in top_indices]
            inst_poses = [inst_poses[i] for i in top_indices]
            inst_bbox2ds = [inst_bbox2ds[i] for i in top_indices]
        elif select_combine:
            vis_scores = np.array(cur_inst_info['vis_area'])
            top_indices = np.argsort(vis_scores)[-8:]
            viewcov_indices = np.argwhere(vis_scores < 0)
            combined_indices = np.unique(np.concatenate(
                (top_indices, viewcov_indices.flatten()), axis=0))

            all_feats = all_feats[combined_indices]
            inst_frames = [inst_frames[i] for i in combined_indices]
            inst_poses = [inst_poses[i] for i in combined_indices]
            inst_bbox2ds = [inst_bbox2ds[i] for i in combined_indices]

            if 'view_map' not in cur_inst_info.keys():
                continue
            view_map = cur_inst_info['view_map']
            occ_cnt = np.count_nonzero(view_map > 0)
            occ_ratio = occ_cnt / (sph_grid_size[0] * sph_grid_size[1])
            logging.info(f"Instance {glo_inst_id} has {occ_ratio*100:.2f}% observed from {obs_num} frames")
        else:
            # select randomly 8 views
            if obs_num > 8:
                rand_indices = np.random.choice(obs_num, 8, replace=False)
                all_feats = all_feats[rand_indices]
                inst_frames = [inst_frames[i] for i in rand_indices]
                inst_poses = [inst_poses[i] for i in rand_indices]
                inst_bbox2ds = [inst_bbox2ds[i] for i in rand_indices]
        

        feat_inst = np.mean(all_feats, axis=0)
        inst_color = inst_to_c[glo_inst_id]

        inst_sem_dict[glo_inst_id] = {
            'feat': feat_inst,
            'frame_id': inst_frames,
            'pose': inst_poses,
            'box_2d': inst_bbox2ds,
            'color': inst_color,
        }
        if select_by_viewcov:
            inst_sem_dict[glo_inst_id]['bbox_c'] = cur_inst_info['bbox_c']
            inst_sem_dict[glo_inst_id]['bbox_s'] = cur_inst_info['bbox_s']
            inst_sem_dict[glo_inst_id]['bbox_rot'] = cur_inst_info['bbox_rot']

    logging.info(f"Totally {len(inst_sem_dict)} instances with {total_query} query features.")
    inst_sem_f = pjoin(exp_results, inst_sem_name)
    with open(inst_sem_f, 'wb') as f:
        pickle.dump(inst_sem_dict, f)
    logging.info(f"Saved instance semantic features to {inst_sem_f}")


if __name__=="__main__":
    # set files path
    args = parse_args()
    main(args)