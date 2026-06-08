import os, sys, time, argparse, pickle, logging, copy
from os.path import join as pjoin
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import torch
import open3d as o3d

from vl_models import VLModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FORMAT = '%(asctime)s.%(msecs)06d %(levelname)-8s: [%(filename)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt='%H:%M:%S')


class PointCloudProcessor:
    def __init__(self, points):
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(points)

    def get_bounding_box(self):
        """
        Computes the oriented bounding box based on the PCA of the convex hull.
        """
        try:
            # Try computing the Oriented Bounding Box (OBB)
            obb = self.pcd.get_oriented_bounding_box()
            position = obb.center
            extent = obb.extent
            rotation = obb.R  # 3x3 rotation matrix
        except RuntimeError:
            print("Warning: OBB computation failed, using AABB instead.")
            
            # Compute Axis-Aligned Bounding Box (AABB)
            aabb = self.pcd.get_axis_aligned_bounding_box()
            max_pos = aabb.max_bound
            min_pos = aabb.min_bound
            position = (max_pos + min_pos) / 2
            extent = np.abs(max_pos - min_pos)
            rotation = np.eye(3)  # Identity matrix (no rotation)

        return position, extent, rotation

    def process(self, voxel_size=0.01):
        # sparsify pcl
        self.pcd = self.pcd.voxel_down_sample(voxel_size)
        # remove outlier
        nb_neighbors=10
        std_thres=2.0
        self.pcd, _ = self.pcd.remove_statistical_outlier(nb_neighbors, std_thres)
        # calculate boundiing box
        return self.get_bounding_box()

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

def update_view_cov_map(
    inst_points, bbox_c, view_map, sph_grid_size, 
    view_overlap_ratio_thres
):
    obj_rays = inst_points - bbox_c.T
    obj_rays = obj_rays / np.linalg.norm(obj_rays, axis=-1, keepdims=True)
    # convert them to sph coord and map to the occ grid
    theta = np.arccos(np.clip(obj_rays[:,2], -1.0, 1.0))  # in [0, pi]
    theta = theta / np.pi * sph_grid_size[0]  # map to [0, grid_H]
    theta = np.floor(theta).astype(np.int32)
    phi = np.arctan2(obj_rays[:,1], obj_rays[:,0])  # in [-pi, pi]
    phi = (phi + np.pi) / (2 * np.pi) * sph_grid_size[1]  # map to [0, grid_W]
    phi = np.floor(phi).astype(np.int32)

    # 4.3. check the overlapping with the existing views
    sph_coords = np.unique(np.stack((theta, phi), axis=1), axis=0)  # (M, 2) removed duplicates
    view_overlap_area = np.count_nonzero(
        view_map[sph_coords[:, 0], sph_coords[:, 1]] > 0)
    view_overlap_ratio = view_overlap_area / sph_coords.shape[0]
    if view_overlap_ratio > view_overlap_ratio_thres:
        # too many overlapping views, skip this instance
        return False, None
    
    # logging.info(f"Add inst {glo_inst_id} with view overlap ratio {view_overlap_ratio:.2f}")
    view_map[sph_coords[:, 0], sph_coords[:, 1]] = 1
    return True, view_map


def main(scene_name):
    print(f"Running scene {scene_name}")

    out_dir = f'/cluster/scratch/dengzi/sem_map/{scene_name}'

    glo_inst_map_dir = f'{out_dir}/glo_2d_map'
    pano_seg_dir = f'{out_dir}/cropformer'
    # rgb_dir = f'/cluster/scratch/dengzi/T_replica/{scene_name}/color'
    rgb_dir = f'/cluster/scratch/dengzi/scans/replica/{scene_name}/results'
    depth_dir = f'/cluster/scratch/dengzi/scans/replica/{scene_name}/results'
    # pose_dir = f'/cluster/scratch/dengzi/T_replica/{scene_name}/pose'
    poses_list = np.loadtxt(f'/cluster/scratch/dengzi/scans/replica/{scene_name}/traj.txt')
    poses_list = poses_list.reshape(-1, 4, 4)

    # H_depth, W_depth = 480, 640 # scannet
    H_depth, W_depth = 680, 1200 # replica
    yx_grid = np.mgrid[0:H_depth, 0:W_depth] # (2, H, W)
    sph_grid_size = (int(H_depth/8), int(W_depth/8))

    # siglip-so400m-14-384 | siglip-l-16-384
    # siglip2-l-16-384 | siglip2-so400m-14-384
    VLM_name = 'siglip-l-16-384' 
    # VLM_name = 'fix_weight' # ovoslam_learned
    inst_sem_name = f'inst_sem_{VLM_name}_incre_combine.pkl'
    vl_model = VLModel(model_name=VLM_name, img_size=(H_depth, W_depth), device=DEVICE)

    # ##########################
    # NOTE they are mutually exclusive
    select_by_vis = False
    select_combine = True
    # ##########################

    vis_area_thres = 1000
    max_top_vis = 10

    ray_cast_max_depth = 50.0
    depth_scale = 6553.5


    if select_combine:
        K_depth = np.array([
            [600.0, 0., 599.5],
            [0., 600.0, 339.5],
            [0.0, 0.0, 1.0]
        ])
        view_overlap_ratio_thres = 0.9

    inst_dict = {}
    judge_time_list = []
    vlm_time_list = []

    num_frame, step = 2000, 10

    glo_inst_color = f'{out_dir}/inst_color_200.pkl'
    glo_inst_color = pickle.load(open(glo_inst_color, 'rb'))
    inst_sem_f = f'{out_dir}/{inst_sem_name}'

    for f_i in tqdm(range(0, num_frame, step)):
        # logging.info(f"==================== Process frame {f_i} ====================")

        # NOTE img is in BGR, depth is in meters, pose is c2w
        # rgb image has been wrapped to depth image
        rgb_img = cv2.imread(pjoin(rgb_dir, f"frame{f_i:06d}.jpg"), cv2.IMREAD_UNCHANGED)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        depth_img = cv2.imread(pjoin(depth_dir, f"depth{f_i:06d}.png"), cv2.IMREAD_UNCHANGED)
        depth_scaled = depth_img.astype(np.float32) / depth_scale
        valid_d_mask = (depth_scaled > 0.0) & (depth_scaled < ray_cast_max_depth)

        pose = poses_list[f_i]

        # local instance map
        inst_seg_f = pjoin(pano_seg_dir, f"frame{f_i:06d}.png")
        inst_seg = cv2.imread(inst_seg_f, cv2.IMREAD_UNCHANGED).astype(np.int32)

        # global instance map
        glo_inst_map_f = pjoin(glo_inst_map_dir, f"frame{f_i:06d}.png")
        glo_inst_map = cv2.imread(glo_inst_map_f, cv2.IMREAD_UNCHANGED).astype(np.int32)

        if not select_by_vis:
            depth_scaled[~valid_d_mask] = 0.0
            points_map = cv2.rgbd.depthTo3d(depth_scaled, K_depth)


        observed_inst_ids = np.unique(glo_inst_map)
        for glo_inst_id in observed_inst_ids:
            if glo_inst_id == 0:
                continue
            
            s_t = time.time()
            glo_inst_mask = (glo_inst_map == glo_inst_id)  # (H, W)
            glo_inst_area = np.count_nonzero(glo_inst_mask)
            # 1. skip too small instance reconstructed in the global map
            if glo_inst_area < 0.5 * vis_area_thres:
                # logging.warning(f"[Skip] inst {glo_inst_id} with inst area {glo_inst_area}.")
                continue

            # NOTE find the majotiry panoptic id in the mask area
            pano_id_map = inst_seg[glo_inst_mask]
            pano_ids, pano_id_count = np.unique(pano_id_map, return_counts=True)
            pano_id = pano_ids[np.argmax(pano_id_count)]

            # this only consider the mask from the panoptic segments
            pano_mask = (inst_seg == pano_id)  # (H, W)
            pano_id_area = np.count_nonzero(pano_mask) 

            # count the pixel in the overlap area
            overlap_area = np.max(pano_id_count)

            if pano_id_area < vis_area_thres:
                # logging.warning(f"[Skip] inst {glo_inst_id} with pano area {pano_id_area}.")
                continue

            if glo_inst_id not in inst_dict.keys():
                cur_inst_info = {
                    'frame_id': [], 'feat': [], 'pose': [],
                    'vis_area': [], 'box_2d': [], 
                }
            else:
                cur_inst_info = inst_dict[glo_inst_id]

            if select_combine:
                inst_d_mask = np.logical_and(glo_inst_mask, valid_d_mask)  # (H, W)
                if np.count_nonzero(inst_d_mask) < 0.1* vis_area_thres:
                    # logging.warning(f"[Skip] Not enough valid depth pts in inst {glo_inst_id}.")
                    continue
                inst_points = points_map[inst_d_mask].astype(np.float32).reshape(-1,3)
                inst_points = inst_points @ pose[:3,:3].T + pose[:3,3:4].T

                # 4.1. Init an object center by estimating the BBOX
                if 'view_map' not in cur_inst_info.keys():
                    # this will add the 3d bbox and the view map into the cur_inst_info
                    success, cur_inst_info = init_view_cov(
                        depth_scaled, inst_d_mask, inst_points, 
                        glo_inst_id, cur_inst_info, 
                        vis_area_thres, sph_grid_size
                    )
                    if not success:
                        continue
                
                # 4.2. Map the rays from obj_c to surface to the spherical grid
                view_map = cur_inst_info['view_map']
                success, view_map = update_view_cov_map(
                    inst_points, cur_inst_info['bbox_c'], 
                    view_map, sph_grid_size, view_overlap_ratio_thres
                )
                # decide whether to select this view based on the view coverage
                if not success:
                    continue # not novel enough?

                # decide whether to select this view based on the vis_area
                past_vis_areas = cur_inst_info['vis_area']
                if len(past_vis_areas) >= max_top_vis:
                    top_vis_s = np.sort(np.array(cur_inst_info['vis_area']))[-max_top_vis:]
                    if overlap_area <= np.min(top_vis_s):                            
                        continue # not visible enough

                cur_inst_info['view_map'] = view_map

            elif select_by_vis:
                past_vis_areas = cur_inst_info['vis_area']
                if len(past_vis_areas) >= max_top_vis:
                    top_vis_s = np.sort(np.array(cur_inst_info['vis_area']))[-max_top_vis:]
                    if overlap_area <= np.min(top_vis_s):
                        continue # not visible enough

            judge_t = time.time() - s_t
            judge_time_list.append(judge_t)


            # =================================== 
            inst_dict[glo_inst_id] = cur_inst_info

            yxs = yx_grid[:, glo_inst_mask] # (2, M)
            y1, x1 = np.min(yxs[0]), np.min(yxs[1])
            y2, x2 = np.max(yxs[0]), np.max(yxs[1])

            obj_mask = np.logical_or(pano_mask, glo_inst_mask)

            roi_feat, process_time = vl_model.encode_image_with_bbox(
                rgb_img, obj_mask, (x1, y1, x2, y2)
            )
            vlm_time_list.append(process_time)

            inst_dict[glo_inst_id]['frame_id'].append(f_i)
            inst_dict[glo_inst_id]['box_2d'].append((x1, y1, x2, y2))
            inst_dict[glo_inst_id]['feat'].append(roi_feat)
            inst_dict[glo_inst_id]['vis_area'].append(overlap_area)

    logging.info(f"Max VMem: {torch.cuda.max_memory_allocated() / 1024**3:.6f} GB")

    inst_sem_dict = {}
    for glo_inst_id in inst_dict.keys():
        cur_inst_info = inst_dict[glo_inst_id]
        inst_frames = cur_inst_info['frame_id']
        obs_num = len(inst_frames)
        if obs_num == 0:
            continue
        inst_bbox2ds = cur_inst_info['box_2d']
        vis_scores = np.array(cur_inst_info['vis_area'])

        all_feats = np.array(cur_inst_info['feat'])

        # if select_by_vis or select_combine:
        # select the top-k with max vis area
        top_indices = np.argsort(vis_scores)[-max_top_vis:]
        all_feats = all_feats[top_indices]
        inst_frames = [inst_frames[i] for i in top_indices]
        inst_bbox2ds = [inst_bbox2ds[i] for i in top_indices]
        vis_scores = vis_scores[top_indices]

        inst_sem_dict[glo_inst_id] = {
            'feat': all_feats, #feat_inst,
            'vis_area': vis_scores,
            'frame_id': inst_frames,
            'box_2d': inst_bbox2ds,
            'color': glo_inst_color[glo_inst_id],
        }

    with open(inst_sem_f, 'wb') as f:
        pickle.dump(inst_sem_dict, f)

    vlm_time_list = np.array(vlm_time_list)
    # t_per_query = np.mean(vlm_time_list)
    t_per_frame = np.sum(vlm_time_list) / (num_frame / step)
    # logging.info(f"Avg time per query: {t_per_query:.4f} s")
    logging.info(f"Avg vlm time per frame: {t_per_frame:.4f} s")

    judge_time_list = np.array(judge_time_list)
    t_per_frame = np.sum(judge_time_list) / (num_frame / step)
    logging.info(f"Avg judge time per frame: {t_per_frame:.4f} s")


if __name__ == '__main__':
    # scene_list = [
    #     'office0', 'office1', 'office2', 'office3', 'office4', 
    # ]
    # scene_list = ['office0']
    scene_list = [
        # 'office0', 'office1', 'office2', 'office3', 
        'office4', 'room0', 'room1', 'room2',
    ]
    for scene_name in scene_list:
        main(scene_name)