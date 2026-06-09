import os, sys, time, argparse, pickle, logging, copy
from os.path import join as pjoin
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import torch
# from scipy.spatial.transform import Rotation as R
# import open3d as o3d
# import matplotlib.pyplot as plt

# self packages
from utils.common_utils import Segment, PointCloudProcessor
from utils.data_loaders import ScannetLoader, ReplicaLoader
from visualizations.vis_utils import vis_id_map

from vl_models import VLModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FORMAT = '%(asctime)s.%(msecs)06d %(levelname)-8s: [%(filename)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt='%H:%M:%S')


def parse_args():
    parse = argparse.ArgumentParser(description='Semantic Mapping-Python') 
    # dataset 
    parse.add_argument("--dataset", type=str, default="scenenn", 
        help="which data set to use, scenenn or scannet ")

    # files path
    parse.add_argument("--scene_num", type=str, required=True, 
        help="which scene for mapping ")
    parse.add_argument("--result_folder", type=str,required=True, 
        help="folder of mapping results")
    parse.add_argument("--data_folder", type=str, required=True, 
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
    parse.add_argument("--num_threads", type=int, default=-1, 
        help="threads to use")
    parse.add_argument("--debug", action="store_true", 
        help="whether to use visualization ")
    parse.add_argument("--preload", action="store_true", 
        help="whether to preload images ")

    parse.add_argument("--use_temp_results", action='store_true', 
        help="use 2D segments results ")
    parse.add_argument("--save_temp_results", action='store_true', 
        help="save intermediate 2D segments results ")
    parse.add_argument("--save_temp_img", action='store_true', 
        help="save intermediate results in images ")
    parse.add_argument("--intermediate_seg_folder", type=str, default='segments', 
        help="folder to save intermediate segments result ")

    parse.add_argument("--use_temp_panoptics", action='store_true', 
        help="use 2D panoptic segments ")
    parse.add_argument("--save_temp_panoptics", action='store_true', 
        help="save 2D panoptic segments ")
    parse.add_argument("--temp_panoptics_folder", type=str, default='segments', 
        help="folder to save 2D panoptic segments ")

    parse.add_argument("--use_temp_geometrics", action='store_true', 
        help="use 2D geometrics segments ")
    parse.add_argument("--save_temp_geometrics", action='store_true', 
        help="save 2D geometrics segments ")
    parse.add_argument("--temp_geometrics_folder", type=str, default='segments', 
        help="folder to save 2D geometrics segments result ")

    parse.add_argument("--task", type=str, default="coco80", 
        help="coco80; nyu13; Nyu40; cocoPano")
 
    parse.add_argument("--data_association", type=int, default=0, 
        help="0 - Ori; 1 - SemMerge; \
        2 - SemMerge+BackgroundMerge+only consider size>1000; \
        3 - SemMerge+BackgroundMerge+only consider size>1000 + consider Sem when register superpoint; \
        4 - no merging; \
        5 - using designated superpoint id for 3D segments")
    parse.add_argument("--inst_association", type=int, default=0, 
        help="0 for Ori; 1 for Label-Sem-Inst; \
        2 for Label-Inst-Sem; 3 for SegGraph ")


    # for SegGraph
    parse.add_argument("--seg_graph_confidence", type=int, default=0, 
        help="0 for all confidence as 1; \
            1 for using inst score; \
            2 for use inst score and overlap ratio; \
            3 for use inst score, overlap ratio and geometric confidence")
    parse.add_argument("--use_inst_label_connect", type=int, default=1, 
        help="")
    parse.add_argument("--connection_ratio_th", type=float, default=0.2, 
        help="")
    parse.add_argument("--test_geometric_confidence", action='store_true', 
        help="try test geometric confidence calculation")

    # NOTE currently not used
    parse.add_argument("--use_2D_confidence", action='store_true', 
        help="use MaskRCNN for 2D data association")
    parse.add_argument("--geo_confidence", action='store_true',
        help="use geometric confidence")
    parse.add_argument("--label_confidence", action='store_true', 
        help="use label confidence")  
    return parse.parse_args()


def make_res_dirs(args):
    res_dirs = {}

    result_root_dir = args.result_folder

    res_dirs['folder'] = result_root_dir
    res_dirs['log'] = os.path.join(result_root_dir, 'log')
    res_dirs['temp_segs'] = args.intermediate_seg_folder
    res_dirs['temp_panoptics'] = args.temp_panoptics_folder
    res_dirs['temp_geometrics'] = args.temp_geometrics_folder

    if not os.path.exists(res_dirs['folder']):
        os.makedirs(res_dirs['folder'])
    if not os.path.exists(res_dirs['log']):
        os.makedirs(res_dirs['log'])

    # Currently we can only use pre-processed panoptic segments
    assert os.path.exists(res_dirs['temp_panoptics']), f"[Error]Temp panoptic folder: {res_dirs['temp_panoptics']} does not exist!"

    if args.use_temp_results:
        assert os.path.exists(res_dirs['temp_segs']), f"[Error]Temp panoptic folder: {res_dirs['temp_segs']} does not exist!"
    if args.save_temp_results and not os.path.exists(res_dirs['temp_segs']):
        os.makedirs(res_dirs['temp_segs'])

    if args.use_temp_geometrics:
        assert os.path.exists(res_dirs['temp_geometrics']), f"[Error]Temp geometrics folder: {res_dirs['temp_geometrics']} does not exist!"
    if args.save_temp_geometrics and not os.path.exists(res_dirs['temp_geometrics']):
        os.makedirs(res_dirs['temp_geometrics'])
    
    return res_dirs


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
    

def main(args):

    import consistent_gsm # type: ignore
    import depth_segmentation_py # type: ignore

    # dataset 
    dataset = args.dataset
    panoptic_node = None
    
    if dataset == "scannet_nyu":
        from utils.common_scannet_nyu import SegmentsGenerator
    elif dataset == "replica":
        from utils.common_scannet_nyu import SegmentsGenerator
    else:
        logging.error("Please choose a suitable dataset!")
        raise NotImplementedError
    
    # set configuration
    use_temp_results = args.use_temp_results
    save_segments = args.save_temp_results
    use_geos = args.use_temp_geometrics
    save_geos = args.save_temp_geometrics
    save_results_img = args.save_temp_img

    task = args.task
    use_geo_confidence = args.geo_confidence
    use_label_confidence = args.label_confidence
    inst_association = args.inst_association
    data_association = args.data_association

    seg_graph_confidence = args.seg_graph_confidence
    use_inst_label_connect = args.use_inst_label_connect
    connection_ratio_th = args.connection_ratio_th

    # input and output configuration
    scene_num = args.scene_num
    result_folder = args.result_folder
    data_path = args.data_folder
    scene_folder = pjoin(data_path, scene_num)

    result_dirs = make_res_dirs(args)

    temp_segs_folder = result_dirs['temp_segs']
    temp_pano_folder = result_dirs['temp_panoptics']
    temp_geos_folder = result_dirs['temp_geometrics']

    log_info = args.log

    # DataLoader
    if dataset == "scannet_nyu":
        data_loader = ScannetLoader(
            scene_folder, args.preload, args.preload)
    elif dataset == "replica":
        data_loader = ReplicaLoader(
            scene_folder, args.preload, args.preload)

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

    num_threads = args.num_threads

    # for scannet exp with limited observations
    # iters = 160
    # end = iters * step + start

    # ========================== initialized integrator ==========================
    log_file = os.path.abspath(result_dirs['log'])
    gsm_node = consistent_gsm.GlobalSegmentMap_py(
        log_file, task, 
        use_geo_confidence, 
        use_label_confidence, 
        inst_association, data_association, 
        num_threads, args.debug, 
        seg_graph_confidence, 
        use_inst_label_connect==1, 
        connection_ratio_th, 
        0.9 # cos_sim_th, not used
    )
    gsm_node.outputLog(log_info)

    # ==========================================================
    # TODO set flag for complete open-set segmentation
    VLM_name = 'siglip-l-16-384'
    exp_results = pjoin(result_dirs['folder'], 'cropformer_inst')
    # temp_feats = pjoin(result_dirs['folder'], 'cropformer_inst', 'temp_feats')
    # use_prev_feat = False

    # NOTE they are mutually exclusive
    select_by_vis = False
    select_by_viewcov = False
    select_combine = True

    # incre_vis | 
    inst_sem_name = f'inst_sem_{VLM_name}_{int(np.ceil((end-start)/step))}_incre_combine.pkl'

    # ==========================================================

    if not os.path.exists(exp_results):
        os.makedirs(exp_results)
    # if not os.path.exists(temp_feats):
    #     os.makedirs(temp_feats)

    logging.info("Using pre-processed instance seg data!")
    if not use_geos:
        dep_segmentor = depth_segmentation_py.DepthSegmentation_py(
            H_depth,W_depth,cv2.CV_32FC1, K_depth
        )
    else:
        logging.info("Using pre-processed depth seg data!")
        dep_segmentor = None
            
    
    # create the segment generator
    segments_generator = SegmentsGenerator(
        gsm_node, dep_segmentor, panoptic_node,
        save_results_img, result_dirs['folder'], 
        save_segments, use_temp_results, temp_segs_folder, 
        panoptics_folder=temp_pano_folder, 
        save_geometrics=save_geos, geometrics_folder=temp_geos_folder
    )

    # Create the VL model for language-aligned image feature extraction
    vl_model = VLModel(model_name=VLM_name, img_size=(H_depth, W_depth), device=DEVICE)
    logging.info(f"VL model {VLM_name} initialized!")

    ray_cast_max_depth = 50.0
    # minimum visible area in the pano seg for an instance to be considered
    vis_area_thres = 1000
    max_top_vis = 8
    if select_by_viewcov:
        view_overlap_ratio_thres = 0.85
    elif select_combine:
        view_overlap_ratio_thres = 0.9

    gsm_node.initializeCameraRayCaster(
        K_depth, H_depth, W_depth, 0.01, ray_cast_max_depth, num_threads
    )

    inst_dict = {}
    yx_grid = np.mgrid[0:H_depth, 0:W_depth] # (2, H, W)
    sph_grid_size = (int(H_depth/8), int(W_depth/8))
    # theta_phi_grid = np.mgrid[0:sph_grid_size[0], 0:sph_grid_size[1]]


    

    # ========================== start mapping ==========================
    time_s = time.time()
    frame_ids = range(start, end, step)
    for f_i in tqdm(frame_ids):
        # NOTE img is in BGR, depth is in meters, pose is c2w
        # rgb image has been wrapped to depth image
        rgb_img, depth_scaled, pose = data_loader.getDataFromIndex(f_i)
        # check data validity
        if(rgb_img is None or depth_scaled is None or pose is None):
            logging.warning(f"[Skip] frame {f_i} is lack of RGB / Depth / Pose.")
            continue
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        valid_d_mask = (depth_scaled > 0.0) & (depth_scaled < ray_cast_max_depth)
        

        # NOTE use instance segs from cropformer
        inst_seg_f = pjoin(temp_pano_folder, 'cropformer', 
            os.path.basename(data_loader.rgb_path_map[f_i]).split('.')[0] + '.png'
        )
        inst_seg = cv2.imread(inst_seg_f, cv2.IMREAD_UNCHANGED)
        if data_loader.mapRGBtoDepth is not None:
            inst_seg = data_loader.mapRGBtoDepth(inst_seg)

        # pybind11 only accepts not unsigned int
        inst_seg = inst_seg.astype(np.int32)

        # perform depth segmentation first
        if dep_segmentor is not None:
            segments_generator.SegmentDepth(depth_scaled, rgb_img, f_i)

        segment_list: list[Segment] = segments_generator.frameToSegmentsCropFormer(
            depth_scaled, K_depth, pose, f_i, inst_seg
        )

        if len(segment_list) == 0:
            logging.warning(f"[Skip] No segment found in frame {f_i}")
            continue
        
        for segment in segment_list:
            if(seg_graph_confidence == 3):
                segment.calculateBBox()

            # all segs except bg are 'thing' and with a 1.0 inst_conf
            gsm_node.insertSegmentsOpen(
                segment.points, segment.box_points, 
                segment.instance_label, segment.class_label, 
                segment.sem_feat, 
                segment.inst_confidence, segment.overlap_ratio, 
                pose, segment.is_thing, segment.segment_label
            )
    
        # update the global segment map with the new segments
        gsm_node.integrateFrame()

        # Ray casting to get the global instance map from the super points
        glo_inst_map = gsm_node.raycastInstancePredictions(
            pose, inst_seg, depth_scaled
        )

        if not select_by_vis:
            depth_scaled[~valid_d_mask] = 0.0
            points_map = cv2.rgbd.depthTo3d(depth_scaled, K_depth)

        # ################# View Selection #################
        observed_inst_ids = np.unique(glo_inst_map)
        for glo_inst_id in observed_inst_ids:
            if glo_inst_id == 0:
                continue
            
            glo_inst_mask = (glo_inst_map == glo_inst_id)  # (H, W)
            glo_inst_area = np.count_nonzero(glo_inst_mask)
            # 1. skip too small instance reconstructed in the global map
            if glo_inst_area < 0.5 * vis_area_thres:
                logging.warning(f"[Skip] inst {glo_inst_id} with inst area {glo_inst_area}.")
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
            overlap_mask = np.logical_and(glo_inst_mask, pano_mask)

            # 2. check the visibility of the instance
            if pano_id_area < vis_area_thres:
                logging.warning(f"[Skip] inst {glo_inst_id} with pano area {pano_id_area}.")
                continue

            # overlap_ratio = dice_coeff(glo_inst_area, pano_id_area, overlap_area)
            # 3. NOTE if smaller, it means the pano seg is oversegmented
            # we allow the global instance to be under-segemented
            # pano_in_inst_ratio = overlap_area / glo_inst_area
            # if pano_in_inst_ratio < 0.5:
            #     logging.warning(f"[Skip] inst {glo_inst_id} with overlap ratio {pano_in_inst_ratio:.4f} < 0.5")
            #     continue

            if glo_inst_id not in inst_dict.keys():
                cur_inst_info = {
                    'frame_id': [], 'feat': [], 'pose': [],
                    'vis_area': [], 'box_2d': [], 
                }
            else:
                cur_inst_info = inst_dict[glo_inst_id]

            # 4. check if it's a good new view to select
            if select_by_viewcov or select_combine:
                inst_d_mask = np.logical_and(glo_inst_mask, valid_d_mask)  # (H, W)
                if np.count_nonzero(inst_d_mask) < 0.1* vis_area_thres:
                    logging.warning(f"[Skip] Not enough valid depth pts in inst {glo_inst_id}.")
                    continue
                inst_points = points_map[inst_d_mask].astype(np.float32).reshape(-1,3)
                inst_points = inst_points @ pose[:3,:3].T + pose[:3,3:4].T

                # 4.1. Init an object center by estimating the BBOX
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
                if select_combine:
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


            inst_dict[glo_inst_id] = cur_inst_info


            # ====================================================================
            # 5. Get visual feature from the VL model
            # detrermine the cropping area by the global inst mask
            yxs = yx_grid[:, glo_inst_mask] # (2, M)
            y1, x1 = np.min(yxs[0]), np.min(yxs[1])
            y2, x2 = np.max(yxs[0]), np.max(yxs[1])
            
            # vis_feat_name = f"{VLM_name}_F_{f_i}_{x1}-{y1}-{x2-x1}-{y2-y1}.npy"
            # vis_feat_path = pjoin(temp_feats, vis_feat_name)
            # if use_prev_feat and os.path.exists(vis_feat_path):
            #     roi_feat = np.load(vis_feat_path) # load the feature from the file
            # else:
            # masked by the union
            obj_mask = np.logical_or(pano_mask, glo_inst_mask)
            roi_feat = vl_model.encode_image_with_bbox(
                rgb_img, obj_mask, (x1, y1, x2, y2)
            )
            # np.save(vis_feat_path, roi_feat)

            # 6. Add the instance to the buffer
            inst_dict[glo_inst_id]['frame_id'].append(f_i)
            inst_dict[glo_inst_id]['box_2d'].append((x1, y1, x2, y2))
            inst_dict[glo_inst_id]['feat'].append(roi_feat)
            inst_dict[glo_inst_id]['pose'].append(pose)
            inst_dict[glo_inst_id]['vis_area'].append(overlap_area)
            # inst_dict[glo_inst_id]['overlap_ratio'].append(pano_in_inst_ratio)
                
        # clean idle mem
        gsm_node.clearTemporaryMemory()

    logging.info(f"Max VMem: {torch.cuda.max_memory_allocated() / 1024**3:.6f} GB")
    gsm_node.outputLog(f"Time taken per frame: {(time.time() - time_s)/len(frame_ids):.2f} seconds")

    # generate log and mesh
    gsm_node.LogLabelInformation()
    gsm_node.LogMeshColors(exp_results)

    logging.info("Start mesh generation!")
    # flags: label_mesh, sem_mesh, inst_mesh
    gsm_node.generateMesh(
        exp_results, str(int(np.ceil((end-start)/step))), 
        False, False, True
    )

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
        vis_scores = np.array(cur_inst_info['vis_area'])

        all_feats = np.array(cur_inst_info['feat'])
        total_query += all_feats.shape[0]

        if select_by_viewcov:
            view_map = cur_inst_info['view_map']
            occ_cnt = np.count_nonzero(view_map > 0)
            occ_ratio = occ_cnt / (sph_grid_size[0] * sph_grid_size[1])
            gsm_node.outputLog(f"Instance {glo_inst_id} has {occ_ratio*100:.2f}% observed from {obs_num} frames")
        elif select_by_vis or select_combine:
            # select the top-k with max vis area
            top_indices = np.argsort(vis_scores)[-max_top_vis:] # indices for values sorted in ascending order
            all_feats = all_feats[top_indices]
            inst_frames = [inst_frames[i] for i in top_indices]
            inst_poses = [inst_poses[i] for i in top_indices]
            inst_bbox2ds = [inst_bbox2ds[i] for i in top_indices]
            vis_scores = vis_scores[top_indices]
        
        inst_color = gsm_node.getInstanceColor(glo_inst_id)

        inst_sem_dict[glo_inst_id] = {
            'feat': all_feats,
            'vis_area': vis_scores,
            'frame_id': inst_frames,
            'pose': inst_poses,
            'box_2d': inst_bbox2ds,
            'color': inst_color,
        }
        if select_by_viewcov:
            inst_sem_dict[glo_inst_id]['bbox_c'] = cur_inst_info['bbox_c']
            inst_sem_dict[glo_inst_id]['bbox_s'] = cur_inst_info['bbox_s']
            inst_sem_dict[glo_inst_id]['bbox_rot'] = cur_inst_info['bbox_rot']

    gsm_node.outputLog(f"Totally {len(inst_sem_dict)} instances with {total_query} queries, avg {total_query / len(inst_sem_dict):.2f}.")

    # dump results
    inst_sem_f = pjoin(exp_results, inst_sem_name)
    with open(inst_sem_f, 'wb') as f:
        pickle.dump(inst_sem_dict, f)
    logging.info(f"Saved instance semantic features to {inst_sem_f}")

    

if __name__=="__main__":
    # set files path
    args = parse_args()
    main(args)