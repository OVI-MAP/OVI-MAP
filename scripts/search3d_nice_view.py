import os, sys, time, argparse, pickle, logging
from os.path import join as pjoin
from tqdm import tqdm

import cv2, copy
from PIL import Image
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import open3d.core as o3c

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# self packages
from utils.common_utils import PointCloudProcessor
from utils.data_loaders import ScannetLoader, SceneNNLoader

from vl_models import VLModel
from search3d_incremental import RayCastScene

from scripts.utils.vis_utils import vis_id_map

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FORMAT = '%(asctime)s.%(msecs)06d %(levelname)-8s: [%(filename)s] %(message)s'
# logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt='%H:%M:%S')



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
    parse.add_argument("--gt_mask_folder", type=str, default="", 
        help="valid when dataset is nyu gt, path to g.t. mask annotations")

    # log
    parse.add_argument("--log", type=str, default='', 
        help="log info ")
    # mapping configuration
    parse.add_argument("--start", type=int, default=0, help="?")
    parse.add_argument("--end", type=int, default=800, help="?")
    parse.add_argument("--step", type=int, default=5, 
        help="use one frame for integration every n_step frames ")
    
    return parse.parse_args()


def main(args):
    dataset = args.dataset
    # input and output configuration
    scene_num = args.scene_num
    
    data_path = args.data_folder
    scene_folder = pjoin(data_path, scene_num)

    result_folder = pjoin(args.result_folder, scene_num, 'search3d')
    os.makedirs(result_folder, exist_ok=True)

    log_info = args.log

    # DataLoader
    traj_filename = args.traj_filename
    if dataset == "scenenn":
        data_loader = SceneNNLoader(scene_folder, traj_filename)
    elif dataset == "scannet" or dataset == "scannet_nyu" \
        or dataset == "scannet_nyu_gt":
        data_loader = ScannetLoader(scene_folder, traj_filename)

    _a,depth_img,_b = data_loader.getDataFromIndex(1)
    H = depth_img.shape[0]
    W = depth_img.shape[1]
    print(f"Depth image size: {W}x{H}")
    K_depth = data_loader.getCameraMatrix()

    start = args.start
    assert (start >= data_loader.indexes[0])
    end = args.end
    if end < 0:
        end = data_loader.index_max
    step = args.step


    # Import the complete instance map for testing
    inst_map_path = '/home/zilong/Disk_data/semantic_mapping_result/scene0001_00/cropformer_inst_res/instance_mesh_160.ply'
    raycast_scene = RayCastScene(inst_map_path)
    
    yx_grid = raycast_scene.init_camera((
        K_depth[0][0], K_depth[1][1], K_depth[0][2], K_depth[1][2], H, W))

    inst_ids = raycast_scene.inst_ids
    num_inst = len(inst_ids)

    # Create the VL model for language-aligned image feature extraction
    vl_model_name = 'siglip-l-16-384'
    vl_model = VLModel(model_name=vl_model_name, img_size=(H, W), device=DEVICE)

    sph_coord_size = (int(H/8), int(W/8))
    theta_phi_grid = np.mgrid[0:sph_coord_size[0], 0:sph_coord_size[1]]
    # For keep track of each instance
    inst_dict = {
        inst_id: {
            'frame_id': [],
            'feat': [],
            'vis_score': [],
            'pose': [], 
            'view_map': np.zeros(sph_coord_size, dtype=np.uint8)
        } for inst_id in inst_ids
    }

    vis_area_thres = 4000 
    overlap_ratio_thres = 0.85

    import rerun as rr
    from scipy.spatial.transform import Rotation as R
    rr.init("nice_view", recording_id="nice_view", spawn=True)

    time_s = time.time()
    frame_ids = range(start, end, step)
    for f_i in tqdm(frame_ids):
        # the return pose is cam2world
        rgb_img, depth_img, pose = data_loader.getDataFromIndex(f_i)
        # check data validity
        if(rgb_img is None or depth_img is None or pose is None):
            logging.info(f"[Error] Skipping frame {f_i}")
            continue
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
        # get 2D instance map from ray casting
        inst_map, dist_hit = raycast_scene.raycast_to_instances2D(pose)
        inst_list_frame = np.unique(inst_map)

        depth_scaled = dist_hit
        points_map = cv2.rgbd.depthTo3d(depth_scaled, K_depth)
        
        valid_d_mask = (depth_scaled > 0.0) & (depth_scaled < 30.0)

        # vis_id_map(inst_map, f'/home/zilong/Downloads/nice_view_{f_i}.png')

        for inst_id in inst_list_frame:
            if inst_id == 0:
                continue

            # only keep the instance with enough pixels
            inst_mask = (inst_map == inst_id)  # (H, W)
            inst_id -= 1  # make it zero-based index

            # 1. check the visibility of the instance
            vis_area = np.count_nonzero(inst_mask)
            if vis_area < vis_area_thres:
                continue

            inst_d_mask = np.logical_and(inst_mask, valid_d_mask)  # (H, W)
            inst_points = points_map[inst_d_mask].astype(np.float32).reshape(-1,3)
            inst_points = inst_points @ pose[:3,:3].T + pose[:3,3:4].T # tranform to world frame

            # 1.5 initialize the object center at the first good frame
            if len(inst_dict[inst_id]['frame_id']) == 0 or \
                vis_area > 2.0 * inst_dict[inst_id]['vis_score']:

                depth_arr = depth_img[inst_d_mask]
                if len(depth_arr) < 0.5 * vis_area_thres:
                    print(f"[Skip] Not enough depth pts to init inst {inst_id} in frame {f_i}")
                    continue

                if len(inst_dict[inst_id]['frame_id']) > 0:
                    print(f"Re-initialize inst {inst_id} in frame {f_i}")
                    # update the view coverage map
                    prev_view_map = inst_dict[inst_id]['view_map']
                    sph_coord_arr = theta_phi_grid[:, prev_view_map > 0]
                    theta = sph_coord_arr[0, :] * np.pi / sph_coord_size[0]
                    phi = sph_coord_arr[1, :] * 2*np.pi / sph_coord_size[1] - np.pi
                    x = np.sin(theta) * np.cos(phi)
                    y = np.sin(theta) * np.sin(phi)
                    z = np.cos(theta)
                    prev_rays = np.stack((x, y, z), axis=-1)
                    prev_surface_pts = inst_dict[inst_id]['bbox_c'] + prev_rays

                PCLprocessor = PointCloudProcessor(inst_points)
                bbox_res = PCLprocessor.process(voxel_size=0.02)
                bbox_pos, bbox_extent, bbox_rot_matrix = bbox_res
                inst_dict[inst_id]['bbox_c'] = bbox_pos
                inst_dict[inst_id]['bbox_s'] = bbox_extent
                inst_dict[inst_id]['bbox_rot'] = np.array(bbox_rot_matrix)

                if len(inst_dict[inst_id]['frame_id']) > 0:
                    print(f"Updating the view map for inst {inst_id}")
                    new_view_map = np.zeros_like(prev_view_map)
                    # update the view map with the new object center
                    new_rays = prev_surface_pts - inst_dict[inst_id]['bbox_c'].T
                    new_rays = new_rays / np.linalg.norm(new_rays, axis=-1, keepdims=True)
                    # convert them to sph coord and map to the occ grid
                    theta = np.arccos(np.clip(new_rays[:,2], -1.0, 1.0))
                    theta = theta / np.pi * sph_coord_size[0]
                    theta = np.floor(theta).astype(np.int32)
                    phi = np.arctan2(new_rays[:,1], new_rays[:,0])
                    phi = (phi + np.pi) / (2 * np.pi) * sph_coord_size[1]
                    phi = np.floor(phi).astype(np.int32)
                    sph_coords = np.stack((theta, phi), axis=1)
                    sph_coords = np.unique(sph_coords, axis=0)
                    new_view_map[sph_coords[:, 0], sph_coords[:, 1]] = 1
                    inst_dict[inst_id]['view_map'] = new_view_map

                inst_dict[inst_id]['vis_score'] = vis_area

                rr.set_time_seconds("stable_time", f_i)
                rr.log(f"mesh/{inst_id}_{f_i}",
                    rr.Points3D(
                        positions=inst_points,
                        colors=raycast_scene.inst_colors[inst_id].tolist(),
                        radii=0.01
                    )
                )
                rr.log(f"bbox/{inst_id}_{f_i}",
                    rr.Boxes3D(
                        centers=bbox_pos, 
                        half_sizes=bbox_extent / 2,
                        quaternions=R.from_matrix(inst_dict[inst_id]['bbox_rot']).as_quat().tolist(),
                        colors=[(255, 0, 0)], 
                        fill_mode='MajorWireframe'
                    )
                )

            # 2. get the ray from object center to hit object surface
            obj_rays = inst_points - inst_dict[inst_id]['bbox_c'].T
            obj_rays = obj_rays / np.linalg.norm(obj_rays, axis=-1, keepdims=True)
            # convert them to sph coord and map to the occ grid
            theta = np.arccos(np.clip(obj_rays[:,2], -1.0, 1.0))  # in [0, pi]
            theta = theta / np.pi * sph_coord_size[0]  # map to [0, H/8]
            theta = np.floor(theta).astype(np.int32)
            phi = np.arctan2(obj_rays[:,1], obj_rays[:,0])  # in [-pi, pi]
            phi = (phi + np.pi) / (2 * np.pi) * sph_coord_size[1]  # map to [0, W/8]
            phi = np.floor(phi).astype(np.int32)


            # 3. check the overlapping with the existing views
            view_map = inst_dict[inst_id]['view_map']
            # remove repeated views
            sph_coords = np.stack((theta, phi), axis=1)  # (M, 2)
            sph_coords = np.unique(sph_coords, axis=0)  # remove duplicates
            overlap_area = np.count_nonzero(
                view_map[sph_coords[:, 0], sph_coords[:, 1]] > 0)
            overlap_ratio = overlap_area / sph_coords.shape[0]
            if overlap_ratio > overlap_ratio_thres:
                # too many overlapping views, skip this instance
                continue
            
            print(f"Add inst {inst_id} in frame {f_i} with sph overlap ratio {overlap_ratio:.2f}")
            # update the view map
            view_map[sph_coords[:, 0], sph_coords[:, 1]] = 1
            inst_dict[inst_id]['view_map'] = view_map
 
            # 4. Extract the features
            yxs = yx_grid[:, inst_mask] # (2, M)
            y1, x1 = np.min(yxs[0]), np.min(yxs[1])
            y2, x2 = np.max(yxs[0]), np.max(yxs[1])
            roi_feat = vl_model.encode_image_with_bbox(
                rgb_img, inst_mask, (x1, y1, x2, y2)
            )

            # 5. Add the instance to the buffer
            inst_dict[inst_id]['frame_id'].append(f_i)
            inst_dict[inst_id]['feat'].append(roi_feat)

        # save the view_map of instance #??
        cv2.imwrite(pjoin(result_folder, f'view_map_inst9_{f_i:04d}.png'), 
            (inst_dict[9]['view_map'] * 255).astype(np.uint8))
        
    logging.info(f"Time taken per frame: {(time.time() - time_s)/len(frame_ids):.2f} seconds")

    # average the features
    inst_sem_dict = {}
    for inst_id in inst_ids:
        obs_num = len(inst_dict[inst_id]['frame_id'])
        if obs_num == 0:
            continue

        all_feats = np.array(inst_dict[inst_id]['feat'])
        feat_inst = np.mean(all_feats, axis=0)

        view_map = inst_dict[inst_id]['view_map']
        occ_cnt = np.count_nonzero(view_map > 0)
        occ_ratio = occ_cnt / (sph_coord_size[0] * sph_coord_size[1])
        print(f"Instance {inst_id} has {occ_ratio*100:.2f}% observed from {obs_num} frames")

        inst_color = raycast_scene.inst_colors[inst_id]
        inst_sem_dict[inst_id] = {
            'feat': feat_inst,
            'frame_id': inst_dict[inst_id]['frame_id'],
            'color': inst_color, 
            'bbox_c': inst_dict[inst_id]['bbox_c'],
            'bbox_s': inst_dict[inst_id]['bbox_s'],
            'bbox_rot': inst_dict[inst_id]['bbox_rot'],
        }
    inst_sem_f = pjoin(result_folder, f'sph_map_inst_sem_{vl_model_name}.pkl')
    with open(inst_sem_f, 'wb') as f:
        pickle.dump(inst_sem_dict, f)

    # # select nice views and average the features
    # for inst_id in inst_ids:
    #     vis_scores = np.array(inst_dict[inst_id]['vis_score'])
    #     # get the top 10 views with the highest visibility scores
    #     top_indices = np.argsort(vis_scores)[-10:]
    #     top_feats = np.array(inst_dict[inst_id]['feat'])[top_indices]

    #     if len(top_feats) == 0:
    #         continue

    #     inst_feats[inst_id] = np.mean(top_feats, axis=0)



if __name__=="__main__":
    # set files path
    args = parse_args()
    main(args)