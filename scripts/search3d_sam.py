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
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.cluster import AgglomerativeClustering

# self packages
import semantics.semantic_utils as semantic_utils
from utils.common_utils import Segment
from utils.data_loaders import ScannetLoader, SceneNNLoader

# image embedding
import clip
# from transformers import AutoProcessor, AutoModel, AutoTokenizer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FORMAT = '%(asctime)s.%(msecs)06d %(levelname)-8s: [%(filename)s] %(message)s'
# logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt='%H:%M:%S')

# python vl_segmentation.py --scene_num scene0001_00 


def parse_args():
    parse = argparse.ArgumentParser(description='Semantic Mapping-Python') 
    # dataset 
    parse.add_argument("--dataset", type=str, default="scannet_nyu", 
        help="which data set to use, scenenn or scannet ")

    # files path
    parse.add_argument("--scene_num", type=str, required=True, 
        help="which scene for mapping ")
    
    parse.add_argument("--result_folder", type=str, 
        default='/home/zilong/Disk_data/sem_seg_temp', 
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
    parse.add_argument("--start", type=int, default=0, 
        help="?")
    parse.add_argument("--end", type=int, default=800, 
        help="?")
    parse.add_argument("--step", type=int, default=5, 
        help="use one frame for integration every n_step frames ")
    parse.add_argument("--num_threads", type=int, default=10, 
        help="threads to use")
    
    return parse.parse_args()


def main(args):
    dataset = args.dataset
    # input and output configuration
    scene_num = args.scene_num
    
    data_path = args.data_folder
    scene_folder = pjoin(data_path, scene_num)

    result_folder = pjoin(args.result_folder, scene_num, 'search3d')
    os.makedirs(result_folder, exist_ok=True)

    ext_file_folder = pjoin('/home/zilong/Downloads/scans', scene_num, 'segments')

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
    num_threads = args.num_threads

    clip_model, prep_clip = clip.load('ViT-B/32', device=DEVICE)
    # siglip_model_name = "google/siglip2-base-patch16-224"
    # siglip_model_name = "google/siglip2-so400m-patch14-384"
    # siglip_model = AutoModel.from_pretrained(siglip_model_name).to(DEVICE)
    # tokenizer = AutoTokenizer.from_pretrained(siglip_model_name)
    # processor = AutoProcessor.from_pretrained(siglip_model_name, use_fast=True)

    inst_map_path = '/home/zilong/Disk_data/semantic_mapping_result/scene0001_00/original_res/instance_mesh_160.ply'
    inst_mesh = o3d.io.read_triangle_mesh(inst_map_path)
    inst_mesh.compute_vertex_normals()
    inst_mesh.compute_triangle_normals()

    vertex_colors = np.asarray(inst_mesh.vertex_colors)
    triangles = np.asarray(inst_mesh.triangles)

    # out_inst_colors = np.ones_like(vertex_colors).astype(np.float32) * 200.0/255 # initial gray
    # vertex_colors = vertex_colors.reshape(-1, 3, 3)
    # vertex_colors = np.repeat(vertex_colors[:, 0:1, :], 3, axis=-2).reshape(-1, 3)
    # inst_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    # Convert to Open3D's tensor-based triangle mesh
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(inst_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)


    label_colors = (vertex_colors * 255.0).astype(np.uint8)
    unique_c_inst = np.unique(label_colors, axis=0)
    num_inst = unique_c_inst.shape[0]
    print(f"Number of instances: {num_inst}")

    # only index the instance by the first vertex's color
    label_colors = label_colors.reshape(-1, 3, 3)[:, 0, :]

    inst_ids = np.arange(num_inst)
    inst_cnt = np.zeros((num_inst,), dtype=np.int32)
    inst_feat = {
        inst_id: [] for inst_id in inst_ids
    }
  
    
    fx = K_depth[0][0]
    fy = K_depth[1][1]
    cx = K_depth[0][2]
    cy = K_depth[1][2]
    # Generate rays for each pixel
    yx_grid = np.mgrid[0:H, 0:W]
    dirs = np.stack([(yx_grid[1] - cx) / fx, (yx_grid[0] - cy) / fy, np.ones((H, W))], axis=-1)  # (H, W, 3)
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)  # Normalize


    for f_i in tqdm(range(start, end, step)):
        # pose is from world to camera
        rgb_img, depth_img, pose = data_loader.getDataFromIndex(f_i)
        # check data validity
        if(rgb_img is None or depth_img is None or pose is None):
            logging.info(f"[Error] Skipping frame {f_i}")
            continue
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)


        open_segs_f = pjoin(ext_file_folder, 'openseg_seg_clip_merge', str(f_i).zfill(5)+'.pkl')
        with open(open_segs_f, 'rb') as f:
            open_segs_dict = pickle.load(f)
        open_seg_meta = open_segs_dict['meta_info']
        seg_map = open_segs_dict['seg_map']
        seg_map = data_loader.mapRGBtoDepth(seg_map)
        
        # ===================== get the projected instance color =====================
        # Transform ray directions to world coordinates
        ray_dirs = dirs.reshape(-1, 3) @ pose[:3, :3].T
        ray_origins = np.broadcast_to(pose[:3, 3], ray_dirs.shape)
        rays = np.concatenate([ray_origins, ray_dirs], axis=-1)
        rays = o3c.Tensor(rays, dtype=o3c.Dtype.Float32)

        # do ray casting and check the primitive ids where the rays hit
        ans = scene.cast_rays(rays)
        triangle_ids = ans['primitive_ids'].numpy() # no hit is scene.INVALID_ID
        hit_mask = (triangle_ids >= 0) & (triangle_ids < len(triangles))
        triangle_ids[~hit_mask] = 0

        vertex_ids = triangles[triangle_ids][:, 0]
        inst_color_img = vertex_colors[vertex_ids]
        inst_color_img[~hit_mask] = 0.0  # Set non-hit pixels to black
        inst_color_img = (inst_color_img * 255.0).astype(np.uint8)

        unique_c_img = np.unique(inst_color_img, axis=0)
        inst_color_img = inst_color_img.reshape(H, W, 3)
        # ============================================================================



        for inst_c in unique_c_img:
            if np.all(inst_c == 0):
                continue
            inst_mask = np.all(inst_color_img == inst_c, axis=-1)
            if np.count_nonzero(inst_mask) < 10000:
                continue
            inst_id = inst_ids[np.all(unique_c_inst == inst_c, axis=1)]
            if len(inst_id) != 1:
                raise ValueError(f"Instance color {inst_c} not found in unique instances.")
            inst_id = inst_id[0]

            sam_seg_ids = seg_map[inst_mask]
            if len(sam_seg_ids) == 1 and sam_seg_ids[0] == 0:
                continue

            inst_area = np.count_nonzero(inst_mask)
            cand_seg_ids = Counter(sam_seg_ids.reshape(-1))
            for sam_seg_id in cand_seg_ids:
                if sam_seg_id == 0:
                    continue

                seg_area = cand_seg_ids[sam_seg_id]
                if seg_area < inst_area * 0.2:
                    break

                seg_feat = open_seg_meta[sam_seg_id]['sem_feat']
                inst_feat[inst_id].append(seg_feat)

                if seg_area > inst_area * 0.5:
                    break

    sim_func = torch.nn.CosineSimilarity(dim=-1)

    clustering = AgglomerativeClustering(
        metric='precomputed',
        linkage='average',
        distance_threshold=0.1,  # adjust based on your data
        n_clusters=None
    )
                
    for inst_id in inst_ids:
        feat_list = inst_feat[inst_id]
        if len(feat_list) == 0:
            inst_feat[inst_id] = None
            continue
        if len(feat_list) < 3:
            inst_feat[inst_id] = feat_list[0]
            continue

        feat_list = np.array(feat_list)
        feats_inst = torch.from_numpy(feat_list).to(DEVICE) # (M, D)
        # perform clustering
        sim_scores = sim_func(feats_inst.unsqueeze(0), feats_inst.unsqueeze(1)) # (M, M)

        labels = clustering.fit_predict(1 - sim_scores.detach().cpu().numpy())  # (M,M)

        label_counts = Counter(labels)
        most_common_label = label_counts.most_common(1)[0][0]
        largest_cluster_indices = np.where(labels == most_common_label)[0]
        cluster_feat = np.mean(feat_list[largest_cluster_indices], axis=0)

        inst_feat[inst_id] = cluster_feat


    

    text_embed_f = pjoin('/home/zilong/Downloads/scans', 'clip_embed_COCO134.pkl')
    with open(text_embed_f, 'rb') as f:
        feat_cand_list = pickle.load(f)
    text_words = []
    text_embeds = []
    for w, em in feat_cand_list.items():
        text_words.append(w)
        text_embeds.append(em)
    text_embeds = torch.from_numpy(np.array(text_embeds)).to(DEVICE)

    legend_handles = []

    for inst_id in inst_ids:
        if inst_feat[inst_id] is None:
            continue
        
        feat_inst = torch.tensor(inst_feat[inst_id]).to(DEVICE)
        sim_scores = sim_func(feat_inst, text_embeds)
        match_idx = torch.argmax(sim_scores).item()

        c = (unique_c_inst[inst_id]).astype('float') / 255.0
        legend_handles.append(mpatches.Patch(
            color=(c[0], c[1], c[2],),
            label=text_words[match_idx],
        ))

        # print(f"Inst id: {inst_id}, name: {text_words[match_idx]}")
    
    plt.legend(handles=legend_handles, loc='center', bbox_to_anchor=(0.5, 0.5), borderaxespad=0.0)
    plt.axis('off')
    plt.show()
    



if __name__=="__main__":
    # set files path
    args = parse_args()
    main(args)