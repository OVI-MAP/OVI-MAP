import os, sys, time, argparse, pickle, glob
from os.path import join as pjoin
import cv2

import numpy as np
import torch
import open3d as o3d
# from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rerun as rr
from scipy.spatial.transform import Rotation as R

# from plyfile import PlyData, PlyElement
from .vis_utils import get_new_pallete
from .data_loaders import ScannetLoader, ReplicaLoader
from .mesh_postprocess_utils import init_label_colormap, match_feature_to_label_embed


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parse = argparse.ArgumentParser(description='Semantic Mapping-Python') 
    # files path
    parse.add_argument("--scene_num", type=str, required=True, 
        help="which scene to process, use 'all' for all scenes")
    
    parse.add_argument("--result_folder", type=str, 
        default='/home/zilong/Disk_data/semantic_mapping_result', 
        help="folder of mapping results")
    
    return parse.parse_args()



# python -m scripts.utils.vis_view_selection --scene_num office0

def main(args):
    scene_num = args.scene_num

    result_folder = args.result_folder
    result_folder = pjoin(result_folder, scene_num)
    result_folder = pjoin(result_folder, 'cropformer_inst')

    # TODO flags to trigger
    vis_class_name  = True
    use_canonical_phrase = True

    task = 'Replica' # NYU40 | CoCoPano | Replica | Scannet200
    text_prompt, _, _, valid_ids = init_label_colormap(task)

    if task == 'Replica':
        scene_folder = pjoin('/media/zilong/Documents/MasterProject/Replica', scene_num)
        data_loader = ReplicaLoader(scene_folder)
    elif task == 'Scannet200' or task == 'NYU40':
        scene_folder = pjoin('/media/zilong/Documents/MasterProject/scannet_v2', scene_num)
        data_loader = ScannetLoader(scene_folder)

    depth_h = data_loader.depth_h
    depth_w = data_loader.depth_w

    text_embed_dir = '/home/zilong/Downloads/scans'
    vl_model_name = 'siglip-l-16-384'

    if use_canonical_phrase:
        from .text_embedding import TextEmbedder
        text_embedder = TextEmbedder(vl_model_name, device=DEVICE)
        text_embeds_canon = text_embedder.get_canonical_text_embed()

    pred_inst_mesh_f = glob.glob(pjoin(result_folder, 'instance_mesh_*.ply'))[0]
    frame_cnt = os.path.basename(pred_inst_mesh_f)[14:-4]
    view_select = 'incre_vis_fix' # top-8 | viewcov | incre_vis | incre_viewcov | incre_combine
    inst_sem_feat_name =  f'inst_sem_{vl_model_name}_{frame_cnt}_{view_select}.pkl'
    with open(pjoin(result_folder,inst_sem_feat_name), 'rb') as f:
        inst_sem_dict = pickle.load(f)    


    inst_mesh = o3d.t.io.read_triangle_mesh(pred_inst_mesh_f)
    vertex_c = inst_mesh.vertex.colors.numpy()
    vertex_pos = inst_mesh.vertex.positions.numpy()
    num_v = vertex_c.shape[0]

    triangles = inst_mesh.triangle.indices.numpy()
    triangle_c = vertex_c[triangles[:, 0]]
    triangle_c = (triangle_c * 255.0).astype(np.uint8) # indexed by triangle-idx
    
    # ####################### Feature Mapping #######################
    text_embed_dir = '/home/zilong/Disk_data/semantic_mapping_result'
    text_embeds_f = pjoin(text_embed_dir, f'{vl_model_name}_{text_prompt}.pkl')
    with open(text_embeds_f, 'rb') as f:
        feat_cand_list = pickle.load(f)
    text_words = []
    text_embeds = []
    for w, em in feat_cand_list.items():
        # ======= Skip the background class ======
        if w == 'background':
            continue
        # ========================================
        text_words.append(w)
        text_embeds.append(em)
    text_embeds = torch.from_numpy(np.array(text_embeds)).to(DEVICE)

    # ## Map the extracted features to text embeddings
    color_to_inst = {}
    legend_handles = []

    rr.init("vis_view_selection", spawn=True)

    vis_inst_list = inst_sem_dict.keys()
    vis_frame_list = []

    for inst_i, glo_inst_id in enumerate(vis_inst_list):
        frame_list = inst_sem_dict[glo_inst_id]['frame_id']
        if len(frame_list) < 2:
            continue

        poses = inst_sem_dict[glo_inst_id]['pose']
        inst_bbox2ds = inst_sem_dict[glo_inst_id]['box_2d']

        feat_inst = torch.tensor(inst_sem_dict[glo_inst_id]['feat']).to(DEVICE)
        match_idx = match_feature_to_label_embed(
            feat_inst, text_embeds, 
            text_embeds_canon, True
        )

        inst_color = inst_sem_dict[glo_inst_id]['color']
        inst_color_tuple = (inst_color[0], inst_color[1], inst_color[2])
        # only in case that the backend gives the same color to different instances
        if inst_color_tuple in color_to_inst:
            print(f"Warning: Instance {glo_inst_id} with color {inst_color_tuple} already exists in color_to_inst.")
        # ======= Skip the background class ======
        real_sem_id = valid_ids[match_idx + 1] # +1 because skipped the bg class

        if vis_class_name:
            print(f"Instance-{glo_inst_id}  matched  {real_sem_id}-{text_words[match_idx]}")


        # =======================================================================
        # Get mask of vertices matching this color
        triangle_mask = np.all(triangle_c == inst_color, axis=-1)
        if not np.any(triangle_mask) or np.count_nonzero(triangle_mask) < 10:
            continue  # Skip too small / empty instances
        inst_triangles = triangles[triangle_mask].flatten() # [M, 3], array of vertices ids

        v_pos = vertex_pos[inst_triangles] # [M*3, 3]
        v_c = vertex_c[inst_triangles] # [M*3, 3]
        triangle_indices = np.arange(inst_triangles.shape[0]).reshape(-1, 3)


        rr.set_time_seconds("stable_time", inst_i)
        rr.log(f"inst_{glo_inst_id}/{text_words[match_idx]}", rr.Mesh3D(
            vertex_positions=v_pos,
            vertex_colors=(v_c * 255.0).astype(np.uint8),
            triangle_indices=triangle_indices,
        ))
       
        if glo_inst_id not in vis_frame_list:
            continue

        for idx, pose in enumerate(poses):
            rr.log(f"inst_{glo_inst_id}/camera_{idx}", rr.Pinhole(focal_length=600, width=depth_w, height=depth_h))
            rr.log(f"inst_{glo_inst_id}/camera_{idx}", rr.Transform3D(
                translation=pose[:3, 3], 
                rotation=rr.Quaternion(xyzw=R.from_matrix(pose[:3, :3]).as_quat())
            ))
        # show the frame with the bbox for this instance
        for idx, f_i in enumerate(frame_list):
            cur_img, _, _ = data_loader.getDataFromIndex(f_i)
            cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
            bbox_2d = inst_bbox2ds[idx] # [x1, y1, x2, y2]
            rr.log(f"inst_{glo_inst_id}/img_{idx}_{f_i}", rr.Boxes2D(
                mins=[bbox_2d[0], bbox_2d[1]], 
                sizes=[bbox_2d[2]-bbox_2d[0], bbox_2d[3]-bbox_2d[1]]
            ))
            rr.log(f"inst_{glo_inst_id}/img_{idx}_{f_i}", rr.Image(image=cur_img, opacity=0.5))
            
        

    #     if vis_class_name:
    #         inst_color = inst_color.astype(np.float32) / 255.0
    #         legend_handles.append(mpatches.Patch(
    #             color=inst_color,
    #             label=text_words[match_idx],
    #         ))
        
    # if vis_class_name:
    #     plt.legend(handles=legend_handles, loc='center', 
    #                bbox_to_anchor=(0.5, 0.5), borderaxespad=0.0)
    #     plt.axis('off')
    #     plt.savefig(
    #         pjoin(result_folder, f'sem_map_{vl_model_name}_{text_prompt}.png'), 
    #         bbox_inches='tight', pad_inches=0.1
    #     )
    #     plt.close()



if __name__=="__main__":
    # set files path
    args = parse_args()
    print(f'Processing scene: {args.scene_num}')
    main(args)