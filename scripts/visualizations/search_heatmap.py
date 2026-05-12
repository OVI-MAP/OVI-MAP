import os, sys, time, argparse, pickle, glob
from os.path import join as pjoin

import numpy as np
import torch

from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt

from plyfile import PlyData
from scripts.visualizations.vis_utils import get_new_pallete
from scripts.utils.text_embedding import TextEmbedder

import rerun as rr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def parse_args():
    parse = argparse.ArgumentParser(description='Semantic Mapping-Python') 
    # files path
    parse.add_argument("--scene_num", type=str, required=True, 
        help="which scene to process, use 'all' for all scenes")
    
    parse.add_argument("--search_text", type=str, required=True, 
        help="text to search for in the scene")

    parse.add_argument("--result_folder", type=str, 
        default='/home/zilong/Disk_data/semantic_mapping_result', 
        help="folder of mapping results")
    parse.add_argument("--data_folder", type=str, 
        default='/media/zilong/Documents/MasterProject/Replica', 
        help="which scene for mapping ")
    
    return parse.parse_args()


# python -m scripts.utils.search_heatmap --scene_num scene0011_00 --search_text sofa

def main(args):
    scene_num = args.scene_num
    data_path = args.data_folder
    scene_folder = pjoin(data_path, scene_num)

    result_folder = args.result_folder
    result_folder = pjoin(result_folder, scene_num)

    result_folder = pjoin(result_folder, 'cropformer_inst')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    search_text = args.search_text
    print(f'Searching for text: {search_text} in scene: {scene_num}')

    # NOTE whether to use the canonical phrase from LERF
    use_rela_score = False

    vl_model_name = 'siglip-l-16-384'
    text_embedder = TextEmbedder(vl_model_name, device=DEVICE)

    pred_inst_mesh_f = glob.glob(
        pjoin(result_folder, 'instance_map_gt_*.ply'))[0]

    f_cnt = os.path.basename(pred_inst_mesh_f)[16:-4]
    view_select = 'top-8' # 'top-8' | 'raycast'
    inst_sem_feat_name =  f'inst_sem_{vl_model_name}_{f_cnt}_{view_select}.pkl'
    # =====================================================
    with open(pjoin(result_folder,inst_sem_feat_name), 'rb') as f:
        inst_sem_dict = pickle.load(f)    

    

    pred_ply = PlyData.read(pred_inst_mesh_f)
    pred_inst_ids = np.array(pred_ply['vertex']['label'])
    pred_inst_set = np.unique(pred_inst_ids)
    v_pos = np.stack([
        pred_ply['vertex']['x'], pred_ply['vertex']['y'], pred_ply['vertex']['z']
    ], axis=-1)

    triangles = np.stack(pred_ply['face']['vertex_indices'])

    new_triangles = []
    # for tri in triangles:
    #     new_triangles.append(np.array([tri[0], tri[1], tri[2]], dtype=np.int32))
    #     new_triangles.append(np.array([tri[0], tri[2], tri[3]], dtype=np.int32))
    # new_triangles = np.stack(new_triangles)
    for tri in triangles:
        new_triangles.append(np.array([tri[0], tri[1], tri[2]], dtype=np.int32))
    # tri_label = pred_inst_ids[triangles[:, 0]]

    text_embeds = text_embedder.get_text_embed([search_text])[0]
    text_embeds_canon = text_embedder.get_canonical_text_embed()

    # ## Map the extracted features to text embeddings
    sim_func = torch.nn.CosineSimilarity(dim=-1) # this will do the normalization

    inst_sim_score = np.zeros(len(pred_inst_set), dtype=np.float32)

    for glo_inst_id in inst_sem_dict.keys():
        frame_list = inst_sem_dict[glo_inst_id]['frame_id']
        if len(frame_list) < 2:
            continue
        feat_inst = torch.tensor(inst_sem_dict[glo_inst_id]['feat']).to(DEVICE)
        sim_s_query = sim_func(feat_inst, text_embeds)

        if use_rela_score:
            sim_s_canon = sim_func(feat_inst, text_embeds_canon) # [L_canon]
            rela_score = torch.exp(sim_s_query) / (torch.exp(sim_s_query) + torch.exp(sim_s_canon))
            score = torch.min(rela_score).item() # [0, 1]
        else:
            score = sim_s_query.item() # [-1, 1]

        print(f'Instance {glo_inst_id} has score: {score:.4f}')
        idx = np.where(pred_inst_set == glo_inst_id)[0]
        inst_sim_score[idx] = score 

    if use_rela_score:
        inst_sim_score[0] = 0.5
    # min-max-norm
    inst_sim_score = (inst_sim_score - inst_sim_score.min()) / (inst_sim_score.max() - inst_sim_score.min())
    # print(inst_sim_score)

    heat_c = np.zeros_like(v_pos, dtype=np.uint8)
    # 'jet', 'plasma', 'coolwarm'
    cmap = plt.get_cmap('jet')
    for i, inst_id in enumerate(pred_inst_set):
        if inst_id == 0:
            continue
        c = np.array(cmap(inst_sim_score[i])[:3])
        heat_c[pred_inst_ids == inst_id] = (c * 255.0).astype(np.uint8)


    vis_name = f'inst_heatmap_{search_text}'
    rr.init(vis_name, recording_id=vis_name, spawn=True)
    rr.log(
        f"heat-map", rr.Mesh3D(
            vertex_positions=v_pos,
            vertex_colors=heat_c,
            triangle_indices=new_triangles,
        )
    )
    # for i, inst_id in enumerate(pred_inst_set):
    #     tri_mask = (tri_label == inst_id)
    #     tris2vertex = triangles[tri_mask].flatten()
    #     tri_idx = np.arange(tris2vertex.shape[0]).reshape(-1, 3)
    #     heat_c = cmap(inst_sim_score[i])[:, :3]
    #     rr.log(
    #         f"heatmap/inst-{inst_id}", rr.Mesh3D(
    #             vertex_positions=v_pos[tris2vertex],
    #             vertex_colors=heat_c,
    #             triangle_indices=tri_idx,
    #         )
    #     )




if __name__=="__main__":
    # set files path
    args = parse_args()

    scene_list = args.scene_num
    if scene_list == 'all':
        scene_list = ['office0', 'office1', 'office2', 'office3', 
                     'office4', 'room0', 'room1', 'room2']
    else:
        scene_list = [scene_list]

    for scene_id in scene_list:
        args.scene_num = scene_id
        print(f'Processing scene: {scene_id}')
        main(args)