import os, sys, time, argparse, pickle, glob
from os.path import join as pjoin

import numpy as np
import torch
import open3d as o3d
import open3d.core as o3c # type: ignore

from .mesh_postprocess_utils import init_label_colormap
from plyfile import PlyData, PlyElement
from .vis_utils import get_new_pallete
from .mesh_postprocess_utils import write_ply_with_labels, match_feature_to_label_embed


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


# python -m scripts.utils.mesh_postprocess_openmask3d --scene_num office0

def main(args):
    scene_num = args.scene_num
    # if needed
    data_path = '/media/zilong/Documents/MasterProject/scannet_v2' # scannet_v2 | Replica
    scene_folder = pjoin(data_path, scene_num)
    # gt_mesh_f = pjoin(scene_folder, '..', f'{scene_num}_mesh.ply')
    gt_mesh_f = pjoin(scene_folder, f'{scene_num}_vh_clean_2.ply')

    result_folder = pjoin(args.result_folder, scene_num)

    task = 'NYU40' # NYU40 | CoCo | CoCoPano | Replica | Scannet200 | Scannet20
    text_prompt, _, color_map, valid_ids = init_label_colormap(task)

    text_embed_dir = '/home/zilong/Disk_data/semantic_mapping_result'
    # VLM_name = 'clip-ViT-L-14-336px' 
    VLM_name = 'siglip-l-16-384' # only for exp with gt_inst

    eval_baseline = 'gt_inst' # mask3d | segment3d | ours | gt_inst
    use_canonical_phrase = True

    if eval_baseline == 'mask3d':
        masks_f = pjoin('/home/zilong/Disk_data/Mask3D/mask3d_masks', 
            f'{scene_num}_vh_clean_2_masks.pt'
        )
        # {scene_num}_mesh_masks.pt
        masks_feature_f = pjoin('/home/zilong/Disk_data/openmask3d/openmask3d/output', 
            scene_num, 'mask3d', f'openmask3d_features.npy')
        inst_mesh_f = pjoin(result_folder, 'mask3d_instance.ply')
        out_mesh_f = pjoin(result_folder, 'mask3d_openmask3d_sem.ply')
    elif eval_baseline == 'segment3d':
        masks_f = pjoin('/home/zilong/Disk_data/Mask3D/segment3d_masks', 
            f'{scene_num}_masks.pt'
        )
        masks_feature_f = pjoin('/home/zilong/Disk_data/openmask3d/openmask3d/output', 
            scene_num, 'segment3d', f'openmask3d_features.npy')
        inst_mesh_f = pjoin(result_folder, 'segment3d_instance.ply')
        out_mesh_f = pjoin(result_folder, 'segment3d_openmask3d_sem.ply')
    elif eval_baseline == 'ours':
        masks_f = pjoin('/home/zilong/Disk_data/Mask3D/ours_masks', f'{scene_num}_masks.pt')
        masks_feature_f = pjoin('/home/zilong/Disk_data/openmask3d/openmask3d/output', 
            scene_num, 'ours', f'openmask3d_features.npy')
        inst_mesh_f = glob.glob(pjoin(
            result_folder, 'cropformer_inst', 'instance_mesh_*.ply'))[0]
        out_mesh_f = pjoin(result_folder, 'ours_openmask3d_sem_nyu40.ply')
    elif eval_baseline == 'gt_inst':
        result_folder = pjoin(result_folder, 'gt_inst_vis')
        view_select = 'incre_vis' # incre_vis | incre_viewcov
        inst_sem_feat_f =  glob.glob(pjoin(
            result_folder, f'inst_sem_{VLM_name}_*_{view_select}.pkl')
        )[0]
        inst_mesh_f = pjoin(result_folder, '..', 'gt_instance_mesh.ply')
        out_mesh_f = pjoin(result_folder, 'gt_inst_sem_nyu40.ply')

    
    if use_canonical_phrase:
        from .text_embedding import TextEmbedder
        text_embedder = TextEmbedder(VLM_name, device=DEVICE)
        text_embeds_canon = text_embedder.get_canonical_text_embed()

    inst_mesh = PlyData.read(inst_mesh_f)
    v_pos = np.stack([
        inst_mesh['vertex']['x'], inst_mesh['vertex']['y'], inst_mesh['vertex']['z']
    ], axis=-1)
    num_vertices = v_pos.shape[0]
    sem_c = np.ones((num_vertices, 3), dtype=np.float32) * 200.0 / 255.0
    sem_label = np.zeros(num_vertices, dtype=np.uint16)
    triangles = inst_mesh['face']['vertex_indices']

    
    # ####################### Feature Mapping #######################
    text_embeds_f = pjoin(text_embed_dir, f'{VLM_name}_{text_prompt}.pkl')
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

    # ====================== only for gt_inst exp ======================
    if eval_baseline == 'gt_inst':
        with open(inst_sem_feat_f, 'rb') as f:
            inst_sem_dict = pickle.load(f)
        gt_inst_labels = inst_mesh['vertex']['label']

        for glo_inst_id in inst_sem_dict.keys():
            frame_list = inst_sem_dict[glo_inst_id]['frame_id']
            # we need at least two views for a valid instance
            if len(frame_list) < 2:
                continue

            feat_inst = torch.tensor(inst_sem_dict[glo_inst_id]['feat']).to(DEVICE)
            match_idx = match_feature_to_label_embed(
                feat_inst, text_embeds, 
                text_embeds_canon, True
            )

            # ======= Skip the background class ======
            real_sem_id = valid_ids[match_idx + 1] # +1 because skipped the bg class
            class_name = text_words[match_idx]
            class_color = color_map[class_name]

            inst_vertices = (glo_inst_id == gt_inst_labels)

            sem_c[inst_vertices] = np.array(class_color, dtype=np.float32) / 255.0
            sem_label[inst_vertices] = real_sem_id

        write_ply_with_labels(out_mesh_f, v_pos, sem_c, sem_label, triangles)
        return
    # ================================================================

    inst_masks = torch.load(masks_f) # (N, M)
    inst_features = np.load(masks_feature_f) # (M, D)
    assert num_vertices == inst_masks.shape[0]
    assert inst_masks.shape[1] == inst_features.shape[0]
    num_inst = inst_masks.shape[1]


    for inst_id in range(num_inst):

        feat_inst = torch.tensor(inst_features[inst_id]).to(DEVICE)
        match_idx = match_feature_to_label_embed(
            feat_inst, text_embeds, 
            text_embeds_canon, True
        )
        
        class_name = text_words[match_idx]
        class_color = color_map[class_name]
        class_id = valid_ids[match_idx + 1]  # +1 because skipped bg class

        inst_vertices = inst_masks[:, inst_id].astype(bool)

        sem_c[inst_vertices] = np.array(class_color, dtype=np.float32) / 255.0
        sem_label[inst_vertices] = class_id
    


    print("Saving the mapped mesh.")
    # ========================= Save the mapped mesh =========================
    if eval_baseline == 'ours':
        # load ours reconstructed mesh and map it to the gt mesh
        gt_mesh = PlyData.read(gt_mesh_f)
        gt_v = np.stack([
            gt_mesh['vertex']['x'], gt_mesh['vertex']['y'], gt_mesh['vertex']['z']
        ], axis=-1)
        gt_tri = gt_mesh['face']['vertex_indices']
        num_v_gt = gt_v.shape[0]

        dist_thresh = 0.05
        pred_inst_vertex = o3d.core.Tensor(v_pos, dtype=o3d.core.float32)
        knn_data = o3c.nns.NearestNeighborSearch(pred_inst_vertex)
        knn_data.knn_index()

        gt_vertex = o3d.core.Tensor(gt_v, dtype=o3d.core.float32)
        indices, distances_sqr = knn_data.knn_search(gt_vertex, 1)
        indices = indices.reshape(-1).numpy()
        matched = (distances_sqr < dist_thresh**2).reshape(-1).numpy()

        out_label = np.zeros((num_v_gt), dtype=np.uint16)
        out_color = np.ones((num_v_gt, 3), dtype=np.float32) * (200.0/255.0)
        idxs = np.arange(num_v_gt)
        out_color[idxs[matched]] = sem_c[indices[matched]]
        out_label[idxs[matched]] = sem_label[indices[matched]]

        write_ply_with_labels(out_mesh_f, gt_v, out_color, out_label, gt_tri)
    else:
        write_ply_with_labels(out_mesh_f, v_pos, sem_c, sem_label, triangles)



if __name__=="__main__":
    # set files path
    args = parse_args()

    scene_list = args.scene_num
    if scene_list == 'all':
        scene_list = ['scene0011_00', 'scene0011_01', 
            'scene0050_00', 'scene0050_01', 'scene0050_02', 
            'scene0084_00', 'scene0084_01', 'scene0084_02', 
            'scene0168_00', 'scene0168_01', 'scene0168_02', 
            'scene0231_00', 'scene0231_01', 'scene0231_02', 
            'scene0378_00', 'scene0378_01', 'scene0378_02', 
            'scene0518_00']
        # scene_list = ['office0', 'office1', 'office2', 'office3', 
        #              'office4', 'room0', 'room1', 'room2']
    else:
        scene_list = [scene_list]

    for scene_id in scene_list:
        args.scene_num = scene_id
        print(f'Processing scene: {scene_id}')
        main(args)