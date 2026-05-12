import os, sys, time, argparse, pickle, glob
from os.path import join as pjoin

import numpy as np
import torch
import open3d as o3d
import open3d.core as o3c # type: ignore

from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from plyfile import PlyData, PlyElement
from .vis_utils import get_new_pallete


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def write_ply_with_labels(
    filename, vertex_positions, 
    vertex_colors, vertex_labels, triangles
):
    """
    Write a PLY file with vertex positions, colors, and labels.
    vertex_positions: Nx3 array of vertex positions
    vertex_colors: Nx3 array of vertex colors (0-1 range)
    vertex_labels: Nx array of vertex labels (uint16)
    triangles: Mx3 array of triangle vertex indices
    """
    vertex_dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ]
    if vertex_labels is not None:
        vertex_dtype.append(('label', 'u2'))
    vertices = np.empty(vertex_positions.shape[0], dtype=vertex_dtype)
    vertices['x'] = vertex_positions[:, 0]
    vertices['y'] = vertex_positions[:, 1]
    vertices['z'] = vertex_positions[:, 2]
    vertices['red'] = (vertex_colors[:, 0] * 255).astype(np.uint8)
    vertices['green'] = (vertex_colors[:, 1] * 255).astype(np.uint8)
    vertices['blue'] = (vertex_colors[:, 2] * 255).astype(np.uint8)
    if vertex_labels is not None:
        vertices['label'] = vertex_labels.astype(np.uint16)

    face_dtype = [('vertex_indices', 'O')]
    faces = np.empty(len(triangles), dtype=face_dtype)
    faces['vertex_indices'] = [row.astype(np.int32) for row in triangles]

    ply_data = PlyData([
        PlyElement.describe(vertices, 'vertex'), 
        PlyElement.describe(faces, 'face')
    ], text=False)
    ply_data.comments.append("MLIB generated")
    # print("Writing PLY file into:", filename)
    ply_data.write(filename)



def project_to_triangle_mesh(
    from_mesh, to_gt_mesh_vertex, color2=None, label2=None, 
    dist_thresh=0.05, 
    device=o3c.Device("CUDA", 0)
):
    from_mesh_vertex_tensor = from_mesh.vertex.positions.to(device)
    to_gt_mesh_vertex_tensor = to_gt_mesh_vertex.to(device)

    # KNN matching
    knn_data = o3c.nns.NearestNeighborSearch(from_mesh_vertex_tensor)
    knn_data.knn_index()
    indices, distances_sqr = knn_data.knn_search(to_gt_mesh_vertex_tensor, 1)
    indices = indices.reshape(-1).numpy()
    matched = (distances_sqr < dist_thresh**2).reshape(-1).numpy()

    from_mesh_color = from_mesh.vertex.colors.numpy()
    from_mesh_label = from_mesh.vertex.labels.numpy()

    num_vertices = to_gt_mesh_vertex_tensor.shape[0]
    output_color = np.ones((num_vertices, 3), dtype=np.float32) * (200.0/255.0)
    output_label = np.zeros((num_vertices,), dtype=np.uint16)

    idxs = np.arange(num_vertices)
    output_color[idxs[matched]] = from_mesh_color[indices[matched]]
    output_label[idxs[matched]] = from_mesh_label[indices[matched]]
    output_color2 = None
    output_label2 = None
    if color2 is not None:
        output_color2 = np.ones((num_vertices, 3), dtype=np.float32) * (200.0/255.0)
        output_color2[idxs[matched]] = color2[indices[matched]]
    if label2 is not None:
        output_label2 = np.zeros((num_vertices,), dtype=np.uint16)
        output_label2[idxs[matched]] = label2[indices[matched]]

    return output_color, output_label, output_color2, output_label2


def match_feature_to_label_embed(feat_inst, text_embeds, 
    text_embeds_canon=None, use_canonical_phrase=False
):
    sim_func = torch.nn.CosineSimilarity(dim=-1)
    sim_s_queries = sim_func(feat_inst, text_embeds) # (L_cand,)
    if use_canonical_phrase:
        sim_s_canon = sim_func(feat_inst, text_embeds_canon) # (L_canon,)
        sim_s_queries = sim_s_queries.unsqueeze(0) # (1, L_cand)
        sim_s_canon = sim_s_canon.unsqueeze(1) # (L_canon, 1)
        rela_scores = torch.exp(sim_s_queries) / (torch.exp(sim_s_queries) + torch.exp(sim_s_canon))
        rela_scores = torch.min(rela_scores, dim=0).values # (L_cand,)
        match_idx = torch.argmax(rela_scores).item()
    else:
        match_idx = torch.argmax(sim_s_queries).item()
    
    return match_idx


def init_label_colormap(task='Replica'):
    """
    Return:
        text_prompt: The text prompt for the task
        labels: The list of text labels for the task
        color_map: The color map for the task, from text label to RGB
        valid_ids: The list of valid IDs for the task
    """
    if task == 'NYU40':
        text_prompt = 'NYU41'
        from .semantic_const import NYU_40, NYU_41_PALETTE
        labels = ['background'] + NYU_40
        valid_ids = list(range(len(labels)))
        color_map = {label: NYU_41_PALETTE[i] for i, label in enumerate(labels)}
    elif task == 'CoCoPano':
        text_prompt = 'COCO134'
        from .semantic_const import COCO_PANOPTIC_133, COCO_PANOPTIC_134_PALETTE
        labels = ['background'] + COCO_PANOPTIC_133
        valid_ids = list(range(len(labels)))
        color_map = {label: COCO_PANOPTIC_134_PALETTE[i] for i, label in enumerate(labels)}
    elif task == 'Replica':
        text_prompt = 'Replica'
        from .semantic_const import REPLICA_51, REPLICA_52_PALETTE
        labels = ['background'] + REPLICA_51
        valid_ids = list(range(len(labels)))
        color_map = {label: REPLICA_52_PALETTE[i] for i, label in enumerate(labels)}
    elif task == 'Scannet200':
        text_prompt = 'Scannet200'
        from .semantic_const import CLASS_LABELS_200, SCANNET_COLOR_MAP_200, VALID_CLASS_IDS_200
        labels = ['background'] + CLASS_LABELS_200
        valid_ids = [0] + VALID_CLASS_IDS_200
        color_map = {labels[i]: SCANNET_COLOR_MAP_200[id] for i, id in enumerate(valid_ids)}
    elif task == 'Scannet20':
        text_prompt = 'Scannet20'
        from .semantic_const import CLASS_LABELS_20, SCANNET_COLOR_MAP_20, VALID_CLASS_IDS_20
        labels = ['background'] + CLASS_LABELS_20
        valid_ids = [0] + VALID_CLASS_IDS_20
        color_map = {labels[i]: SCANNET_COLOR_MAP_20[id] for i, id in enumerate(valid_ids)}
    else:
        raise NotImplementedError(f'Unknown task: {task}')
    
    return text_prompt, labels, color_map, valid_ids





def parse_args():
    parse = argparse.ArgumentParser(description='Semantic Mapping-Python') 
    # files path
    parse.add_argument("--scene_num", type=str, required=True, 
        help="which scene to process, use 'all' for all scenes")
    
    parse.add_argument("--result_folder", type=str, 
        default='/home/zilong/Disk_data/semantic_mapping_result', 
        help="folder of mapping results")
    
    return parse.parse_args()






# python -m scripts.utils.mesh_postprocess_utils --scene_num office0

def main(args):
    scene_num = args.scene_num

    result_folder = args.result_folder
    result_folder = pjoin(result_folder, scene_num)
    result_folder = pjoin(result_folder, 'cropformer_inst_rt')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    task = 'Replica' # NYU40 | CoCo | CoCoPano | Replica | Scannet200 | Scannet20
    VLM_name = 'siglip-l-16-384'

    # TODO flags to trigger =====================
    output_mask_file = False         # if save the mask for openmask3d
    map_to_gt_mesh = True
    use_canonical_phrase = True

    show_mapping = False
    remove_raw_recon = False # NOTE Attention !!!!!
    # ===========================================

    if use_canonical_phrase:
        from .text_embedding import TextEmbedder
        text_embedder = TextEmbedder(VLM_name, device=DEVICE)
        text_embeds_canon = text_embedder.get_canonical_text_embed()
    

    gt_inst_mesh_f = pjoin(result_folder, '..', 'gt_instance_mesh.ply')
    pred_inst_mesh_f = sorted(glob.glob(pjoin(result_folder, 'instance_mesh_*.ply')))[0]
    
    if output_mask_file:
        mask_out_dir = '/home/zilong/Disk_data/Mask3D/ours_masks'
        os.makedirs(mask_out_dir, exist_ok=True)
        mask_save_f = pjoin(mask_out_dir, f'{scene_num}_masks.pt') 

    frame_cnt = os.path.basename(pred_inst_mesh_f)[14:-4]
    view_select = 'real_time' # random-8 | viewcov | incre_vis | incre_combine | incre_vis_normal
    inst_sem_feat_name =  f'inst_sem_{VLM_name}_{frame_cnt}_{view_select}.pkl'

    out_inst_mesh_f = pjoin(
        result_folder, f'instance_map_gt_{frame_cnt}.ply')
    out_sem_mesh_f = pjoin(
        result_folder, f'semantic_map_gt_{frame_cnt}.ply') # _nyu40

    # =================== parameters setting ends here ===================




    text_embed_dir = '/home/zilong/Disk_data/semantic_mapping_result'
    with open(pjoin(result_folder, inst_sem_feat_name), 'rb') as f:
        inst_sem_dict = pickle.load(f)

    text_prompt, labels, color_map, valid_ids = init_label_colormap(task)
    # plt.figure(figsize=(20, 6))
    # for i, label in enumerate(labels):
    #     plt.bar(i, 0.5, color=np.array(color_map[label]) / 255.0, label=label)
    # plt.xticks(range(len(labels)), labels, rotation=90)
    # plt.title(f'Color Map for {task}')
    # plt.savefig(pjoin(result_folder, f'color_map_{task}.png'))
    # plt.close()
    
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
    
    color_to_inst = {}

    for glo_inst_id in inst_sem_dict.keys():
        frame_list = inst_sem_dict[glo_inst_id]['frame_id']
        # we need at least two views for a valid instance
        if len(frame_list) < 2:
            continue
        
        # for weighted averaging, [M, D]
        feat_inst = torch.tensor(inst_sem_dict[glo_inst_id]['feat']).to(DEVICE)
        vis_scores = torch.tensor(inst_sem_dict[glo_inst_id]['vis_area']).to(DEVICE)

        # NOTE =================== weigthed sum ===================
        vis_scores = vis_scores[-8:]
        feat_inst = feat_inst[-8:, :] # [M, D]
        vis_scores = vis_scores / (torch.sum(vis_scores) + 1e-6)
        feat_inst = torch.sum(feat_inst * vis_scores.unsqueeze(-1), dim=0)
        # ==========================================================

        # # simple averaging
        # feat_inst = torch.mean(feat_inst, dim=0) # [1, D]

        match_idx = match_feature_to_label_embed(
            feat_inst, text_embeds, 
            text_embeds_canon, True
        )

        inst_color = inst_sem_dict[glo_inst_id]['color']
        inst_color_tuple = (inst_color[0], inst_color[1], inst_color[2])
        if inst_color_tuple in color_to_inst:
            print(f"Warning: Instance {glo_inst_id} with color {inst_color_tuple} already exists in color_to_inst.")
        # ======= Skip the background class ======
        real_sem_id = valid_ids[match_idx + 1] # +1 because skipped the bg class

        color_to_inst[inst_color_tuple] = {
            'glo_inst_id': glo_inst_id,
            'class_id': real_sem_id, 
            'class_name': text_words[match_idx],
        }
        if show_mapping:
            print(f"Instance-{glo_inst_id}  matched  {real_sem_id}-{text_words[match_idx]}")

    # ####################### Load the instance mesh #######################
    inst_mesh = o3d.t.io.read_triangle_mesh(pred_inst_mesh_f)
    vertex_colors = inst_mesh.vertex.colors.numpy()
    num_v = vertex_colors.shape[0]

    triangles = inst_mesh.triangle.indices.numpy()
    triangle_colors = vertex_colors[triangles[:, 0]]
    triangle_colors = (triangle_colors * 255.0).astype(np.uint8) # indexed by triangle-idx

    # ####################### Create the semantic mesh #######################
    out_mesh_colors = np.ones_like(vertex_colors, dtype=np.uint8) * 200
    out_vertex_labels = np.zeros(num_v, dtype=np.uint16)
    out_vertex_inst_id = np.zeros(num_v, dtype=np.uint16)
    
    if output_mask_file:
        masks = np.zeros((num_v, len(color_to_inst)), dtype=np.uint8) # (N, M)

    # Color the instance in the mesh according to the class name
    for inst_i, inst_item in enumerate(color_to_inst.items()):
        inst_color_tuple, inst_info = inst_item
        inst_color = np.array(inst_color_tuple, dtype=np.uint8)
        class_color = color_map[inst_info['class_name']]

        # Get mask of vertices matching this color
        triangle_mask = np.all(triangle_colors == inst_color, axis=-1)
        if not np.any(triangle_mask):
            continue  # Skip empty instances
        inst_vertices = triangles[triangle_mask].flatten() # [M, 3] -> [M*3] Array of vertex ids

        if output_mask_file:
            masks[inst_vertices, inst_i] = 1

        # [M*3, 3]
        out_mesh_colors[inst_vertices] = np.array(class_color, dtype=np.uint8)
        out_vertex_labels[inst_vertices] = inst_info['class_id']
        out_vertex_inst_id[inst_vertices] = inst_info['glo_inst_id']

    if output_mask_file:
        torch.save(masks, mask_save_f)
    
    # =========================================================================
    if map_to_gt_mesh:
        out_mesh_colors = out_mesh_colors.astype(np.float32) / 255.0
        # color and label the vertices semantically
        inst_mesh.vertex.colors = o3c.Tensor(out_mesh_colors, 
            dtype=o3c.float32, device=inst_mesh.device)
        inst_mesh.vertex.labels = o3c.Tensor(out_vertex_labels, 
            dtype=o3c.uint16, device=inst_mesh.device)
        # o3d.t.io.write_triangle_mesh(
        #     pjoin(result_folder, f'semantic_mesh_{text_prompt}.ply'), inst_mesh
        # )

        plydata = PlyData.read(gt_inst_mesh_f)
        map_vertices = np.stack([
            plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']
        ], axis=-1)
        map_vertices = o3c.Tensor(map_vertices, 
            dtype=o3c.float32, device=inst_mesh.device)

        triangle_indices = plydata['face']['vertex_indices']

        map_sem_c, map_sem_label, map_inst_c, map_inst_id = project_to_triangle_mesh(
            inst_mesh, map_vertices, 
            color2=vertex_colors, label2=out_vertex_inst_id,
            dist_thresh=0.05, device=inst_mesh.device
        )
        # write it via Plydata
        write_ply_with_labels(out_sem_mesh_f, 
            map_vertices.numpy(), map_sem_c, map_sem_label, 
            triangle_indices
        )
        write_ply_with_labels(out_inst_mesh_f, 
            map_vertices.numpy(), map_inst_c,  map_inst_id, 
            triangle_indices
        )

    if remove_raw_recon:
        os.remove(pred_inst_mesh_f)




if __name__=="__main__":
    # set files path
    args = parse_args()

    scene_list = args.scene_num
    if scene_list == 'all':
        # scene_list = ['scene0011_00', 'scene0011_01', 
        #     'scene0050_00', 'scene0050_01', 'scene0050_02', 
        #     'scene0084_00', 'scene0084_01', 'scene0084_02', 
        #     'scene0168_00', 'scene0168_01', 'scene0168_02', 
        #     'scene0231_00', 'scene0231_01', 'scene0231_02', 
        #     'scene0378_00', 'scene0378_01', 'scene0378_02', 
        #     'scene0518_00']
        scene_list = ['office0', 'office1', 'office2', 'office3',
                     'office4', 'room0', 'room1', 'room2']
    else:
        scene_list = [scene_list]

    for scene_id in scene_list:
        args.scene_num = scene_id
        print(f'Processing scene: {scene_id}')
        main(args)