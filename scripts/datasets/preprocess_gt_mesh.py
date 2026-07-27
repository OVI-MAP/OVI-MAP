import os, argparse, copy, json, glob, tqdm
from os.path import join as pjoin
from plyfile import PlyData, PlyElement

import numpy as np
import open3d as o3d
import open3d.core as o3c # type: ignore

from scripts.utils.mesh_postprocess_utils import write_ply_with_labels
from scripts.visualizations.vis_utils import get_new_pallete


def convert_replica_sem_inst_mesh(data_folder, scene_num, res_folder,
                                   gt_sem_folder=None, gt_inst_folder=None):
    from ..utils.semantic_const import Replica_map_to_reduced, REPLICA_52_PALETTE

    if gt_sem_folder is None:
        gt_sem_folder = pjoin(data_folder, '..', 'replica_gt_semantics')
    if gt_inst_folder is None:
        gt_inst_folder = pjoin(data_folder, '..', 'replica_gt_instances')

    gt_sem_list_f = pjoin(gt_sem_folder, f'semantic_labels_{scene_num}.txt')
    gt_sem_list = np.loadtxt(gt_sem_list_f, dtype=int)
    gt_inst_list_f = pjoin(gt_inst_folder, f'instance_labels_{scene_num}.txt')
    gt_inst_list = np.loadtxt(gt_inst_list_f, dtype=int)
    gt_mesh_f = pjoin(data_folder, '..', f'{scene_num}_mesh.ply')

    gt_sem_set = np.unique(gt_sem_list)
    gt_inst_set = np.unique(gt_inst_list)

    plydata = PlyData.read(gt_mesh_f)
    num_vertex = len(plydata['vertex']['x'])
    vertex_pos = np.zeros((num_vertex, 3), dtype=np.float32)
    vertex_pos[:, 0] = plydata['vertex']['x']
    vertex_pos[:, 1] = plydata['vertex']['y']
    vertex_pos[:, 2] = plydata['vertex']['z']

    faces = plydata['face']['vertex_indices']

    inst_colors = np.zeros((num_vertex, 3), dtype=np.float32)
    inst_labels = np.zeros((num_vertex,), dtype=np.uint16)

    sem_colors = np.zeros((num_vertex, 3), dtype=np.float32)
    sem_labels = np.zeros((num_vertex,), dtype=np.uint16)

    for sem_id in gt_sem_set:
        sem_mask = (gt_sem_list == sem_id)
        # NOTE we shift the mapped sem label by +1, and bg id becomes 0
        map_sem_id = Replica_map_to_reduced[sem_id] + 1
        sem_colors[sem_mask] = np.array(REPLICA_52_PALETTE[map_sem_id], dtype='float') / 255.0
        sem_labels[sem_mask] = map_sem_id

    inst_color_map = get_new_pallete(len(gt_inst_set), return_type='float')
    for i, inst_i in enumerate(gt_inst_set):
        inst_mask = (gt_inst_list == inst_i)
        inst_colors[inst_mask] = inst_color_map[i, :]
        inst_labels[inst_mask] = i

    # save the new mesh
    write_ply_with_labels(
        pjoin(res_folder, 'gt_semantic_mesh.ply'), 
        vertex_pos, sem_colors, sem_labels, faces
    )
    write_ply_with_labels(
        pjoin(res_folder, 'gt_instance_mesh.ply'),
        vertex_pos, inst_colors, inst_labels, faces
    )
        


def convert_scannet_mesh(data_folder, scene_num, res_folder, map_scannet200=True):
    if map_scannet200:
        # for coloring the mesh as much as possible
        from ..utils.semantic_const import SCANNET_COLOR_MAP_200
    else:
        from ..utils.semantic_const import NYU_41_PALETTE
    import pandas as pd

    gt_sem_mesh_f = pjoin(
        data_folder, f'{scene_num}_vh_clean_2.labels.ply')
    gt_segs_f = pjoin(
        data_folder, f'{scene_num}_vh_clean_2.0.010000.segs.json')
    gt_inst_to_segs_f = pjoin(
        data_folder, f'{scene_num}.aggregation.json')
    labels_table = pd.read_csv(
        pjoin(data_folder, '..', 'scannetv2-labels.combined.tsv'), 
        sep="\t"
    )

    gt_sem_mesh = PlyData.read(gt_sem_mesh_f) # it's label in NYU40
    gt_inst_mesh = copy.deepcopy(gt_sem_mesh)

    out_inst_mesh_f = pjoin(res_folder, 'gt_instance_mesh.ply')
    out_sem_mesh_f = pjoin(res_folder, 'gt_semantic_mesh.ply')
    if not map_scannet200:
        out_sem_mesh_f = pjoin(res_folder, 'gt_semantic_mesh_nyu40.ply')

    # load the per vertex segment registration
    with open(gt_segs_f, 'r') as f:
        gt_segs = json.load(f)
    vertex_to_segs = np.array(gt_segs['segIndices'])
    # load the segments in gt instances
    with open(gt_inst_to_segs_f, 'r') as f:
        gt_inst_to_segs = json.load(f)

    gt_insts = gt_inst_to_segs['segGroups']
    inst_color_map = get_new_pallete(len(gt_insts))

    num_vertex = len(gt_sem_mesh['vertex']['label'])
    inst_labels = np.zeros(num_vertex, dtype=np.uint16)
    inst_colors = np.ones((num_vertex, 3), dtype=np.uint8) * 200
    sem_labels = np.zeros(num_vertex, dtype=np.uint16)
    sem_colors = np.ones((num_vertex, 3), dtype=np.uint8) * 200

    for inst_i in gt_insts:
        inst_id = inst_i['objectId']
        sem_label = inst_i['label']
        segs_in_inst = inst_i['segments']

        inst_mask = np.zeros_like(inst_labels, dtype=bool)
        for seg_id in segs_in_inst:
            seg_mask = (vertex_to_segs == seg_id)
            inst_mask |= seg_mask

        inst_labels[inst_mask] = inst_id
        inst_colors[inst_mask] = inst_color_map[inst_id]

        if map_scannet200:
            raw_sem_id = labels_table[labels_table["raw_category"] == sem_label]["id"].values[0]
            if raw_sem_id not in SCANNET_COLOR_MAP_200.keys():
                print(f"[Warning] Semantic label {sem_label} not found in SCANNET200.")
                # map_sem_id = 0  # background
                map_color = SCANNET_COLOR_MAP_200[0]
            else:
                # map_sem_id = list(SCANNET_COLOR_MAP_200.keys()).index(raw_sem_id)
                map_color = SCANNET_COLOR_MAP_200[raw_sem_id]
        else:
            # map to NYU 40 set
            raw_sem_id = labels_table[labels_table["raw_category"] == sem_label]["nyu40id"].values[0]
            map_color = NYU_41_PALETTE[raw_sem_id]
        
        sem_labels[inst_mask] = raw_sem_id # we store the real id
        sem_colors[inst_mask] = np.array(map_color, dtype='uint8')

    gt_inst_mesh['vertex']['label'] = inst_labels
    gt_inst_mesh['vertex']['red'] = inst_colors[:, 0]
    gt_inst_mesh['vertex']['green'] = inst_colors[:, 1]
    gt_inst_mesh['vertex']['blue'] = inst_colors[:, 2]
    gt_inst_mesh.write(out_inst_mesh_f)

    gt_inst_mesh['vertex']['label'] = sem_labels
    gt_inst_mesh['vertex']['red'] = sem_colors[:, 0]
    gt_inst_mesh['vertex']['green'] = sem_colors[:, 1]
    gt_inst_mesh['vertex']['blue'] = sem_colors[:, 2]
    gt_inst_mesh.write(out_sem_mesh_f)



def crop_gt_mesh(gt_mesh, recon_mesh, out_mesh_f):
    dist_thresh = 0.05

    gt_vertex = np.stack([
        gt_mesh['vertex']['x'], gt_mesh['vertex']['y'], gt_mesh['vertex']['z']
    ], axis=-1)
    gt_color = np.stack([
        gt_mesh['vertex']['red'], gt_mesh['vertex']['green'], gt_mesh['vertex']['blue']
    ], axis=-1).astype(np.float32) / 255.0  # [0, 1]
    gt_tri = gt_mesh['face']['vertex_indices']
    num_vertices = gt_vertex.shape[0]

    # ========== map the gt mesh to our reconstructed mesh ==========
    gt_vertex = o3d.core.Tensor(gt_vertex, dtype=o3d.core.float32)
    knn_data = o3c.nns.NearestNeighborSearch(gt_vertex)
    knn_data.knn_index()
    indices, distances_sqr = knn_data.knn_search(recon_mesh.vertex.positions, 1)
    indices = indices.reshape(-1).numpy()
    matched = (distances_sqr < dist_thresh**2).reshape(-1).numpy()

    m_idx = np.unique(indices[matched])
    inv_mapping = np.full(num_vertices, -1, dtype=np.int32)
    inv_mapping[m_idx] = np.arange(len(m_idx))

    new_triangles = []
    for face in tqdm.tqdm(gt_tri):
        new_face = inv_mapping[face]
        # only keep the triangles has all vertices matched
        if np.all(new_face >= 0):
            new_triangles.append(new_face)

    write_ply_with_labels(out_mesh_f, 
        vertex_positions=gt_vertex[m_idx].numpy(),
        vertex_colors=gt_color[m_idx],
        vertex_labels=None,
        triangles=new_triangles
    )


def map_part_to_gt(gt_mesh, pred_mesh, out_mesh_f):
    dist_thresh = 0.05

    gt_vertex = np.stack([
        gt_mesh['vertex']['x'], gt_mesh['vertex']['y'], gt_mesh['vertex']['z']
    ], axis=-1)
    gt_color = np.stack([
        gt_mesh['vertex']['red'], gt_mesh['vertex']['green'], gt_mesh['vertex']['blue']
    ], axis=-1).astype(np.float32) / 255.0  # [0, 1]
    gt_tri = gt_mesh['face']['vertex_indices']
    num_vertices = gt_vertex.shape[0]

    pred_vertex = np.stack([
        pred_mesh['vertex']['x'], pred_mesh['vertex']['y'], pred_mesh['vertex']['z']
    ], axis=-1)
    pred_color = np.stack([
        pred_mesh['vertex']['red'], pred_mesh['vertex']['green'], pred_mesh['vertex']['blue']
    ], axis=-1).astype(np.float32) / 255.0  # [0, 1]
    pred_label = pred_mesh['vertex']['label']
    pred_vertex = o3d.core.Tensor(pred_vertex, dtype=o3d.core.float32)

    output_label = np.zeros((num_vertices,), dtype=np.uint16)
    output_color = np.ones((num_vertices, 3), dtype=np.float32) * (200.0/255.0)

    knn_data = o3c.nns.NearestNeighborSearch(pred_vertex)
    knn_data.knn_index()

    gt_vertex = o3d.core.Tensor(gt_vertex, dtype=o3d.core.float32)
    indices, distances_sqr = knn_data.knn_search(gt_vertex, 1)
    indices = indices.reshape(-1).numpy()
    matched = (distances_sqr < dist_thresh**2).reshape(-1).numpy()

    idxs = np.arange(num_vertices)
    output_color[idxs[matched]] = pred_color[indices[matched]]
    output_label[idxs[matched]] = pred_label[indices[matched]]

    write_ply_with_labels(out_mesh_f, 
        vertex_positions=gt_vertex.numpy(),
        vertex_colors=output_color,
        vertex_labels=output_label,
        triangles=gt_tri
    )

    
    

# python -m scripts.datasets.preprocess_gt_mesh --scene_num office0

def main(args):
    scene_num = args.scene_num
    data_path = args.data_folder
    scene_folder = pjoin(data_path, scene_num)

    result_folder = args.result_folder
    result_folder = pjoin(result_folder, scene_num)
    os.makedirs(result_folder, exist_ok=True)

    # # for scannet
    # convert_scannet_mesh(scene_folder, scene_num, result_folder, map_scannet200=False)
    # for replica
    convert_replica_sem_inst_mesh(
        scene_folder, scene_num, result_folder,
        gt_sem_folder=args.gt_sem_folder,
        gt_inst_folder=args.gt_inst_folder,
    )



def parse_args():
    parse = argparse.ArgumentParser(description='Semantic Mapping-Python')
    # files path
    parse.add_argument("--scene_num", type=str, required=True,
        help="which scene to process, use 'all' for all scenes")

    parse.add_argument("--data_folder", type=str, required=True,
        help="path to the dataset folder (e.g., .../Replica)")
    parse.add_argument("--result_folder", type=str, required=True,
        help="path to the output folder for mapped GT meshes")
    parse.add_argument("--gt_sem_folder", type=str, default=None,
        help="path to Replica ground-truth semantic labels folder (default: <data_folder>/../replica_gt_semantics)")
    parse.add_argument("--gt_inst_folder", type=str, default=None,
        help="path to Replica ground-truth instance labels folder (default: <data_folder>/../replica_gt_instances)")

    return parse.parse_args()

# python -m scripts.datasets.preprocess_gt_mesh --scene_num all --data_folder /data/Datasets/Replica --result_folder /data/semantic_mapping_result --gt_sem_folder /workspace/scripts/datasets/replica_gt_semantics --gt_inst_folder /workspace/scripts/datasets/replica_gt_instances

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