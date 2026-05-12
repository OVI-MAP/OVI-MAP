import os, time, argparse, glob
from os.path import join as pjoin

import numpy as np
from plyfile import PlyData
import open3d as o3d
import open3d.core as o3c # type: ignore

def calculate_chamfer_distance(gt_inst_mesh_f, pred_inst_mesh_f, dist_thres=0.05):

    gt_ply = PlyData.read(gt_inst_mesh_f)
    num_gt_vertices = gt_ply['vertex']['label'].shape[0]
    gt_vertices = np.zeros((num_gt_vertices, 3), dtype=np.float32)
    gt_vertices[:, 0] = gt_ply['vertex']['x']
    gt_vertices[:, 1] = gt_ply['vertex']['y']
    gt_vertices[:, 2] = gt_ply['vertex']['z']
    gt_vertices = o3c.Tensor(gt_vertices, dtype=o3c.float32)

    inst_mesh = o3d.t.io.read_triangle_mesh(pred_inst_mesh_f)
    pred_ply = PlyData.read(pred_inst_mesh_f)
    num_pred_vertices = pred_ply['vertex']['label'].shape[0]
    pred_vertices = np.zeros((num_pred_vertices, 3), dtype=np.float32)
    pred_vertices[:, 0] = pred_ply['vertex']['x']
    pred_vertices[:, 1] = pred_ply['vertex']['y']
    pred_vertices[:, 2] = pred_ply['vertex']['z']
    pred_vertices = inst_mesh.vertex.positions


    knn_pred = o3c.nns.NearestNeighborSearch(pred_vertices)
    knn_pred.knn_index()
    knn_gt = o3c.nns.NearestNeighborSearch(gt_vertices)
    knn_gt.knn_index()

    indices, distances_sqr = knn_gt.knn_search(pred_vertices, 1)
    indices = indices.reshape(-1)
    distances_sqr = distances_sqr.reshape(-1)
    matched = distances_sqr < dist_thres**2
    distances_matched_sqr = distances_sqr[matched]
    dist_pred2gt = o3c.Tensor.mean(distances_matched_sqr).cpu().numpy()

    indices, distances_sqr = knn_pred.knn_search(gt_vertices, 1)
    indices = indices.reshape(-1)
    distances_sqr = distances_sqr.reshape(-1)
    matched = distances_sqr < dist_thres**2
    distances_matched_sqr = distances_sqr[matched]
    dist_gt2pred = o3c.Tensor.mean(distances_matched_sqr).cpu().numpy()

    chamfer_distance = dist_pred2gt + dist_gt2pred
    print(f'{chamfer_distance:.4f}')


def parse_args():
    parse = argparse.ArgumentParser(description='Semantic Mapping-Python') 
    # files path
    parse.add_argument("--scene_num", type=str, required=True, 
        help="which scene for mapping ")
    
    parse.add_argument("--result_folder", type=str, 
        default='/home/zilong/Disk_data/semantic_mapping_result', 
        help="folder of mapping results")

    return parse.parse_args()


# python -m scripts.eval_chamfer_dist --scene_num office0


def main(args):
    scene_num = args.scene_num
    result_folder = args.result_folder

    # 'original_res_close-sem' | 'cropformer_inst_res'
    result_folder = pjoin(result_folder, scene_num, 'cropformer_inst')
    if not os.path.exists(result_folder):
        raise FileNotFoundError(f"Result folder {result_folder} does not exist.")

    gt_inst_mesh_f = pjoin(result_folder, '..', 'gt_instance_mesh.ply')
    pred_inst_mesh_f = glob.glob(pjoin(result_folder, 'instance_mesh_*.ply'))[0]

    # # for segment3D results
    # pred_inst_mesh_f = pjoin(result_folder, '..', 'segment3d_instance.ply')
    # for mask3D results
    # pred_inst_mesh_f = pjoin(result_folder, '..', 'mask3d_instance.ply')

    calculate_chamfer_distance(gt_inst_mesh_f, pred_inst_mesh_f, dist_thres=0.05)



if __name__=="__main__":
    args = parse_args()

    scene_list = args.scene_num
    if scene_list == 'all':
        scene_list = ['office0', 'office1', 'office2', 'office3', 
                     'office4', 'room0', 'room1', 'room2']
    else:
        scene_list = [scene_list]

    for scene_id in scene_list:
        args.scene_num = scene_id
        # print(f'Processing scene: {scene_id}')
        main(args)

        