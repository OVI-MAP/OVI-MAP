import os, time, argparse, logging, glob
from os.path import join as pjoin

import numpy as np
from plyfile import PlyData


def assign_pred_inst_to_gt_inst(gt_inst_mesh_f, pred_inst_mesh_f, res_folder):
  
    gt_ply = PlyData.read(gt_inst_mesh_f)
    gt_inst_ids = gt_ply['vertex']['label']
    gt_inst_set = np.unique(gt_inst_ids)
    pred_ply = PlyData.read(pred_inst_mesh_f)
    pred_inst_ids = pred_ply['vertex']['label']
    pred_inst_set = np.unique(pred_inst_ids)

    vis_mapping = False

    if vis_mapping:
        gt_v_pos = np.stack([
            gt_ply['vertex']['x'], gt_ply['vertex']['y'], gt_ply['vertex']['z']
        ], axis=-1)
        gt_v_c = np.stack([
            gt_ply['vertex']['red'], gt_ply['vertex']['green'], gt_ply['vertex']['blue']
        ], axis=-1) # [0, 255]
        gt_tri = np.stack(gt_ply['face']['vertex_indices'])
        gt_tri_c = gt_v_c[gt_tri[:, 0]]
        pred_v_pos = np.stack([
            pred_ply['vertex']['x'], pred_ply['vertex']['y'], pred_ply['vertex']['z']
        ], axis=-1)
        pred_v_c = np.stack([
            pred_ply['vertex']['red'], pred_ply['vertex']['green'], pred_ply['vertex']['blue']
        ], axis=-1) # [0, 255]
        pred_tri = np.stack(pred_ply['face']['vertex_indices'])
        pred_tri_c = pred_v_c[pred_tri[:, 0]]

        import rerun as rr
        rr.init("inst_overlap", recording_id="inst_overlap", spawn=True)

    min_region_sizes = 10

    # for each gt instance, find a matching predicted instance with max overlap
    inst_mapping = {}
    gt_inst_iou_scores = []
    area_gt_inst = []

    for i, gt_inst_id in enumerate(gt_inst_set):
        if gt_inst_id == 0:
            continue
        gt_inst_mask = (gt_inst_ids == gt_inst_id)
        gt_inst_area = np.count_nonzero(gt_inst_mask)
        
        if gt_inst_area < min_region_sizes:
            continue

        area_gt_inst.append(gt_inst_area)
        
        mapped_pred_inst_id = pred_inst_ids[gt_inst_mask]
        mapped_pred_inst_set, area_intersects = np.unique(mapped_pred_inst_id, return_counts=True)
        area_intersects[mapped_pred_inst_set == 0] = 0  # ignore background

        # max_area_pred_inst = np.argmax(area_map_pred_inst)
        # inst_mapping[gt_inst_id] = max_area_pred_inst

        # intersect = area_map_pred_inst[max_area_pred_inst]
        # pred_inst_mask = (pred_inst_ids == max_area_pred_inst)
        # pred_inst_area = np.count_nonzero(pred_inst_mask)
        # iou_score = intersect / (gt_inst_area + pred_inst_area - intersect)

        # find the one with maximum IoU instead of maximum intersection
        area_pred_inst = []
        for pred_inst_id in mapped_pred_inst_set:
            if pred_inst_id == 0:
                area_pred_inst.append(0)
                continue
            area_pred_inst.append(np.count_nonzero(pred_inst_ids == pred_inst_id))
        area_pred_inst = np.array(area_pred_inst)

        iou_scores = area_intersects / (gt_inst_area + area_pred_inst - area_intersects)
        max_iou_idx = np.argmax(iou_scores)
        max_iou_score = iou_scores[max_iou_idx]
        best_pred_inst_id = mapped_pred_inst_set[max_iou_idx]
        inst_mapping[gt_inst_id] = best_pred_inst_id
        gt_inst_iou_scores.append(max_iou_score)


        if vis_mapping:
            rr.set_time_seconds("stable_time", i)
            tri_mask = np.all(gt_tri_c == gt_v_c[gt_inst_mask][0], axis=-1)
            tris = gt_tri[tri_mask].flatten()
            tri_idx = np.arange(tris.shape[0]).reshape(-1, 3)
            rr.log(
                f"mesh/gt_inst", rr.Mesh3D(
                    vertex_positions=gt_v_pos[tris],
                    vertex_colors=gt_v_c[tris],
                    triangle_indices=tri_idx,
                )
            )
            tri_mask = np.all(pred_tri_c == pred_v_c[pred_inst_mask][0], axis=-1)
            tris = pred_tri[tri_mask].flatten()
            tri_idx = np.arange(tris.shape[0]).reshape(-1, 3)
            rr.log(
                f"mesh/pred_inst", rr.Mesh3D(
                    vertex_positions=pred_v_pos[tris],
                    vertex_colors=pred_v_c[tris],
                    triangle_indices=tri_idx,
                )
            )

    area_gt_inst = np.array(area_gt_inst)
    area_weight = area_gt_inst / np.sum(area_gt_inst)

    gt_inst_iou_scores = np.array(gt_inst_iou_scores)
    weighted_iou_scores = gt_inst_iou_scores * area_weight

    tp_25, fp_25 = 0, 0
    tp_50, fp_50 = 0, 0
    tp_75, fp_75 = 0, 0
    for i, pred_inst_id in enumerate(pred_inst_set):
        if pred_inst_id == 0:
            continue
        pred_inst_mask = (pred_inst_ids == pred_inst_id)
        pred_inst_area = np.count_nonzero(pred_inst_mask)
        if pred_inst_area < min_region_sizes:
            continue
        
        mapped_gt_inst_id = gt_inst_ids[pred_inst_mask]
        mapped_gt_inst_set, area_intersects = np.unique(mapped_gt_inst_id, return_counts=True)
        area_intersects[mapped_gt_inst_set == 0] = 0  # ignore background

        # find the one with maximum IoU instead of maximum intersection
        area_gt_inst = []
        for gt_inst_id in mapped_gt_inst_set:
            if gt_inst_id == 0:
                area_gt_inst.append(0)
                continue
            area_gt_inst.append(np.count_nonzero(gt_inst_ids == gt_inst_id))
        area_gt_inst = np.array(area_gt_inst)

        iou_scores = area_intersects / (pred_inst_area + area_gt_inst - area_intersects)
        max_iou_idx = np.argmax(iou_scores)
        max_iou_score = iou_scores[max_iou_idx]

        if max_iou_score >= 0.25:
            tp_25 += 1
        else:
            fp_25 += 1
        if max_iou_score >= 0.50:
            tp_50 += 1
        else:
            fp_50 += 1
        if max_iou_score >= 0.75:
            tp_75 += 1
        else:
            fp_75 += 1

    rec_25 = tp_25 / len(gt_inst_iou_scores) # tp / (tp + fn)
    prec_25 = tp_25 / (tp_25 + fp_25)
    rec_50 = tp_50 / len(gt_inst_iou_scores)
    prec_50 = tp_50 / (tp_50 + fp_50)
    rec_75 = tp_75 / len(gt_inst_iou_scores)
    prec_75 = tp_75 / (tp_75 + fp_75)

    print(f'{np.mean(gt_inst_iou_scores):.4f}\t{np.sum(weighted_iou_scores):.4f}\t{prec_75:.4f}\t{rec_75:.4f}\t{prec_50:.4f}\t{rec_50:.4f}\t{prec_25:.4f}\t{rec_25:.4f}')



def parse_args():
    parse = argparse.ArgumentParser(description='Semantic Mapping-Python') 
    # files path
    parse.add_argument("--scene_num", type=str, required=True, 
        help="which scene for mapping ")
    
    parse.add_argument("--result_folder", type=str, 
        default='/home/zilong/Disk_data/semantic_mapping_result', 
        help="folder of mapping results")

    return parse.parse_args()


# python -m scripts.eval_inst_seg --scene_num office0


def main(args):
    scene_num = args.scene_num
    result_folder = args.result_folder

    result_folder = pjoin(result_folder, scene_num, 'cropformer_inst')
    if not os.path.exists(result_folder):
        raise FileNotFoundError(f"Result folder {result_folder} does not exist.")

    gt_inst_mesh_f = pjoin(
        result_folder, '..', 'gt_instance_mesh.ply')
    pred_inst_mesh_f = glob.glob(
        pjoin(result_folder, 'instance_map_gt_*.ply'))[0]

    assign_pred_inst_to_gt_inst(gt_inst_mesh_f, pred_inst_mesh_f, result_folder)



if __name__=="__main__":
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

    print('mIoU\twIoU\tmP@75\tmR@75\tmP@50\tmR@50\tmP@25\tmR@25')

    for scene_id in scene_list:
        args.scene_num = scene_id
        # print(f'Processing scene: {scene_id}')
        main(args)

        