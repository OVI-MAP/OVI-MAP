import os, sys, time, argparse, pickle, logging, glob
from os.path import join as pjoin
import json

import numpy as np
import torch
from plyfile import PlyData

from .eval_utils import init, evaluate
from scripts.utils.mesh_postprocess_utils import init_label_colormap

def map_gt_mesh(cfg_gt_mesh):
    res_folder = cfg_gt_mesh["res_folder"]
    
    gt_inst_mesh = PlyData.read(cfg_gt_mesh["inst_mesh_f"])
    gt_sem_mesh = PlyData.read(cfg_gt_mesh["sem_mesh_f"])

    inst_ids = gt_inst_mesh['vertex']['label']
    sem_labels = gt_sem_mesh['vertex']['label']
    sem_label_set = np.unique(sem_labels)

    vertex_num = sem_labels.shape[0]
    sem_inst_ids = np.zeros(vertex_num, dtype=np.int32)
    for sem_label in sem_label_set:
        if sem_label == 0:
            continue
        mapped_insts = inst_ids[sem_labels == sem_label]

        inclass_obj_num = 0
        for inst_id in np.unique(mapped_insts):
            if inst_id == 0:
                continue
            # assign instance id to vertexes
            inst_w_sem_id = int(sem_label * 1000 + inclass_obj_num)
            sem_inst_ids[inst_ids == inst_id] = inst_w_sem_id
            inclass_obj_num += 1

    out_gt_sem_inst_id_f = pjoin(res_folder, 'gt_sem_inst_id.npy')
    np.save(out_gt_sem_inst_id_f, sem_inst_ids)
    return out_gt_sem_inst_id_f



def map_pred_mesh(cfg_pred_mesh):

    res_folder = cfg_pred_mesh["res_folder"]

    pred_inst_mesh = PlyData.read(cfg_pred_mesh["inst_mesh_f"])
    pred_sem_mesh = PlyData.read(cfg_pred_mesh["sem_mesh_f"])

    inst_ids = pred_inst_mesh['vertex']['label']
    sem_labels = pred_sem_mesh['vertex']['label']
    inst_id_set = np.unique(inst_ids)

    min_region_sizes = 100

    sem_max_vertices_in_inst = {}
    inst_info = {}
    inst_id = 0

    for inst_id in inst_id_set:
        # skip the unseen part
        if inst_id == 0:
            continue

        inst_mask = (inst_ids == inst_id)
        area_inst = np.count_nonzero(inst_mask)
        if area_inst < min_region_sizes:
            continue
        # fing the semantic class id for this instance
        mapped_sem_labels = sem_labels[inst_mask]
        cnt_sem_labels = np.bincount(mapped_sem_labels)
        sem_class_id = np.argmax(cnt_sem_labels)
        if sem_class_id != 0:
            inst_info[inst_id] = {
                'mask':inst_mask, 'area':area_inst, 'class':sem_class_id,
                'confidence':0, 'file': None
            }
        # print(f'instance id: {inst_id}, semantic class id: {list(color_map)[sem_class_id]}, area: {area_inst}')
        
        if sem_class_id not in sem_max_vertices_in_inst:
            sem_max_vertices_in_inst[sem_class_id] = area_inst
        else:
            sem_max_vertices_in_inst[sem_class_id] = max(sem_max_vertices_in_inst[sem_class_id], area_inst)


    # calculate confidence for each instance
    for inst_id in inst_info.keys():
        inst_info[inst_id]['confidence'] = np.sum(inst_info[inst_id]['area']) * 1.0 / sem_max_vertices_in_inst[inst_info[inst_id]['class']]

    out_pred_inst_info_f = pjoin(res_folder, 'pred_inst_sem_mapping.txt')
    with open(out_pred_inst_info_f, 'w') as f:
        for inst_id, info in inst_info.items():
            out_inst_info_f = f"inst-{inst_id}_label-{info['class']}.npy"
            inst_info[inst_id]['file'] = pjoin(res_folder, out_inst_info_f)

            f.write(f"{out_inst_info_f} {info['class']} {info['confidence']:.6f}\n")
        f.close()

    # save the instance masks to npy files
    for inst_id in inst_info.keys():
        inst_save_path = inst_info[inst_id]['file']
        if(inst_save_path is not None):
            mask_to_write = inst_info[inst_id]['mask'].reshape(-1,1)
            np.save(inst_save_path, mask_to_write)

    return out_pred_inst_info_f


def eval_per_class_IoU(cfg_gt_mesh_list, cfg_pred_mesh_list, valid_ids, labels):
    """
    Evaluate per class IoU and accuracy on every vertex
    valid_ids: list of valid class ids, including the background class (0)
    """
    # we only take the valid classes into account
    num_classes = len(valid_ids)
    confusion = np.zeros([num_classes, num_classes], dtype=np.ulonglong)

    for i in range(len(cfg_gt_mesh_list)):

        gt_ply = PlyData.read(cfg_gt_mesh_list[i]['sem_mesh_f'])
        gt_sem_ids = gt_ply['vertex']['label']
        pred_ply = PlyData.read(cfg_pred_mesh_list[i]['sem_mesh_f'])
        pred_sem_ids = pred_ply['vertex']['label']
        gt_sem_id_set = np.unique(gt_sem_ids)
        pred_sem_id_set = np.unique(pred_sem_ids)

        valid_gt_mask = (gt_sem_ids != 0)
        gt_sem_ids = gt_sem_ids[valid_gt_mask]
        pred_sem_ids = pred_sem_ids[valid_gt_mask]

        # map the real sem ids to ordinal ids
        mapped_gt_sem_ids = np.zeros_like(gt_sem_ids)
        mapped_pred_sem_ids = np.zeros_like(pred_sem_ids)
        valid_id_mask = np.ones_like(gt_sem_ids).astype(bool)

        for gt_id in gt_sem_id_set:
            m = (gt_sem_ids == gt_id)
            if gt_id in valid_ids:
                mapped_id = valid_ids.index(gt_id)
                mapped_gt_sem_ids[m] = mapped_id
                continue
            valid_id_mask &= np.logical_not(gt_sem_ids == gt_id)
        # the labels in the pred mesh are always valid, guaranteed by the prev step
        for pred_id in pred_sem_id_set:
            m = (pred_sem_ids == pred_id)
            if pred_id not in valid_ids:
                mapped_id = 0
            else:
                mapped_id = valid_ids.index(pred_id)
            mapped_pred_sem_ids[m] = mapped_id

        # for (gt_val, pred_val) in zip(gt_sem_ids, pred_sem_ids):
        #     confusion[gt_val][pred_val] += 1
        mapped_gt_sem_ids = mapped_gt_sem_ids[valid_id_mask]
        mapped_pred_sem_ids = mapped_pred_sem_ids[valid_id_mask]
        np.add.at(confusion, (mapped_gt_sem_ids, mapped_pred_sem_ids), 1)

        sys.stdout.write("\rscans processed: {}".format(i+1))
        sys.stdout.flush()
    
    class_ious = {}
    # print('\nclasses \t IoU \t Acc')
    for i, label_id in enumerate(valid_ids):
        if label_id == 0:
            continue
        if np.longlong(confusion[i, :].sum()) == 0:
            # print(f'{labels[i]} has no gt labels, skip')
            continue
        
        label_name = labels[i]
        tp = np.longlong(confusion[i, i])
        fn = np.longlong(confusion[i, :].sum()) - tp
        fp = np.longlong(confusion[:, i].sum()) - tp
        denom = float(tp + fp + fn)
        # if denom == 0:
        #     class_ious[label_name] = (0.0, 0.0) #float('nan')
        # else:
        iou = tp / denom if denom > 0 else 0.0
        acc = tp / float(tp + fn) if tp + fn > 0 else 0.0
        class_ious[label_name] = (iou, acc)
        # print(f'{label_name:<14s} - {iou:>5.2%} {acc:>6.2%}')

    iou_values = np.array([i[0] for i in class_ious.values()])
    acc_values = np.array([i[1] for i in class_ious.values()])
    print(f'\nmIoU\tmAcc')
    print(f'{np.mean(iou_values):.4f}\t{np.mean(acc_values):.4f}')



def parse_args():
    parse = argparse.ArgumentParser(description='Semantic Mapping-Python') 
    
    parse.add_argument("--result_folder", type=str, 
        default='/home/zilong/Disk_data/semantic_mapping_result', 
        help="folder of mapping results")
    
    return parse.parse_args()




# python -m scripts.eval_sem_seg

def main(args):

    result_folder = args.result_folder

    # TODO modify the task
    task = 'Replica' # NYU40 | CoCoPano | Replica | Scannet200 | Scannet20
    _, labels, _, valid_ids = init_label_colormap(task)

    # scene_list = ['scene0011_00', 'scene0011_01', 
    #     'scene0050_00', 'scene0050_01', 'scene0050_02', 
    #     'scene0084_00', 'scene0084_01', 'scene0084_02', 
    #     'scene0168_00', 'scene0168_01', 'scene0168_02', 
    #     'scene0231_00', 'scene0231_01', 'scene0231_02', 
    #     'scene0378_00', 'scene0378_01', 'scene0378_02', 
    #     'scene0518_00']
    scene_list = ['office0', 'office1', 'office2', 'office3', 
                  'office4', 'room0', 'room1', 'room2']
    cfg_gt_mesh_list = []
    cfg_pred_mesh_list = []
    
    gt_files = []
    pred_files = [] 
    
    eval_per_point_semantic = True
    load_prev = False

    for scene_num in scene_list:
        print(f'Processing scene: {scene_num}')
        # cropformer_inst | cropformer_inst_combine | gt_inst_vis
        seq_folder = pjoin(result_folder, scene_num, 'cropformer_inst_rt')
        #  | eval_gt_inst_incre_vis
        eval_folder = pjoin(seq_folder, 'eval_ours')
        # eval_folder = pjoin(seq_folder, '..', 'eval_ours_OM3D')

        res_f = os.path.join(result_folder, 
            "results_replica_ours_real_time.json")

        if not os.path.exists(eval_folder):
            os.makedirs(eval_folder, exist_ok=True)
        else:
            load_prev = True
        load_prev = False

        # remember to change the mapped semantics  _nyu40
        gt_inst_mesh_f = pjoin(seq_folder, '..', 'gt_instance_mesh.ply')
        gt_sem_mesh_f = pjoin(seq_folder, '..', 'gt_semantic_mesh.ply')

        
        pred_inst_mesh_f = sorted(glob.glob(pjoin(seq_folder, 'instance_mesh_*.ply')))[0]
        frame_cnt = os.path.basename(pred_inst_mesh_f)[14:-4]
        # frame_cnt = 160
        pred_inst_mesh_f = pjoin(seq_folder, f'instance_map_gt_{frame_cnt}.ply')
        pred_sem_mesh_f = pjoin(seq_folder, f'semantic_map_gt_{frame_cnt}.ply')
        # pred_sem_mesh_f = pjoin(seq_folder, '..', 'ours_openmask3d_sem.ply')

        # pred_inst_mesh_f = pjoin(seq_folder, '..', 'gt_instance_mesh.ply')
        # pred_sem_mesh_f = pjoin(seq_folder, 'gt_inst_sem.ply')

        # pred_inst_mesh_f = pjoin(seq_folder, '..', 'segment3d_instance.ply')
        # pred_sem_mesh_f = pjoin(seq_folder, '..', 'segment3d_openmask3d_sem.ply')
        
        # ========================================================================

        
        cfg_gt_mesh = {
            "sem_mesh_f": gt_sem_mesh_f,
            "inst_mesh_f": gt_inst_mesh_f,
            "res_folder": eval_folder,
        }
        cfg_gt_mesh_list.append(cfg_gt_mesh)
        cfg_pred_mesh = {
            "sem_mesh_f": pred_sem_mesh_f,
            "inst_mesh_f": pred_inst_mesh_f,
            "res_folder": eval_folder,
        }
        cfg_pred_mesh_list.append(cfg_pred_mesh)

        if load_prev:
            gt_file_path = pjoin(eval_folder, 'gt_sem_inst_id.npy')
            pred_file_path = pjoin(eval_folder, 'pred_inst_sem_mapping.txt')
        else:
            gt_file_path = map_gt_mesh(cfg_gt_mesh)
            pred_file_path = map_pred_mesh(cfg_pred_mesh)
        gt_files.append(gt_file_path)
        pred_files.append(pred_file_path)

    if eval_per_point_semantic:
        print(f"Evaluating per vertex semantic results...")
        eval_per_class_IoU(cfg_gt_mesh_list, cfg_pred_mesh_list, valid_ids, labels)

    init(task)
    print(f"Evaluating {len(pred_files)} semantic mapping results...")
    avgs = evaluate(
        result_folder, pred_files, gt_files, 
        result_folder
    )

    # Save results
    with open(res_f, "w") as f:
        json.dump(avgs, f)



if __name__=="__main__":
    args = parse_args()
    main(args)

        