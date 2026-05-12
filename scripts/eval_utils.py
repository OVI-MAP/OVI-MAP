import os, json, copy
from os.path import join as pjoin

import numpy as np


def init(task):

    global overlaps
    overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
    # overlaps = np.array([0.75, 0.50, 0.25])
    global min_region_sizes
    min_region_sizes = np.array([100])
    global dist_threshes
    dist_threshes = np.array([float('inf')])
    global dist_confs
    dist_confs = np.array([-float('inf')])

    global CLASS_LABELS
    global VALID_CLASS_IDS
    if task == 'NYU40':
        from scripts.utils.semantic_const import NYU_40
        CLASS_LABELS = NYU_40
        VALID_CLASS_IDS = [i+1 for i in range(len(CLASS_LABELS))]
    elif task == 'CoCo':
        from scripts.utils.semantic_const import COCO_PANOPTIC_133
        CLASS_LABELS = COCO_PANOPTIC_133
        VALID_CLASS_IDS = [i+1 for i in range(len(CLASS_LABELS))]
    elif task == 'Replica':
        from scripts.utils.semantic_const import REPLICA_51, CLASS_LABELS_REPLICA
        all_labels = ['background'] + REPLICA_51
        CLASS_LABELS = []
        VALID_CLASS_IDS = []
        for label in CLASS_LABELS_REPLICA:
            CLASS_LABELS.append(label)
            if label in all_labels:
                VALID_CLASS_IDS.append(all_labels.index(label))
    elif task == 'Scannet200':
        from scripts.utils.semantic_const import CLASS_LABELS_200, VALID_CLASS_IDS_200
        CLASS_LABELS = CLASS_LABELS_200
        VALID_CLASS_IDS = VALID_CLASS_IDS_200
    elif task == 'Scannet20':
        from scripts.utils.semantic_const import CLASS_LABELS_20, VALID_CLASS_IDS_20
        CLASS_LABELS = CLASS_LABELS_20
        VALID_CLASS_IDS = VALID_CLASS_IDS_20
    else:
        raise ValueError(" No matced task!")

    global ID_TO_LABEL # real id, not ordinal
    global LABEL_TO_ID
    ID_TO_LABEL = {}
    LABEL_TO_ID = {}
    # skip background
    for i in range(len(VALID_CLASS_IDS)):
        LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
        ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]
    


# ------------ Instance Utils ------------ #

class Instance(object):
    instance_id = 0
    label_id = 0
    vert_count = 0
    med_dist = -1
    dist_conf = 0.0

    def __init__(self, mesh_vert_instances, instance_id):
        if (instance_id == -1):
            return
        self.instance_id     = int(instance_id)
        self.label_id    = int(self.get_label_id(instance_id))
        self.vert_count = int(self.get_instance_verts(mesh_vert_instances, instance_id))

    def get_label_id(self, instance_id):
        return int(instance_id // 1000)

    def get_instance_verts(self, mesh_vert_instances, instance_id):
        return (mesh_vert_instances == instance_id).sum()

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def to_dict(self):
        dict = {}
        dict["instance_id"] = self.instance_id
        dict["label_id"]    = self.label_id
        dict["vert_count"]  = self.vert_count

        dict["med_dist"]    = self.med_dist
        dict["dist_conf"]   = self.dist_conf
        return dict

    def from_json(self, data):
        self.instance_id     = int(data["instance_id"])
        self.label_id        = int(data["label_id"])
        self.vert_count      = int(data["vert_count"])
        if ("med_dist" in data):
            self.med_dist    = float(data["med_dist"])
            self.dist_conf   = float(data["dist_conf"])

    def __str__(self):
        return "("+str(self.instance_id)+")"

def get_gt_instances(ids, class_ids, class_labels, id2label):
    instances = {}
    for label in class_labels:
        instances[label] = []
    instance_ids = np.unique(ids)
    for id in instance_ids:
        if id == 0:
            continue
        inst = Instance(ids, id) # get the class by dividing by 1000
        if inst.label_id in class_ids:
            instances[id2label[inst.label_id]].append(inst.to_dict())    
    return instances

def read_instance_prediction_file(filename, pred_path):
    lines = open(filename).read().splitlines()
    instance_info = {}
    abs_pred_path = os.path.abspath(pred_path)
    for line in lines:
        parts = line.split(' ')
        if len(parts) != 3:
            print('invalid instance prediction file. Expected (per line): [rel path prediction] [label id prediction] [confidence prediction]')
        if os.path.isabs(parts[0]):
            print('invalid instance prediction file. First entry in line must be a relative path')
        mask_file = os.path.join(os.path.dirname(filename), parts[0])
        mask_file = os.path.abspath(mask_file)
        # check that mask_file lives inside prediction path
        if os.path.commonprefix([mask_file, abs_pred_path]) != abs_pred_path:
            print('predicted mask {} in prediction text file {} points outside of prediction path.'.format(mask_file,filename))

        info            = {}
        info["label_id"] = int(float(parts[1]))
        info["conf"]    = float(parts[2])
        instance_info[mask_file]  = info
    return instance_info


def assign_instances_for_scan(pred_path, pred_file, gt_file):
    # get gt instances
    gt_ids = np.load(gt_file).astype(np.int32).reshape(-1)
    gt_instances = get_gt_instances(gt_ids, VALID_CLASS_IDS, CLASS_LABELS, ID_TO_LABEL)

    pred_inst_info = read_instance_prediction_file(pred_file, pred_path)

    # associate
    gt2pred = copy.deepcopy(gt_instances) # a dict of "sem_label": [inst1, ...]
    for label_name in gt2pred:
        for inst in gt2pred[label_name]:
            inst['matched_pred'] = []
    
    pred2gt = {}
    for label in CLASS_LABELS:
        pred2gt[label] = []

    
    # mask for not-counted labels
    bool_void = np.logical_not(np.in1d(gt_ids // 1000, VALID_CLASS_IDS))
    pred_inst_i = 0
    # go thru all prediction masks
    for pred_mask_file in pred_inst_info:
        label_id = int(pred_inst_info[pred_mask_file]['label_id'])
        conf = pred_inst_info[pred_mask_file]['conf']
        if not label_id in ID_TO_LABEL:
            # print(f'(Skip) Label id {label_id} not in ID_TO_LABEL')
            continue
        label_name = ID_TO_LABEL[label_id]

        pred_inst_mask = np.load(pred_mask_file).astype(np.int32).reshape(-1)
        if len(pred_inst_mask) != len(gt_ids):
            print(f'Length of pred mesh and gt mesh do not match: {len(pred_inst_mask)} vs {len(gt_ids)}')

        pred_inst_mask = (pred_inst_mask > 0)
        pred_inst_area = np.count_nonzero(pred_inst_mask)
        if pred_inst_area < min_region_sizes[0]:
            continue

        pred_instance = {}
        pred_instance['filename'] = pred_mask_file
        pred_instance['pred_id'] = pred_inst_i
        pred_instance['label_id'] = label_id
        pred_instance['vert_count'] = pred_inst_area
        pred_instance['confidence'] = conf
        pred_instance['void_intersection'] = np.count_nonzero(np.logical_and(bool_void, pred_inst_mask))

        # matched gt instances
        matched_gt = []
        # go thru all gt instances with matched semantic label
        for (gt_inst_i, gt_inst) in enumerate(gt2pred[label_name]):
            intersection = np.count_nonzero(
                np.logical_and(gt_ids == gt_inst['instance_id'], pred_inst_mask))
            if intersection > 0:
                gt_copy = gt_inst.copy()
                pred_copy = pred_instance.copy()
                gt_copy['intersection']   = intersection
                pred_copy['intersection'] = intersection
                matched_gt.append(gt_copy)
                gt2pred[label_name][gt_inst_i]['matched_pred'].append(pred_copy)
        pred_instance['matched_gt'] = matched_gt
        pred2gt[label_name].append(pred_instance)
        pred_inst_i += 1

    return gt2pred, pred2gt



def evaluate_matches(matches):
    # results: class x overlap
    ap = np.zeros((len(dist_threshes), len(CLASS_LABELS), len(overlaps)), float)

    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
            zip(min_region_sizes, dist_threshes, dist_confs)):
        for oi, overlap_th in enumerate(overlaps):
            pred_visited = {}
            for m in matches:
                for p in matches[m]['pred']:
                    for label_name in CLASS_LABELS:
                        for p in matches[m]['pred'][label_name]:
                                pred_visited[p['filename']] = False
            
            # iterate over all classes
            for li, label_name in enumerate(CLASS_LABELS):
                y_true = np.empty(0)
                y_score = np.empty(0)
                hard_false_negatives = 0
                has_gt = False
                has_pred = False
                for m in matches:
                    pred_instances = matches[m]['pred'][label_name]
                    gt_instances = matches[m]['gt'][label_name]
                    # filter groups in ground truth
                    gt_instances = [
                        gt for gt in gt_instances if
                        gt['instance_id'] >= 1000 and gt['vert_count'] >= min_region_size and \
                        gt['med_dist'] <= distance_thresh and gt['dist_conf'] >= distance_conf
                    ]
                    if gt_instances:
                        has_gt = True
                    if pred_instances:
                        has_pred = True

                    cur_true = np.ones(len(gt_instances))
                    cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                    cur_match = np.zeros(len(gt_instances), dtype=bool)
                    # collect matches
                    for (gti, gt) in enumerate(gt_instances):
                        found_match = False
                        num_pred = len(gt['matched_pred'])
                        for pred in gt['matched_pred']:
                            # greedy assignments
                            if pred_visited[pred['filename']]:
                                continue
                            union = gt['vert_count'] + pred['vert_count'] - pred['intersection']
                            overlap = float(pred['intersection']) / union
                            
                            if overlap > overlap_th:
                                confidence = pred['confidence']
                                # if already have a prediction for this gt,
                                # the one with the lower score is a FP
                                if cur_match[gti]:
                                    max_score = max(cur_score[gti], confidence)
                                    min_score = min(cur_score[gti], confidence)
                                    cur_score[gti] = max_score
                                    # append false positive
                                    cur_true = np.append(cur_true, 0)
                                    cur_score = np.append(cur_score, min_score)
                                    cur_match = np.append(cur_match, True)
                                # otherwise set score
                                else:
                                    found_match = True
                                    cur_match[gti] = True
                                    cur_score[gti] = confidence
                                    pred_visited[pred['filename']] = True
                        if not found_match:
                            hard_false_negatives += 1
                    # remove non-matched ground truth instances
                    cur_true = cur_true[cur_match == True]
                    cur_score = cur_score[cur_match == True]

                    # collect non-matched predictions as false positive
                    for pred in pred_instances:
                        found_gt = False
                        for gt in pred['matched_gt']:
                            union = gt['vert_count'] + pred['vert_count'] - gt['intersection']
                            overlap = float(gt['intersection']) / union
                            if overlap > overlap_th:
                                found_gt = True
                                break
                        if not found_gt:
                            num_ignore = pred['void_intersection']
                            for gt in pred['matched_gt']:
                                # group?
                                if gt['instance_id'] < 1000:
                                    num_ignore += gt['intersection']
                                # small ground truth instances
                                if gt['vert_count'] < min_region_size or gt['med_dist'] > distance_thresh or gt[
                                    'dist_conf'] < distance_conf:
                                    num_ignore += gt['intersection']
                            proportion_ignore = float(num_ignore) / pred['vert_count']
                            # if not ignored append false positive
                            if proportion_ignore <= overlap_th:
                                cur_true = np.append(cur_true, 0)
                                confidence = pred["confidence"]
                                cur_score = np.append(cur_score, confidence)

                    # append to overall results
                    y_true = np.append(y_true, cur_true)
                    y_score = np.append(y_score, cur_score)

                # compute average precision
                if has_gt and has_pred:
                    # compute precision recall curve first

                    # sorting and cumsum
                    score_arg_sort = np.argsort(y_score)
                    y_score_sorted = y_score[score_arg_sort]
                    y_true_sorted = y_true[score_arg_sort]
                    y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                    # unique thresholds
                    (thresholds, unique_indices) = np.unique(y_score_sorted, return_index=True)
                    num_prec_recall = len(unique_indices) + 1

                    # prepare precision recall
                    num_examples = len(y_score_sorted)
                    # https://github.com/ScanNet/ScanNet/pull/26
                    # all predictions are non-matched but also all of them are ignored and not counted as FP
                    # y_true_sorted_cumsum is empty
                    num_true_examples = y_true_sorted_cumsum[-1] \
                        if len(y_true_sorted_cumsum) > 0 else 0
                    precision = np.zeros(num_prec_recall)
                    recall = np.zeros(num_prec_recall)

                    # deal with the first point
                    y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                    # deal with remaining
                    for idx_res, idx_scores in enumerate(unique_indices):
                        cumsum = y_true_sorted_cumsum[idx_scores - 1]
                        tp = num_true_examples - cumsum
                        fp = num_examples - idx_scores - tp
                        fn = cumsum + hard_false_negatives
                        p = float(tp) / (tp + fp)
                        r = float(tp) / (tp + fn)
                        precision[idx_res] = p
                        recall[idx_res] = r

                    # first point in curve is artificial
                    precision[-1] = 1.
                    recall[-1] = 0.

                    # compute average of precision-recall curve
                    recall_for_conv = np.copy(recall)
                    recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                    recall_for_conv = np.append(recall_for_conv, 0.)

                    stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5], 'valid')
                    # integrate is now simply a dot product
                    ap_current = np.dot(precision, stepWidths)

                elif has_gt:
                    ap_current = 0.0
                else:
                    ap_current = float('nan')
                ap[di, li, oi] = ap_current
    return ap

def compute_averages(aps):
    d_inf = 0
    o50 = np.where(np.isclose(overlaps, 0.5))
    o25 = np.where(np.isclose(overlaps, 0.25))
    
    avg_dict = {}
    oAllBut25 = np.where(np.logical_not(np.isclose(overlaps, 0.25)))
    avg_dict['all_ap'] = np.nanmean(aps[d_inf, :, oAllBut25]) # from 0.5 to 0.95 with step 0.05
    avg_dict['all_ap_50%'] = np.nanmean(aps[d_inf, :, o50])
    avg_dict['all_ap_25%'] = np.nanmean(aps[d_inf, :, o25])

    avg_dict["classes"] = {}
    for (li, label_name) in enumerate(CLASS_LABELS):
        avg_dict["classes"][label_name] = {}
        avg_dict["classes"][label_name]["ap"] = np.average(aps[d_inf, li, oAllBut25])
        avg_dict["classes"][label_name]["ap50%"] = np.average(aps[d_inf, li, o50])
        avg_dict["classes"][label_name]["ap25%"] = np.average(aps[d_inf, li, o25])

    return avg_dict

def evaluate(
    pred_path, pred_files, gt_files, 
    eval_folder
):
    matches = {}
    for i, pred_f in enumerate(pred_files):
        gt_f = gt_files[i]
        matches_key = os.path.abspath(gt_f)
        # assign gt to predictions
        gt2pred, pred2gt = assign_instances_for_scan(pred_path, pred_f, gt_f)

        # for label_name, insts in gt2pred.items():
        #     if len(insts) == 0:
        #         continue
        #     print(label_name)
        #     print(insts)
        # print('\n\n\n')

        # for label_name, insts in pred2gt.items():
        #     if len(insts) == 0:
        #         continue
        #     print(label_name)
        #     print(insts)

        matches[matches_key] = {}
        matches[matches_key]['gt'] = gt2pred
        matches[matches_key]['pred'] = pred2gt

    
    ap_scores = evaluate_matches(matches)
    avgs = compute_averages(ap_scores)

    return avgs

