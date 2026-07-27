import os, sys, time, threading
import numpy as np
from scipy.spatial.transform import Slerp, Rotation
import cv2
from collections import Counter

import sys
sys.path.append("/usr/lib/python3/dist-packages")

# self packages
from utils.semantic_const import NYU_40, COCO_PANOPTIC_133
from utils.common_utils import class_id_to_one_hot
from utils.common_utils import Segment

NYU_41_text = ['Background'] + NYU_40
COCO_134_text = ['Background'] + COCO_PANOPTIC_133
BackgroundSemId = 0

class SegmentDepthWrapper(threading.Thread):

    def __init__(self, depth_segmentor, depth_img_scaled,  rgb_img):
        threading.Thread.__init__(self)

        self.depth_img_scaled = depth_img_scaled
        self.depth_segmentor = depth_segmentor
        self.rgb_img = rgb_img.astype(np.float32)
    
        self.depth_map = None
        self.segment_masks_list = []

    def run(self):
        self.depth_segmentor.depthSegment(self.depth_img_scaled, self.rgb_img)
        # get depth map and segment masks
        self.depth_map = self.depth_segmentor.get_depthMap()
        self.segment_masks_list = self.depth_segmentor.get_segmentMasks()



class SegmentsGenerator:
    """
    Class Descriptions
    ?
    """
    def __init__(self, gsm_node, depth_segmentor, panoptic_segmentor, \
        save_resutls_img=False, img_folder = None, \
        save_segments = False, use_segments = False, segments_folder = None,
        save_panoptics = False, use_panoptics = False, panoptics_folder = None,
        save_geometrics = False, use_geometrics = False, geometrics_folder = None):
        
        self.depth_segmentor = depth_segmentor
        self.panoptic_segmentor = panoptic_segmentor

        self.save_resutls_img = save_resutls_img
        self.img_folder = img_folder
        if self.save_resutls_img:
            self.semantic_folder = os.path.join(self.img_folder, 'panoptic_seg')
            if not os.path.exists(self.semantic_folder):
                os.makedirs(self.semantic_folder)

        self.save_segments = save_segments
        self.use_segments =use_segments
        self.segments_folder = segments_folder
        if(self.save_segments and self.segments_folder is not None):
            if not os.path.exists(self.segments_folder):
                os.makedirs(self.segments_folder)
        else:
            self.save_segments = False

        self.save_panoptics = save_panoptics
        self.use_panoptics = use_panoptics
        self.panoptics_folder = panoptics_folder
        if(self.save_panoptics and self.panoptics_folder is not None):
            if not os.path.exists(self.panoptics_folder):
                os.makedirs(self.panoptics_folder)
        else:
            self.save_panoptics = False

        self.save_geometrics = save_geometrics
        self.use_geometrics =use_geometrics
        self.geometrics_folder = geometrics_folder
        if(self.save_geometrics and self.geometrics_folder is not None):
            if not os.path.exists(self.geometrics_folder):
                os.makedirs(self.geometrics_folder)
        else:
            self.save_geometrics = False

        return None

    
    def SegmentDepth(self, depth_img, rgb_img, frame_i):
        depth_segmentor_thread = SegmentDepthWrapper(
            self.depth_segmentor, depth_img, rgb_img)
        depth_segmentor_thread.start()
        depth_segmentor_thread.join()
        
        # NOTE depth_seg_masks is a multi-layer mask, each seg a layer
        depth_seg_masks =  depth_segmentor_thread.segment_masks_list
        if self.save_geometrics and (self.geometrics_folder is not None):
            self.save2DGeometricSegs(depth_seg_masks, frame_i)
    
    
    def save2DGeometricSegs(self, segment_masks, frame_i):
        """save geometric information"""
        geometric_seg_mask = np.zeros(segment_masks.shape[1:], dtype=np.uint8)
        for seg_idx in range(len(segment_masks)):
            seg_mask = segment_masks[seg_idx].astype(bool)
            geometric_seg_mask[seg_mask] = seg_idx+1
        geometric_seg_mask_f = os.path.join(self.geometrics_folder, str(frame_i).zfill(5)+"_mask.png")
        cv2.imwrite(geometric_seg_mask_f, geometric_seg_mask)

    def load2DGeometricSegs(self, frame_i):
        # load geometric information
        geometric_seg_mask_f = os.path.join(self.geometrics_folder, str(frame_i).zfill(5)+"_mask.png")
        if not os.path.isfile(geometric_seg_mask_f):
            return None
        geometric_seg_mask = cv2.imread(geometric_seg_mask_f, cv2.IMREAD_UNCHANGED)
        return geometric_seg_mask
    
    
    def frameToSegmentsCropFormer(self, 
        depth_img, camera_K, pose_i, frame_i, seg_map
    ):
        points_map = cv2.rgbd.depthTo3d(depth_img, camera_K)
        depth_segs = self.load2DGeometricSegs(frame_i)
        depth_inst_ids = np.unique(depth_segs)

        segments_list = []

        sem_depth_segs = []
        bg_segs = []

        # find a pano-seg id for each seg from depth
        for m_id in depth_inst_ids:
            if m_id == 0:
                continue
            depth_seg_mask = (depth_segs==m_id)
            depth_seg_area = np.count_nonzero(depth_seg_mask)
            # remove small segments
            if depth_seg_area < 100:
                continue


            cand_pairs = Counter(seg_map[depth_seg_mask].reshape(-1))
            max_overlap_area = 0
            max_cand_id = 0 
            for pano_inst_id in cand_pairs:
                # skip w/o pixels instance label
                if(pano_inst_id == 0):
                    continue
                
                cand_area = cand_pairs[pano_inst_id]
                inst_area = np.count_nonzero(seg_map == pano_inst_id)

                area_ratio_pano = 0.9 
                area_ratio_depth = 0.5 
                area_pano_lowthres = area_ratio_pano * inst_area
                area_depth_upthres = area_ratio_depth * depth_seg_area

                # if depth-undersegment 
                # (one pano seg instance only occupy a few of the depth seg)
                if(cand_area > area_pano_lowthres and cand_area < area_depth_upthres):
                    # further seg if the instance is not majority in depth seg
                    overlap_mask = np.logical_and(depth_seg_mask, (seg_map == pano_inst_id))
                    sem_depth_segs.append({
                        'mask': overlap_mask, 
                        'inst_id': pano_inst_id, 'is_thing': True, 
                        'inst_score': 1.0, 
                        'overlap_r':cand_area * 1.0 / inst_area
                    })
                    # further determine remaining part
                    depth_seg_area -= cand_area # reduce the area from the whole
                    depth_seg_mask[overlap_mask] = False
                else:
                    # update the overlap area of not splited instance
                    if max_overlap_area < cand_area:
                        max_overlap_area = cand_area
                        max_cand_id = pano_inst_id
            
            area_ratio_label = 0.2
            set_label_lowthres = area_ratio_label * depth_seg_area
            if(max_overlap_area >= set_label_lowthres):
                sem_depth_segs.append({
                    'mask': depth_seg_mask, 
                    'inst_id': max_cand_id, 'is_thing': True, 
                    'inst_score': 1.0, 'sem_id': max_cand_id,
                    'overlap_r': max_overlap_area* 1.0 / depth_seg_area
                })
            else:
                bg_segs.append({
                    'mask': depth_seg_mask, 
                    'inst_id': BackgroundSemId, 'is_thing': False, 
                    'inst_score': 0.5, 
                    'overlap_r': cand_pairs[0] * 1.0 / depth_seg_area
                })
        
        fused_seg = np.zeros_like(depth_segs, dtype=np.uint8) # [H, W]
        
        seg_index = 0
        for inst_seg in sem_depth_segs:
            inst_mask = inst_seg['mask']
            points = points_map[inst_mask].astype(np.float32).reshape(-1,3)
            is_thing = inst_seg['is_thing']
            instance_label = inst_seg['inst_id']
            
            semantic_label = 1
            sem_feat = class_id_to_one_hot(1, num_classes=2)

            inst_score = inst_seg['inst_score']
            overlap_ratio = inst_seg['overlap_r']
            segment = Segment(
                points, is_thing, instance_label, semantic_label, 
                inst_score, overlap_ratio, pose_i, seg_index, sem_feat=sem_feat
            )
            # segment.calculateConfidenceDefault()
            segments_list.append(segment)
            seg_index += 1
            fused_seg[inst_mask] = seg_index
        
        for bg_seg in bg_segs:
            inst_mask = bg_seg['mask']
            points = points_map[inst_mask].astype(np.float32).reshape(-1,3)
            is_thing = bg_seg['is_thing']
            instance_label = bg_seg['inst_id']

            semantic_label = BackgroundSemId
            sem_feat = class_id_to_one_hot(0, num_classes=2)

            inst_score = bg_seg['inst_score']
            overlap_ratio = bg_seg['overlap_r']
            segment = Segment(
                points, is_thing, instance_label, semantic_label, 
                inst_score, overlap_ratio, pose_i, seg_index, sem_feat=sem_feat
            )
            # segment.calculateConfidenceDefault()
            segments_list.append(segment)
            seg_index += 1
            fused_seg[inst_mask] = seg_index
        # vis_id_map(fused_seg, f'/home/zilong/Downloads/fused_seg_{frame_i}.png')

        return segments_list
