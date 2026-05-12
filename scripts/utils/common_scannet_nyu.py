import os, sys, time, threading, copy, pickle, random
import numpy as np
from scipy.spatial.transform import Slerp, Rotation
import cv2
from collections import Counter
from multiprocessing import Process
import matplotlib.pyplot as plt

import sys
sys.path.append("/usr/lib/python3/dist-packages")

# self packages
from semantics.semantic_utils import NYU_40, COCO_133
from semantics.pano_scannet_nyu_colormap import *
from utils.common_utils import class_id_to_one_hot, dictToHd5, hd5ToDict
from utils.common_utils import Segment
from utils.vis_utils import vis_id_map

NYU_41_text = ['Background'] + NYU_40
COCO_134_text = ['Background'] + COCO_133

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

    def Segment2D(self, depth_img, rgb_img, frame_i):
        result = None
        depth_segmentor_thread = SegmentDepthWrapper(
            self.depth_segmentor, depth_img, rgb_img)
        # depth segmentation subprocess
        depth_segmentor_thread.start()

        # ======== Panoptic Segmentation ========
        panoptic_result = None
        # NOTE If use Mask2Former, it's false by default
        if self.panoptic_segmentor.use_gt:
            panoptic_result = self.panoptic_segmentor.forward(frame_i)
        else:
            panoptic_result = self.panoptic_segmentor.forward(rgb_img)
        
        if len(panoptic_result['info']) == 0:
            print("Nothing from panoptic segmentor!")
            depth_segmentor_thread.join()
            return result

        # ======== Depth Segmentation ========
        # waiting for depth seg result
        depth_segmentor_thread.join()
        depth_map = depth_segmentor_thread.depth_map
        # NOTE depth_seg_masks is a multi-layer mask, each seg a layer
        depth_seg_masks =  depth_segmentor_thread.segment_masks_list
        if len(depth_seg_masks) == 0:
            print("Nothing from depth segmentor!")
            return result
        
        # extract instance/stuff information 
        id2info_instance = {}
        id2info_stuff = {}
        for id_info in panoptic_result['info']:
            id = id_info['id']
            is_thing = id_info['isthing']
            if is_thing:
                # instance
                id2info_instance[id] = id_info
            else:
                # stuff
                id2info_stuff[id] = id_info

        seg_map = panoptic_result['seg_map']
        result = {
            # from panoptic seg
            'seg_map': seg_map, 
            'id2info_instance':id2info_instance, 
            'id2info_stuff':id2info_stuff, 
            # from depth seg
            'segment_masks': depth_seg_masks, 
            'depth_map': depth_map
        }

        if self.save_panoptics and (self.panoptics_folder is not None):
            self.save2DPanopticSegs(seg_map, panoptic_result['info'], frame_i)
        if self.save_geometrics and (self.geometrics_folder is not None):
            self.save2DGeometricSegs(depth_seg_masks, frame_i)
        
        return result
    
    def SegmentDepth(self, depth_img, rgb_img, frame_i):
        depth_segmentor_thread = SegmentDepthWrapper(
            self.depth_segmentor, depth_img, rgb_img)
        depth_segmentor_thread.start()
        depth_segmentor_thread.join()
        
        # NOTE depth_seg_masks is a multi-layer mask, each seg a layer
        depth_seg_masks =  depth_segmentor_thread.segment_masks_list
        if self.save_geometrics and (self.geometrics_folder is not None):
            self.save2DGeometricSegs(depth_seg_masks, frame_i)
    
    def save2DPanopticSegs(self, seg_map, pano_info, frame_i):
        """save panoptic information"""
        panoptic_mask = seg_map
        panoptic_mask_f = os.path.join(self.panoptics_folder, str(frame_i).zfill(5)+"_mask.png")
        cv2.imwrite(panoptic_mask_f, panoptic_mask)

        pano_info_dict = {
            'ids': [sem_info['id'] for sem_info in pano_info],
            "is_thing": [sem_info['isthing'] for sem_info in pano_info],
            "cates": [sem_info['category_id'] for sem_info in pano_info],
            "areas": [sem_info['area'] for sem_info in pano_info]
        }
        pano_info_f = os.path.join(self.panoptics_folder, str(frame_i).zfill(5)+"_info.h5")
        dictToHd5(pano_info_f, pano_info_dict)
    
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

    def load2DPanopticGeometricSegs(self, camera_K, depth_img_scaled, frame_i):
        # load panoptic information
        panoptic_mask_f = os.path.join(self.panoptics_folder, str(frame_i).zfill(5)+"_mask.png")
        panoptic_info_f = os.path.join(self.panoptics_folder, str(frame_i).zfill(5)+"_info.h5")
        if( (not os.path.isfile(panoptic_mask_f)) or 
            (not os.path.isfile(panoptic_info_f))):
            return None
        panoptic_info_dict = hd5ToDict(panoptic_info_f)
        id2info = {}
        id2info_instance = {}
        id2info_stuff = {}
        for sem_idx in range(len(panoptic_info_dict['ids'])):
            id = panoptic_info_dict['ids'][sem_idx]
            is_thing = panoptic_info_dict['is_thing'][sem_idx]
            sem_info = {
                'id': id, 'isthing':is_thing, 
                'category_id': panoptic_info_dict['cates'][sem_idx],
                'area': panoptic_info_dict['areas'][sem_idx]
            }
            if is_thing:
                id2info_instance[id] = sem_info
            else:
                id2info_stuff[id] = sem_info

            id2info[id] = sem_info

        panoptic_mask = cv2.imread(panoptic_mask_f, cv2.IMREAD_UNCHANGED)

        # load geometric information
        geometric_seg_mask_f = os.path.join(self.geometrics_folder, str(frame_i).zfill(5)+"_mask.png")
        if( (not os.path.isfile(geometric_seg_mask_f)) ):
            return None
        geometric_seg_mask = cv2.imread(geometric_seg_mask_f, cv2.IMREAD_UNCHANGED)

        segment_masks = []
        segs_ids = np.unique(geometric_seg_mask)
        for seg_id in segs_ids:
            if seg_id == 0:
                continue
            seg_mask = np.zeros(panoptic_mask.shape, dtype=bool)
            seg_mask[geometric_seg_mask==seg_id] = True
            segment_masks.append(seg_mask)
        
        # get depth map
        depth_map = cv2.rgbd.depthTo3d(depth=depth_img_scaled,K=camera_K)
        result = {
            'seg_map': panoptic_mask, 
            'id2info': id2info,
            'id2info_instance': id2info_instance, 
            'id2info_stuff': id2info_stuff,
            'segment_masks': np.array(segment_masks).reshape(-1,panoptic_mask.shape[0], panoptic_mask.shape[-1]), 
            'depth_map': depth_map
        }
        
        return result
    
    def generateSegments(self, seg_result_2D, pose, frame_i):
        # get panoptic segmentation result
        segment_list = []
        
        # panoptic segs on RGB 
        seg_map = seg_result_2D['seg_map']
        id2info_ins = seg_result_2D['id2info_instance']
        id2info_stuff = seg_result_2D['id2info_stuff']
        ## Get masks for each id from panoptic seg
        num_panoptic_segs = len(id2info_ins)+len(id2info_stuff)
        pano_seg_masks = np.zeros((seg_map.shape[0], seg_map.shape[1], num_panoptic_segs+1), dtype=bool)
        mask_idxs_2D = np.indices(seg_map.shape)
        # shape [H, W, num_pano_segs]
        pano_seg_masks[mask_idxs_2D[0], mask_idxs_2D[1], seg_map] = True

        
        # depth segs
        segment_masks = seg_result_2D['segment_masks']
        ## generate segments candidates
        sem_depth_segments = []
        extra_instances = []
        background_segments = []

        # find a pano-seg ids for each seg from depth
        for m_id in range(segment_masks.shape[0]):
            depth_seg_mask = segment_masks[m_id,:,:].copy()
            depth_seg_mask = depth_seg_mask.astype(bool)
            depth_seg_area = np.sum(depth_seg_mask)
            # NOTE remove small segments, thresholded by 100 pixels
            if depth_seg_area < 100:
                continue
            
            cand_pairs = Counter(seg_map[depth_seg_mask].reshape(-1))
            max_overlap_area = 0
            max_cand_id = 0 
            for pano_seg_id in cand_pairs:
                ## NOTE skip background
                if(pano_seg_id == 0):
                    continue
                cand_area = cand_pairs[pano_seg_id]

                if pano_seg_id in id2info_ins:
                    # parameters from voxbloxpp
                    area_ratio_pano = 0.9 
                    area_ratio_depth = 0.5 
                    area_pano_lowthres = area_ratio_pano * id2info_ins[pano_seg_id]['area']
                    area_depth_upthres = area_ratio_depth * depth_seg_area

                    # if depth-undersegment 
                    # (pano seg instance only takes less than xx% of the depth seg)
                    if(cand_area > area_pano_lowthres and cand_area < area_depth_upthres):
                        # further seg it if the instance is not majority in the depth seg
                        overlap_mask = np.logical_and(
                            depth_seg_mask, pano_seg_masks[:,:,pano_seg_id])
                        extra_instances.append({
                            'mask': overlap_mask, 
                            'id': pano_seg_id, 'is_thing': True, 
                            'inst_score': 1.0, 
                            'overlap_r':cand_area * 1.0 / id2info_ins[pano_seg_id]['area']
                        })
                        # further determine remaining part
                        depth_seg_area -= cand_area # reduce the area from the whole
                        depth_seg_mask[overlap_mask] = False
                    else:
                        # update the overlap area of not splited instance
                        if max_overlap_area < cand_area:
                            max_overlap_area = cand_area
                            max_cand_id = pano_seg_id

                elif pano_seg_id in id2info_stuff:
                    # parameters from voxbloxpp
                    # area_ratio_pano = 0.05 
                    # area_ratio_depth = 0.8 
                    # area_pano_lowthres = area_ratio_pano * id2info_stuff[pano_seg_id]['area']
                    # area_depth_upthres = area_ratio_depth * depth_seg_area
                    
                    if(False): # voxbloxpp, Han et al
                    # if(cand_area > area_pano_lowthres and cand_area < area_depth_upthres):
                        overlap_mask = np.logical_and(
                            depth_seg_mask, pano_seg_masks[:,:,pano_seg_id])
                        inst_score = 0.5
                        extra_instances.append({
                            'mask': overlap_mask, 
                            'id': pano_seg_id, 'is_thing': False,
                            'inst_score': 0.5, 
                            'overlap_r':cand_area * 1.0 / id2info_stuff[pano_seg_id]['area']
                        })
                        # further determine remaining part
                        depth_seg_area -= cand_area
                        depth_seg_mask[overlap_mask] = False
                    else:
                        # update the overlap area of not splited instance
                        if max_overlap_area < cand_area:
                            max_overlap_area = cand_area
                            max_cand_id = pano_seg_id
            
            area_ratio_label = 0.2
            set_label_lowthres = area_ratio_label * depth_seg_area
            # set semantic label for depth_seg
            if(max_overlap_area >= set_label_lowthres):
                is_thing = (max_cand_id in id2info_ins)
                # inst_score = id2info_instance[max_cand_id]['score'] if is_thing else 0.5
                sem_depth_segments.append({
                    'mask': depth_seg_mask, 
                    'id': max_cand_id, 'is_thing': is_thing, 
                    'inst_score': 1.0 if is_thing else 0.5, 
                    'overlap_r': max_overlap_area* 1.0 / depth_seg_area
                })
            else:
                background_segments.append({
                    'mask': depth_seg_mask, 
                    'id': BackgroundSemId, 'is_thing': False, 
                    'inst_score': 0.5, 
                    'overlap_r': cand_pairs[0] * 1.0 / depth_seg_area
                })
        
        
        # generate segments
        mask_seg_frame = np.zeros_like(seg_map, dtype=np.uint8) # [H, W]
        depth_map = seg_result_2D['depth_map']
        seg_index = 0
        for sem_depth_seg in sem_depth_segments:
            points = depth_map[sem_depth_seg['mask']].astype(np.float32).reshape(-1,3)
            is_thing = sem_depth_seg['is_thing']
            instance_label = sem_depth_seg['id']
            semantic_label = id2info_ins[instance_label]['category_id'] if is_thing \
                else id2info_stuff[instance_label]['category_id']
            semantic_label = semantic_map(semantic_label)

            inst_score = sem_depth_seg['inst_score']
            overlap_ratio = sem_depth_seg['overlap_r']
            segment = Segment(
                points, is_thing, instance_label, semantic_label, 
                inst_score, overlap_ratio, pose, seg_index
            )
            # segment.calculateConfidenceDefault()
            segment_list.append(segment)
            seg_index += 1
            mask_seg_frame[sem_depth_seg['mask']] = seg_index
        
        for extra_instance_seg in extra_instances:
            points = depth_map[extra_instance_seg['mask']].astype(np.float32).reshape(-1,3)
            is_thing = extra_instance_seg['is_thing']
            instance_label = extra_instance_seg['id']
            semantic_label = id2info_ins[instance_label]['category_id'] if is_thing \
                else id2info_stuff[instance_label]['category_id']
            semantic_label = semantic_map(semantic_label)

            inst_score = extra_instance_seg['inst_score']
            overlap_ratio = extra_instance_seg['overlap_r']
            segment = Segment(
                points, is_thing, instance_label, semantic_label, 
                inst_score, overlap_ratio, pose, seg_index
            )
            # segment.calculateConfidenceDefault()
            segment_list.append(segment)
            seg_index += 1 
            mask_seg_frame[extra_instance_seg['mask']] = seg_index
        
        for background_seg in background_segments:
            points = depth_map[background_seg['mask']].astype(np.float32).reshape(-1,3)
            is_thing = background_seg['is_thing']
            instance_label = background_seg['id']
            semantic_label = BackgroundSemId # background semantic label

            inst_score = background_seg['inst_score']
            overlap_ratio = background_seg['overlap_r']
            segment = Segment(
                points, is_thing, instance_label, semantic_label, 
                inst_score, overlap_ratio, pose, seg_index
            )
            # segment.calculateConfidenceDefault()
            segment_list.append(segment)
            seg_index += 1
            mask_seg_frame[background_seg['mask']] = seg_index
        
        if self.save_segments:
            mask_f = os.path.join(self.segments_folder, str(frame_i).zfill(5)+"_mask.png")
            cv2.imwrite(mask_f, mask_seg_frame)
        
        return segment_list



    def outlierRemove(self, segment_list, neighbor_dist_th = 0.05):
        # TODO
        # instance_to_seg_pair = {}   
        # # get instance-segment map
        # for seg in segment_list:
        #     if not seg.is_thing:
        #         continue
            # if seg.instance_label in instance_to_seg_pair:
            #     instance_to_seg_pair[seg.instance_label].append(seg.index)
            # else:
            #     instance_to_seg_pair[seg.instance_label] = [seg.index]

        # for instance_label in instance_to_seg_pair:
        #     # get neighbor map
        #     instance_seg_list = instance_to_seg_pair[instance_label]
        #     neighber_map = { seg_index:[] for seg_index in instance_seg_list}
        #     for i, seg_i in enumerate(instance_seg_list):
        #         for j in range(seg_i+1, )
        return segment_list


    def frameToSegments(self, depth_img, rgb_img, pose, frame_i, camera_K = None):
        segment_list = []

        seg_result_2D = None
        if self.use_panoptics and self.use_geometrics and (camera_K is not None):
            seg_result_2D = self.load2DPanopticGeometricSegs(
                camera_K, depth_img, frame_i
            )
        else:
            seg_result_2D = self.Segment2D(depth_img, rgb_img, frame_i)

        if seg_result_2D is None:
            return segment_list 
        
        # refine the depth segments and assign seg ids, then pack all segs
        segment_list = self.generateSegments(seg_result_2D, pose, frame_i)

        # save segments information
        if self.save_segments:
            seg_info = {
                'is_thing':[], 'instance_label':[], 'class_label':[], 
                'inst_confidence':[], 'overlap_ratio': [], 
                'pose':[], 'center':[], 'seg_num':0
            }
            for seg in segment_list:
                seg_info['is_thing'].append(seg.is_thing)
                seg_info['instance_label'].append(seg.instance_label)
                seg_info['class_label'].append(seg.class_label)
                seg_info['inst_confidence'].append(seg.inst_confidence)
                seg_info['overlap_ratio'].append(seg.overlap_ratio)
                seg_info['pose'].append(seg.pose)
                seg_info['center'].append(seg.center)
            seg_info['seg_num'] = len(segment_list)
            
            seg_info_f = os.path.join(self.segments_folder, str(frame_i).zfill(5)+"_seg_info.h5")
            # print(f"Saved Seg Result to {seg_info_f}")
            dictToHd5(seg_info_f, seg_info)

        return segment_list
    

    def gtSemSegToSegments(self, 
        depth_img, camera_K, pose_i, frame_i, gt_sem_segs, text_embed=None
    ):
        points_map = cv2.rgbd.depthTo3d(depth_img, camera_K)
        depth_segs = self.load2DGeometricSegs(frame_i)
        depth_inst_ids = np.unique(depth_segs)

        segments_list = []

        ## generate segments candidates
        sem_depth_segs = []
        bg_segs = []

        # find a pano-seg id for each seg from depth
        for m_id in depth_inst_ids:
            if m_id == 0:
                continue
            depth_seg_mask = (depth_segs==m_id)
            depth_seg_area = np.count_nonzero(depth_seg_mask)
            # NOTE remove small segments, thresholded by 100 pixels
            if depth_seg_area < 100:
                continue
            
            sem_cnts = np.bincount(gt_sem_segs[depth_seg_mask])
            sem_id = np.argmax(sem_cnts)
            overlap_area = sem_cnts[sem_id]

            if sem_id != 0:
                sem_depth_segs.append({
                    'mask': depth_seg_mask, 
                    'inst_id': sem_id, 'is_thing': True, 
                    'inst_score': 1.0, 
                    'overlap_r': overlap_area* 1.0 / depth_seg_area
                })
            else:
                bg_segs.append({
                    'mask': depth_seg_mask, 
                    'inst_id': BackgroundSemId, 'is_thing': False, 
                    'inst_score': 0.5, 
                    'overlap_r': overlap_area * 1.0 / depth_seg_area
                })
        
        
        # generate segments
        mask_seg_frame = np.zeros_like(gt_sem_segs, dtype=np.uint8) # [H, W]
        seg_index = 0
        for inst_seg in sem_depth_segs:
            inst_mask = inst_seg['mask']
            points = points_map[inst_mask].astype(np.float32).reshape(-1,3)
            is_thing = inst_seg['is_thing']
            instance_label = inst_seg['inst_id']
            
            # find the major gt_sem_label in the mask area
            semantic_label = inst_seg['inst_id']
            # sem_feat = open_seg_meta[instance_label]['sem_feat']
            # sem_feat = np.array(sem_feat, dtype=np.float32)
            if text_embed is not None:
                sem_feat = text_embed[NYU_41_text[semantic_label]]
                sem_feat = np.array(sem_feat, dtype=np.float32)
            else:
                sem_feat = class_id_to_one_hot(semantic_label, num_classes=512)

            inst_score = inst_seg['inst_score']
            overlap_ratio = inst_seg['overlap_r']
            segment = Segment(
                points, is_thing, instance_label, semantic_label, 
                inst_score, overlap_ratio, pose_i, seg_index, sem_feat=sem_feat
            )
            # segment.calculateConfidenceDefault()
            segments_list.append(segment)
            seg_index += 1
            mask_seg_frame[inst_mask] = seg_index
        
        for bg_seg in bg_segs:
            inst_mask = bg_seg['mask']
            points = points_map[inst_mask].astype(np.float32).reshape(-1,3)
            is_thing = bg_seg['is_thing']
            instance_label = bg_seg['inst_id']

            semantic_label = BackgroundSemId # background semantic label
            # sem_feat = np.zeros(1152, dtype=np.float32)
            # sem_feat[0] = 1.0
            if text_embed is not None:
                sem_feat = text_embed['Background']
                sem_feat = np.array(sem_feat, dtype=np.float32)
            else:
                sem_feat = class_id_to_one_hot(semantic_label, num_classes=512)

            inst_score = bg_seg['inst_score']
            overlap_ratio = bg_seg['overlap_r']
            segment = Segment(
                points, is_thing, instance_label, semantic_label, 
                inst_score, overlap_ratio, pose_i, seg_index, sem_feat=sem_feat
            )
            # segment.calculateConfidenceDefault()
            segments_list.append(segment)
            seg_index += 1
            mask_seg_frame[inst_mask] = seg_index

        return segments_list
    
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
    

    def frameToSegmentsOpenset(self, 
        depth_img, camera_K, pose_i, frame_i, 
        gt_sem_segs, seg_map, open_seg_meta
    ):
        points_map = cv2.rgbd.depthTo3d(depth_img, camera_K)
        depth_segs = self.load2DGeometricSegs(frame_i)
        depth_inst_ids = np.unique(depth_segs)

        segments_list = []
        
        # pano_ids = np.unique(seg_map)
        # num_seg_ids = len(pano_ids)
        # pano_seg_masks = np.zeros(seg_map.shape+(num_seg_ids,), dtype=bool)
        # mask_idxs_2D = np.indices(seg_map.shape)
        # # shape [H, W, num_seg_ids]
        # pano_seg_masks[mask_idxs_2D[0], mask_idxs_2D[1], seg_map] = True

        ## generate segments candidates
        sem_depth_segs = []
        extra_insts = []
        bg_segs = []
        
        # find a pano-seg id for each seg from depth
        for m_id in depth_inst_ids:
            if m_id == 0:
                continue
            depth_seg_mask = (depth_segs==m_id)
            depth_seg_area = np.count_nonzero(depth_seg_mask)
            # NOTE remove small segments, thresholded by 100 pixels
            if depth_seg_area < 100:
                continue
            
            # this is instance id, not semnatic id
            cand_pairs = Counter(seg_map[depth_seg_mask].reshape(-1))
            max_overlap_area = 0
            max_cand_id = 0 
            for pano_inst_id in cand_pairs:
                ## NOTE skip background
                if(pano_inst_id == 0):
                    continue
                
                mask_inst = (seg_map == pano_inst_id)
                inst_area = np.count_nonzero(mask_inst)
                cand_area = cand_pairs[pano_inst_id]

                # parameters from voxbloxpp
                area_ratio_pano = 0.9 
                area_ratio_depth = 0.5 
                area_pano_lowthres = area_ratio_pano * inst_area
                area_depth_upthres = area_ratio_depth * depth_seg_area

                # if depth-undersegment 
                # (one pano seg instance only occupy a few of the depth seg)
                if(cand_area > area_pano_lowthres and cand_area < area_depth_upthres):
                    # further seg if the instance is not majority in depth seg
                    overlap_mask = np.logical_and(depth_seg_mask, mask_inst)
                    extra_insts.append({
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
                    'inst_score': 1.0, 
                    'overlap_r': max_overlap_area* 1.0 / depth_seg_area
                })
            else:
                bg_segs.append({
                    'mask': depth_seg_mask, 
                    'inst_id': BackgroundSemId, 'is_thing': False, 
                    'inst_score': 0.5, 
                    'overlap_r': cand_pairs[0] * 1.0 / depth_seg_area
                })
        
        
        
        # generate segments
        mask_seg_frame = np.zeros_like(seg_map, dtype=np.uint8) # [H, W]
        seg_index = 0
        for inst_seg in sem_depth_segs+extra_insts:
            inst_mask = inst_seg['mask']
            points = points_map[inst_mask].astype(np.float32).reshape(-1,3)
            is_thing = inst_seg['is_thing']
            instance_label = inst_seg['inst_id']
            
            # find the major gt_sem_label in the mask area
            semantic_label = np.argmax(np.bincount(gt_sem_segs[inst_mask]))
            # get the sem_feat
            sem_feat = open_seg_meta[instance_label]['sem_feat']
            sem_feat = np.array(sem_feat, dtype=np.float32)#[0]

            inst_score = inst_seg['inst_score']
            overlap_ratio = inst_seg['overlap_r']
            segment = Segment(
                points, is_thing, instance_label, semantic_label, 
                inst_score, overlap_ratio, pose_i, seg_index, sem_feat=sem_feat
            )
            # segment.calculateConfidenceDefault()
            segments_list.append(segment)
            seg_index += 1
            mask_seg_frame[inst_mask] = seg_index
        
        for bg_seg in bg_segs:
            inst_mask = bg_seg['mask']
            points = points_map[inst_mask].astype(np.float32).reshape(-1,3)
            is_thing = bg_seg['is_thing']
            instance_label = bg_seg['inst_id']

            semantic_label = BackgroundSemId # background semantic label
            sem_feat = class_id_to_one_hot(0, num_classes=512)


            inst_score = bg_seg['inst_score']
            overlap_ratio = bg_seg['overlap_r']
            segment = Segment(
                points, is_thing, instance_label, semantic_label, 
                inst_score, overlap_ratio, pose_i, seg_index, sem_feat=sem_feat
            )
            # segment.calculateConfidenceDefault()
            segments_list.append(segment)
            seg_index += 1
            mask_seg_frame[inst_mask] = seg_index
        
        return segments_list


    def loadSegments(self, depth_img, camera_K, frame_i):
        segments_list = []
        mask_f = os.path.join(self.segments_folder, str(frame_i).zfill(5)+"_mask.png")
        seg_info_f = os.path.join(self.segments_folder, str(frame_i).zfill(5)+"_seg_info.h5")
        if( (not os.path.isfile(mask_f)) or (not os.path.isfile(seg_info_f)) ):
            print(f"Lacking mask or seg file of frame {frame_i}")
            return segments_list

        mask = cv2.imread(mask_f, cv2.IMREAD_UNCHANGED)
        seg_info= hd5ToDict(seg_info_f)
        points_all = cv2.rgbd.depthTo3d(depth_img, camera_K)

        for seg_i in range(seg_info['seg_num']):
            seg_mask = (mask==(seg_i+1))
            # create p3d from depths
            points_seg = points_all[seg_mask].astype(np.float32).reshape(-1,3)

            is_thing = seg_info['is_thing'][seg_i]
            instance_label = seg_info['instance_label'][seg_i]
            class_label = int(seg_info['class_label'][seg_i])
            class_onehot = class_id_to_one_hot(class_label)
            inst_confidence = seg_info['inst_confidence'][seg_i]
            overlap_ratio = seg_info['overlap_ratio'][seg_i]
            pose = seg_info['pose'][seg_i]
            center = seg_info['center'][seg_i]

            segment_label = 0
            if 'segment_label' in seg_info:
                segment_label = seg_info['segment_label'][seg_i]
            

            segment = Segment(
                points_seg, is_thing, instance_label, class_label, 
                inst_confidence, overlap_ratio, pose, seg_i, center, segment_label, 
                # sem_feat=class_onehot
            )
            if(segment.points.shape[0] < 1):
                print(f"Not enough points in seg-{seg_i}")
                continue
            # segment.calculateConfidenceDefault()
            segments_list.append(segment)
        
        return segments_list



