"""
View selection strategies for OVI-MAP.
Decides which views (frames) to keep per instance.
"""

import logging
import numpy as np

from utils.common_utils import PointCloudProcessor


def init_view_cov(
    depth_scaled, inst_d_mask, inst_points,
    glo_inst_id, cur_inst_info,
    vis_area_thres, sph_grid_size
):
    """Initialize the per-instance 3D bounding box and spherical view-coverage map.

    Returns:
        (success: bool, cur_inst_info: dict or None)
    """
    depth_arr = depth_scaled[inst_d_mask]
    if len(depth_arr) < 0.5 * vis_area_thres:
        logging.warning(f"[Skip] Not enough depth pts to init inst {glo_inst_id}")
        return False, None

    PCLprocessor = PointCloudProcessor(inst_points)
    bbox_res = PCLprocessor.process(voxel_size=0.01)
    bbox_pos, bbox_extent, bbox_rot_matrix = bbox_res

    view_map = np.zeros(sph_grid_size, dtype=np.uint8)
    cur_inst_info['view_map'] = view_map
    cur_inst_info['bbox_c'] = bbox_pos
    cur_inst_info['bbox_s'] = bbox_extent
    cur_inst_info['bbox_rot'] = np.array(bbox_rot_matrix)

    return True, cur_inst_info


def update_view_cov_map(
    inst_points, bbox_c, view_map, sph_grid_size,
    view_overlap_ratio_thres
):
    """Map object surface rays to a spherical grid and check view overlap.

    Returns:
        (novel_view: bool, view_map: np.ndarray or None)
    """
    obj_rays = inst_points - bbox_c.T
    obj_rays = obj_rays / np.linalg.norm(obj_rays, axis=-1, keepdims=True)
    # convert them to sph coord and map to the occ grid
    theta = np.arccos(np.clip(obj_rays[:, 2], -1.0, 1.0))  # in [0, pi]
    theta = theta / np.pi * sph_grid_size[0]  # map to [0, grid_H]
    theta = np.floor(theta).astype(np.int32)
    phi = np.arctan2(obj_rays[:, 1], obj_rays[:, 0])  # in [-pi, pi]
    phi = (phi + np.pi) / (2 * np.pi) * sph_grid_size[1]  # map to [0, grid_W]
    phi = np.floor(phi).astype(np.int32)

    # 4.3. check the overlapping with the existing views
    sph_coords = np.unique(np.stack((theta, phi), axis=1), axis=0)  # (M, 2) removed duplicates
    view_overlap_area = np.count_nonzero(
        view_map[sph_coords[:, 0], sph_coords[:, 1]] > 0)
    view_overlap_ratio = view_overlap_area / sph_coords.shape[0]
    if view_overlap_ratio > view_overlap_ratio_thres:
        # too many overlapping views, skip this instance
        return False, None

    view_map[sph_coords[:, 0], sph_coords[:, 1]] = 1
    return True, view_map


# ---------------------------------------------------------------------------
# Per-frame view selection entry point
# ---------------------------------------------------------------------------

def select_views_for_frame(
    glo_inst_map, inst_seg, depth_scaled, valid_d_mask, pose, points_map,
    inst_dict,
    vis_area_thres, max_top_vis, sph_grid_size,
    view_select_strategy, view_overlap_ratio_thres,
):
    """Run view selection for all observed instances in the current frame.

    Modifies *inst_dict* in place to update per-instance state (view_map + [3d_bbox]) 
    and decides which instances are selected for feature extraction.

    Strategies (passed as ``view_select_strategy``):
        - 'vis':      Top-K by visible pixel area.
        - 'viewcov':  Spherical view-coverage map; skip views with >85% overlap.
        - 'combine':  Both view coverage gate + top-K visibility ranking (default).

    Args:
        glo_inst_map:        uint16[H, W]  — global instance ID per pixel.
        inst_seg:            int32[H, W]   — panoptic segmentation of current RGB.
        depth_scaled:        float32[H, W] — depth in meters.
        valid_d_mask:        bool[H, W]    — pixels with valid depth.
        pose:                float32[4, 4] — camera-to-world transform.
        points_map:          float32[H, W, 3] — 3D points back-projected from depth.
        inst_dict:           dict[int, dict]  — global per-instance accumulated state
        vis_area_thres:      int           — minimum visible pixel area.
        max_top_vis:         int           — max views per instance.
        sph_grid_size:       (int, int)    — spherical grid (H, W) for view coverage.
        view_select_strategy: str          — 'vis', 'viewcov', or 'combine'.
        view_overlap_ratio_thres: float    — skip if overlap exceeds this.

    Returns:
        list of dicts with keys:
          glo_inst_id, glo_inst_mask, pano_mask, overlap_area
    """
    selected = []

    observed_inst_ids = np.unique(glo_inst_map)
    for glo_inst_id in observed_inst_ids:
        # skip background
        if glo_inst_id == 0:
            continue

        glo_inst_mask = (glo_inst_map == glo_inst_id)  # (H, W)
        glo_inst_area = np.count_nonzero(glo_inst_mask)
        # 1. skip too small instance reconstructed in the global map
        if glo_inst_area < 0.5 * vis_area_thres:
            logging.warning(f"[Skip] inst {glo_inst_id} with inst area {glo_inst_area}.")
            continue

        # NOTE find the majority panoptic id in current frame's panoptic segmentation
        pano_id_map = inst_seg[glo_inst_mask]
        pano_ids, pano_id_count = np.unique(pano_id_map, return_counts=True)
        pano_id = pano_ids[np.argmax(pano_id_count)]

        # only consider the mask within the local segments
        pano_mask = (inst_seg == pano_id)  # (H, W)
        pano_id_area = np.count_nonzero(pano_mask)

        # count the pixel in the overlap area
        overlap_area = np.max(pano_id_count)
        overlap_mask = np.logical_and(glo_inst_mask, pano_mask)

        # 2. check the visibility of the instance in current frame
        if pano_id_area < vis_area_thres:
            logging.warning(f"[Skip] inst {glo_inst_id} with pano area {pano_id_area}.")
            continue

        # 3. NOTE if smaller, it means the pano seg is oversegmented
        # we allow the global instance to be under-segemented
        # pano_in_inst_ratio = overlap_area / glo_inst_area
        # if pano_in_inst_ratio < 0.5:
        #     logging.warning(f"[Skip] inst {glo_inst_id} with overlap ratio {pano_in_inst_ratio:.4f} < 0.5")
        #     continue

        if glo_inst_id not in inst_dict.keys():
            cur_inst_info = {
                'frame_id': [], 'feat': [], 'pose': [],
                'vis_area': [], 'box_2d': [],
            }
        else:
            cur_inst_info = inst_dict[glo_inst_id]

        # 4. check if it's a novel view to select
        if view_select_strategy in ('viewcov', 'combine'):
            inst_d_mask = np.logical_and(glo_inst_mask, valid_d_mask)  # (H, W)
            if np.count_nonzero(inst_d_mask) < 0.1 * vis_area_thres:
                logging.warning(f"[Skip] Not enough valid depth pts in inst {glo_inst_id}.")
                continue
            inst_points = points_map[inst_d_mask].astype(np.float32).reshape(-1, 3)
            inst_points = inst_points @ pose[:3, :3].T + pose[:3, 3:4].T

            # 4.1. Init an object center by estimating the BBOX
            if 'view_map' not in cur_inst_info.keys():
                success, cur_inst_info = init_view_cov(
                    depth_scaled, inst_d_mask, inst_points,
                    glo_inst_id, cur_inst_info,
                    vis_area_thres, sph_grid_size
                )
                if not success:
                    continue

            # 4.2. Map the rays from obj_c to surface to the spherical grid
            view_map = cur_inst_info['view_map']
            success, view_map = update_view_cov_map(
                inst_points, cur_inst_info['bbox_c'],
                view_map, sph_grid_size, view_overlap_ratio_thres
            )
            if not success:
                continue  # this view is not novel enough

            # decide whether to select this view based on the vis_area
            if view_select_strategy == 'combine':
                past_vis_areas = cur_inst_info['vis_area']
                if len(past_vis_areas) >= max_top_vis:
                    top_vis_s = np.sort(np.array(cur_inst_info['vis_area']))[-max_top_vis:]
                    if overlap_area <= np.min(top_vis_s):
                        continue  # not visible enough

            cur_inst_info['view_map'] = view_map

        elif view_select_strategy == 'vis':
            past_vis_areas = cur_inst_info['vis_area']
            if len(past_vis_areas) >= max_top_vis:
                top_vis_s = np.sort(np.array(cur_inst_info['vis_area']))[-max_top_vis:]
                if overlap_area <= np.min(top_vis_s):
                    continue  # not visible enough

        # write back the cur_inst_info to update view_map for this instance
        inst_dict[glo_inst_id] = cur_inst_info

        selected.append({
            'glo_inst_id': glo_inst_id,
            'glo_inst_mask': glo_inst_mask,
            'pano_mask': pano_mask,
            'overlap_area': overlap_area,
        })

    return selected
