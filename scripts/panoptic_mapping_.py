import os, sys, time, argparse, pickle, logging, copy
from os.path import join as pjoin
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import subprocess, json, zlib, threading, base64

# self packages
from options import parse_args
from view_selection import select_views_for_frame
from utils.common_scannet_nyu import SegmentsGenerator
from utils.common_utils import Segment
from utils.data_loaders import ScannetLoader, ReplicaLoader
# for cross env 
from ipc_utils import create_shm_slots, compress_mask, cleanup_shm_slots

# Perception worker handles all GPU inference; reconstruction is CPU-only.
# Use "cuda" as default device string for the worker (set to "cpu" if no GPU).
DEVICE = "cuda"

FORMAT = '%(asctime)s.%(msecs)06d %(levelname)-8s: [%(filename)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt='%H:%M:%S')

# # Suppress noisy library loggers
# _log_suppress = ["PIL", "matplotlib", "transformers", "urllib3", "matplotlib.font_manager"]
# for _mod in _log_suppress:
#     logging.getLogger(_mod).setLevel(logging.WARNING)


def make_res_dirs(args):
    res_dirs = {}

    result_root_dir = args.result_folder

    res_dirs['folder'] = result_root_dir
    res_dirs['log'] = os.path.join(result_root_dir, 'log')
    res_dirs['temp_segs'] = args.intermediate_seg_folder
    res_dirs['temp_panoptics'] = args.temp_panoptics_folder
    res_dirs['temp_geometrics'] = args.temp_geometrics_folder

    if not os.path.exists(res_dirs['folder']):
        os.makedirs(res_dirs['folder'])
    if not os.path.exists(res_dirs['log']):
        os.makedirs(res_dirs['log'])

    # Currently we can only use pre-processed panoptic segments
    assert os.path.exists(res_dirs['temp_panoptics']), \
        f"[Error]Temp panoptic folder: {res_dirs['temp_panoptics']} does not exist!"

    if args.use_temp_geometrics:
        assert os.path.exists(res_dirs['temp_geometrics']), \
            f"[Error]Temp geometrics folder: {res_dirs['temp_geometrics']} does not exist!"
    if args.save_temp_geometrics and not os.path.exists(res_dirs['temp_geometrics']):
        os.makedirs(res_dirs['temp_geometrics'])

    # if args.use_temp_results:
    #     assert os.path.exists(res_dirs['temp_segs']), \
    #         f"[Error]Temp fused segmentations folder: {res_dirs['temp_segs']} does not exist!"
    # if args.save_temp_results and not os.path.exists(res_dirs['temp_segs']):
    #     os.makedirs(res_dirs['temp_segs'])
    
    return res_dirs


def init_feature_extractor(args, H_depth, W_depth, VLM_name, DEVICE, inst_dict):
    """Launch the perception worker and return IPC objects."""

    # Resolve worker script path relative to this file
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.isabs(args.perception_worker):
        worker_script = args.perception_worker
    else:
        worker_script = os.path.join(_this_dir, os.path.basename(args.perception_worker))

    NUM_SHM_SLOTS = 4
    shm_slots = create_shm_slots(H_depth, W_depth, num_slots=NUM_SHM_SLOTS)
    shm_slot_names = [s[0] for s in shm_slots]

    # Build a clean env for the worker: remove ROS devel-lib paths from
    # PYTHONPATH to avoid namespace-package conflicts (e.g. google.protobuf).
    worker_env = os.environ.copy()
    ros_workspace_path = None
    for p in worker_env.get("PYTHONPATH", "").split(":"):
        if "mapping_ros_ws/devel/lib" in p:
            ros_workspace_path = p
            break
    if ros_workspace_path:
        worker_env["PYTHONPATH"] = worker_env.get("PYTHONPATH", "").replace(
            ":" + ros_workspace_path, ""
        ).replace(ros_workspace_path + ":", "").replace(ros_workspace_path, "")

    worker_proc = subprocess.Popen(
        [args.perception_python, worker_script,
         "--shm-names", ",".join(shm_slot_names),
         "--img-height", str(H_depth),
         "--img-width", str(W_depth),
         "--model-name", VLM_name,
         "--device", str(DEVICE)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,  # worker stderr inherits parent's stderr
        env=worker_env,
    )
    worker_stdin = worker_proc.stdin
    worker_stdout = worker_proc.stdout

    # Track which shm slots have been written for the current frame
    slot_frame_map = {}  # slot_idx -> f_i

    # ---- Result collector (background thread) ----
    pending_count = [0]  # mutable container shared between threads

    def result_collector():
        """Read responses from worker stdout, fill inst_dict placeholders."""
        for line in worker_stdout:
            line = line.decode("utf-8").strip()
            if not line:
                continue
            response = json.loads(line)
            if response.get("cmd") == "shutdown_ack":
                break
            gid = response["glo_inst_id"]
            ridx = response["request_idx"]
            if response["status"] == "ok":
                feat_bytes = base64.b64decode(response["feat_b64"])
                roi_feat = np.frombuffer(feat_bytes, dtype=np.float32)
                if gid in inst_dict and ridx < len(inst_dict[gid]['feat']):
                    inst_dict[gid]['feat'][ridx] = roi_feat
                # logging.info("[Async] Feature extracted for inst %s frame %s (request %d).", gid, response.get("f_i", "?"), ridx)
            else:
                logging.warning(
                    "[Async] Feature extraction failed for inst %s "
                    "frame %s: %s",
                    gid, response.get("f_i", "?"),
                    response.get("error", "unknown"))
            pending_count[0] -= 1

    collector_thread = threading.Thread(target=result_collector, daemon=True)
    collector_thread.start()

    def wait_for_slot(slot_idx):
        """Backpressure: block if worker is falling too far behind."""
        while pending_count[0] > 50:
            time.sleep(0.01)

    logging.info("Perception worker launched (PID %d) with model '%s' on %s",
                 worker_proc.pid, VLM_name, DEVICE)

    return {
        'worker_proc': worker_proc,
        'worker_stdin': worker_stdin,
        'collector_thread': collector_thread,
        'pending_count': pending_count,
        'shm_slots': shm_slots,
        'slot_frame_map': slot_frame_map,
        'wait_for_slot': wait_for_slot,
        'NUM_SHM_SLOTS': NUM_SHM_SLOTS,
    }



def main(args):
    if args.quiet:
        logging.disable(logging.WARNING)
    import consistent_gsm # type: ignore
    import depth_segmentation_py # type: ignore

    # dataset
    dataset = args.dataset
    panoptic_node = None
    
    # set configuration
    use_temp_results = args.use_temp_results
    save_segments = args.save_temp_results
    use_geos = args.use_temp_geometrics
    save_geos = args.save_temp_geometrics
    save_results_img = args.save_temp_img

    task = args.task
    use_geo_confidence = args.geo_confidence
    use_label_confidence = args.label_confidence
    inst_association = args.inst_association
    data_association = args.data_association

    seg_graph_confidence = args.seg_graph_confidence
    use_inst_label_connect = args.use_inst_label_connect
    connection_ratio_th = args.connection_ratio_th

    # input and output configuration
    scene_num = args.scene_num
    result_folder = args.result_folder
    data_path = args.data_folder
    scene_folder = pjoin(data_path, scene_num)

    result_dirs = make_res_dirs(args)

    temp_segs_folder = result_dirs['temp_segs']
    temp_pano_folder = result_dirs['temp_panoptics']
    temp_geos_folder = result_dirs['temp_geometrics']

    log_info = args.log

    # DataLoader
    if dataset == "scannet_nyu":
        data_loader = ScannetLoader(
            scene_folder, args.preload, args.preload)
    elif dataset == "replica":
        data_loader = ReplicaLoader(
            scene_folder, args.preload, args.preload)
    else:
        logging.error("Please choose a suitable dataset!")
        raise NotImplementedError

    H_depth = data_loader.depth_h
    W_depth = data_loader.depth_w
    K_depth = data_loader.getDepthCameraMatrix()

    start = args.start
    assert (start >= data_loader.indexes[0])
    end = args.end
    if end < 0:
        end = data_loader.index_max + 1
    step = args.step
    # use -1 to set the step size to 1/200 of the total frames (for Scannet Dataset)
    if step < 0:
        step = int((end-start) // 200)
    num_frames = int(np.ceil((end-start)/step))
    logging.info(f"Running scene {scene_num} from frame {start} to frame {end-1} with step {step}.")
    
    # for scannet exp with limited observations
    # iters = 160
    # end = iters * step + start


    # =========== initialized the global segment integrator ===========
    log_file = os.path.abspath(result_dirs['log'])
    gsm_node = consistent_gsm.GlobalSegmentMap_py(
        log_file, task, 
        use_geo_confidence, 
        use_label_confidence, 
        inst_association, data_association, 
        args.num_threads, args.debug, 
        seg_graph_confidence, 
        use_inst_label_connect==1, 
        connection_ratio_th, 
        0.9 # cos_sim_th, not used
    )
    gsm_node.outputLog(log_info)

    # ==========================================================
    VLM_name = 'siglip-l-16-384'
    exp_results = pjoin(result_dirs['folder'], 'cropformer_inst')
    # temp_feats = pjoin(result_dirs['folder'], 'cropformer_inst', 'temp_feats')
    # use_prev_feat = False

    # View selection strategy: 'vis' | 'viewcov' | 'combine' (default)
    view_select_strategy = 'combine'

    # NOTE the final pkl file stores the instance semantic features for each instance, 
    # including the features, poses, 2D bbox, and visibility scores
    inst_sem_name = f'inst_sem_{VLM_name}_{num_frames}_incre_combine.pkl'
    inst_sem_f = pjoin(exp_results, inst_sem_name)

    # max depth for ray casting to get the 2D proj. of global instance map, in meters
    ray_cast_max_depth = 50.0
    # minimum visible area in the pano seg for an instance to be considered
    vis_area_thres = 1000
    # NOTE buffer size for each instance to store the meta from different views
    max_top_vis = 10

    if view_select_strategy == 'viewcov':
        view_overlap_ratio_thres = 0.85
    else:
        view_overlap_ratio_thres = 0.9

    # grid size for spherical view coverage map (H, W)
    yx_grid = np.mgrid[0:H_depth, 0:W_depth] # (2, H, W)
    sph_grid_size = (int(H_depth/8), int(W_depth/8))

    # ==========================================================

    if not os.path.exists(exp_results):
        os.makedirs(exp_results)
    # if not os.path.exists(temp_feats):
    #     os.makedirs(temp_feats)

    logging.info("Using pre-processed instance seg data!")
    if not use_geos:
        dep_segmentor = depth_segmentation_py.DepthSegmentation_py(
            H_depth,W_depth,cv2.CV_32FC1, K_depth
        )
    else:
        logging.info("Using pre-processed depth seg data!")
        dep_segmentor = None
            
    # ---- Initialize instance-semantic manager ----
    inst_dict = {}

    # create the segment generator
    segments_generator = SegmentsGenerator(
        gsm_node, dep_segmentor, panoptic_node,
        save_results_img, result_dirs['folder'], 
        save_segments, use_temp_results, temp_segs_folder, 
        panoptics_folder=temp_pano_folder, 
        save_geometrics=save_geos, geometrics_folder=temp_geos_folder
    )
    gsm_node.initializeCameraRayCaster(
        K_depth, H_depth, W_depth, 0.01, ray_cast_max_depth, args.num_threads
    )

    if not args.skip_feature_extraction:
        feature_extractor = init_feature_extractor(args, H_depth, W_depth, VLM_name, DEVICE, inst_dict)
    else:
        feature_extractor = None
        logging.info("Feature extraction skipped (--skip_feature_extraction).")


    # ========================== start mapping ==========================
    time_s = time.time()
    frame_ids = range(start, end, step)
    for f_i in tqdm(frame_ids):
        # NOTE img is in BGR, depth is in meters, pose is c2w
        # rgb image has been wrapped to depth image
        rgb_img, depth_scaled, pose = data_loader.getDataFromIndex(f_i)
        # check data validity
        if(rgb_img is None or depth_scaled is None or pose is None):
            logging.warning(f"[Skip] frame {f_i} is lack of RGB / Depth / Pose.")
            continue
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        valid_d_mask = (depth_scaled > 0.0) & (depth_scaled < ray_cast_max_depth)
        

        # TODO use instance segs from cropformer
        inst_seg_f = pjoin(temp_pano_folder, 
            os.path.basename(data_loader.rgb_path_map[f_i]).split('.')[0] + '.png')
        inst_seg = cv2.imread(inst_seg_f, cv2.IMREAD_UNCHANGED)
        if data_loader.mapRGBtoDepth is not None:
            inst_seg = data_loader.mapRGBtoDepth(inst_seg)
        inst_seg = inst_seg.astype(np.int32) # pybind11 only accepts not unsigned int


        # perform depth segmentation of not using pre-processed depth segments
        if dep_segmentor is not None:
            # NOTE depth segmentation is done in a separate thread to avoid blocking the main thread
            # results must be dump locally and will be loaded in the frameToSegmentsCropFormer
            segments_generator.SegmentDepth(depth_scaled, rgb_img, f_i)
        
        # fuse the depth segments and panoptic segments
        segment_list: list[Segment] = segments_generator.frameToSegmentsCropFormer(
            depth_scaled, K_depth, pose, f_i, inst_seg
        )
        if len(segment_list) == 0:
            logging.warning(f"[Skip] No segment found in frame {f_i}")
            continue

        # lift the fused segments to 3D and insert them into the global segment map
        for segment in segment_list:
            if(seg_graph_confidence == 3):
                segment.calculateBBox()

            # all segs except bg are 'thing' and with a 1.0 inst_conf
            gsm_node.insertSegmentsOpen(
                segment.points, segment.box_points, 
                segment.instance_label, segment.class_label, 
                segment.sem_feat, 
                segment.inst_confidence, segment.overlap_ratio, 
                pose, segment.is_thing, segment.segment_label
            )
    
        # update the global segment map with the new segments
        gsm_node.integrateFrame()

        # Ray casting to get the global instance map from the super points
        glo_inst_map = gsm_node.raycastInstancePredictions(
            pose, inst_seg, depth_scaled
        )

        if view_select_strategy != 'vis':
            depth_scaled[~valid_d_mask] = 0.0
            points_map = cv2.rgbd.depthTo3d(depth_scaled, K_depth)

        # ################# View Selection #################
        selected_views = select_views_for_frame(
            glo_inst_map, inst_seg, depth_scaled, valid_d_mask, pose, points_map,
            inst_dict,
            vis_area_thres, max_top_vis, sph_grid_size,
            view_select_strategy, view_overlap_ratio_thres,
        )

        # ################# Async Feature Extraction #################
        for sv in selected_views:
            glo_inst_id = sv['glo_inst_id']
            glo_inst_mask = sv['glo_inst_mask']
            pano_mask = sv['pano_mask']
            overlap_area = sv['overlap_area']

            # detrermine the cropping area by the global inst mask
            yxs = yx_grid[:, glo_inst_mask] # (2, M)
            y1, x1 = np.min(yxs[0]), np.min(yxs[1])
            y2, x2 = np.max(yxs[0]), np.max(yxs[1])

            # masked by the union of local instance mask and the global instance mask
            # will be used to create pure color bg masking for the visual feature extractor
            obj_mask = np.logical_or(pano_mask, glo_inst_mask)

            # ---- Synchronous metadata (needed by future view selections) ----
            inst_dict[glo_inst_id]['frame_id'].append(f_i)
            inst_dict[glo_inst_id]['box_2d'].append((x1, y1, x2, y2))
            inst_dict[glo_inst_id]['pose'].append(pose)
            inst_dict[glo_inst_id]['vis_area'].append(overlap_area)
            # placeholder, filled async
            inst_dict[glo_inst_id]['feat'].append(None)  
            request_idx = len(inst_dict[glo_inst_id]['feat']) - 1

            if not args.skip_feature_extraction:
                # -------- Async: feature extraction --------
                # wait for a free shared memory slot to sync rgb
                slot_idx = f_i % feature_extractor['NUM_SHM_SLOTS']
                if slot_idx not in feature_extractor['slot_frame_map'] or feature_extractor['slot_frame_map'][slot_idx] != f_i:
                    feature_extractor['wait_for_slot'](slot_idx)
                    np.copyto(feature_extractor['shm_slots'][slot_idx][2], rgb_img)
                    feature_extractor['slot_frame_map'][slot_idx] = f_i

                # send feature extraction request to perception worker
                mask_bytes = compress_mask(obj_mask)
                request = json.dumps({
                    "f_i": int(f_i), "glo_inst_id": int(glo_inst_id),
                    "slot_idx": int(slot_idx), "request_idx": int(request_idx),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)], "mask_len": len(mask_bytes)
                })
                feature_extractor['worker_stdin'].write((request + "\n").encode())
                feature_extractor['worker_stdin'].write(mask_bytes)
                feature_extractor['worker_stdin'].flush()
                feature_extractor['pending_count'][0] += 1
                # --------------------------------------------
                
        # clean idle mem
        gsm_node.clearTemporaryMemory()

    if not args.skip_feature_extraction:
        # ---- Shutdown perception worker and drain remaining results ----
        logging.info("Sending shutdown to perception worker (pending: %d)...", feature_extractor['pending_count'][0])
        try:
            feature_extractor['worker_stdin'].write('{"cmd": "shutdown"}\n'.encode())
            feature_extractor['worker_stdin'].flush()
            feature_extractor['worker_stdin'].close()
        except (BrokenPipeError, OSError):
            pass

        # Wait for collector thread to process remaining responses
        feature_extractor['collector_thread'].join(timeout=300)
        if feature_extractor['collector_thread'].is_alive():
            logging.warning("Result collector did not finish within timeout; "
                            "some features may be missing.")

        # Terminate worker if still running
        if feature_extractor['worker_proc'].poll() is None:
            feature_extractor['worker_proc'].wait(timeout=10)
            if feature_extractor['worker_proc'].poll() is None:
                logging.warning("Worker did not exit gracefully; killing.")
                feature_extractor['worker_proc'].kill()
                feature_extractor['worker_proc'].wait()

        # Cleanup shared memory
        cleanup_shm_slots(feature_extractor['shm_slots'])
        logging.info("Perception worker shut down. Pending at exit: %d", feature_extractor['pending_count'][0])
        # GPU memory tracking is handled by the perception worker process
    gsm_node.outputLog(f"Time taken per frame: {(time.time() - time_s)/len(frame_ids):.2f} seconds")
    # -------------------- Shutdown complete --------------------

    # generate log and mesh
    gsm_node.LogLabelInformation()
    gsm_node.LogMeshColors(exp_results)

    logging.info("Start mesh generation!")
    # flags for saving: label_mesh, sem_mesh, inst_mesh
    gsm_node.generateMesh(
        exp_results, str(num_frames), 
        False, False, True
    )

    if not args.skip_feature_extraction:
        np.random.seed(0)

        # sum up the instance features and save them
        inst_sem_dict = {}
        total_query = 0
        for glo_inst_id in inst_dict.keys():
            cur_inst_info = inst_dict[glo_inst_id]
            inst_frames = cur_inst_info['frame_id']
            obs_num = len(inst_frames)
            if obs_num == 0:
                continue
            inst_poses = cur_inst_info['pose']
            inst_bbox2ds = cur_inst_info['box_2d']
            vis_scores = np.array(cur_inst_info['vis_area'])

            # Filter out entries where async feature extraction failed (None placeholder)
            valid_idx = [i for i, f in enumerate(cur_inst_info['feat']) if f is not None]
            if len(valid_idx) == 0:
                continue
            all_feats = np.array([cur_inst_info['feat'][i] for i in valid_idx])
            vis_scores = vis_scores[valid_idx]
            inst_frames = [inst_frames[i] for i in valid_idx]
            inst_poses = [inst_poses[i] for i in valid_idx]
            inst_bbox2ds = [inst_bbox2ds[i] for i in valid_idx]

            total_query += all_feats.shape[0]

            if view_select_strategy == 'viewcov':
                view_map = cur_inst_info['view_map']
                occ_cnt = np.count_nonzero(view_map > 0)
                occ_ratio = occ_cnt / (sph_grid_size[0] * sph_grid_size[1])
                gsm_node.outputLog(f"Instance {glo_inst_id} has {occ_ratio*100:.2f}% observed from {obs_num} frames")
            else:
                # select the top-k with max vis area
                top_indices = np.argsort(vis_scores)[-max_top_vis:] # indices for values sorted in ascending order
                all_feats = all_feats[top_indices]
                inst_frames = [inst_frames[i] for i in top_indices]
                inst_poses = [inst_poses[i] for i in top_indices]
                inst_bbox2ds = [inst_bbox2ds[i] for i in top_indices]
                vis_scores = vis_scores[top_indices]

            inst_color = gsm_node.getInstanceColor(glo_inst_id)

            inst_sem_dict[glo_inst_id] = {
                'feat': all_feats,
                'vis_area': vis_scores,
                'frame_id': inst_frames,
                'pose': inst_poses,
                'box_2d': inst_bbox2ds,
                'color': inst_color,
            }
            # if view_select_strategy == 'viewcov':
            #     inst_sem_dict[glo_inst_id]['bbox_c'] = cur_inst_info['bbox_c']
            #     inst_sem_dict[glo_inst_id]['bbox_s'] = cur_inst_info['bbox_s']
            #     inst_sem_dict[glo_inst_id]['bbox_rot'] = cur_inst_info['bbox_rot']

        gsm_node.outputLog(f"Total {len(inst_sem_dict)} instances with {total_query} queries, avg {total_query / len(inst_sem_dict):.2f}.")

        # dump results
        with open(inst_sem_f, 'wb') as f:
            pickle.dump(inst_sem_dict, f)
        logging.info(f"Saved instance semantic features to {inst_sem_f}")
    else:
        logging.info("Feature extraction skipped. Mesh generation only.")

    

if __name__=="__main__":
    # set files path
    args = parse_args()
    main(args)