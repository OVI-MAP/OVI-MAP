import argparse

def parse_args():
    parse = argparse.ArgumentParser(description='Semantic Mapping-Python') 
    # dataset 
    parse.add_argument("--dataset", type=str, default="scenenn", 
        help="which data set to use, scenenn or scannet ")

    # files path
    parse.add_argument("--scene_num", type=str, required=True, 
        help="which scene for mapping ")
    parse.add_argument("--result_folder", type=str,required=True, 
        help="folder of mapping results")
    parse.add_argument("--data_folder", type=str, required=True, 
        help="which scene for mapping ")
    parse.add_argument("--traj_filename", type=str, default="trajectory.log", 
        help="which trajectory_to_use ")

    # log
    parse.add_argument("--quiet", action='store_true',
        help="suppress WARNING messages from third-party libraries")
    parse.add_argument("--log", type=str, default='',
        help="log info ")
    # mapping configuration
    parse.add_argument("--start", type=int,default=0, 
        help="start of the sequence, 0 means starting from the beginning")
    parse.add_argument("--end", type=int, default=-1, 
        help="end of the sequence, -1 means using all of them")
    parse.add_argument("--step", type=int, default=-1, 
        help="use one frame for integration every n_step frames ")
    parse.add_argument("--num_threads", type=int, default=-1, 
        help="threads to use")
    parse.add_argument("--debug", action="store_true", 
        help="whether to use visualization ")
    parse.add_argument("--preload", action="store_true", 
        help="whether to preload images ")

    parse.add_argument("--use_temp_results", action='store_true', 
        help="use 2D segments results ")
    parse.add_argument("--save_temp_results", action='store_true', 
        help="save intermediate 2D segments results ")
    parse.add_argument("--save_temp_img", action='store_true', 
        help="save intermediate results in images ")
    parse.add_argument("--intermediate_seg_folder", type=str, default='segments', 
        help="folder to save intermediate segments result ")

    parse.add_argument("--use_temp_panoptics", action='store_true', 
        help="use 2D panoptic segments ")
    parse.add_argument("--save_temp_panoptics", action='store_true', 
        help="save 2D panoptic segments ")
    parse.add_argument("--temp_panoptics_folder", type=str, default='segments', 
        help="folder to save 2D panoptic segments ")

    parse.add_argument("--use_temp_geometrics", action='store_true', 
        help="use 2D geometrics segments ")
    parse.add_argument("--save_temp_geometrics", action='store_true', 
        help="save 2D geometrics segments ")
    parse.add_argument("--temp_geometrics_folder", type=str, default='segments', 
        help="folder to save 2D geometrics segments result ")

    parse.add_argument("--task", type=str, default="coco80", 
        help="coco80; nyu13; Nyu40; cocoPano")
 
    parse.add_argument("--data_association", type=int, default=0, 
        help="0 - Ori; 1 - SemMerge; \
        2 - SemMerge+BackgroundMerge+only consider size>1000; \
        3 - SemMerge+BackgroundMerge+only consider size>1000 + consider Sem when register superpoint; \
        4 - no merging; \
        5 - using designated superpoint id for 3D segments")
    parse.add_argument("--inst_association", type=int, default=0, 
        help="0 for Ori; 1 for Label-Sem-Inst; \
        2 for Label-Inst-Sem; 3 for SegGraph ")


    # for SegGraph
    parse.add_argument("--seg_graph_confidence", type=int, default=0, 
        help="0 for all confidence as 1; \
            1 for using inst score; \
            2 for use inst score and overlap ratio; \
            3 for use inst score, overlap ratio and geometric confidence")
    parse.add_argument("--use_inst_label_connect", type=int, default=1, 
        help="")
    parse.add_argument("--connection_ratio_th", type=float, default=0.2, 
        help="")
    parse.add_argument("--test_geometric_confidence", action='store_true', 
        help="try test geometric confidence calculation")

    # NOTE currently not used
    parse.add_argument("--use_2D_confidence", action='store_true', 
        help="use MaskRCNN for 2D data association")
    parse.add_argument("--geo_confidence", action='store_true',
        help="use geometric confidence")
    parse.add_argument("--label_confidence", action='store_true',
        help="use label confidence")

    # Feature extraction
    parse.add_argument("--skip_feature_extraction", action='store_true',
        help="skip VLM feature extraction (only reconstruct, no open-vocab querying)")
    # Async perception worker
    parse.add_argument("--perception_python", type=str, 
        default="/home/zilong-ubuntu/miniconda3/envs/ovimap-perception-py310/bin/python",
        help="Path to Python interpreter for the perception worker. ")
    parse.add_argument("--perception_worker", type=str,
        default="scripts/perception_worker.py",
        help="Path to perception_worker.py script.")

    return parse.parse_args()