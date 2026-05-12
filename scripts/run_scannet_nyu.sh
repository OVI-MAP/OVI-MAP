#!/bin/bash

# which scene to run
SceneNum=scene0001_00
# data source dir
DataFolder=/media/zilong/Documents/MasterProject/scannet_v2

# folder of mapping results
ResultFolder=/home/zilong/Disk_data/semantic_mapping_result/${SceneNum}
# folder to save intermediate segments result
IntermediateSegsFolder=/home/zilong/Disk_data/fuse_seg_temp/${SceneNum}
# folder to save 2D panoptic segments
PanopticSegsFolder=/home/zilong/Disk_data/sem_seg_temp/${SceneNum}
# folder to save 2D geometrics segments
GeometricSegsFolder=/home/zilong/Disk_data/geo_seg_temp/${SceneNum}

ThreadNum=10
logPrefix=info

export PYTHONPATH=${PYTHONPATH}:mapping_ros_ws/devel/lib

python scripts/panoptic_mapping_.py \
--dataset scannet_nyu \
--task Nyu40 \
--scene_num ${SceneNum} \
--data_folder ${DataFolder} \
--result_folder ${ResultFolder} \
--start 0 \
--end 800 \
--step 5 \
--data_association 2 \
--inst_association 4 \
--seg_graph_confidence 3 \
--use_temp_results \
--save_temp_results \
--intermediate_seg_folder ${IntermediateSegsFolder} \
--temp_panoptics_folder ${PanopticSegsFolder} \
--use_temp_geometrics \
--save_temp_geometrics \
--temp_geometrics_folder ${GeometricSegsFolder} \
--num_threads ${ThreadNum} \
--log "${logPrefix}"

# --preload \
# --use_temp_results \
# --use_temp_geometrics \