#!/bin/bash

SceneList=("office0")
# SceneList=("office0" "office1" "office2" "office3" "office4" "room0" "room1" "room2")

# workspace dir
WS=/home/zilong-ubuntu/Data

# data source dir
DataFolder=${WS}/Datasets/Replica

ThreadNum=10
logPrefix=info

export PYTHONPATH=${PYTHONPATH}:mapping_ros_ws/devel/lib

# Loop through each scene
for SceneNum in "${SceneList[@]}"; do
    echo "Processing Replica scene: $SceneNum"

    ResultFolder=${WS}/semantic_mapping_result/${SceneNum}
    IntermediateSegsFolder=${WS}/fuse_seg_temp/${SceneNum}
    PanopticSegsFolder=${WS}/sem_seg_temp/${SceneNum}
    GeometricSegsFolder=${WS}/geo_seg_temp/${SceneNum}

    python scripts/panoptic_mapping_.py \
        --dataset replica --task Nyu40 \
        --scene_num ${SceneNum} \
        --data_folder ${DataFolder} \
        --result_folder ${ResultFolder} \
        --start 0 --end 2000 --step 10 \
        --data_association 2 \
        --inst_association 4 \
        --seg_graph_confidence 3 \
        --use_temp_results --save_temp_results \
        --intermediate_seg_folder ${IntermediateSegsFolder} \
        --temp_panoptics_folder ${PanopticSegsFolder} \
        --use_temp_geometrics --save_temp_geometrics \
        --temp_geometrics_folder ${GeometricSegsFolder} \
        --num_threads ${ThreadNum} \
        --log "${logPrefix}"
done

# --preload \
# --use_temp_results \
# --use_temp_geometrics \