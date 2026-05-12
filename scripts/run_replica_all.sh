#!/bin/bash

SceneList=("office0")
# SceneList=("office0" "office1" "office2" "office3" "office4" "room0" "room1" "room2")
# data source dir
DataFolder=/home/zilong-ubuntu/Data/Datasets/Replica

ThreadNum=10
logPrefix=info

export PYTHONPATH=${PYTHONPATH}:mapping_ros_ws/devel/lib

# Loop through each scene
for SceneNum in "${SceneList[@]}"; do
    echo "Processing Replica scene: $SceneNum"

    ResultFolder=/home/zilong-ubuntu/Data/semantic_mapping_result/${SceneNum}
    IntermediateSegsFolder=/home/zilong-ubuntu/Data/fuse_seg_temp/${SceneNum}
    PanopticSegsFolder=/home/zilong-ubuntu/Data/sem_seg_temp/${SceneNum}
    GeometricSegsFolder=/home/zilong-ubuntu/Data/geo_seg_temp/${SceneNum}

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