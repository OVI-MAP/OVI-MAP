#!/bin/bash

SceneList=("scene0011_00" "scene0011_01" "scene0050_00" "scene0050_01" "scene0050_02" "scene0084_00" "scene0084_01" "scene0084_02" "scene0168_00" "scene0168_01" "scene0168_02" "scene0231_00" "scene0231_01" "scene0231_02" "scene0378_00" "scene0378_01" "scene0378_02" "scene0518_00")
# SceneList=("scene0378_01" "scene0378_02" "scene0518_00")

# workspace dir
WS=/home/zilong-ubuntu/Data

# data source dir
DataFolder=${WS}/Datasets/ScanNet

ThreadNum=10
logPrefix=info

export PYTHONPATH=${PYTHONPATH}:mapping_ros_ws/devel/lib

# Loop through each scene
for SceneNum in "${SceneList[@]}"; do
    echo "Processing Scannet scene: $SceneNum"

    ResultFolder=${WS}/semantic_mapping_result/${SceneNum}
    IntermediateSegsFolder=${WS}/fuse_seg_temp/${SceneNum}
    PanopticSegsFolder=${WS}/sem_seg_temp/${SceneNum}
    GeometricSegsFolder=${WS}/geo_seg_temp/${SceneNum}

    python scripts/panoptic_mapping_.py \
        --dataset scannet_nyu --task Nyu40 \
        --scene_num ${SceneNum} \
        --data_folder ${DataFolder} \
        --result_folder ${ResultFolder} \
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

# --preload
# --use_temp_results
# --use_temp_geometrics