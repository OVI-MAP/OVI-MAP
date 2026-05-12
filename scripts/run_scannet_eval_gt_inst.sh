#!/bin/bash

SceneList=("scene0011_00" "scene0011_01" "scene0050_00" "scene0050_01" "scene0050_02" "scene0084_00" "scene0084_01" "scene0084_02" "scene0168_00" "scene0168_01" "scene0168_02" "scene0231_00" "scene0231_01" "scene0231_02" "scene0378_00" "scene0378_01" "scene0378_02" "scene0518_00")
# SceneList=("scene0231_00")
# data source dir
DataFolder=/media/zilong/Documents/MasterProject/scannet_v2

ThreadNum=10
logPrefix=info

export PYTHONPATH=${PYTHONPATH}:mapping_ros_ws/devel/lib

# Loop through each scene
for SceneNum in "${SceneList[@]}"; do
    echo "Processing Scannet scene: $SceneNum"

    ResultFolder=/home/zilong/Disk_data/semantic_mapping_result/${SceneNum}
    IntermediateSegsFolder=/home/zilong/Disk_data/fuse_seg_temp/${SceneNum}
    PanopticSegsFolder=/home/zilong/Disk_data/sem_seg_temp/${SceneNum}
    GeometricSegsFolder=/home/zilong/Disk_data/geo_seg_temp/${SceneNum}

    python scripts/test_view_selection.py \
        --dataset scannet_nyu \
        --scene_num ${SceneNum} \
        --data_folder ${DataFolder} \
        --result_folder ${ResultFolder} \
        --log "${logPrefix}"
done