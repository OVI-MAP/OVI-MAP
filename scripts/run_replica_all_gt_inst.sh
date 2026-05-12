#!/bin/bash

# SceneList=("office2")
SceneList=("office0" "office1" "office2" "office3" "office4" "room0" "room1" "room2")
# data source dir
DataFolder=/media/zilong/Documents/MasterProject/Replica

ThreadNum=10
logPrefix=info

export PYTHONPATH=${PYTHONPATH}:mapping_ros_ws/devel/lib

# Loop through each scene
for SceneNum in "${SceneList[@]}"; do
    echo "Processing Replica scene: $SceneNum"

    ResultFolder=/home/zilong/Disk_data/semantic_mapping_result/${SceneNum}
    IntermediateSegsFolder=/home/zilong/Disk_data/fuse_seg_temp/${SceneNum}
    PanopticSegsFolder=/home/zilong/Disk_data/sem_seg_temp/${SceneNum}
    GeometricSegsFolder=/home/zilong/Disk_data/geo_seg_temp/${SceneNum}

    python scripts/test_view_selection.py \
        --dataset replica \
        --scene_num ${SceneNum} \
        --data_folder ${DataFolder} \
        --result_folder ${ResultFolder} \
        --start 0 --end 2000 --step 10 \
        --log "${logPrefix}"
done
