#!/bin/bash
# =============================================================================
# OVI-MAP Async Pipeline Launcher
# =============================================================================
# Runs the split pipeline: 
# reconstruction (Python 3.8 + ROS) + perception worker (Python 3.10+).

# Examples:
#   bash scripts/run_pipeline_async.sh office0          # single scene
#   bash scripts/run_pipeline_async.sh                  # all default scenes
#
# Requirements:
#   - ovi-map conda env with ROS packages and sourced (Python 3.8 for reconstruction)
#   - ovimap-perception-py310 conda env created (Python 3.10+ for perception)
# =============================================================================
set -e

# Workspace root (where datasets and results live)
WS=${WS:-/data}
DataFolder=${DataFolder:-${WS}/Datasets/ScanNet}
# Result folder prefix
ResultPrefix=${ResultPrefix:-${WS}/semantic_mapping_result}

if [ $# -gt 0 ]; then
    SceneList=("$@")
else
    # SceneList=("scene0011_00")
    SceneList=("scene0011_00" "scene0011_01" "scene0050_00" "scene0050_01" "scene0050_02" "scene0084_00" "scene0084_01" "scene0084_02" "scene0168_00" "scene0168_01" "scene0168_02" "scene0231_00" "scene0231_01" "scene0231_02" "scene0378_00" "scene0378_01" "scene0378_02" "scene0518_00")
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Source ROS workspace (sets PYTHONPATH, LD_LIBRARY_PATH, ROS_PACKAGE_PATH, etc.)
source "${REPO_ROOT}/mapping_ros_ws/devel/setup.bash"

# Perception worker script location
PERCEPTION_WORKER="${REPO_ROOT}/scripts/perception_worker.py"
# Path to the Python interpreter for the perception worker
PERCEPTION_PYTHON="/opt/conda/envs/ovimap-perception-py310/bin/python"


for SceneNum in "${SceneList[@]}"; do
    echo ""
    echo ">>> Processing Scannet scene: ${SceneNum}"

    ResultFolder="${ResultPrefix}/${SceneNum}"
    IntermediateSegsFolder="${WS}/fuse_seg_temp/${SceneNum}"
    PanopticSegsFolder="${WS}/sem_seg_temp/${SceneNum}/cropformer"
    GeometricSegsFolder="${WS}/geo_seg_temp/${SceneNum}"

    python scripts/panoptic_mapping_.py \
        --dataset scannet_nyu --task Nyu40 \
        --scene_num "${SceneNum}" \
        --data_folder "${DataFolder}" \
        --result_folder "${ResultFolder}" \
        --data_association 2 \
        --inst_association 4 \
        --seg_graph_confidence 3 \
        --use_temp_results --save_temp_results --intermediate_seg_folder "${IntermediateSegsFolder}" \
        --temp_panoptics_folder "${PanopticSegsFolder}" \
        --use_temp_geometrics --save_temp_geometrics --temp_geometrics_folder "${GeometricSegsFolder}" \
        --num_threads 10 --log "ovimap" \
        --perception_python "${PERCEPTION_PYTHON}" \
        --perception_worker "${PERCEPTION_WORKER}"

    echo "<<< Finished scene: ${SceneNum}"
done
# --use_temp_geometrics \
# --skip_feature_extraction \
echo ""
echo "All scenes processed."
