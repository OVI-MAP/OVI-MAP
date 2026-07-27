#!/bin/bash
# =============================================================================
# OVI-MAP Docker entrypoint
# Sources ROS + conda env, then executes the user command.
# =============================================================================
set -e

# Source ROS Noetic
if [ -f /opt/ros/noetic/setup.bash ]; then
    source /opt/ros/noetic/setup.bash
fi

# Source the compiled ROS workspace (if built)
if [ -f /workspace/mapping_ros_ws/devel/setup.bash ]; then
    source /workspace/mapping_ros_ws/devel/setup.bash
fi

# Activate the reconstruction conda environment
if [ -d /opt/conda/envs/ovimap-py38 ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate ovimap-py38
fi

exec "$@"
