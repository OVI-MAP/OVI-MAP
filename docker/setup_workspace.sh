#!/bin/bash
# =============================================================================
# First-time ROS workspace setup (run INSIDE the container).
# =============================================================================
# Usage:
#   bash docker/setup_workspace.sh              # run inside container
# =============================================================================
set -e

# Ensure we're in a conda env with ROS sourced
if [ -z "$ROS_DISTRO" ]; then
    echo "ERROR: ROS not sourced. Run this inside the Docker container."
    exit 1
fi

echo "=== Setting up ROS workspace ==="

cd /workspace/mapping_ros_ws

# Initialize catkin workspace
echo "--- catkin init ---"
catkin init
catkin config --extend /opt/ros/noetic --merge-devel
catkin config --cmake-args -DCMAKE_CXX_STANDARD=14 -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3

# Fetch third-party source dependencies
echo "--- wstool init & update ---"
wstool init src
cd src
wstool merge -t . consistent_panoptic_mapping/consistent_panoptic_mapping.rosinstall
wstool update
cd ..

echo ""
echo "=== Building ROS packages ==="
catkin build consistent_gsm depth_segmentation_py

