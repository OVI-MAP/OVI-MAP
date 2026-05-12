#ifndef GLOBAL_SEGMENT_MAP_PY_COMMON_H_
#define GLOBAL_SEGMENT_MAP_PY_COMMON_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "cvnp/cvnp.h"

#include <global_segment_map_node/controller.h>
#include <global_segment_map_node/conversions.h>
#include <global_segment_map/segment.h>
#include <geometry_msgs/TransformStamped.h>
#include <global_segment_map/label_voxel.h>
#include <global_segment_map/utils/file_utils.h>
#include <global_segment_map/utils/map_utils.h>

#include <cmath>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <glog/logging.h>
#include <minkindr_conversions/kindr_tf.h>
#include <voxblox/alignment/icp.h>
#include <voxblox/core/common.h>
#include <voxblox/io/sdf_ply.h>
#include <voxblox_ros/mesh_vis.h>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include<opencv2/opencv.hpp>

using namespace voxblox;
using namespace voxblox::voxblox_gsm;



#endif  // GLOBAL_SEGMENT_MAP_PY_COMMON_H_
