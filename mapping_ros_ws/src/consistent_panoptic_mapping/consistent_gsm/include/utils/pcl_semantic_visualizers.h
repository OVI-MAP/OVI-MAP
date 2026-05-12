#ifndef GLOBAL_SEGMENT_MAP_PY_UTILS_PCL_VISUALIZER_H_
#define GLOBAL_SEGMENT_MAP_PY_UTILS_PCL_VISUALIZER_H_
#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>
#include <glog/logging.h>
#include <omp.h>

#include "global_segment_map/label_tsdf_map.h"
#include "global_segment_map/label_voxel.h"
#include "global_segment_map/meshing/label_tsdf_mesh_integrator.h"
#include "global_segment_map/meshing/instance_color_map.h"
#include "global_segment_map/meshing/label_color_map.h"
#include "global_segment_map/meshing/semantic_color_map.h"
#include "global_segment_map/semantic_instance_label_fusion.h"

#include <voxblox/core/common.h>
#include <voxblox/core/layer.h>
#include <voxblox/core/voxel.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace voxblox {

struct PCLSemVisualizerConfig {
EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	int thread_num = 10;
	SemanticColorMap::ClassTask class_task = SemanticColorMap::ClassTask::kCoco80;

};

class PCLSemVisualizer{
public:
	PCLSemVisualizer(const PCLSemVisualizerConfig& visualizer_configure, LabelTsdfMap* map,
		std::vector<double>& camera_position, std::vector<double>& clip_distances);
	void createPointcloudFromTsdfLayer();
	void updatePointClouds();
	void visualizePointClouds();

	PCLSemVisualizerConfig pcl_sem_visualizer_configure_;

	// map layers and colors
	const Layer<LabelVoxel>* label_layer_const_ptr_;
	const Layer<TsdfVoxel>* sdf_layer_const_ptr_;
	const SemanticInstanceLabelFusion* semantic_instance_label_fusion_ptr_;
	InstanceColorMap instance_color_map_;
	LabelColorMap label_color_map_;
	SemanticColorMap semantic_color_map_;

	// colored point clouds
	int num_pcl_visual_ = 3;
	std::mutex pcls_mutex_;
	std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pointclouds_;
	
	// visualizers'things
	std::atomic_bool pcls_updated_ = ATOMIC_VAR_INIT(false);;
	std::vector<double> camera_position_;
	std::vector<double> clip_distances_;
	int update_interval_ms_ = 1000;
};


}

#endif