#include "utils/pcl_semantic_visualizers.h"

namespace voxblox {

PCLSemVisualizer::PCLSemVisualizer(const PCLSemVisualizerConfig& visualizer_configure, LabelTsdfMap* map,
	std::vector<double>& camera_position, std::vector<double>& clip_distances):
	pcl_sem_visualizer_configure_(visualizer_configure),
	sdf_layer_const_ptr_(map->getTsdfLayerPtr()),
	label_layer_const_ptr_(map->getLabelLayerPtr()),
	semantic_instance_label_fusion_ptr_(map->getSemanticInstanceLabelFusionPtr()),
	label_color_map_(),
	instance_color_map_(),
	semantic_color_map_(SemanticColorMap::create(visualizer_configure.class_task)),
	camera_position_(camera_position),
	clip_distances_(clip_distances)
{
	pointclouds_.clear();
	pointclouds_.reserve(num_pcl_visual_);
	// instance pcl; semantic pcl; label pcl
	for(int pcl_i=0; pcl_i<num_pcl_visual_; pcl_i++)
	{
		pointclouds_.push_back(pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>));
	}
	omp_set_num_threads(visualizer_configure.thread_num);

}

void PCLSemVisualizer::createPointcloudFromTsdfLayer()
{
	BlockIndexList tsdf_blocks_indexs;
	sdf_layer_const_ptr_->getAllAllocatedBlocks(&tsdf_blocks_indexs);
	// Cache layer settings.
	size_t vps = sdf_layer_const_ptr_->voxels_per_side();
	size_t num_voxels_per_block = vps * vps * vps;
	constexpr float kMinWeight = 0;
	int max_num_points = tsdf_blocks_indexs.size()*num_voxels_per_block;

	// 0 for instance; 1 for label; 2 for semantics
	std::lock_guard<std::mutex> lock(pcls_mutex_);
	for(auto pcl_ptr:pointclouds_)
	{
		pcl_ptr->clear();
		pcl_ptr->reserve(max_num_points);
	}

	#pragma omp parallel
	{
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_thread_inst_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_thread_label_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_thread_seman_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);

		int nums_thread = omp_get_num_threads();
		// allocate pcls for each thread
		int jobs_per_thread = int(max_num_points*1.1/nums_thread);
		pcl_thread_inst_ptr->reserve(jobs_per_thread);
		pcl_thread_label_ptr->reserve(jobs_per_thread);
		pcl_thread_seman_ptr->reserve(jobs_per_thread);

		// iterate all blocks and get colored pointcloud
		#pragma omp for
		for(const BlockIndex& block_index : tsdf_blocks_indexs)
		{
			const Block<TsdfVoxel>& tsdf_block = sdf_layer_const_ptr_->getBlockByIndex(block_index);
			const Block<LabelVoxel>& label_block = label_layer_const_ptr_->getBlockByIndex(block_index);
			for(size_t linear_index = 0; linear_index < num_voxels_per_block; 
				++linear_index)
			{
				
				const TsdfVoxel& tsdf_voxel = tsdf_block.getVoxelByLinearIndex(linear_index);
				if (tsdf_voxel.weight > kMinWeight)
				{
					Point coord = tsdf_block.computeCoordinatesFromLinearIndex(linear_index);
					const LabelVoxel& label_voxel = label_block.getVoxelByLinearIndex(linear_index);
					// get color
					Label seg_label = label_voxel.label;
					InstanceLabel inst_label = semantic_instance_label_fusion_ptr_->getInstanceLabel(seg_label);
					SemanticLabel seman_label = semantic_instance_label_fusion_ptr_->getSemanticLabel(seg_label);
					Color color_inst, color_label, color_seman;
					instance_color_map_.getColor(inst_label,&color_inst);
					label_color_map_.getColor(seg_label, &color_label);
					semantic_color_map_.getColor(seman_label, &color_seman);
					// create colored point
					float loc_x = coord.x();
					float loc_y = coord.y();
					float loc_z = coord.z();
					// instancepoint
					pcl_thread_inst_ptr->emplace_back(
						loc_x, loc_y, loc_z, color_inst.r, color_inst.g, color_inst.b);
					pcl_thread_label_ptr->emplace_back(
						loc_x, loc_y, loc_z, color_label.r, color_label.g, color_label.b);
					pcl_thread_seman_ptr->emplace_back(
						loc_x, loc_y, loc_z, color_seman.r, color_seman.g, color_seman.b);
				}
			}
		}

		// merge all the thread to the merged pcls
		#pragma omp critical
		{
			*pointclouds_[0] += *pcl_thread_inst_ptr;
			*pointclouds_[1] += *pcl_thread_label_ptr;
			*pointclouds_[2] += *pcl_thread_seman_ptr;
		}
	}	
}

void PCLSemVisualizer::updatePointClouds()
{
	createPointcloudFromTsdfLayer();
	pcls_updated_.store(true) ;
}

void PCLSemVisualizer::visualizePointClouds()
{
	std::vector<std::string> visual_names = {"instance", "label", "semantic"};
	// initialize PCL visualizers
	uint8_t n_visualizers = num_pcl_visual_;
	std::vector<std::shared_ptr<pcl::visualization::PCLVisualizer>> pcl_visualizers;
	pcl_visualizers.reserve(n_visualizers);
	for (int index = 0; index < n_visualizers; ++index)
	{
		// PCLVisualizer class can NOT be used across multiple threads, thus need to
    	// create instances of it in the same thread they will be used in.
		std::shared_ptr<pcl::visualization::PCLVisualizer> visualizer =
        	std::make_shared<pcl::visualization::PCLVisualizer>();
		std::string name = "PointClouds_" + visual_names[index];
		visualizer->setWindowName(name.c_str());
		visualizer->setBackgroundColor(255, 255, 255);
		visualizer->initCameraParameters();

		if (camera_position_.size()) {
      		visualizer->setCameraPosition(
				camera_position_[0], camera_position_[1], camera_position_[2],
				camera_position_[3], camera_position_[4], camera_position_[5],
				camera_position_[6], camera_position_[7], camera_position_[8]);
    	}
		if (clip_distances_.size())
		{
			visualizer->setCameraClipDistances(clip_distances_[0], clip_distances_[1]);
		}
		pcl_visualizers.push_back(visualizer);
	}
	while(true)
	{
		for (int index = 0; index < n_visualizers; ++index) 
		{
			pcl_visualizers[index]->spinOnce(update_interval_ms_);
		}
		if(pcls_updated_.load())
		{
			std::lock_guard<std::mutex> lock(pcls_mutex_);
			for (int index = 0; index < n_visualizers; ++index) 
			{
				pcl_visualizers[index]->removePointCloud("pointclouds");
				if (!pcl_visualizers[index]->updatePointCloud(pointclouds_[index],"pointclouds"))
				{
					pcl_visualizers[index]->addPointCloud(pointclouds_[index],"pointclouds");
					pcl_visualizers[index]->setPointCloudRenderingProperties
						(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "pointclouds");
				}
          			
           
			}
			pcls_updated_.store(false);
		}
		// else
		// {
		// 	std::this_thread::sleep_for(std::chrono::milliseconds(10));
		// }
	}
}

} // namespace voxblox