#include "consistent_mapping/consistent_gsm_mapper.h"
#include <global_segment_map/common.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

using namespace voxblox;
using namespace voxblox::voxblox_gsm;

ConsistentGSMMapper::ConsistentGSMMapper(const std::string &configFile)
{
    // read in settings
    settings_ptr_ = std::make_shared<SemanticMappingSettings>(configFile);

    // map configuration
    map_config_.voxel_size = settings_ptr_->getVoxelSize();
    map_config_.inst_association = settings_ptr_->getInstAssociateMode();
    map_config_.voxels_per_side = settings_ptr_->getVoxelBlockLen();
    map_config_.use_inst_label_connect = settings_ptr_->getUseInstLabelConnect();
    map_config_.connection_ratio_th = settings_ptr_->getSegGraphConnectRatioTH();
    map_.reset(new LabelTsdfMap(map_config_));

    // map integrator configuration
    // TSDF
    tsdf_integrator_config_.voxel_carving_enabled =
        settings_ptr_->getUseVoxelCarving();
    tsdf_integrator_config_.allow_clear = settings_ptr_->getAllowClear();
    tsdf_integrator_config_.max_ray_length_m = settings_ptr_->getMaxRayLen();
    tsdf_integrator_config_.min_ray_length_m = settings_ptr_->getMinRayLen();
    tsdf_integrator_config_.default_truncation_distance =
        map_config_.voxel_size * settings_ptr_->getTruncateDistFactor();
    tsdf_integrator_config_.enable_anti_grazing = settings_ptr_->getUseAntiGrazing();
    // label
    label_tsdf_integrator_config_.merging_min_overlap_ratio =
        settings_ptr_->getMergingMinOverlapRation();
    label_tsdf_integrator_config_.merging_min_frame_count =
        settings_ptr_->getMergingMinFrame();
    label_tsdf_integrator_config_.enable_semantic_instance_segmentation =
        settings_ptr_->getUseSemInstSegmentation();
    // Task
    std::string class_task = settings_ptr_->getTask();
    if (class_task.compare("coco80") == 0)
    {
        label_tsdf_mesh_config_.class_task = SemanticColorMap::ClassTask ::kCoco80;
        BackgroundSemLabel = 0u;
    }
    else if (class_task.compare("nyu13") == 0)
    {
        label_tsdf_mesh_config_.class_task = SemanticColorMap::ClassTask ::kNyu13;
    }
    else if (class_task.compare("Nyu40") == 0)
    {
        label_tsdf_mesh_config_.class_task = SemanticColorMap::ClassTask ::Nyu40;
        BackgroundSemLabel = 0u;
    }
    else if (class_task.compare("CoCoPano") == 0)
    {
        BackgroundSemLabel = 80u;
        label_tsdf_mesh_config_.class_task = SemanticColorMap::ClassTask ::kCocoPano;
    }
    else
    {

        label_tsdf_mesh_config_.class_task = SemanticColorMap::ClassTask::kCoco80;
    }
    // map integrator
    integrator_.reset(new LabelTsdfConfidenceIntegrator(
        tsdf_integrator_config_,
        label_tsdf_integrator_config_,
        map_.get(),
        settings_ptr_->getUseGeoConfidence(),
        settings_ptr_->getUseLabelConfidence(),
        settings_ptr_->getInstAssociateMode(),
        settings_ptr_->getDataAssociateMode(),
        settings_ptr_->getSegGraphMode()));

    // mesh layer and integrator configuration.
    mesh_label_layer_.reset(new MeshLayer(map_->block_size()));
    mesh_semantic_layer_.reset(new MeshLayer(map_->block_size()));
    mesh_instance_layer_.reset(new MeshLayer(map_->block_size()));

    label_tsdf_mesh_config_.color_scheme =
        MeshLabelIntegrator::ColorScheme::kLabel;
    mesh_label_integrator_.reset(new MeshLabelIntegrator(
        mesh_config_, label_tsdf_mesh_config_, map_.get(),
        mesh_label_layer_.get(), &need_full_remesh_));
    label_tsdf_mesh_config_.color_scheme =
        MeshLabelIntegrator::ColorScheme::kSemantic;
    mesh_semantic_integrator_.reset(new MeshLabelIntegrator(
        mesh_config_, label_tsdf_mesh_config_, map_.get(),
        mesh_semantic_layer_.get(), &need_full_remesh_));
    label_tsdf_mesh_config_.color_scheme =
        MeshLabelIntegrator::ColorScheme::kInstance;
    mesh_instance_integrator_.reset(new MeshLabelIntegrator(
        mesh_config_, label_tsdf_mesh_config_, map_.get(),
        mesh_instance_layer_.get(), &need_full_remesh_));

    std::vector<std::shared_ptr<MeshLayer>> mesh_layers;
    mesh_layers.push_back(mesh_instance_layer_);
    mesh_layers.push_back(mesh_label_layer_);
    mesh_layers.push_back(mesh_semantic_layer_);
    mesh_layer_updated_ = false;

    // Log settings
    FLAGS_alsologtostderr = false;
    std::string log_folder = settings_ptr_->getResultFolder() + "/log";
    FLAGS_log_dir = log_folder;
    std::string command = "mkdir -p " + FLAGS_log_dir;
    int res = system(command.c_str());
    google::InitGoogleLogging("gms_py");

    LOG(INFO) << "  BackgroundSemLabel: " 
              << int(BackgroundSemLabel);
    LOG(INFO) << "  use geometric confidence: "
              << settings_ptr_->getUseGeoConfidence();
    LOG(INFO) << "  use label confidence: "
              << settings_ptr_->getUseLabelConfidence();
    LOG(INFO) << "  inst association: " 
              << integrator_->semantic_instance_label_fusion_ptr_->inst_association_;
    LOG(INFO) << "  data association: " 
              << integrator_->data_association_;
    LOG(INFO) << "  seg_graph_confidence: " 
              << integrator_->seg_graph_confidence_;
    int inst_association = settings_ptr_->getInstAssociateMode();
    if (inst_association == 3 || inst_association == 4 || inst_association == 6 || inst_association == 7)
    {
        integrator_->semantic_instance_label_fusion_ptr_->initSegGraph();
        LOG(INFO) << "  use_inst_label_connect: " 
                  << integrator_->semantic_instance_label_fusion_ptr_->use_inst_label_connect_;
        LOG(INFO) << "  connection_ratio_th_: " 
                  << integrator_->semantic_instance_label_fusion_ptr_->connection_ratio_th_;
    }
}

bool ConsistentGSMMapper::integrateSegments(
    std::vector<SegmentConfidence *> &segments_ptrs)
{
    // move pointers to in-class vectors
    segments_to_integrate_.clear();
    segments_to_integrate_.reserve(segments_ptrs.size());
    for (SegmentConfidence *seg_ptr : segments_ptrs)
    {
        segments_to_integrate_.emplace_back(seg_ptr);
    }
    segments_ptrs.clear();
    // compute segments-superpoints association candidates
    for (Segment *&segment_ptr : segments_to_integrate_)
    {
        integrator_->computeSegmentLabelCandidatesConfidence(
            segment_ptr, &segment_label_candidates, &segment_merge_candidates_);
    }
    // decide super-point labels for segments
    if (settings_ptr_->getUseLabelPropagation())
    {
        integrator_->decideLabelPointCloudsConfidence(&segments_to_integrate_,
                                                      &segment_label_candidates, &segment_merge_candidates_);
    }
    // poses of current segments
    Transformation T_G_C = segments_to_integrate_.at(0)->T_G_C_;
    Transformation T_Gicp_C = T_G_C;

    // integrate segments into map
    {
        std::lock_guard<std::mutex> label_tsdf_layers_lock(label_tsdf_layers_mutex_);
        for (Segment *segment : segments_to_integrate_)
        {
            CHECK_NOTNULL(segment);
            segment->T_G_C_ = T_Gicp_C;
            integrator_->integratePointCloudConfidence(
                segment->T_G_C_, segment->points_C_,
                dynamic_cast<SegmentConfidence *>(segment)->geometry_confidence_,
                dynamic_cast<SegmentConfidence *>(segment)->seg_label_confidence_,
                segment->colors_, segment->label_,
                kIsFreespacePointcloud_);
        }
        // merge super-points if necessary
        if (settings_ptr_->getDataAssociateMode() != 0) {
            integrator_->mergeLabelConfidence(&merges_to_publish_);
        }
        else {
            integrator_->mergeLabels(&merges_to_publish_);
        }

        integrator_->getLabelsToPublish(&segment_labels_to_publish_);
    }
    return true;
}
void ConsistentGSMMapper::clearFrameSegsCache()
{
    // clear cached memory
    segment_merge_candidates_.clear();
    segment_label_candidates.clear();
    for (Segment *segment : segments_to_integrate_) {
        delete segment;
    }
    segments_to_integrate_.clear();
    merges_to_publish_.clear();
    segment_labels_to_publish_.clear();
}

bool ConsistentGSMMapper::generateMesh(
    std::string mesh_file_folder,
    std::string frame_num)
{
    bool clear_mesh = true; // default

    std::lock_guard<std::mutex> mesh_layer_lock(mesh_layer_mutex_);
    std::lock_guard<std::mutex> label_tsdf_layers_lock(
        label_tsdf_layers_mutex_);

    bool only_mesh_updated_blocks = false;
    constexpr bool clear_updated_flag = true;
    // mesh_merged_integrator_->generateMesh(only_mesh_updated_blocks,
    //                                       clear_updated_flag);
    mesh_label_integrator_->generateMesh(only_mesh_updated_blocks,
                                         clear_updated_flag);
    mesh_semantic_integrator_->generateMesh(only_mesh_updated_blocks,
                                            clear_updated_flag);
    mesh_instance_integrator_->generateMesh(only_mesh_updated_blocks,
                                            clear_updated_flag);
    // mesh_confidence_integrator_->generateMesh(only_mesh_updated_blocks,
    //                                     clear_updated_flag);
    mesh_layer_updated_ = true;

    // bool success = outputMeshLayerAsPly(mesh_file_folder + "/confidence_mesh_"+frame_num+".ply", false,
    //                                     *mesh_confidence_layer_);
    bool success = outputMeshLayerAsPly(mesh_file_folder + "/label_mesh_" + frame_num + ".ply", false,
                                        *mesh_label_layer_);
    success &= outputMeshLayerAsPly(mesh_file_folder + "/semantic_mesh_" + frame_num + ".ply", false,
                                    *mesh_semantic_layer_);
    success &= outputMeshLayerAsPly(mesh_file_folder + "/instance_mesh_" + frame_num + ".ply", false,
                                    *mesh_instance_layer_);
    
    if (success)
        LOG(INFO) << "Output file as PLY: " << mesh_file_folder.c_str();
    else
        LOG(INFO) << "Failed to output mesh as PLY: " << mesh_file_folder.c_str();

    return success;
}

inline SemanticLabel ConsistentGSMMapper::quarySemanticPoint(
    Point &query_point)
{
    LabelVoxel *label_voxel_ptr =
        integrator_->label_layer_->getVoxelPtrByCoordinates(query_point);
    if (label_voxel_ptr != nullptr)
    {
        return integrator_->semantic_instance_label_fusion_ptr_->getSemanticLabel(
            label_voxel_ptr->label);
    }
    else
        return BackgroundSemLabel;
}

void ConsistentGSMMapper::quarySemanticPointCloud(
    std::vector<Point> query_pcl,
    std::vector<SemanticLabel> &semantics_list)
{
    semantics_list.clear();
    semantics_list.reserve(query_pcl.size());
    for (Point &query_point : query_pcl)
    {
        semantics_list.push_back(quarySemanticPoint(query_point));
    }
}

Label ConsistentGSMMapper::quarySuperpointID(Point query_point)
{
    LabelVoxel *label_voxel_ptr =
        integrator_->label_layer_->getVoxelPtrByCoordinates(query_point);
    if (label_voxel_ptr != nullptr)
    {
        return label_voxel_ptr->label;
    }
    else
        return BackgroundLabel;
}