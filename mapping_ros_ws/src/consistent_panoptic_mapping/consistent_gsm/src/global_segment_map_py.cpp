#include "consistent_mapping/global_segment_map_py.h"
#include <global_segment_map/common.h>
#include "voxblox/core/color.h"

#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <chrono>  // chrono::system_clock
#include <ctime>   // localtime
#include <iomanip> // put_time
#include <fstream>

using namespace voxblox;
using namespace voxblox::voxblox_gsm;

GlobalSegmentMap_py::GlobalSegmentMap_py(
    std::string log_file, 
    std::string task,
    bool use_geo_confidence, 
    bool use_label_confidence,
    int inst_association, 
    int data_association,
    int num_threads, 
    bool debug, 
    int seg_graph_confidence, 
    bool use_inst_label_connect,
    float connection_ratio_th, 
    float cos_sim_threshold
) : integrated_frames_count_(0u),
    task_(task),
    use_geo_confidence_(use_geo_confidence),
    use_label_confidence_(use_label_confidence),
    inst_association_(inst_association),
    data_association_(data_association),
    enable_semantic_instance_segmentation_(true),
    use_label_propagation_(true),
    log_file_(log_file),
    debug_(debug),
    seg_graph_confidence_(seg_graph_confidence),
    use_inst_label_connect_(use_inst_label_connect),
    connection_ratio_th_(connection_ratio_th), 
    cos_sim_threshold_(cos_sim_threshold)
{
    FLAGS_alsologtostderr = true;
    FLAGS_log_dir = log_file_;
    google::InitGoogleLogging("gms_py");

    systemMem_ = getValue();
    LOG(INFO) << "  Memory used by system at the beginning: " << systemMem_ << " KB";

    if (num_threads > 0 && num_threads <= tsdf_integrator_config_.integrator_threads) {
        tsdf_integrator_config_.integrator_threads = num_threads;
        mesh_config_.integrator_threads = num_threads;
        visualizer_config_.thread_num = num_threads;
    }

    size_t integrator_threads = tsdf_integrator_config_.integrator_threads;
    LOG(INFO) << "integrator_threads: " << integrator_threads;
    map_config_.voxel_size = 0.01; // TODO yaml
    map_config_.inst_association = inst_association_;
    map_config_.voxels_per_side = 8u; // TODO yaml
    map_config_.use_inst_label_connect = use_inst_label_connect_;
    map_config_.connection_ratio_th = connection_ratio_th_;
    map_.reset(new LabelTsdfMap(map_config_));

    // Determine TSDF Label integrator parameters.
    // TSDF
    tsdf_integrator_config_.voxel_carving_enabled = false;
    tsdf_integrator_config_.allow_clear = true;
    FloatingPoint truncation_distance_factor = 5.0f; // TODO yaml
    tsdf_integrator_config_.max_ray_length_m = 10.0f; // TODO yaml
    tsdf_integrator_config_.min_ray_length_m = 0.1f; // TODO yaml
    tsdf_integrator_config_.default_truncation_distance = map_config_.voxel_size * truncation_distance_factor;
    std::string method("merged");
    tsdf_integrator_config_.enable_anti_grazing = false; // TODO yaml
    // Label
    if (use_label_confidence_)
        label_tsdf_integrator_config_.merging_min_overlap_ratio = 0.1; // TODO yaml
    else
        label_tsdf_integrator_config_.merging_min_overlap_ratio = 0.15;         // TODO yaml
    label_tsdf_integrator_config_.merging_min_frame_count = 10;                 // TODO yaml
    label_tsdf_integrator_config_.enable_semantic_instance_segmentation = true; // TODO yaml
    label_tsdf_integrator_config_.label_register_min_overlap_ratio = 0.0;
    if (data_association_ == 4)
        label_tsdf_integrator_config_.enable_pairwise_confidence_merging = false;

    // Task
    std::string class_task = task_;
    if (class_task.compare("coco80") == 0) {
        label_tsdf_mesh_config_.class_task = SemanticColorMap::ClassTask ::kCoco80;
        BackgroundSemLabel = 0u;
    }
    else if (class_task.compare("nyu13") == 0) {
        label_tsdf_mesh_config_.class_task = SemanticColorMap::ClassTask ::kNyu13;
    }
    else if (class_task.compare("Nyu40") == 0) {
        label_tsdf_mesh_config_.class_task = SemanticColorMap::ClassTask ::Nyu40;
        BackgroundSemLabel = 0u;
    }
    else if (class_task.compare("cocoPano") == 0) {
        BackgroundSemLabel = 80;
        label_tsdf_mesh_config_.class_task = SemanticColorMap::ClassTask ::kCocoPano;
    }
    else {
        label_tsdf_mesh_config_.class_task = SemanticColorMap::ClassTask::kCoco80;
    }

    integrator_.reset(new LabelTsdfConfidenceIntegrator(
        tsdf_integrator_config_, 
        label_tsdf_integrator_config_, 
        map_.get(), 
        use_geo_confidence_, 
        use_label_confidence_, 
        inst_association_,
        data_association_, 
        seg_graph_confidence_
    ));
    integrator_->InitMetaSemantics(class_task);

    // mesh layer and integrator settings.
    // mesh_merged_layer_.reset(new MeshLayer(map_->block_size()));
    mesh_label_layer_.reset(new MeshLayer(map_->block_size()));
    mesh_semantic_layer_.reset(new MeshLayer(map_->block_size()));
    mesh_instance_layer_.reset(new MeshLayer(map_->block_size()));
    mesh_confidence_layer_.reset(new MeshLayer(map_->block_size()));

    // label_tsdf_mesh_config_.color_scheme = MeshLabelIntegrator::ColorScheme::kMerged;
    // mesh_merged_integrator_.reset(new MeshLabelIntegrator(
    //     mesh_config_, label_tsdf_mesh_config_, 
    //     map_.get(), mesh_merged_layer_.get(), 
    //     &need_full_remesh_
    // ));

    label_tsdf_mesh_config_.color_scheme = MeshLabelIntegrator::ColorScheme::kLabel;
    mesh_label_integrator_.reset(new MeshLabelIntegrator(
        mesh_config_, label_tsdf_mesh_config_, 
        map_.get(), mesh_label_layer_.get(), 
        &need_full_remesh_
    ));

    // to vis the open-set semantic labels, set the color scheme to kOpenSemantic
    label_tsdf_mesh_config_.color_scheme = MeshLabelIntegrator::ColorScheme::kSemantic;
    mesh_semantic_integrator_.reset(new MeshLabelIntegrator(
        mesh_config_, label_tsdf_mesh_config_, 
        map_.get(), mesh_semantic_layer_.get(), 
        &need_full_remesh_
    ));

    label_tsdf_mesh_config_.color_scheme = MeshLabelIntegrator::ColorScheme::kInstance;
    mesh_instance_integrator_.reset(new MeshLabelIntegrator(
        mesh_config_, label_tsdf_mesh_config_, 
        map_.get(), mesh_instance_layer_.get(), 
        &need_full_remesh_
    ));

    label_tsdf_mesh_config_.color_scheme = MeshLabelIntegrator::ColorScheme::kLabelConfidence;
    mesh_confidence_integrator_.reset(new MeshLabelIntegrator(
        mesh_config_, label_tsdf_mesh_config_, 
        map_.get(), mesh_confidence_layer_.get(), 
        &need_full_remesh_
    ));

    // meshes
    std::vector<std::shared_ptr<MeshLayer>> mesh_layers;
    mesh_layers.push_back(mesh_instance_layer_);
    mesh_layers.push_back(mesh_label_layer_);
    mesh_layers.push_back(mesh_semantic_layer_);
    // mesh_layers.push_back(mesh_confidence_layer_);
    // mesh_layers.push_back(mesh_merged_layer_);
    mesh_layer_updated_ = false;
    LOG(INFO) << "  BackgroundSemLabel: " << int(BackgroundSemLabel);
    LOG(INFO) << "  use geometric confidence: " << use_geo_confidence_;
    LOG(INFO) << "  use label confidence: " << use_label_confidence_;
    LOG(INFO) << "  inst association: " << integrator_->semantic_instance_label_fusion_ptr_->inst_association_;
    LOG(INFO) << "  data association: " << integrator_->data_association_;
    LOG(INFO) << "  seg_graph_confidence: " << integrator_->seg_graph_confidence_;
    if (inst_association_ == 3 || inst_association_ == 4 || inst_association == 6 || inst_association == 7) {
        integrator_->semantic_instance_label_fusion_ptr_->initSegGraph();
        LOG(INFO) << "  use_inst_label_connect: " << integrator_->semantic_instance_label_fusion_ptr_->use_inst_label_connect_;
        LOG(INFO) << "  connection_ratio_th_: " << integrator_->semantic_instance_label_fusion_ptr_->connection_ratio_th_;
    }

    // open-set feature 
    highest_sem_label_ = 0u;

    // visualizer
    std::vector<double> camera_position = {
        4, 4, 8,
        4, 4, 0,
        0.64278761, 0.76604444, 0};                      // TODO yaml
    std::vector<double> clip_distances = {0.1, 8.86051}; // TODO yaml
    // std::vector<double> clip_distances = {1.79126, 8.86051}; // TODO yaml
    double update_mesh_every_n_sec = 0.0;

    bool save_visualizer_frames = true;
    visualizer_mesh_ = std::make_shared<Visualizer>(
        mesh_layers, 
        &mesh_layer_updated_, 
        &mesh_layer_mutex_,
        camera_position, 
        clip_distances, 
        save_visualizer_frames, 
        log_file_
    );
    visualizer_pcl_ = std::make_shared<PCLSemVisualizer>(
        visualizer_config_, 
        map_.get(), 
        camera_position, 
        clip_distances
    );
    
    if (debug_) {
        viz_mesh_thread_ = std::thread(&Visualizer::visualizeMesh, visualizer_mesh_);
        // viz_pcl_thread_ = std::thread(&PCLSemVisualizer::visualizePointClouds, visualizer_pcl_);
    }
    int curMem = getValue()-systemMem_;
    LOG(INFO) << "  Memory usage at init: " << curMem << " KB |" << curMem / 1024.0 << " MB";
}


// new function for open-set segments
void GlobalSegmentMap_py::insertSegmentsOpen(
    pybind11::array &points, 
    // pybind11::array& geometry_confidence, //float
    pybind11::array &b_box, 
    InstanceLabel instance_label, 
    SemanticLabel semantic_label, 
    pybind11::array_t<float> &sem_feature, 
    ObjSegConfidence inst_confidence,
    ObjSegConfidence obj_seg_confidence,
    pybind11::array &T_G_C,
    bool is_thing,
    Label desginated_label)
{
    // LOG(INFO) << "  Memory usage before insertSegments: " << getValue() << " kB";
    cv::Mat T_G_C_mat = cvnp::nparray_to_mat(T_G_C);
    Eigen::Matrix<float, 4, 4> T_G_C_eigen;
    cv::cv2eigen(T_G_C_mat, T_G_C_eigen);
    Transformation T_G_C_voxblox(T_G_C_eigen);

    cv::Mat points_mat = cvnp::nparray_to_mat(points);
    cv::Mat b_box_mat = cvnp::nparray_to_mat(b_box);

    Segment *segment = nullptr;
    segment = new SegmentConfidence(
        &points_mat, &b_box_mat, instance_label, semantic_label, 
        T_G_C_voxblox, inst_confidence, obj_seg_confidence, 
        is_thing, desginated_label
    );
    // object of SegmentConfidence, where has a object of Segment (from voxblox)
    segments_to_integrate_.push_back(segment); 
}


void GlobalSegmentMap_py::insertSegments(
    pybind11::array &points, // float
    // pybind11::array& geometry_confidence, //float
    pybind11::array &b_box,       // float
    InstanceLabel instance_label, // uint16_t
    SemanticLabel semantic_label, // uint8_t
    ObjSegConfidence inst_confidence,
    ObjSegConfidence obj_seg_confidence,
    pybind11::array &T_G_C,
    bool is_thing,
    Label desginated_label)
{
    cv::Mat T_G_C_mat = cvnp::nparray_to_mat(T_G_C);
    Eigen::Matrix<float, 4, 4> T_G_C_eigen;
    cv::cv2eigen(T_G_C_mat, T_G_C_eigen);
    Transformation T_G_C_voxblox(T_G_C_eigen);

    cv::Mat points_mat = cvnp::nparray_to_mat(points);
    // cv::Mat geometry_confidence_mat = cvnp::nparray_to_mat(geometry_confidence);
    cv::Mat b_box_mat = cvnp::nparray_to_mat(b_box);

    Segment *segment = nullptr;
    segment = new SegmentConfidence(
        &points_mat, &b_box_mat,
        instance_label, semantic_label, 
        T_G_C_voxblox, inst_confidence, obj_seg_confidence, 
        is_thing, desginated_label
    );
    segments_to_integrate_.push_back(segment);
}

void GlobalSegmentMap_py::insertSegmentsPoseConfidence(
    pybind11::array &points, // float
    // pybind11::array& colors, // rgba uint8_t
    // pybind11::array& geometry_confidence, //float
    pybind11::array &b_box,       // float
    InstanceLabel instance_label, // uint16_t
    SemanticLabel semantic_label, // uint8_t
    ObjSegConfidence inst_confidence,
    ObjSegConfidence obj_seg_confidence,
    pybind11::array &T_G_C,
    float pose_confidence,
    bool is_thing,
    Label desginated_label)
{
    // LOG(INFO) << "  Memory usage before insertSegments: " << getValue() << " kB";
    cv::Mat T_G_C_mat = cvnp::nparray_to_mat(T_G_C);
    Eigen::Matrix<float, 4, 4> T_G_C_eigen;
    cv::cv2eigen(T_G_C_mat, T_G_C_eigen);
    Transformation T_G_C_voxblox(T_G_C_eigen);

    cv::Mat points_mat = cvnp::nparray_to_mat(points);
    // cv::Mat colors_mat = cvnp::nparray_to_mat(colors);
    // cv::Mat geometry_confidence_mat = cvnp::nparray_to_mat(geometry_confidence);
    cv::Mat b_box_mat = cvnp::nparray_to_mat(b_box);
    Segment *segment = nullptr;
    segment = new SegmentConfidence(
        &points_mat, &b_box_mat,
        instance_label, semantic_label, 
        T_G_C_voxblox, pose_confidence,
        inst_confidence, obj_seg_confidence, 
        is_thing, desginated_label
    );
    segments_to_integrate_.push_back(segment);
    // LOG(INFO) << "  Memory usage after insertSegments: " << getValue() << " kB";
}


float GlobalSegmentMap_py::cosineSimilarity(
    const SemanticFeature& a, const SemanticFeature& b ) 
{
    float dot_product = std::inner_product(a.begin(), a.end(), b.begin(), 0.0f);
    float norm_a = std::sqrt(std::inner_product(a.begin(), a.end(), a.begin(), 0.0f));
    float norm_b = std::sqrt(std::inner_product(b.begin(), b.end(), b.begin(), 0.0f));

    return (norm_a > 0 && norm_b > 0) ? (dot_product / (norm_a * norm_b)) : 0.0f;
}



bool GlobalSegmentMap_py::integrateFrame()
{
    bool whether_merge_alias = false;
    LOG(INFO) << "Integrating frame n." << ++integrated_frames_count_;
    int curMem = getValue()-systemMem_;
    // LOG(INFO) << "  Memory usage before integrateFrame: " << curMem << " KB | " << curMem / 1024.0 << " MB";

    auto time_start = std::chrono::system_clock::now();

    if (data_association_ == 5) {
        std::set<Segment *, SegmentConfidence::PtrCompare> labeled_segments;
        // use designated superpoint id directly from instance label
        for (Segment *segment : segments_to_integrate_)
            labeled_segments.insert(segment);
        
        integrator_->updateInstanceConfidence(&labeled_segments);
    }
    else {
        for (Segment *segment : segments_to_integrate_) {
            integrator_->computeSegmentLabelCandidatesConfidence(
                segment, &segment_label_candidates_, &segment_merge_candidates_);
        }
        // auto time_end = std::chrono::system_clock::now();
        // auto duration = std::chrono::duration<double>(time_end - time_start).count();
        // LOG(INFO) << "  computeSegmentLabelCandidatesConfidence cost: " << duration << " seconds";

        // LOG(INFO) << "  Confidence candidate. ";
        // for (auto label_it = segment_label_candidates_.begin();
        //         label_it != segment_label_candidates_.end();++label_it) {
        //         for (auto segment_it = label_it->second.begin();
        //             segment_it != label_it->second.end(); segment_it++) {
        //         LOG(INFO) << "    Label " << label_it->first << " - seg.size "
        //           << (segment_it->first)->points_C_.size() << " - confi " << segment_it->second;
        //     }
        // }

        // true by default
        if (use_label_propagation_) {
            time_start = std::chrono::system_clock::now();
            integrator_->decideLabelPointCloudsConfidence(
                &segments_to_integrate_,
                &segment_label_candidates_,
                &segment_merge_candidates_
            );

            // time_end = std::chrono::system_clock::now();
            // duration = std::chrono::duration<double>(time_end - time_start).count();
            // LOG(INFO) << "  decideLabelPointCloudsConfidence cost: " << duration << " seconds";
        }
    }

    constexpr bool kIsFreespacePointcloud = false;
    Transformation T_G_C = segments_to_integrate_.at(0)->T_G_C_;
    Transformation T_Gicp_C = T_G_C;

    {
        size_t seg_count = 0;
        auto time_start = std::chrono::system_clock::now();

        std::lock_guard<std::mutex> label_tsdf_layers_lock(
            label_tsdf_layers_mutex_);

        for (Segment *segment : segments_to_integrate_) {
            CHECK_NOTNULL(segment);
            segment->T_G_C_ = T_Gicp_C;
            integrator_->integratePointCloudConfidence(
                segment->T_G_C_, segment->points_C_,
                dynamic_cast<SegmentConfidence *>(segment)->geometry_confidence_,
                dynamic_cast<SegmentConfidence *>(segment)->seg_label_confidence_,
                segment->colors_, segment->label_,
                kIsFreespacePointcloud); // TODO for confidence

            // LOG(INFO) << "Integrate segment " << int(segment->label_) << " with confidence "
            //   <<  dynamic_cast<SegmentConfidence*>(segment)->seg_label_confidence_;

            // if(segment->instance_label_!=0) {
            //   LOG_EVERY_N(INFO, 1) << " segment segseg confidence: " <<
            //     dynamic_cast<SegmentConfidence*>(segment)->seg_label_confidence_;
            //   LOG_EVERY_N(INFO, 1) << " segment obj_seg_confidence confidence: " <<
            //     dynamic_cast<SegmentConfidence*>(segment)->obj_seg_confidence_;
            // }
        }

        auto time_end = std::chrono::system_clock::now();
        auto duration = std::chrono::duration<double>(time_end - time_start).count();
        LOG(INFO) << "  integratePointCloudConfidence cost: " << duration << " seconds";

        // LOG_EVERY_N(INFO, 1) << "  Integrated " << segments_to_integrate_.size()
        //     << " pointclouds in " << duration << " secs. ";

        LOG_EVERY_N(INFO, 1) << "  The map contains "
            << map_->getTsdfLayerPtr()->getNumberOfAllocatedBlocks() << " tsdf and "
            << map_->getLabelLayerPtr()->getNumberOfAllocatedBlocks() << " label blocks.";

        if (data_association_ != 0)
            whether_merge_alias = integrator_->mergeLabelConfidence(&merges_to_publish_);
        else
            integrator_->mergeLabels(&merges_to_publish_);

        integrator_->getLabelsToPublish(&segment_labels_to_publish_);
    }
    // LOG(INFO) << " PairWiseConfidence: ";

    // for( LLMapIt label_it = integrator_-> pairwise_confidence_.begin();
    //       label_it!=integrator_-> pairwise_confidence_.end(); label_it++) {
    //   for(LMapIt pair_it=label_it->second.begin(); pair_it!=label_it->second.end(); pair_it++) {
    //       LOG(INFO) << "    Label " <<  int(label_it->first) << "- "<< int(pair_it->first) <<
    //           ": " << pair_it->second;
    //   }
    // }
    
    curMem = getValue()-systemMem_;
    LOG(INFO) << "  Memory usage after integrateFrame: " << curMem << " KB | " << curMem / 1024.0 << " MB";
    
    return whether_merge_alias;
}


void GlobalSegmentMap_py::LogSegmentsLabels()
{
    LOG(INFO) << " LogSegmentsInformation: ";
    for (Segment *segment : segments_to_integrate_)
    {
        Label global_label = segment->label_;
        SemanticLabel seman_label = integrator_->semantic_instance_label_fusion_ptr_->getSemanticLabel(global_label);
        LOG(INFO) << " Label" << int(global_label) << " ; size: " << segment->points_C_.size() << " ; semantic label: " << int(seman_label);
    }
}

void GlobalSegmentMap_py::clearTemporaryMemory()
{
    segment_merge_candidates_.clear();
    segment_label_candidates_.clear();
    for (Segment *segment : segments_to_integrate_)
    {
        delete segment;
    }
    segments_to_integrate_.clear();
    merges_to_publish_.clear();
    segment_labels_to_publish_.clear();
    // int curMem = getValue() - systemMem_;
    // LOG(INFO) << "  Memory usage after clearTemporaryMemory: " << curMem << " kB";
}


bool GlobalSegmentMap_py::generateMesh(
    std::string mesh_file_folder, 
    std::string frame_num, 
    bool save_label_mesh, 
    bool save_sem_mesh, 
    bool save_inst_mesh
)
{
    bool clear_mesh = true; // default

    std::lock_guard<std::mutex> mesh_layer_lock(mesh_layer_mutex_);

    std::lock_guard<std::mutex> label_tsdf_layers_lock(
        label_tsdf_layers_mutex_);

    bool only_mesh_updated_blocks = false;
    constexpr bool clear_updated_flag = true;
    
    if (save_label_mesh)
        mesh_label_integrator_->generateMesh(only_mesh_updated_blocks, clear_updated_flag);
    if (save_sem_mesh)
        mesh_semantic_integrator_->generateMesh(only_mesh_updated_blocks,clear_updated_flag);
    if (save_inst_mesh)
        mesh_instance_integrator_->generateMesh(only_mesh_updated_blocks, clear_updated_flag);
    // mesh_confidence_integrator_->generateMesh(
    //     only_mesh_updated_blocks, clear_updated_flag);
    // mesh_merged_integrator_->generateMesh(
    //     only_mesh_updated_blocks, clear_updated_flag);
      
    mesh_layer_updated_ = true;
    bool success = true;
    
    if (save_label_mesh)
        success &= outputMeshLayerAsPly(
            mesh_file_folder + "/label_mesh_" + frame_num + ".ply",
            false, *mesh_label_layer_
        );
    if (save_sem_mesh)
        success &= outputMeshLayerAsPly(
            mesh_file_folder + "/semantic_mesh_" + frame_num + ".ply", 
            false, *mesh_semantic_layer_
        );
    if (save_inst_mesh)
        success &= outputMeshLayerAsPly(
            mesh_file_folder + "/instance_mesh_" + frame_num + ".ply", 
            false, *mesh_instance_layer_
        );
    // success = outputMeshLayerAsPly(
    //     mesh_file_folder + "/confidence_mesh_"+frame_num+".ply", false, *mesh_confidence_layer_);
    
    if (success)
        LOG(INFO) << "Output file as PLY: " << mesh_file_folder.c_str();
    else
        LOG(INFO) << "Failed to output mesh as PLY.";

    return success;
}

void GlobalSegmentMap_py::updateVisualization()
{
    std::lock_guard<std::mutex> mesh_layer_lock(mesh_layer_mutex_);
    {
        std::lock_guard<std::mutex> label_tsdf_layers_lock(
            label_tsdf_layers_mutex_);

        bool need_full_remesh_ = true;
        bool only_mesh_updated_blocks = true;
        if (need_full_remesh_) {
            only_mesh_updated_blocks = false;
            need_full_remesh_ = false;
        }

        bool clear_updated_flag = false;
        mesh_layer_updated_ |= mesh_label_integrator_->generateMesh(
            only_mesh_updated_blocks, clear_updated_flag);
        mesh_layer_updated_ |= mesh_instance_integrator_->generateMesh(
            only_mesh_updated_blocks, clear_updated_flag);
        mesh_layer_updated_ |= mesh_semantic_integrator_->generateMesh(
            only_mesh_updated_blocks, clear_updated_flag);
        clear_updated_flag = true;

        // mesh_layer_updated_ |= mesh_merged_integrator_->generateMesh(
        //     only_mesh_updated_blocks, clear_updated_flag);
        // mesh_layer_updated_ |= mesh_confidence_integrator_->generateMesh(
        //     only_mesh_updated_blocks, clear_updated_flag);
    }
}

void GlobalSegmentMap_py::updateVisualizationPCL()
{
    visualizer_pcl_->updatePointClouds();
}


/**
 * Log the semantic and intance labels of each labeled piece
 */
void GlobalSegmentMap_py::LogLabelInformation()
{
    integrator_->cleanStaleLabels();
    SemanticInstanceLabelFusion *sem_inst_label_fuse_ptr_ =
        integrator_->semantic_instance_label_fusion_ptr_;
    LOG(INFO) << "Log Label Information: ";

    bool inst_asso_flag = (inst_association_ == 3 || inst_association_ == 4 || 
        inst_association_ == 6 || inst_association_ == 7);

    for (auto label_it = sem_inst_label_fuse_ptr_->label_frames_count_.begin();
             label_it != sem_inst_label_fuse_ptr_->label_frames_count_.end(); label_it++)
    {
        InstanceLabel instance_label = sem_inst_label_fuse_ptr_->getInstanceLabel(label_it->first, 0.1f);
        if (instance_label != BackgroundLabel)
        {
            // SemanticLabel semantic_label = sem_inst_label_fuse_ptr_->getSemanticLabel(label_it->first);
            // LOG(INFO) << "  Label: " << int(label_it->first) << " Sem: " << int(semantic_label) << " Inst: " << std::setfill('0') << std::setw(5) << int(instance_label);
            if (!inst_asso_flag)
                sem_inst_label_fuse_ptr_->logLabelSemanticInstanceCountInfo(label_it->first);
        }
    }
    if (inst_asso_flag) {
        sem_inst_label_fuse_ptr_->logSegGraphInfo(log_file_);
        LogLabelInitialGuess(log_file_);
    }
}

void GlobalSegmentMap_py::LogMeshColors(std::string log_path)
{
    std::set<InstanceLabel> inst_label_set;
    std::set<SemanticLabel> sem_label_set;
    const SemanticInstanceLabelFusion *sem_inst_label_fuse_ptr_ =
        integrator_->semantic_instance_label_fusion_ptr_;


    LOG(INFO) << "LogLabelColor: ";
    Color label_color;
    for (auto label_it = sem_inst_label_fuse_ptr_->label_frames_count_.begin();
        label_it != sem_inst_label_fuse_ptr_->label_frames_count_.end(); label_it++)
    {
        InstanceLabel instance_label = sem_inst_label_fuse_ptr_->getInstanceLabel(label_it->first, 0.1f);
        if (instance_label != BackgroundLabel) {
            if (inst_label_set.find(instance_label) == inst_label_set.end())
                inst_label_set.insert(instance_label);

            Color label_color;
            mesh_label_integrator_->label_color_map_.getColor(label_it->first, &label_color);
            // LOG(INFO) << "  Label: " << int(label_it->first) << " Color: ("
            //     << int(label_color.r) << "," << int(label_color.g) << "," << int(label_color.b) << ")";
        }

        SemanticLabel sem_label = sem_inst_label_fuse_ptr_->getSemanticLabel(label_it->first);
        if (sem_label_set.find(sem_label) == sem_label_set.end())
            sem_label_set.insert(sem_label);
    }

    LOG(INFO) << "LogInstanceColor: ";
    for (auto inst_it = inst_label_set.begin(); 
        inst_it != inst_label_set.end(); inst_it++)
    {
        Color inst_color;
        mesh_instance_integrator_->instance_color_map_.getColor(*inst_it, &inst_color);
        LOG(INFO) << "  Instance: " << int(*inst_it) << " Color: ("
            << int(inst_color.r) << "," << int(inst_color.g) << "," << int(inst_color.b) << ")";
    }

    LOG(INFO) << "LogSemanticColor: ";
    for (auto sem_it = sem_label_set.begin(); 
        sem_it != sem_label_set.end(); sem_it++)
    {
        Color sem_c;
        // convert the semantic label to open-semantic label
        mesh_semantic_integrator_->open_sem_color_map_.getColor(
            static_cast<OpenSemanticLabel>(*sem_it), &sem_c);
        LOG(INFO) << "  Semantic: " << int(*sem_it) << " Color: ("
            << int(sem_c.r) << "," << int(sem_c.g) << "," << int(sem_c.b) << ")";
    }

    if (inst_association_ == 3) {
        // log sem connection map
    }
}

pybind11::array_t<uint8_t> GlobalSegmentMap_py::getInstanceColor(
    int instance_label)
{
    Color inst_color;
    mesh_instance_integrator_->instance_color_map_.getColor(
        static_cast<InstanceLabel>(instance_label), 
        &inst_color);

    pybind11::array_t<uint8_t> color_array(3);
    auto buf = color_array.request();
    uint8_t* ptr = static_cast<uint8_t*>(buf.ptr);
    ptr[0] = inst_color.r;
    ptr[1] = inst_color.g;
    ptr[2] = inst_color.b;

    return color_array;
}

void GlobalSegmentMap_py::LogLabelInitialGuess(std::string log_path)
{
    std::string label_inital_log = log_path + "/LabelInitialGuess.txt";
    LOG(INFO) << "Label initial guess saved to LabelInitialGuess.txt: ";
    std::ofstream log_file_io;
    log_file_io.open(label_inital_log.c_str());
    if (log_file_io.is_open())
    {
        log_file_io << "# label initial instance guess " << std::endl;
        log_file_io << "# format: label semantic_label instance_label r g b " << std::endl;

        SemanticInstanceLabelFusion *sem_inst_label_fuse_ptr_ =
            integrator_->semantic_instance_label_fusion_ptr_;
        for (auto label_it = sem_inst_label_fuse_ptr_->label_frames_count_.begin();
             label_it != sem_inst_label_fuse_ptr_->label_frames_count_.end(); label_it++)
        {
            InstanceLabel instance_label = sem_inst_label_fuse_ptr_->getInstanceLabel(label_it->first, 0.1f);
            SemanticLabel semantic_label = sem_inst_label_fuse_ptr_->getSemanticLabel(label_it->first);
            Color label_color;
            mesh_label_integrator_->label_color_map_.getColor(label_it->first, &label_color);
            log_file_io << std::setfill('0') << std::setw(5) << int(label_it->first) << " "
                << int(semantic_label) << " "
                << std::setfill('0') << std::setw(5) << int(instance_label) << " "
                << int(label_color.r) << " " << int(label_color.g) << " " << int(label_color.b) << std::endl;
        }
    }
}



void GlobalSegmentMap_py::initializeCameraRayCaster(
    pybind11::array &camera_K, int img_height,
    int img_width, float range_min,
    float range_max, int thread_num)
{
    if (cam_ray_generator_ == nullptr) {
        cv::Mat camera_K_mat = cvnp::nparray_to_mat(camera_K);
        cam_ray_generator_ = new CameraRayGenerator(
            camera_K_mat, img_height, img_width,
            range_min, range_max, thread_num);
        LOG(INFO) << "CameraRayGenerator initialized with img_height: " << img_height
                  << ", img_width: " << img_width;
    }
}



/**
 * Get the global instance map 
 */
pybind11::array_t<uint16_t> GlobalSegmentMap_py::raycastInstancePredictions(
    pybind11::array &T_G_C,            // float
    pybind11::array &instance_mask,    // int
    pybind11::array &depth_img_scaled  // float
) {
    cv::Mat T_G_C_mat = cvnp::nparray_to_mat(T_G_C);
    Eigen::Matrix<float, 4, 4> T_G_C_eigen;
    cv::cv2eigen(T_G_C_mat, T_G_C_eigen);
    Transformation T_G_C_voxblox(T_G_C_eigen);

    cv::Mat instance_mask_mat = cvnp::nparray_to_mat(instance_mask);
    instance_mask_mat.convertTo(instance_mask_mat, CV_16U);
    // cv::imwrite("mask.png", instance_mask_mat);
    cv::Mat rayCastInstImg(instance_mask_mat.size(), CV_16U, cv::Scalar(0));

    cv::Mat depth_img_scaled_mat = cvnp::nparray_to_mat(depth_img_scaled);
    // cv::imwrite("depth.png", depth_img_scaled_mat);
    

    float search_length = 0.4f;

    integrator_->raycastInstancePredictions(
        T_G_C_voxblox, 
        instance_mask_mat, 
        depth_img_scaled_mat, 
        search_length, 
        cam_ray_generator_, 
        &rayCastInstImg
    );

    // Ensure it's continuous (if not, clone it)
    return cvnp::mat_to_nparray(rayCastInstImg.clone(), false);
}




void GlobalSegmentMap_py::raycastPanopticPredictions(
    pybind11::array &T_G_C,            // float
    pybind11::array &panoptic_mask,    // uint8_t
    pybind11::array &inst_sem_labels,  // uint8_t
    pybind11::array &depth_img_scaled, // float
    const float search_length,
    float pose_confidence = 1.0
) {
    cv::Mat T_G_C_mat = cvnp::nparray_to_mat(T_G_C);
    Eigen::Matrix<float, 4, 4> T_G_C_eigen;
    cv::cv2eigen(T_G_C_mat, T_G_C_eigen);
    Transformation T_G_C_voxblox(T_G_C_eigen);

    cv::Mat panoptic_mask_mat = cvnp::nparray_to_mat(panoptic_mask);
    cv::Mat depth_img_scaled_mat = cvnp::nparray_to_mat(depth_img_scaled);

    // map the panoptic instance id to close-set semantic label
    std::map<InstanceLabel, SemanticLabel> inst_sem_map;
    int num_inst_labels = inst_sem_labels.shape()[0];
    for (int inst_i = 0; inst_i < num_inst_labels; inst_i++) {
        inst_sem_map[inst_i + 1] = ((uint8_t *)inst_sem_labels.data())[inst_i];
        LOG(INFO) << "    panoptic id " << int(inst_i+1) << " ; sem: " << int( inst_sem_map[inst_i+1]);
    }


    std::map<Label, std::map<InstanceLabel, int>> label_instances_cout;
    auto time_start = std::chrono::system_clock::now();
    assert((inst_association_ == 6) || (inst_association_ == 7));
    if (inst_association_ == 6)
        integrator_->raycastPanopticPredictions(
            T_G_C_voxblox, panoptic_mask_mat, inst_sem_map, 
            depth_img_scaled_mat, search_length, cam_ray_generator_, 
            label_instances_cout
        );
    else
        integrator_->raycastPanopticPredictions(
            T_G_C_voxblox, panoptic_mask_mat, inst_sem_map, 
            depth_img_scaled_mat, search_length, cam_ray_generator_, 
            label_instances_cout, pose_confidence
        );

    
    auto time_end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration<double>(time_end - time_start).count();
    LOG(INFO) << "  ray cast cost: " << duration << " seconds";
}



void GlobalSegmentMap_py::outputLog(std::string log_info)
{
    LOG(INFO) << log_info;
}





PYBIND11_MODULE(consistent_gsm, m)
{
    m.doc() = "pybind11 for consistent global segmentation map"; // optional module docstring

    pybind11::class_<GlobalSegmentMap_py>(m, "GlobalSegmentMap_py")
        .def(pybind11::init<std::string, std::string, bool, bool, int, int, int, bool, int, bool, float, float>())
        .def("insertSegments", &GlobalSegmentMap_py::insertSegments)
        .def("insertSegmentsOpen", &GlobalSegmentMap_py::insertSegmentsOpen)
        .def("insertSegmentsPoseConfidence", &GlobalSegmentMap_py::insertSegmentsPoseConfidence)
        .def("integrateFrame", &GlobalSegmentMap_py::integrateFrame)
        .def("LogSegmentsLabels", &GlobalSegmentMap_py::LogSegmentsLabels)
        .def("clearTemporaryMemory", &GlobalSegmentMap_py::clearTemporaryMemory)
        .def("generateMesh", &GlobalSegmentMap_py::generateMesh)
        .def("outputLog", &GlobalSegmentMap_py::outputLog)
        .def("updateVisualization", &GlobalSegmentMap_py::updateVisualization)
        .def("updateVisualizationPCL", &GlobalSegmentMap_py::updateVisualizationPCL)
        .def("LogLabelInformation", &GlobalSegmentMap_py::LogLabelInformation)
        .def("LogMeshColors", &GlobalSegmentMap_py::LogMeshColors)
        .def("getInstanceColor", &GlobalSegmentMap_py::getInstanceColor)
        // .def("mapTextEmbedding", &GlobalSegmentMap_py::mapTextEmbedding)
        // .def("initFeatureMap", &GlobalSegmentMap_py::initFeatureMap)
        // .def("LogOpensetFeatureDistribution", &GlobalSegmentMap_py::LogOpensetFeatureDistribution)
        .def("initializeCameraRayCaster", &GlobalSegmentMap_py::initializeCameraRayCaster)
        .def("raycastInstancePredictions", &GlobalSegmentMap_py::raycastInstancePredictions)
        .def("raycastPanopticPredictions", &GlobalSegmentMap_py::raycastPanopticPredictions);
}