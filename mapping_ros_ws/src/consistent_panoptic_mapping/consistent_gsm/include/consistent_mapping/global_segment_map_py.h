#ifndef GLOBAL_SEGMENT_MAP_PY_H_
#define GLOBAL_SEGMENT_MAP_PY_H_

// #include "consistent_mapping/common.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "cvnp/cvnp.h"

#include <cmath>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <glog/logging.h>
#include <glog/log_severity.h>

#include "consistent_mapping/label_tsdf_confidence_integrator.h"
#include <consistent_mapping/segment_confidence.h>
#include <consistent_mapping/virtual_memory_log.h>
#include "utils/pcl_semantic_visualizers.h"
#include <global_segment_map_node/controller.h>
#include <global_segment_map_node/conversions.h>
#include <global_segment_map/common.h>
#include <global_segment_map/label_tsdf_integrator.h>
#include <global_segment_map/label_tsdf_map.h>
#include "global_segment_map/semantic_instance_label_fusion.h"
#include <global_segment_map/label_voxel.h>
#include <global_segment_map/meshing/label_tsdf_mesh_integrator.h>
#include <global_segment_map/meshing/label_color_map.h>
#include <global_segment_map/label_voxel.h>
#include <global_segment_map/utils/file_utils.h>
#include <global_segment_map/utils/map_utils.h>
#include <global_segment_map/utils/visualizer.h>

#include <utils/camera_ray_generator.h>

#include <voxblox_ros/conversions.h>
#include <voxblox/alignment/icp.h>
#include <voxblox/core/common.h>
#include <voxblox/io/sdf_ply.h>
#include <voxblox/io/mesh_ply.h>
#include <voxblox_ros/mesh_vis.h>
#include <minkindr_conversions/kindr_tf.h>

#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>


using namespace voxblox;
using namespace voxblox::voxblox_gsm;


class GlobalSegmentMap_py{
public:
GlobalSegmentMap_py(
    std::string log_file, 
    std::string task = "coco80",
    bool use_geo_confidence=false, 
    bool use_label_confidence=false, 
    int inst_association=0, 
    int data_association=0, 
    int num_threads=-1, bool debug=false, 
    int seg_graph_confidence=0, 
    bool use_inst_label_connect = true,
    float connection_ratio_th = 0.2, 
    float cos_sim_threshold = 0.8
);

~GlobalSegmentMap_py()
{
    if(debug_)
    {
        viz_mesh_thread_.join(); 
        // viz_pcl_thread_.join();
    }    
    if(cam_ray_generator_)
    {
        delete cam_ray_generator_;
    }
}

void insertSegments(
    pybind11::array& points,        // float
    // pybind11::array& colors,     // rgba uint8_t
    // pybind11::array& geometry_confidence, //float
    pybind11::array& b_box,         // float
    InstanceLabel instance_label,   // uint16_t
    SemanticLabel semantic_label,   // uint8_t
    ObjSegConfidence inst_confidence,
    ObjSegConfidence obj_seg_confidence,
    pybind11::array &T_G_C,
    bool is_thing,
    Label desginated_label = BackgroundLabel
);

void insertSegmentsOpen(
    pybind11::array& points,        // float
    // pybind11::array& colors,     // rgba uint8_t
    // pybind11::array& geometry_confidence, //float
    pybind11::array& b_box,         // float
    InstanceLabel instance_label,   // uint16_t
    SemanticLabel semantic_label,   // uint8_t
    pybind11::array_t<float>& sem_feature,   // float
    ObjSegConfidence inst_confidence,
    ObjSegConfidence obj_seg_confidence,
    pybind11::array &T_G_C,
    bool is_thing,
    Label desginated_label = BackgroundLabel
);

void insertSegmentsPoseConfidence(
    pybind11::array& points, // float
    // pybind11::array& colors, // rgba uint8_t
    // pybind11::array& geometry_confidence, //float
    pybind11::array& b_box, //float
    InstanceLabel instance_label, //uint16_t
    SemanticLabel semantic_label, //uint8_t
    ObjSegConfidence inst_confidence,
    ObjSegConfidence obj_seg_confidence,
    pybind11::array &T_G_C,
    float pose_confidence,
    bool is_thing,
    Label desginated_label = BackgroundLabel
);

float cosineSimilarity(
    const SemanticFeature& a, 
    const SemanticFeature& b
);

// float searchAndInsertFeature(
//     const SemanticFeature& seg_feature, OpenSemanticLabel *label
// );

// void initFeatureMap(
//     pybind11::array& text_embedding, const int cand_num
// );

bool integrateFrame();

void LogSegmentsLabels();

void clearTemporaryMemory();

bool generateMesh(
    std::string mesh_file_folder, 
    std::string frame_num="",
    bool save_label_mesh=false, 
    bool save_sem_mesh=true, 
    bool save_inst_mesh=true
);

void updateVisualization();
void updateVisualizationPCL();

void LogLabelInformation();
void LogMeshColors(std::string log_path);
void LogLabelInitialGuess(std::string log_path);
pybind11::array_t<uint8_t> getInstanceColor(
    int instance_label
);

void outputLog(std::string log_info);

void initializeCameraRayCaster(
    pybind11::array &camera_K, 
    int img_height, int img_width, 
    float range_min, 
    float range_max, 
    int thread_num = 1
);

pybind11::array_t<uint16_t> raycastInstancePredictions(
    pybind11::array &T_G_C,            // float
    pybind11::array &instance_mask,    // int
    pybind11::array &depth_img_scaled  // float
);

void raycastPanopticPredictions(
    pybind11::array &T_G_C, 
    pybind11::array& panoptic_mask,
    pybind11::array& inst_sem_labels,
    pybind11::array& depth_img_scaled,
    const float search_length,
    float pose_confidence
);

private:
    std::string log_file_; // log file path

/* config parameters */
    std::string task_;
    bool enable_semantic_instance_segmentation_;
    bool use_label_propagation_;
    bool use_geo_confidence_ = false;
    bool use_label_confidence_ = false;
    int inst_association_ = 0;
    int data_association_ = 0;

    int seg_graph_confidence_ = 0;
    bool use_inst_label_connect_ = true;
    float connection_ratio_th_ = 0.2;

    LabelTsdfMap::Config map_config_;
    LabelTsdfIntegrator::Config tsdf_integrator_config_;
    LabelTsdfIntegrator::LabelTsdfConfig label_tsdf_integrator_config_;

    /* data during processing */
    bool integration_on_;
    size_t integrated_frames_count_;

    std::vector<Label> segment_labels_to_publish_;
    std::map<Label, std::set<Label>> merges_to_publish_;
    // Semantic labels.
    std::map<Label, std::map<SemanticLabel, int>>* label_class_count_ptr_;
    // Current frame label propagation.
    std::vector<Segment*> segments_to_integrate_;
    std::map<Label, std::map<Segment*, SegSegConfidence>> segment_label_candidates_;
    std::map<Segment*, std::vector<Label>> segment_merge_candidates_;

    std::mutex label_tsdf_layers_mutex_;
    std::mutex mesh_layer_mutex_;
    bool mesh_layer_updated_;
    bool need_full_remesh_;


    /* map, meshes and integrators */
    std::shared_ptr<LabelTsdfMap> map_;
    std::shared_ptr<LabelTsdfConfidenceIntegrator> integrator_;

    MeshIntegratorConfig mesh_config_;
    MeshLabelIntegrator::LabelTsdfConfig label_tsdf_mesh_config_;

    MeshLabelIntegrator::ColorScheme mesh_color_scheme_;
    std::string mesh_filename_;

    std::shared_ptr<MeshLayer> mesh_label_layer_;
    std::shared_ptr<MeshLayer> mesh_semantic_layer_;
    std::shared_ptr<MeshLayer> mesh_instance_layer_;
    std::shared_ptr<MeshLayer> mesh_merged_layer_;
    std::shared_ptr<MeshLayer> mesh_confidence_layer_;
    std::shared_ptr<MeshLabelIntegrator> mesh_label_integrator_;
    std::shared_ptr<MeshLabelIntegrator> mesh_semantic_integrator_;
    std::shared_ptr<MeshLabelIntegrator> mesh_instance_integrator_;
    std::shared_ptr<MeshLabelIntegrator> mesh_merged_integrator_;
    std::shared_ptr<MeshLabelIntegrator> mesh_confidence_integrator_;

    /* ray caster*/
    CameraRayGenerator* cam_ray_generator_ = nullptr;

    /* visualiz*/
    PCLSemVisualizerConfig visualizer_config_;
    std::shared_ptr<PCLSemVisualizer> visualizer_pcl_;
    std::shared_ptr<Visualizer> visualizer_mesh_;
    std::thread viz_pcl_thread_;
    std::thread viz_mesh_thread_;
    bool debug_=false;


    inline OpenSemanticLabel getFreshSemLabel() {
        CHECK_LT(highest_sem_label_, std::numeric_limits<unsigned int>::max());
        return ++(highest_sem_label_);
    }
    // a uint16_t var
    OpenSemanticLabel highest_sem_label_;

    /* feature vectors */
    std::map<OpenSemanticLabel, SemanticFeature> feature_map_;
    std::map<SemanticLabel, std::set<OpenSemanticLabel>> close_to_open_;
    std::map<OpenSemanticLabel, int> open_set_count_;
    float cos_sim_threshold_=0.1f;

    int systemMem_ = 0; // in KB

};

#endif  // GLOBAL_SEGMENT_MAP_PY_H_