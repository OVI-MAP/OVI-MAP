#ifndef CONSISTENT_GSM_MAPPER_H_
#define CONSISTENT_GSM_MAPPER_H_

#include <cmath>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <glog/logging.h>
#include <glog/log_severity.h>
#include <stdlib.h>

#include "consistent_mapping/label_tsdf_confidence_integrator.h"
#include "consistent_mapping/segment_confidence.h"
#include "global_segment_map/segment.h"
#include "consistent_mapping/SettingsSemantic.h"

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

class ConsistentGSMMapper{
public:
    ConsistentGSMMapper() = delete;
    ConsistentGSMMapper(const std::string &configFile);

    bool integrateSegments(std::vector<SegmentConfidence*>& segments_ptrs);
    void clearFrameSegsCache();
    bool generateMesh(std::string mesh_file_folder, std::string frame_num="");

    inline SemanticLabel quarySemanticPoint(Point& query_point);
    void quarySemanticPointCloud(std::vector<Point> query_pcl,
        std::vector<SemanticLabel>& semantics_list);
    Label quarySuperpointID(Point query_point);

private:
    // settings
    std::shared_ptr<SemanticMappingSettings> settings_ptr_;
    LabelTsdfMap::Config map_config_;
    LabelTsdfIntegrator::Config tsdf_integrator_config_;
    LabelTsdfIntegrator::LabelTsdfConfig label_tsdf_integrator_config_;
    MeshIntegratorConfig mesh_config_;
    MeshLabelIntegrator::LabelTsdfConfig label_tsdf_mesh_config_;
    bool kIsFreespacePointcloud_ = false;

    /* data during processing */
    bool integration_on_;
    size_t integrated_frames_count_;
    std::vector<Label> segment_labels_to_publish_;
    std::map<Label, std::set<Label>> merges_to_publish_;
    // Semantic labels.
    std::map<Label, std::map<SemanticLabel, int>>* label_class_count_ptr_;
    // Current frame label propagation.
    std::vector<Segment*> segments_to_integrate_;
    std::map<Label, std::map<Segment*, SegSegConfidence>> segment_label_candidates;
    std::map<Segment*, std::vector<Label>> segment_merge_candidates_;

    std::mutex label_tsdf_layers_mutex_;
    std::mutex mesh_layer_mutex_;
    bool mesh_layer_updated_;
    bool need_full_remesh_;

    // map, meshes and integrators
    std::shared_ptr<LabelTsdfMap> map_;
    std::shared_ptr<LabelTsdfConfidenceIntegrator> integrator_;

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
};


#endif  // GLOBAL_SEGMENT_MAP_PY_H_