#ifndef GLOBAL_SEGMENT_MAP_SEMANTIC_LABEL_FUSION_H_
#define GLOBAL_SEGMENT_MAP_SEMANTIC_LABEL_FUSION_H_

#include <map>
#include "global_segment_map/segment_graph.h"
#include "global_segment_map/common.h"

namespace voxblox {

class SemanticInstanceLabelFusion {
 public:
  SemanticInstanceLabelFusion()
  {}

// for original
  void increaseLabelInstanceCount(const Label& label,
                                  const InstanceLabel& instance_label);

  void decreaseLabelInstanceCount(const Label& label,
                                  const InstanceLabel& instance_label);

  void increaseLabelFramesCount(const Label& label, 
    const SegmentObserveConfidence& confidence = 1.0);
  void increaseLabelClassCount(const Label& label,
                               const SemanticLabel& semantic_label);

// for confidence
  void increaseLabelInstanceConfidence(const Label& label,
                                      const InstanceLabel& instance_label,
                                      const LabelConfidence& confidence);
  void increaseLabelClassConfidence(const Label& label,
                               const SemanticLabel& semantic_label,
                               const SegmentObserveConfidence& semantic_label_confidence);
  std::pair<SemanticLabel,SegmentObserveConfidence>  getSemanticLabelWithProb(const Label& label) const;

  // for label-class-instance-count
  void  increaseLabelClassInstanceConfidence(const Label& label,
                               const SemanticLabel& semantic_label,
                               const InstanceLabel& instance_label,
                               const SegmentObserveConfidence& semantic_label_confidence);       
  
  InstanceLabel getInstanceLabelWithSemantic(
      const Label& label, const SemanticLabel& semantic_label,
      const std::set<InstanceLabel>& assigned_instances) const;

  // for label-instance-class-count
  void increaseLabelInstanceClassConfidence(const Label& label,
                                const InstanceLabel& instance_label,
                                const SemanticLabel& semantic_label,
                                const SegmentObserveConfidence& semantic_label_confidence);  

  // for SegGraph
  void initSegGraph()
  { 
    seg_graph_ptr_.reset(new SegGraph(use_inst_label_connect_, connection_th_, connection_ratio_th_)); 
  }
  
  void insertInstanceToSegGraph(const std::vector<Label>& labels_vector, 
        const std::vector<ObjSegConfidence>& labels_confidence, const SemanticLabel& semantic_label );
  void insertInstanceToSegGraph(const std::vector<Label>& labels_vector, 
        LLConfidenceMap& labels_confidence_map, const SemanticLabel& semantic_label, bool is_thing);
  void increaseSegGraphConfidence(const Label& label_a, const SemanticLabel& semantic_label, 
          const Label& label_b, const ObjSegConfidence& confidence)
      { seg_graph_ptr_->increaseConfidenceMapUnit(label_a, semantic_label,label_b, confidence);}

  void logSegGraphInfo(std::string log_path);
  void checkLabelInFrameCount();
  void logAllLabels();

  // for common use
  InstanceLabel getInstanceLabel(
      const Label& label, const std::set<InstanceLabel>& assigned_instances =
                              std::set<InstanceLabel>()) const;
  InstanceLabel getInstanceLabel(
      const Label& label, const float count_threshold_factor,
      const std::set<InstanceLabel>& assigned_instances =
          std::set<InstanceLabel>()) const;

  SemanticLabel getSemanticLabel(const Label& label) const;
  void logLabelSemanticInstanceCountInfo(const Label& label)const;
  void mergeLabels(const Label& label_a, const Label& label_b);   // merge

  int inst_association_ = 0;
  std::map<Label, std::map<SemanticLabel, std::map<InstanceLabel, SegmentObserveConfidence>>> label_class_instance_count_;
  std::map<Label, std::map<InstanceLabel, std::map<SemanticLabel, SegmentObserveConfidence>>> label_instance_class_count_;
  LSLMap label_class_count_;
  std::map<Label, SegmentObserveConfidence> label_frames_count_;

  // for seg-graph
  bool use_inst_label_connect_ = true;
  ObjSegConfidence connection_th_ = 3.0; // connection threshold
  ObjSegConfidence connection_ratio_th_ = 0.2; // connection ratio threshold

 protected:
  std::map<Label, std::map<InstanceLabel, SegmentObserveConfidence>> label_instance_count_;
  std::shared_ptr<SegGraph> seg_graph_ptr_;

  // Per frame voxel count of semantic label.


  
};

}  // namespace voxblox

#endif  // GLOBAL_SEGMENT_MAP_SEMANTIC_LABEL_FUSION_H_
