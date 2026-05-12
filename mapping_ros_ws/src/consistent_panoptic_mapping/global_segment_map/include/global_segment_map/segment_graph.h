#ifndef SEGMENT_GRAPH_H
#define SEGMENT_GRAPH_H

#include <queue>
#include <memory>
#include <fstream>
#include <iomanip>
#include "global_segment_map/common.h"

namespace voxblox
{

typedef float InstanceConfidence;

typedef std::map<Label, ObjSegConfidence> LabelConfiMap;
typedef std::map<Label, ObjSegConfidence>::iterator LabelConfiMapIt;
typedef std::map<SemanticLabel, LabelConfiMap> SemLabelConfiMap;
typedef std::map<SemanticLabel, LabelConfiMap>::iterator  SemLabelConfiMapIt;
typedef std::map<Label, SemLabelConfiMap> LabelSemLabelConfiMap;
typedef std::map<Label, SemLabelConfiMap>::iterator LabelSemLabelConfiMapIt;

typedef std::map<Label, LSet> LabelConnectMap;
typedef std::map<Label, LSet>::iterator LabelConnectMapIt;
typedef std::map<SemanticLabel, LabelConnectMap > SemLabelConnectMap;
typedef std::map<SemanticLabel, LabelConnectMap >::iterator SemLabelConnectMapIt;

typedef std::map<Label, LabelConfiMap> LLConfidenceMap;

class SegGraphInstance
{
public:
    using Ptr = std::shared_ptr<SegGraphInstance>;
    using ConstPtr = std::shared_ptr<const SegGraphInstance>;

    SegGraphInstance(const Label& init_label,const SemanticLabel& init_sem, const ObjSegConfidence& label_confidence,
        const bool& is_thing, InstanceLabel id)
    : sem_(init_sem), is_thing_(is_thing), id_(id)
    {
        seg_labels_ = LSet({init_label});
        max_seg_confidence_ = std::max(label_confidence, max_seg_confidence_);
    };

    // compute confidence for instance based on subject segments
    // for high-level usage
    void computeInstConfidence(){}; // TODO
    void insertLabel(const Label& seg_label, const SemanticLabel& seg_sem, const ObjSegConfidence& label_confidence,
        const bool& seg_is_thing)
    { 
        CHECK_EQ(is_thing_, seg_is_thing);
        CHECK_EQ(seg_sem, sem_);
        seg_labels_.insert(seg_label); 
        max_seg_confidence_ = std::max(label_confidence, max_seg_confidence_);
    };
    bool isThing() const{ return is_thing_; };
    InstanceLabel getInstID() const{ return id_; };
    SemanticLabel getInstSem() const{ return sem_; };
    LSet getLabels() const{ return seg_labels_; };
    ObjSegConfidence getMaxSegConfidence() const{ return max_seg_confidence_; };
protected:
    LSet seg_labels_;
    InstanceConfidence inst_confidence_ = 0.;
    bool is_thing_ = true;
    InstanceLabel id_;
    SemanticLabel sem_;

    ObjSegConfidence max_seg_confidence_ = 0.;
};

class SegGraph
{
public:
    using Ptr = std::shared_ptr<SegGraph>;
    using ConstPtr = std::shared_ptr<const SegGraph>;

    SegGraph(const bool use_inst_connect, const ObjSegConfidence connection_th, 
        const ObjSegConfidence connection_ratio_th)
    : use_inst_label_connect_(use_inst_connect), 
    connection_th_(connection_th),
    connection_ratio_th_(connection_ratio_th){};
    
    // increase label_a--semantic--label_b--confidence
    void increaseConfidenceMapUnit(const Label& label_a, const SemanticLabel& semantic_label, 
        const Label& label_b, const ObjSegConfidence& confidence);
    // SET label_a--semantic--label_b--confidence
    void updateConfidenceMapUnit(const Label& label_a, const SemanticLabel& semantic_label, 
        const Label& label_b, const ObjSegConfidence& confidence);
    // insert segments of a instance into SegGraph
    void insertInstanceToSegGraph(const std::vector<Label>& instance_labels, 
        const std::vector<ObjSegConfidence>& labels_confidence, const SemanticLabel& semantic_label );
    void insertInstanceToSegGraph(const std::vector<Label>& instance_labels, 
        LLConfidenceMap& labels_confidence_map, const SemanticLabel& semantic_label, bool is_thing);
        
    // extract semantic for seg label based on seg_graph_confidence_map_
    SemanticLabel extractSemantic(const Label& label);
    // update label_semantic_map_ with observation of each frame
    void updateLabelSemanticMap();
    
    void queryConfidence(const Label& label_a, const SemanticLabel& semantic_label, 
        const Label& label_b, ObjSegConfidence* confidence);
    SemanticLabel getSemantic(const Label& label);
    InstanceLabel getInstance(const Label& label);
    inline bool isLabelABConnected(Label label_a, Label label_b, SemanticLabel semantic_label);
    bool isInstanceLabelConnected(const SegGraphInstance::Ptr& inst_ptr, const Label& label_query, 
        const SemanticLabel& semantic_label);

    void generateSemLabelConnectMap();
    inline bool isThing(SemanticLabel semantic_label) const; 
    void updateAllMap();
    void extractInstances();
    // merge segment label_b to label_a
    LSet getAllLabels();
    void mergeLabels(const Label& label_a, const Label& label_b);
    void removeLabel(const Label& label_remvove);

    // log information
    void logSegGraph(std::string log_file);
    void logConfidenceMap(std::string log_file);
    void logLabelConfidenceMap(Label label);

private:
    LabelSemLabelConfiMap seg_graph_confidence_map_;
    SemLabelConnectMap sem_labels_connect_map;
    std::map<Label, SemanticLabel> label_semantic_map_;
    std::map<Label, InstanceLabel> label_instance_map_;
    LSet updated_segments_;
    bool semantic_up_to_date_ = false;
    bool connection_map_up_to_date = false;

    std::vector<SegGraphInstance::Ptr> instances_;
    InstanceLabel num_instance_ = 0;
    bool instance_up_to_date_ = false;

    bool use_inst_label_connect_ = true;
    ObjSegConfidence connection_th_ = 3.0; // connection threshold
    ObjSegConfidence connection_ratio_th_ = 0.2; // connection ratio threshold
};

}
#endif