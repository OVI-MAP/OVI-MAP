#include "global_segment_map/segment_graph.h"

namespace voxblox
{

void SegGraph::increaseConfidenceMapUnit(const Label& label_a, const SemanticLabel& semantic_label, 
        const Label& label_b, const ObjSegConfidence& confidence)
{
    // increase label_a--semantic--label_b--confidence
    // label_a level
    LabelSemLabelConfiMapIt label_a_it = seg_graph_confidence_map_.find(label_a);
    if(label_a_it!=seg_graph_confidence_map_.end())
    {
        // label_a--semantic level
        SemLabelConfiMapIt semantic_it = label_a_it->second.find(semantic_label);
        if(semantic_it!=label_a_it->second.end())
        {
            // label_a--semantic--label_b level
            LabelConfiMapIt label_b_it = semantic_it->second.find(label_b);
            if(label_b_it!=semantic_it->second.end())
            {
                label_b_it->second += confidence;
            }
            else
            {   
                semantic_it->second.emplace(label_b, confidence);
            }
        }
        else
        {
            LabelConfiMap label_b_confidence;
            label_b_confidence.emplace(label_b, confidence);
            label_a_it->second.emplace(semantic_label, label_b_confidence);
        }
    }
    else
    {
        LabelConfiMap label_b_confidence;
        label_b_confidence.emplace(label_b, confidence);
        SemLabelConfiMap sem_label_b_confidence;
        sem_label_b_confidence.emplace(semantic_label, label_b_confidence);
        seg_graph_confidence_map_.emplace(label_a, sem_label_b_confidence);
    }
    // record update for label_a
    updated_segments_.insert(label_a);
    // set data outdate
    semantic_up_to_date_ = false;
    connection_map_up_to_date = false;
    instance_up_to_date_ = false;

    // LOG(INFO) << "  Get Label " << int(label_a) <<  " and Label " << int(label_b)
    //     << " with Sem " << int(semantic_label) << " with confidence " << confidence;
}

void SegGraph::updateConfidenceMapUnit(const Label& label_a, const SemanticLabel& semantic_label, 
        const Label& label_b, const ObjSegConfidence& confidence)
{
    // increase label_a--semantic--label_b--confidence
    // label_a level
    CHECK_GE(confidence, 0.);
    LabelSemLabelConfiMapIt label_a_it = seg_graph_confidence_map_.find(label_a);
    if(label_a_it!=seg_graph_confidence_map_.end())
    {
        // label_a--semantic level
        SemLabelConfiMapIt semantic_it = label_a_it->second.find(semantic_label);
        if(semantic_it!=label_a_it->second.end())
        {
            // label_a--semantic--label_b level
            LabelConfiMapIt label_b_it = semantic_it->second.find(label_b);
            if(label_b_it!=semantic_it->second.end())
            {
                label_b_it->second = confidence;
            }
            else
            {   
                semantic_it->second.emplace(label_b, confidence);
            }
        }
        else
        {
            LabelConfiMap label_b_confidence;
            label_b_confidence.emplace(label_b, confidence);
            label_a_it->second.emplace(semantic_label, label_b_confidence);
        }
    }
    else
    {
        LabelConfiMap label_b_confidence;
        label_b_confidence.emplace(label_b, confidence);
        SemLabelConfiMap sem_label_b_confidence;
        sem_label_b_confidence.emplace(semantic_label, label_b_confidence);
        seg_graph_confidence_map_.emplace(label_a, sem_label_b_confidence);
    }
    // record update for label_a
    updated_segments_.insert(label_a);
    // set data outdate
    semantic_up_to_date_ = false;
    connection_map_up_to_date = false;
    instance_up_to_date_ = false;

    // LOG(INFO) << "  Get Label " << int(label_a) <<  " and Label " << int(label_b)
    //     << " with Sem " << int(semantic_label) << " with confidence " << confidence;
}

void SegGraph::insertInstanceToSegGraph(const std::vector<Label>& instance_labels, 
        const std::vector<ObjSegConfidence>& labels_confidence, const SemanticLabel& semantic_label )
{   
    const size_t num_labels = instance_labels.size();
    CHECK_GT(num_labels, 0);

    const bool is_thing = isThing(semantic_label);

    for(size_t label_a_i=0; label_a_i<num_labels; label_a_i++)
    {
        const Label& label_a = instance_labels[label_a_i];
        const ObjSegConfidence& label_a_confidence = labels_confidence[label_a_i];
        // add internal edges
        increaseConfidenceMapUnit(label_a, semantic_label, label_a, label_a_confidence);
        ObjSegConfidence label_a_semantic_confidence;
        queryConfidence(label_a, semantic_label, label_a, &label_a_semantic_confidence);
        // LOG(INFO) << "      increase Label " << std::setfill('0') << std::setw(5) <<int(label_a)
        //     << "    semantic " << int(semantic_label) << "; count " << label_a_semantic_confidence;
        if(is_thing) {
            //add external edges
            for(size_t label_b_i=label_a_i+1; label_b_i<num_labels; label_b_i++)
            {
                const Label& label_b = instance_labels[label_b_i];
                const ObjSegConfidence& label_b_confidence = labels_confidence[label_b_i];
                const ObjSegConfidence cross_confidence = label_a_confidence*label_b_confidence;
                increaseConfidenceMapUnit(label_a, semantic_label, label_b, cross_confidence);
                increaseConfidenceMapUnit(label_b, semantic_label, label_a, cross_confidence);

            }
        }
    }
}

void SegGraph::insertInstanceToSegGraph(const std::vector<Label>& instance_labels, 
        LLConfidenceMap& labels_confidence_map, const SemanticLabel& semantic_label, bool is_thing )
{   
    const size_t num_labels = instance_labels.size();
    CHECK_GT(num_labels, 0);

    // const bool is_thing = isThing(semantic_label);

    for(size_t label_a_i=0; label_a_i<num_labels; label_a_i++)
    {
        Label label_a = instance_labels[label_a_i];
        const ObjSegConfidence& label_a_confidence = labels_confidence_map[label_a][label_a];
        // add internal edges
        increaseConfidenceMapUnit(label_a, semantic_label, label_a, label_a_confidence);
        ObjSegConfidence label_a_semantic_confidence;
        queryConfidence(label_a, semantic_label, label_a, &label_a_semantic_confidence);
        // LOG(INFO) << "      increase Label " << std::setfill('0') << std::setw(5) <<int(label_a)
        //     << "    semantic " << int(semantic_label) << "; count " << label_a_semantic_confidence;
        if(is_thing)
        {
            //add external edges
            for(size_t label_b_i=label_a_i+1; label_b_i<num_labels; label_b_i++)
            {
                const Label& label_b = instance_labels[label_b_i];
                const ObjSegConfidence cross_confidence = labels_confidence_map[label_a][label_b];
                increaseConfidenceMapUnit(label_a, semantic_label, label_b, cross_confidence);
                increaseConfidenceMapUnit(label_b, semantic_label, label_a, cross_confidence);
                // LOG(INFO) << "      increase confidence " << cross_confidence << " for label pair" 
                // << std::setfill('0') << std::setw(5) <<int(label_a) << " - "
                // << std::setfill('0') << std::setw(5) <<int(label_b);
            }
        }
    }
}

SemanticLabel SegGraph::extractSemantic(const Label& label)
{
    // use internal edges to determine semantic label of segment
    SemanticLabel semantic_label = BackgroundSemLabel;
    SegmentObserveConfidence max_confidence = 0.;

    LabelSemLabelConfiMapIt label_it = seg_graph_confidence_map_.find(label);
    if(label_it!=seg_graph_confidence_map_.end())
    {
        for(SemLabelConfiMapIt sem_it = label_it->second.begin();
            sem_it!=label_it->second.end(); sem_it++)
        {
            LabelConfiMapIt sem_label_it = sem_it->second.find(label);
            if(sem_label_it!=sem_it->second.end())
            {
                bool greater_than_max_confidence = (sem_label_it->second > max_confidence);
                if(greater_than_max_confidence)
                {
                    semantic_label = sem_it->first;
                    max_confidence = sem_label_it->second;
                }
            }
        }
    }
    // maybe we can also try to use external edges 
    return semantic_label;
}

void SegGraph::updateLabelSemanticMap()
{
    for(const Label& updated_label:updated_segments_)
    {
        const SemanticLabel updated_semantic_label = extractSemantic(updated_label);
        std::map<Label, SemanticLabel>::iterator label_semantic_it = 
            label_semantic_map_.find(updated_label); 
        if(label_semantic_it!=label_semantic_map_.end())
            label_semantic_it->second = updated_semantic_label;
        else
            label_semantic_map_.emplace(updated_label, updated_semantic_label);
    }
    updated_segments_.clear();
    semantic_up_to_date_ = true;
}

void SegGraph::queryConfidence(const Label& label_a, const SemanticLabel& semantic_label, 
        const Label& label_b, ObjSegConfidence* confidence)
{
    *confidence = 0.;
    LabelSemLabelConfiMapIt label_a_it = seg_graph_confidence_map_.find(label_a);
    if(label_a_it!=seg_graph_confidence_map_.end())
    {
        // label_a--semantic level
        SemLabelConfiMapIt semantic_it = label_a_it->second.find(semantic_label);
        if(semantic_it!=label_a_it->second.end())
        {
            // label_a--semantic--label_b level
            LabelConfiMapIt label_b_it = semantic_it->second.find(label_b);
            if(label_b_it!=semantic_it->second.end())
            {
                *confidence = label_b_it->second;
            }
        }
    }
}

SemanticLabel SegGraph::getSemantic(const Label& label)
{
    // if label_semantic_map_ is not up-to-date, update it first
    if(!semantic_up_to_date_)
        updateLabelSemanticMap();

    // get semantic label of segment from label_semantic_map_
    SemanticLabel semantic_label = BackgroundSemLabel;

    std::map<Label, SemanticLabel>::iterator label_it = label_semantic_map_.find(label);
    if(label_it!=label_semantic_map_.end())
    {
        semantic_label = label_it->second;
    }
    return semantic_label;
}

InstanceLabel SegGraph::getInstance(const Label& label)
{
    // if label_semantic_map_ is not up-to-date, update it first
    if(!semantic_up_to_date_)
        updateLabelSemanticMap();
    // if sem_labels_connect_map is not up-to-date, update it first
    if(!connection_map_up_to_date)
        generateSemLabelConnectMap();
    // if label_instance_map_ is not up-to-date, update it first
    if(!instance_up_to_date_)
        extractInstances();

    InstanceLabel instance_label = BackgroundLabel;
    std::map<Label, InstanceLabel>::iterator label_inst_it = label_instance_map_.find(label);
    if(label_inst_it!=label_instance_map_.end())
        instance_label = label_inst_it->second;

    return instance_label;
}

inline bool SegGraph::isLabelABConnected(Label label_a, Label label_b, SemanticLabel semantic_label)
{

    ObjSegConfidence a_a_sem_confidence = 0.;
    ObjSegConfidence b_b_sem_confidence = 0.;
    ObjSegConfidence a_b_sem_confidence = 0.;
    ObjSegConfidence b_a_sem_confidence = 0.;
    queryConfidence(label_a, semantic_label, label_a, &a_a_sem_confidence);
    queryConfidence(label_b, semantic_label, label_b, &b_b_sem_confidence);
    queryConfidence(label_a, semantic_label, label_b, &a_b_sem_confidence);
    queryConfidence(label_b, semantic_label, label_a, &b_a_sem_confidence);
    bool confidence_greater_than_th = (a_b_sem_confidence > connection_th_);
    bool inverse_confidence_greater_than_th = (b_a_sem_confidence > connection_th_);
    bool ratio_greater_than_th = (a_b_sem_confidence > connection_ratio_th_*a_a_sem_confidence);
    bool ratio_greater_than_th_inverse = (b_a_sem_confidence > connection_ratio_th_*b_b_sem_confidence);
    if(confidence_greater_than_th!=inverse_confidence_greater_than_th)
    {
        LOG(INFO) << "Quering Label " << int(label_a) << " Label " << int(label_b)
            << "  Semantics: " << int(semantic_label);
        LOG(INFO) << " a_b_sem_confidence: " << a_b_sem_confidence;
        LOG(INFO) << " b_a_sem_confidence: " << b_a_sem_confidence;
        logLabelConfidenceMap(label_a);
        logLabelConfidenceMap(label_b);
    }
    CHECK_EQ(confidence_greater_than_th, inverse_confidence_greater_than_th);
    return (confidence_greater_than_th&&ratio_greater_than_th&&ratio_greater_than_th_inverse);
}

bool SegGraph::isInstanceLabelConnected(const SegGraphInstance::Ptr& inst_ptr, const Label& label_query, 
        const SemanticLabel& semantic_label)
{
    if(use_inst_label_connect_)
    {
        const LSet labels_in_inst = inst_ptr->getLabels();
        const ObjSegConfidence inst_max_confidence = inst_ptr->getMaxSegConfidence();
        for(const Label& label:labels_in_inst)
        {
            ObjSegConfidence label_query_connected_confidence;
            queryConfidence(label_query, semantic_label, label, &label_query_connected_confidence);
            if(label_query_connected_confidence > connection_ratio_th_*inst_max_confidence)
            return true;
        }
        return false;
    }
    // if do not use rule of inst_label_connect, always return true
    return true;
}

void SegGraph::generateSemLabelConnectMap()
{
    LOG(INFO) << "generateSemLabelConnectMap";
    // aggregate segments according to their semantic label 
    //  and create connection map within semantic-segment sets
    sem_labels_connect_map.clear();
    for(LabelSemLabelConfiMapIt label_it = seg_graph_confidence_map_.begin();
        label_it!=seg_graph_confidence_map_.end(); label_it++)
    {
        const Label label = label_it->first;
        const SemanticLabel semantic_label = getSemantic(label);
        LSet connected_labels;
        // only consider confidence connection of determined semantic label
        SemLabelConfiMapIt label_sem_it = label_it->second.find(semantic_label);
        bool has_semantic = ( label_sem_it != label_it->second.end() ) ||
            (semantic_label == BackgroundSemLabel);
        CHECK_EQ(has_semantic, true);
        for(LabelConfiMapIt label_sem_label_it = label_sem_it->second.begin();
            label_sem_label_it!=label_sem_it->second.end(); label_sem_label_it++)
        {
            Label label_to_check = label_sem_label_it->first;
            bool semantic_consistent = (semantic_label == getSemantic(label_to_check));
            if(semantic_consistent)
            {
                bool is_connected = isLabelABConnected(label, label_to_check, semantic_label);
                if(is_connected)
                {   connected_labels.insert(label_to_check); }
            }
        }
        // insert sem-label-connected set into sem_labels_connect_map
        SemLabelConnectMapIt sem_it = sem_labels_connect_map.find(semantic_label);
        if(sem_it != sem_labels_connect_map.end())
        {
            LabelConnectMapIt sem_label_it = sem_it->second.find(label);
            bool has_no_label = ( sem_label_it == sem_it->second.end() );
            CHECK_EQ(has_no_label, true);
            sem_it->second.emplace(label, connected_labels);
        }
        else
        {
            LabelConnectMap label_connect_map;
            label_connect_map.emplace(label, connected_labels);
            sem_labels_connect_map.emplace(semantic_label, label_connect_map);
        }
    }
    connection_map_up_to_date = true;
}

inline bool SegGraph::isThing(SemanticLabel semantic_label) const
{
    // determine whether segments is "thing"; for "thing" object, we do instance segmentation;
    //  otherwise, we regard "stuff" segments of each semantic label as one instance.

    // for panoptic segmentation "Detectron2"
    if(semantic_label<BackgroundSemLabel)
        return true;
    else
        return false;
} 
void SegGraph::updateAllMap()
{
    // if label_semantic_map_ is not up-to-date, update it first
    if(!semantic_up_to_date_)
        updateLabelSemanticMap();
    // if sem_labels_connect_map is not up-to-date, update it first
    if(!connection_map_up_to_date)
        generateSemLabelConnectMap();
    // if label_instance_map_ is not up-to-date, update it first
    if(!instance_up_to_date_)
        extractInstances();
}

void SegGraph::extractInstances()
{
    // extracted instances with sem_labels_connect_map
    // save results in instances_

    // make sure used data is up-to-date
    if(!semantic_up_to_date_)
        updateLabelSemanticMap();
    if(!connection_map_up_to_date)
        generateSemLabelConnectMap();

    //clear up existing instances
    instances_.clear();
    label_instance_map_.clear();
    num_instance_ = 0;

    // for each set of segments with same semantic label, do instance segmentation
    for(SemLabelConnectMapIt sem_it=sem_labels_connect_map.begin();
        sem_it!=sem_labels_connect_map.end(); sem_it++)
    {
        const SemanticLabel semantic_label = sem_it->first;
        const bool is_thing = isThing(semantic_label);
        if(is_thing)
        {
            // use breadth-first dpoc to create instance and assign segment labels
            LabelConnectMap& semantic_connect_map = sem_it->second;
            const std::size_t num_labels = semantic_connect_map.size();
            std::set<Label> assigned_seg_labels;
            // done when all segment labels are assigned instance label
            for(LabelConnectMapIt label_it = semantic_connect_map.begin(); 
                label_it!=semantic_connect_map.end();  label_it++)
            {
                const Label label = label_it->first;
                if(assigned_seg_labels.find(label) != assigned_seg_labels.end())
                    continue;

                // create new instance for unassigned segment
                ObjSegConfidence label_confidence;
                queryConfidence(label, semantic_label, label, &label_confidence);
                SegGraphInstance::Ptr inst_thing_ptr = std::make_shared<SegGraphInstance>(
                    label, semantic_label, label_confidence, is_thing, ++num_instance_);
                const InstanceLabel inst_id = inst_thing_ptr->getInstID();
                label_instance_map_.emplace(label, inst_id);
                assigned_seg_labels.insert(label);

                // breadth-first dpoc, assign connected segments to the newly created instance
                std::queue<Label> labels_to_query;
                std::set<Label> visited_labels;
                labels_to_query.push(label);
                visited_labels.insert(label);
                while(labels_to_query.size()!=0)
                {
                    const Label& label_to_query = labels_to_query.front();
                    const LSet& labels_to_check = semantic_connect_map.find(label_to_query)->second;
                    for(const Label& label_to_check:labels_to_check)
                    {
                        const bool not_equal_self = (label_to_check!=label_to_query);
                        const bool not_assigned = (assigned_seg_labels.find(label_to_check)
                            == assigned_seg_labels.end());
                        const bool not_visited = (visited_labels.find(label_to_check)
                            == visited_labels.end());
                        const bool is_inst_label_connected = isInstanceLabelConnected(inst_thing_ptr, 
                                label_to_check, getSemantic(label_to_check) );

                        visited_labels.insert(label_to_check);
                        if(not_equal_self && not_assigned && not_visited && is_inst_label_connected)
                        {
                            labels_to_query.push(label_to_check);
                            // add segment label into instance
                            const SemanticLabel label_check_semantic = getSemantic(label_to_check);
                            ObjSegConfidence label_check_confidence;
                            queryConfidence(label_to_check, label_check_semantic, label_to_check, &label_check_confidence);
                            inst_thing_ptr->insertLabel(label_to_check, label_check_semantic, 
                                label_check_confidence, is_thing);
                            label_instance_map_.emplace(label_to_check, inst_id);
                            assigned_seg_labels.insert(label_to_check);
                        }
                    }
                    labels_to_query.pop();
                }
                // add instance
                instances_.push_back(std::move(inst_thing_ptr));
            }
        }
        else
        {
            // initialize instance with one seg
            LabelConnectMapIt sem_label_it = sem_it->second.begin();
            const Label label_to_add = sem_label_it->first;

            ObjSegConfidence label_to_add_confidence;
            queryConfidence(label_to_add, semantic_label, label_to_add, &label_to_add_confidence);
            SegGraphInstance::Ptr inst_stuff_ptr = std::make_shared<SegGraphInstance>(
                label_to_add, semantic_label, label_to_add_confidence, is_thing, ++num_instance_);
            const InstanceLabel inst_id = inst_stuff_ptr->getInstID();
            label_instance_map_.emplace(label_to_add, inst_id);
            sem_label_it++;
            // add remaining segs to the instance
            for( ; sem_label_it!=sem_it->second.end(); sem_label_it++)
            {
                const Label label_to_add = sem_label_it->first;

                ObjSegConfidence label_to_add_confidence;
                queryConfidence(label_to_add, semantic_label, label_to_add, &label_to_add_confidence);
                inst_stuff_ptr->insertLabel(label_to_add, semantic_label, label_to_add_confidence, is_thing);
                label_instance_map_.emplace(label_to_add, inst_id);
            }
            // add instance
            instances_.push_back(std::move(inst_stuff_ptr));
        }
    }

    instance_up_to_date_ = true;
}
LSet SegGraph::getAllLabels()
{
    LOG(INFO) << "          getAllLabels STEP1; seg_graph_confidence_map_.SIZE(): "
        << int(seg_graph_confidence_map_.size());
    LSet labels;
    for(LabelSemLabelConfiMapIt label_it = seg_graph_confidence_map_.begin();
        label_it != seg_graph_confidence_map_.end(); label_it++)
    {
        labels.insert(label_it->first);
    }
    LOG(INFO) << "          getAllLabels STEP2";
    return labels;
}

void SegGraph::mergeLabels(const Label& label_a, const Label& label_b)
{
    LabelSemLabelConfiMapIt label_a_it = seg_graph_confidence_map_.find(label_a);
    LabelSemLabelConfiMapIt label_b_it = seg_graph_confidence_map_.find(label_b);
    const bool label_a_exist = ( label_a_it != seg_graph_confidence_map_.end() );
    const bool label_b_exist = ( label_b_it != seg_graph_confidence_map_.end() );

    // merge only when two labels exist in seg_graph_confidence_map_
    if(label_a_exist && label_b_exist)
    {
        LOG(INFO) << "BeforeMerging,  Label: " <<  int(label_a) << 
            "  Sem: " << int(getSemantic(label_a));
        LOG(INFO) << "  SemanticsCount: ";
        LabelSemLabelConfiMapIt label_it = seg_graph_confidence_map_.find(label_a);
        // for(SemLabelConfiMapIt sem_it=label_it->second.begin();
        //     sem_it != label_it->second.end(); sem_it++)
        // {
        //     LOG(INFO) << "  Semantics: " << int(sem_it->first);
        //     for(LabelConfiMapIt sem_label_it = sem_it->second.begin();
        //         sem_label_it != sem_it->second.end(); sem_label_it++)
        //     {
        //         ObjSegConfidence inverse_confidence;
        //         queryConfidence(sem_label_it->first, sem_it->first, label_a, &inverse_confidence);
        //         LOG(INFO) <<"       LabelOfSemantics: " << int(sem_label_it->first)
        //                 <<" Count: " << sem_label_it->second
        //                 << " Inverse Count: " << inverse_confidence;
        //         if(sem_label_it->first != label_b)
        //             CHECK_EQ(inverse_confidence, sem_label_it->second);
        //     }
        // }
        // LOG(INFO) << "BeforeMerging,  Label: " <<  int(label_b) << 
        //     "  Sem: " << int(getSemantic(label_b));
        // LOG(INFO) << "  SemanticsCount: ";
        // label_it = seg_graph_confidence_map_.find(label_b);
        // for(SemLabelConfiMapIt sem_it=label_it->second.begin();
        //     sem_it != label_it->second.end(); sem_it++)
        // {
        //     LOG(INFO) << "  Semantics: " << int(sem_it->first);
        //     for(LabelConfiMapIt sem_label_it = sem_it->second.begin();
        //         sem_label_it != sem_it->second.end(); sem_label_it++)
        //     {
        //         LOG(INFO) <<"       LabelOfSemantics: " << int(sem_label_it->first)
        //                 <<" Count: " << sem_label_it->second;
        //     }
        // }
        // for all internal and external edges, use maximum between two
        for(SemLabelConfiMapIt label_b_sem_it = label_b_it->second.begin(); 
            label_b_sem_it != label_b_it->second.end(); label_b_sem_it++)
        {
            const SemanticLabel semantic_label_b = label_b_sem_it->first;
            // SemLabelConfiMapIt label_a_sem_it = label_a_it->second.find(semantic_label_b);
            // if(label_a_sem_it != label_a_it->second.end())
            // {
            for(LabelConfiMapIt label_b_sem_label_it = label_b_sem_it->second.begin();
                label_b_sem_label_it != label_b_sem_it->second.end(); label_b_sem_label_it++)
            {
                
                Label label_to_add = label_b_sem_label_it->first;
                
                if(label_to_add == label_b)// internal edges
                {
                    // internal edges of label_b should be added to internal edges of label_a
                    ObjSegConfidence label_a_sem_internal;
                    queryConfidence(label_a, semantic_label_b, label_a, &label_a_sem_internal);
                    ObjSegConfidence label_b_sem_internal;
                    queryConfidence(label_b, semantic_label_b, label_b, &label_b_sem_internal);
                    ObjSegConfidence max_confidence_sem_internal = 
                        std::max(label_a_sem_internal, label_b_sem_internal);
                    updateConfidenceMapUnit(label_a, semantic_label_b, label_a, max_confidence_sem_internal);
                    // LOG(INFO) <<"       Update: Label " << int(label_a) <<" Sem: " << int(semantic_label_b)
                    //     << " Label: " << int(label_a) << "  Confidence: " << max_confidence_sem_internal;
                }
                else // external edges
                {
                    // add label_a_--semantic--label_to_add--max(*) confidence
                    ObjSegConfidence confidence_label_a_label_to_add;
                    queryConfidence(label_a, semantic_label_b, label_to_add, &confidence_label_a_label_to_add);
                    ObjSegConfidence confidence_max = 
                        std::max(confidence_label_a_label_to_add, label_b_sem_label_it->second);
                    updateConfidenceMapUnit(label_a, semantic_label_b, label_to_add, confidence_max);
                    // LOG(INFO) <<"       Update: Label " << int(label_a) <<" Sem: " << int(semantic_label_b)
                    //     << " Label: " << label_to_add << "  Confidence: " << confidence_max;
                    ObjSegConfidence confidence_label_to_add_label_a;
                    queryConfidence(label_to_add, semantic_label_b, label_a, &confidence_label_to_add_label_a);
                    ObjSegConfidence confidence_max_inverse = 
                        std::max(confidence_label_to_add_label_a, label_b_sem_label_it->second);
                    updateConfidenceMapUnit(label_to_add, semantic_label_b, label_a, confidence_max_inverse);
                    // LOG(INFO) <<"       Update: Label " << int(label_to_add) <<" Sem: " << int(semantic_label_b)
                    //     << " Label: " << label_a << "  Confidence: " << confidence_max_inverse;
                    CHECK_EQ(confidence_max, confidence_max_inverse);
                }
            }
            // }
            // else
            // {
            //     label_a_it->second.emplace(semantic_label_b, label_b_sem_it->second);
            // }
        }
        // record update for label_a
        updated_segments_.insert(label_a);
        // set current data outdate
        semantic_up_to_date_ = false;
        connection_map_up_to_date = false;
        instance_up_to_date_ = false;

        // LOG(INFO) << "AfterMerging,  Label: " <<  int(label_a) << 
        //     "  Sem: " << int(getSemantic(label_a));
        // LOG(INFO) << "  SemanticsCount: ";
        // label_it = seg_graph_confidence_map_.find(label_a);
        // for(SemLabelConfiMapIt sem_it=label_it->second.begin();
        //     sem_it != label_it->second.end(); sem_it++)
        // {
        //     LOG(INFO) << "  Semantics: " << int(sem_it->first);
        //     for(LabelConfiMapIt sem_label_it = sem_it->second.begin();
        //         sem_label_it != sem_it->second.end(); sem_label_it++)
        //     {
        //         ObjSegConfidence inverse_confidence;
        //         queryConfidence(sem_label_it->first, sem_it->first, label_a, &inverse_confidence);
        //         LOG(INFO) <<"       LabelOfSemantics: " << int(sem_label_it->first)
        //                 <<" Count: " << sem_label_it->second
        //                 << " Inverse Count: " << inverse_confidence;
        //         if(sem_label_it->first != label_b)
        //             CHECK_EQ(inverse_confidence, sem_label_it->second);
        //     }
        // }
        LOG(INFO) << "  Erasing from SegGraph label " << std::setfill('0') << std::setw(5) << int(label_b);
        removeLabel(label_b);
    }
}

void SegGraph::logSegGraph(std::string log_file)
{
    updateAllMap();
    std::ofstream log_file_io;
    log_file_io.open(log_file.c_str());
    if ( log_file_io.is_open() )
    {
        log_file_io << "instances_: " << std::endl;
        for(SegGraphInstance::Ptr& inst_ptr:instances_)
        {
            log_file_io << "    Inst ID: " << std::setfill('0') << std::setw(5) << int(inst_ptr->getInstID()) << std::endl;
            log_file_io << "    Inst Semantics: " << int(inst_ptr->getInstSem()) << std::endl;
            log_file_io << "    Inst Max Seg Confidence: " << inst_ptr->getMaxSegConfidence() << std::endl;
            LSet inst_labels = inst_ptr->getLabels();
            for(Label label:(inst_labels))
            {
                log_file_io << "        inst-label: " << int(label) <<  std::endl;
            }
        }

        log_file_io << "SemLabelConnectMap: " << std::endl;
        for(SemLabelConnectMapIt sem_it = sem_labels_connect_map.begin();
            sem_it != sem_labels_connect_map.end(); sem_it++)
        {
            log_file_io << "Semantics: " << int(sem_it->first) << std::endl;
            for(LabelConnectMapIt label_it = sem_it->second.begin();
                label_it != sem_it->second.end(); label_it++)
            {
                log_file_io << "  ConnectMap-Label: " << int(label_it->first) << std::endl;
                
                for(LSetIt label_label_it = label_it->second.begin();
                    label_label_it != label_it->second.end(); label_label_it++)
                {
                    log_file_io << "    Connected Labels: " << 
                    std::setfill('0') << std::setw(5) <<  int(*label_label_it) << std::endl;
                }
            }
        }

        log_file_io << "seg_graph_confidence_map_: " << std::endl;
        for(LabelSemLabelConfiMapIt label_it = seg_graph_confidence_map_.begin();
            label_it != seg_graph_confidence_map_.end(); label_it++)
        {
            log_file_io << "Label: " << int(label_it->first) << std::endl;
            for(SemLabelConfiMapIt sem_it = label_it->second.begin();
                sem_it != label_it->second.end(); sem_it++)
            {
                log_file_io << "    Label-Semantics: " << int(sem_it->first) << std::endl;
                log_file_io << "        Label-Semantics-Label: " << std::endl;
                for(LabelConfiMapIt sem_label_it = sem_it->second.begin();
                    sem_label_it != sem_it->second.end(); sem_label_it++)
                {
                    log_file_io << "            Label-Semantics-Label-Confi: " 
                        << int(sem_label_it->first) << "    " << sem_label_it->second << std::endl;
                }
            }
        }
    
    }
    log_file_io.close();
}

void SegGraph::logConfidenceMap(std::string log_file)
{
    updateAllMap();
    std::ofstream log_file_io;
    log_file_io.open(log_file.c_str());
    if ( log_file_io.is_open() )
    {
        log_file_io << "# seg_graph_confidence_map_ " << std::endl;
        log_file_io << "# format: label semantic label confidence " << std::endl;
        for(LabelSemLabelConfiMapIt label_it = seg_graph_confidence_map_.begin();
            label_it != seg_graph_confidence_map_.end(); label_it++)
        {
            for(SemLabelConfiMapIt sem_it = label_it->second.begin();
                sem_it != label_it->second.end(); sem_it++)
            {
                for(LabelConfiMapIt sem_label_it = sem_it->second.begin();
                    sem_label_it != sem_it->second.end(); sem_label_it++)
                {
                    log_file_io << std::setfill('0') << std::setw(5) << int(label_it->first) << " "
                        << int(sem_it->first) << " "
                        << std::setfill('0') << std::setw(5) << int(sem_label_it->first) << " "
                        << sem_label_it->second << std::endl;
                }
            }
        }
    
    }
    log_file_io.close();
}

void SegGraph::logLabelConfidenceMap(Label label)
{
    LOG(INFO) << "  Label: " <<  int(label) << 
        "  Sem: " << int(getSemantic(label));
    CHECK_EQ(getSemantic(label), extractSemantic(label));
    LOG(INFO) << "  SemanticsCount: ";
    LabelSemLabelConfiMapIt label_it = seg_graph_confidence_map_.find(label);
    for(SemLabelConfiMapIt sem_it=label_it->second.begin();
        sem_it != label_it->second.end(); sem_it++)
    {
        LOG(INFO) << "  LabelOfSemantics: " << int(sem_it->first);
        for(LabelConfiMapIt sem_label_it = sem_it->second.begin();
            sem_label_it != sem_it->second.end(); sem_label_it++)
        {
            LOG(INFO) <<"       LabelOfSemantics: " << int(sem_label_it->first)
                    <<" CountOfLabelOfSemantics: " << sem_label_it->second;
        }
    }
}

void SegGraph::removeLabel(const Label& label_remvove)
{
    seg_graph_confidence_map_.erase(label_remvove);
    for(LabelSemLabelConfiMapIt label_it = seg_graph_confidence_map_.begin();
        label_it != seg_graph_confidence_map_.end(); label_it++)
    {
        for(SemLabelConfiMapIt sem_it = label_it->second.begin();
            sem_it != label_it->second.end(); sem_it++)
        {
            sem_it->second.erase(label_remvove);
        }
    }
    semantic_up_to_date_ = false;
    connection_map_up_to_date = false;
    instance_up_to_date_ = false;
}

}