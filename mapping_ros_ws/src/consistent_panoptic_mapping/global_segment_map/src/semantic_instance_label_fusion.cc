#include "global_segment_map/semantic_instance_label_fusion.h"

namespace voxblox {

// for original
void SemanticInstanceLabelFusion::increaseLabelInstanceCount(
    const Label& label, const InstanceLabel& instance_label) {
  auto label_it = label_instance_count_.find(label);
  if (label_it != label_instance_count_.end()) {
    auto instance_it = label_it->second.find(instance_label);
    if (instance_it != label_it->second.end()) {
      ++instance_it->second;
    } else {
      label_it->second.emplace(instance_label, 1);
    }
  } else {
    std::map<InstanceLabel, SegmentObserveConfidence> instance_count;
    instance_count.emplace(instance_label, 1);
    label_instance_count_.emplace(label, instance_count);
  }
}

void SemanticInstanceLabelFusion::increaseLabelClassCount(
    const Label& label, const SemanticLabel& semantic_label) {
  auto label_it = label_class_count_.find(label);
  if (label_it != label_class_count_.end()) {
    auto class_it = label_it->second.find(semantic_label);
    if (class_it != label_it->second.end()) {
      ++class_it->second;
    } else {
      label_it->second.emplace(semantic_label, 1);
    }
  } else {
    SLMap class_points_count;
    class_points_count.emplace(semantic_label, 1);
    label_class_count_.emplace(label, class_points_count);
  }
}

void SemanticInstanceLabelFusion::decreaseLabelInstanceCount(
    const Label& label, const InstanceLabel& instance_label) {
  auto label_it = label_instance_count_.find(label);
  if (label_it != label_instance_count_.end()) {
    auto instance_it = label_it->second.find(instance_label);
    if (instance_it != label_it->second.end()) {
      --instance_it->second;
    } else {
      LOG(FATAL) << "Decreasing a non existing label-instance count.";
    }
  } else {
    LOG(FATAL) << "Decreasing a non existing label-instance count.";
  }
}

void SemanticInstanceLabelFusion::increaseLabelFramesCount(const Label& label, 
  const SegmentObserveConfidence& confidence) {
  auto label_count_it = label_frames_count_.find(label);
  if (label_count_it != label_frames_count_.end()) {
    label_count_it->second += confidence;
  } else {
    label_frames_count_.insert(std::make_pair(label, confidence));
  }
}

// for confidence
void SemanticInstanceLabelFusion::increaseLabelInstanceConfidence(const Label& label,
  const InstanceLabel& instance_label,const LabelConfidence& confidence){
  auto label_it = label_instance_count_.find(label);
  if (label_it != label_instance_count_.end()) {
    auto instance_it = label_it->second.find(instance_label);
    if (instance_it != label_it->second.end()) {
      instance_it->second += confidence;
    } else {
      label_it->second.emplace(instance_label, confidence);
    }
  } else {
    std::map<InstanceLabel, SegmentObserveConfidence> instance_count;
    instance_count.emplace(instance_label, confidence);
    label_instance_count_.emplace(label, instance_count);
  }
}
void SemanticInstanceLabelFusion::increaseLabelClassConfidence(
  const Label& label, const SemanticLabel& semantic_label,
  const SegmentObserveConfidence& semantic_label_confidence){
  auto label_it = label_class_count_.find(label);
  if (label_it != label_class_count_.end()) {
    auto class_it = label_it->second.find(semantic_label);
    if (class_it != label_it->second.end()) {
      class_it->second += semantic_label_confidence;
    } else {
      label_it->second.emplace(semantic_label, semantic_label_confidence);
    }
  } else {
    SLMap class_points_count;
    class_points_count.emplace(semantic_label, semantic_label_confidence);
    label_class_count_.emplace(label, class_points_count);
  }
}
std::pair<SemanticLabel,SegmentObserveConfidence> 
  SemanticInstanceLabelFusion::getSemanticLabelWithProb(const Label& label) const {

  SegmentObserveConfidence p_c_given_l = 0.;
  SemanticLabel semantic_label = BackgroundSemLabel;
  SegmentObserveConfidence total_confidence = 0.;
  SegmentObserveConfidence max_confidence = 0.;
  auto label_it = label_class_count_.find(label);
  if (label_it != label_class_count_.end()) {
    for (auto const& class_count : label_it->second) {
      total_confidence += class_count.second;
      if (class_count.second > max_confidence &&
          class_count.first != BackgroundSemLabel) {
        semantic_label = class_count.first;
        max_confidence = class_count.second;
      }
    }
  }
  if(semantic_label!=BackgroundSemLabel)
    p_c_given_l = max_confidence/total_confidence;

  return std::make_pair(semantic_label, p_c_given_l);
}

// for label-class-instance-count
void  SemanticInstanceLabelFusion::increaseLabelClassInstanceConfidence(const Label& label,
                              const SemanticLabel& semantic_label,
                              const InstanceLabel& instance_label,
                              const SegmentObserveConfidence& label_instance_confidence){
    auto label_it = label_class_instance_count_.find(label);
    if (label_it != label_class_instance_count_.end())
    {
        auto sem_it = label_it->second.find(semantic_label);
        if (sem_it != label_it->second.end())
        {
            auto inst_it = sem_it->second.find(instance_label);
            if (inst_it != sem_it->second.end())
                inst_it->second += label_instance_confidence;
            else
                sem_it->second.insert(std::make_pair(instance_label, label_instance_confidence));
        }
        else
        {
            std::map<InstanceLabel, SegmentObserveConfidence> 
              inst_map = {std::make_pair(instance_label, label_instance_confidence)}; 
            std::pair<SemanticLabel, std::map<InstanceLabel, SegmentObserveConfidence>> 
              sem_pair = std::make_pair(semantic_label, inst_map);
            label_it->second.insert(sem_pair);
        }
    }
    else
    {
        std::map<InstanceLabel, SegmentObserveConfidence> 
          inst_map = {std::make_pair(instance_label, label_instance_confidence)}; 
        std::map<SemanticLabel, std::map<InstanceLabel, SegmentObserveConfidence>> 
          sem_map = {std::make_pair(semantic_label, inst_map)};
        std::pair<Label, std::map<SemanticLabel, std::map<InstanceLabel, SegmentObserveConfidence>>> 
          label_pair = std::make_pair(label, sem_map);
        label_class_instance_count_.insert(label_pair);
    } 
}
InstanceLabel SemanticInstanceLabelFusion::getInstanceLabelWithSemantic(
    const Label& label, const SemanticLabel& semantic_label,
    const std::set<InstanceLabel>& assigned_instances) const
{
    InstanceLabel instance_label = BackgroundLabel;
    SegmentObserveConfidence max_confidence = 0;
    auto label_it = label_class_instance_count_.find(label);
    if (label_it != label_class_instance_count_.end())
    {
        auto sem_it = label_it->second.find(semantic_label);
        if (sem_it != label_it->second.end())
        {
            for (auto const& instance_count : sem_it->second)
            {
                if (instance_count.second > max_confidence && instance_count.first != BackgroundLabel &&
                    assigned_instances.find(instance_count.first) == assigned_instances.end())
                {

                  instance_label = instance_count.first;
                  max_confidence = instance_count.second;
                    
                }
            }
        }                
    }
    return instance_label;
}

// for label-instance-class-count
void SemanticInstanceLabelFusion::increaseLabelInstanceClassConfidence(const Label& label,
                              const InstanceLabel& instance_label,
                              const SemanticLabel& semantic_label,
                              const SegmentObserveConfidence& semantic_label_confidence){
    auto label_it = label_instance_class_count_.find(label);
    if (label_it != label_instance_class_count_.end())
    {
        auto inst_it = label_it->second.find(instance_label);
        if (inst_it != label_it->second.end())
        {
            auto sem_it = inst_it->second.find(semantic_label);
            if (sem_it != inst_it->second.end())
                sem_it->second += semantic_label_confidence;
            else
                inst_it->second.insert(std::make_pair(semantic_label, semantic_label_confidence));
        }
        else
        {
            std::map<SemanticLabel, SegmentObserveConfidence> 
              sem_map = {std::make_pair(semantic_label, semantic_label_confidence)}; 
            std::pair<InstanceLabel, std::map<SemanticLabel, SegmentObserveConfidence>> 
              inst_pair = std::make_pair(instance_label, sem_map);
            label_it->second.insert(inst_pair);
        }
    }
    else
    {
        std::map<SemanticLabel, SegmentObserveConfidence> 
          sem_map = {std::make_pair(semantic_label, semantic_label_confidence)}; 
        std::map<InstanceLabel, std::map<SemanticLabel, SegmentObserveConfidence>> 
          inst_map = {std::make_pair(instance_label, sem_map)};
        std::pair<Label, std::map<InstanceLabel, std::map<SemanticLabel, SegmentObserveConfidence>>> 
          label_pair = std::make_pair(label, inst_map);
        label_instance_class_count_.insert(label_pair);
    } 
}               


  // for SegGraph
void SemanticInstanceLabelFusion::insertInstanceToSegGraph(const std::vector<Label>& labels_vector, 
        const std::vector<ObjSegConfidence>& labels_confidence, const SemanticLabel& semantic_label )
{
  seg_graph_ptr_->insertInstanceToSegGraph(labels_vector, labels_confidence, semantic_label);
  if(inst_association_!=4)
  {
    for(const Label& label_in_inst:labels_vector)
      increaseLabelFramesCount(label_in_inst);
  }
}

/**
 * We are using this one
 */
void SemanticInstanceLabelFusion::insertInstanceToSegGraph(const std::vector<Label>& labels_vector, 
        LLConfidenceMap& labels_confidence_map, const SemanticLabel& semantic_label ,bool is_thing )
{
  seg_graph_ptr_->insertInstanceToSegGraph(labels_vector, labels_confidence_map, semantic_label, is_thing);
  if(inst_association_!=4)
  {
    for(const Label& label_in_inst:labels_vector)
      increaseLabelFramesCount(label_in_inst);
  }
}

void SemanticInstanceLabelFusion::logSegGraphInfo(std::string log_path)
{
    // std::string seg_graph_log = log_path + "/SegGraphInfo.txt";
    // LOG(INFO) << "SegGraphInfo into SegGraphInfo.txt: "; 
    // seg_graph_ptr_->logSegGraph(seg_graph_log);
    // checkLabelInFrameCount();
    std::string seg_confidence_log = log_path + "/ConfidenceMap.txt";
    LOG(INFO) << "SegConfidenceInfo into ConfidenceMap.txt: "; 
    seg_graph_ptr_->logConfidenceMap(seg_confidence_log);
    
}

void SemanticInstanceLabelFusion::checkLabelInFrameCount()
{
  const LSet labels = seg_graph_ptr_->getAllLabels();
  for(const Label& label:labels )
  {
    const bool is_label_in_frame_count = (label_frames_count_.find(label) 
        != label_frames_count_.end() );
    if(!is_label_in_frame_count)
    {
      LOG(ERROR) << " error label: " << int(label);
      logAllLabels();
    }
    CHECK_EQ(is_label_in_frame_count, true);
  }
}
void SemanticInstanceLabelFusion::logAllLabels()
{
  LOG(INFO) << "label_frames_count_.size(): " << label_frames_count_.size();
  for(auto label_it = label_frames_count_.begin(); 
    label_it != label_frames_count_.end(); label_it ++ )
  {
    LOG(INFO) << "  Label: " << std::setfill('0') << std::setw(5) 
      << int(label_it->first);
  }

}

// for common use
InstanceLabel SemanticInstanceLabelFusion::getInstanceLabel(
    const Label& label,
    const std::set<InstanceLabel>& assigned_instances) const {
  
  return getInstanceLabel(label, 0.0f, assigned_instances);
}

InstanceLabel SemanticInstanceLabelFusion::getInstanceLabel(
    const Label& label, const float count_threshold_factor,
    const std::set<InstanceLabel>& assigned_instances) const {
  InstanceLabel instance_label = BackgroundLabel;

  // LOG(INFO) << "  getInstanceLabel: ";
  if(inst_association_ == 1 || inst_association_ == 4 || inst_association_ == 6 || inst_association_ == 7) //use_sem_inst_confidence_
  {
    // get semantic first
    SegmentObserveConfidence semantic_prob = 0.;
    auto sem_prob_pair = getSemanticLabelWithProb(label);
    SemanticLabel semantic_label = sem_prob_pair.first;
    SegmentObserveConfidence probabilty = sem_prob_pair.second;

    if(semantic_label != BackgroundSemLabel){
      SegmentObserveConfidence max_inst_confidence = 0.;
      auto label_it = label_class_instance_count_.find(label);
      if (label_it != label_class_instance_count_.end())
      {
        auto sem_it = label_it->second.find(semantic_label);
        if (sem_it != label_it->second.end())
        {
          for (auto const& instance_count : sem_it->second)
          {
            if (instance_count.second > max_inst_confidence && 
              instance_count.first != BackgroundLabel &&
              assigned_instances.find(instance_count.first) == assigned_instances.end())
            {
              ObjSegConfidence frames_count = 0;
              auto label_count_it = label_frames_count_.find(label);
              if (label_count_it != label_frames_count_.end())
                frames_count = label_count_it->second;
              if (instance_count.second > count_threshold_factor * (float)(frames_count - instance_count.second))
              {
                instance_label = instance_count.first;
                max_inst_confidence = instance_count.second;
              } 
            }
          }
        }                
      }
    }
  }
  else if(inst_association_ == 3) // segment graph
  {
    const SemanticLabel semantic_label = seg_graph_ptr_->getSemantic(label);
    ObjSegConfidence label_sem_confidence;
    seg_graph_ptr_->queryConfidence(label, semantic_label, label, &label_sem_confidence);

    ObjSegConfidence frames_count = 0;
    auto label_count_it = label_frames_count_.find(label);
    if (label_count_it != label_frames_count_.end())
      frames_count = label_count_it->second;
    if(label_sem_confidence > count_threshold_factor*(frames_count-label_sem_confidence))
      instance_label = seg_graph_ptr_->getInstance(label);
  }
  else // 2-use_inst_sem_count or 0-orginal method
  {
    SegmentObserveConfidence max_count = 0;
    auto label_it = label_instance_count_.find(label);
    if (label_it != label_instance_count_.end()) {
      for (auto const& instance_count : label_it->second) {
        if (instance_count.second > max_count && instance_count.first != 0u &&
            assigned_instances.find(instance_count.first) ==
                assigned_instances.end()) {
          ObjSegConfidence frames_count = 0;
          auto label_count_it = label_frames_count_.find(label);
          if (label_count_it != label_frames_count_.end()) {
            frames_count = label_count_it->second;
          }
          if (instance_count.second >
              count_threshold_factor *
                  (float)(frames_count - instance_count.second)) {
            instance_label = instance_count.first;
            max_count = instance_count.second;
          }
        }
      }
    } else {
      // LOG(ERROR) << "No semantic class for label?";
    }
    // TODO(margaritaG): handle this remeshing!!
    // auto prev_instance_it = label_instance_map_.find(label);
    //   if (prev_instance_it != label_instance_map_.end()) {
    //     if (prev_instance_it->second != instance_label) {
    //       *remesh_ptr_ = true;
    //     }
    //   }
    //   label_instance_map_[label] = instance_label;
    //   return instance_label;
  }

  // LOG(INFO) << "  getInstanceLabel Return ";
  return instance_label;
}





SemanticLabel SemanticInstanceLabelFusion::getSemanticLabel(
    const Label& label) const {
  SemanticLabel semantic_label = BackgroundSemLabel;

  if(inst_association_ == 2) //2-use_inst_sem_confidence_
  {
    InstanceLabel inst_label = getInstanceLabel(label);
    if(inst_label != BackgroundLabel)
    {
      SegmentObserveConfidence max_sem_confidence = 0.;
      auto label_it = label_instance_class_count_.find(label);
      if (label_it != label_instance_class_count_.end())
      {
          auto inst_it = label_it->second.find(inst_label);
          if (inst_it != label_it->second.end())
          {
              for (auto const& sem_count : inst_it->second)
              {
                  if (sem_count.second > max_sem_confidence && sem_count.first != BackgroundSemLabel )
                  {
                    semantic_label = sem_count.first;
                    max_sem_confidence = sem_count.second;
                      
                  }
              }
          }                
      }
    }
  }
  else if (inst_association_ == 3) // 3-seg_graph
  {
    semantic_label = seg_graph_ptr_->getSemantic(label);
  }
  
  else  // 0-Ori; 1-use_sem_inst_confidence; 4 -use_sem_inst_confidence as initial guess and then use seg-graph to refine it 
  {
    if (getInstanceLabel(label) == BackgroundLabel) {
      return semantic_label;
    }
    SegmentObserveConfidence max_count = 0;
    auto label_it = label_class_count_.find(label);
    if (label_it != label_class_count_.end()) {
      for (auto const& class_count : label_it->second) {
        if (class_count.second > max_count &&
            class_count.first != BackgroundSemLabel) {
          semantic_label = class_count.first;
          max_count = class_count.second;
        }
      }
    }
  }

  // LOG(INFO) << "  getSemanticLabel Return ";
  return semantic_label;
}

void SemanticInstanceLabelFusion::logLabelSemanticInstanceCountInfo(const Label& label) const
{

  if(inst_association_ == 0) //0-Ori
  {
    auto label_inst_it = label_instance_count_.find(label);
    auto label_sem_it = label_class_count_.find(label);
    for(auto inst_it=label_inst_it->second.begin(); inst_it!=label_inst_it->second.end();
      inst_it++)
    {
      LOG(ERROR) << "   Label:" << int(label) << " Inst:"<< int(inst_it->first)
        <<" Count:" << inst_it->second;
    }
    for(auto sem_it=label_sem_it->second.begin(); sem_it!=label_sem_it->second.end();
      sem_it++)
    {
      LOG(ERROR) << "   Label:" << int(label) << " Sem:"<< int(sem_it->first)
        <<" Count:" << sem_it->second;
    }
  }
  if(inst_association_ == 1) //1-use_sem_inst_confidence_
  {
    auto label_it = label_class_instance_count_.find(label);
    for(auto sem_it=label_it->second.begin(); sem_it!=label_it->second.end(); sem_it++)
      for(auto inst_it=sem_it->second.begin(); inst_it!=sem_it->second.end(); inst_it++)
      {
        LOG(ERROR) << "   Label:" << int(label_it->first) << " Sem:"<< int(sem_it->first)
          << " Inst:"<< int(inst_it->first)<<" Count:" << inst_it->second;
      }
  }
  else if(inst_association_ == 2) //2-use_inst_sem_confidence_
  {
    auto label_it = label_instance_class_count_.find(label);
    for(auto inst_it=label_it->second.begin(); inst_it!=label_it->second.end(); inst_it++)
      for(auto sem_it=inst_it->second.begin(); sem_it!=inst_it->second.end(); sem_it++)
      {
        LOG(ERROR) << "   Label:" << int(label_it->first) << " Inst:"<< int(inst_it->first)
          << " Sem:"<< int(sem_it->first)<<" Count:" << sem_it->second;
      }
  }  
  // else if(inst_association_ == 2)
  // {

  // }
  else // use 0-Ori as default
  {
    auto label_inst_it = label_instance_count_.find(label);
    auto label_sem_it = label_class_count_.find(label);
    for(auto inst_it=label_inst_it->second.begin(); inst_it!=label_inst_it->second.end();
      inst_it++)
    {
      LOG(ERROR) << "   Label:" << int(label) << " Inst:"<< int(inst_it->first)
        <<" Count:" << inst_it->second;
    }
    for(auto sem_it=label_sem_it->second.begin(); sem_it!=label_sem_it->second.end();
      sem_it++)
    {
      LOG(ERROR) << "   Label:" << int(label) << " Sem:"<< int(sem_it->first)
        <<" Count:" << sem_it->second;
    }
  }
}

void SemanticInstanceLabelFusion::mergeLabels(const Label& label_a, const Label& label_b)
{
  LOG(INFO) << "start SemanticInstanceLabelFusion::mergeLabels; label_frames_count_.size(): " << label_frames_count_.size();
  if(inst_association_ == 3 || inst_association_ == 4) // SegGraph
  {
    seg_graph_ptr_->mergeLabels(label_a, label_b);
  }
  LOG(INFO) << "after seg_graph_ptr_->mergeLabels; label_frames_count_.size(): " << label_frames_count_.size();
  if (inst_association_ == 1 || inst_association_ == 4) // LSI
  {
    LOG(INFO) << "1; label_frames_count_.size(): " << label_frames_count_.size();
    // merge and delete label-class count
    const bool label_a_exist_LS = ( label_class_count_.find(label_a)!=label_class_count_.end() );
    const bool label_b_exist_LS = ( label_class_count_.find(label_b)!=label_class_count_.end() );
    if( label_a_exist_LS && label_b_exist_LS)
    {
        auto label_a_it_LS = label_class_count_.find(label_a);
        auto label_b_it_LS = label_class_count_.find(label_b);
        for(auto sem_it = label_b_it_LS->second.begin(); 
          sem_it != label_b_it_LS->second.end(); sem_it++)
        {
          const SemanticLabel semantic_label = sem_it->first;
          auto label_a_sem_it = label_a_it_LS->second.find(semantic_label);
          if(label_a_sem_it != label_a_it_LS->second.end())
          {
            label_a_sem_it->second = std::max(label_a_sem_it->second, sem_it->second);
          }
          else
          {
            label_a_it_LS->second.emplace(semantic_label, sem_it->second);
          }
        }
        label_class_count_.erase(label_b);
    }
    LOG(INFO) << "2; label_frames_count_.size(): " << label_frames_count_.size();
    // merge and delete label-class-instance count
    auto label_a_it = label_class_instance_count_.find(label_a);
    auto label_b_it = label_class_instance_count_.find(label_b);
    const bool label_a_exist_LSI = ( label_a_it != label_class_instance_count_.end() );
    const bool label_b_exist_LSI = ( label_b_it != label_class_instance_count_.end() );
    LOG(INFO) << "3; label_frames_count_.size(): " << label_frames_count_.size();
    if(label_a_exist_LSI && label_b_exist_LSI)
    {
      // for all internal and external edges, use maximum between two
      for(auto label_b_sem_it = label_b_it->second.begin(); 
            label_b_sem_it != label_b_it->second.end(); label_b_sem_it++)
      {
        const SemanticLabel semantic_label_b = label_b_sem_it->first;

        auto label_a_sem_it = label_a_it->second.find(semantic_label_b);
        if(label_a_sem_it != label_a_it->second.end())
        {
          for(auto label_b_sem_inst_it = label_b_sem_it->second.begin(); 
            label_b_sem_inst_it != label_b_sem_it->second.end(); label_b_sem_inst_it++)
          {
            const InstanceLabel sem_inst_label = label_b_sem_inst_it->first;
            auto label_a_sem_inst_it = label_a_sem_it->second.find(sem_inst_label);
            if(label_a_sem_inst_it != label_a_sem_it->second.end())
            {
              ObjSegConfidence max_confidence = std::max(label_a_sem_inst_it->second, 
                label_b_sem_inst_it->second);
              label_a_sem_inst_it->second = max_confidence;
            }
            else
            {
              label_b_sem_it->second.emplace(sem_inst_label, label_b_sem_inst_it->second);
            }
          }
        }
        else
        {
          label_a_it->second.emplace(semantic_label_b, label_b_sem_it->second);
        }  
      }
      LOG(INFO) << "4; label_frames_count_.size(): " << label_frames_count_.size();
      label_class_instance_count_.erase(label_b);
      LOG(INFO) << "5; label_frames_count_.size(): " << label_frames_count_.size();
    }
  }
  LOG(INFO) << "after LSI mergeLabels; label_frames_count_.size(): " << label_frames_count_.size();
  // merge and delete label_frame_count of old label
  if(label_frames_count_.find(label_b)!=label_frames_count_.end() &&
    label_frames_count_.find(label_a)!=label_frames_count_.end())
  {
    label_frames_count_.find(label_a)->second = std::max(label_frames_count_.find(label_a)->second, 
      label_frames_count_.find(label_b)->second);

    LOG(INFO) << "label_frames_count_.size(): " << label_frames_count_.size();
    // label_frames_count_.erase(label_b);
    LOG(INFO) << "label_frames_count_.size() after erase: " << label_frames_count_.size();
    LOG(INFO) << "  Erasing from label_frame_count label " << std::setfill('0') << std::setw(5) << int(label_b);
    LOG(INFO) << "  label_a " << int(label_a) << " label_b " << int(label_b);
  }

  LOG(INFO) << "after SemanticInstanceLabelFusion::mergeLabels; label_frames_count_.size(): " << label_frames_count_.size();
}



}  // namespace voxblox
