#include "consistent_mapping/label_tsdf_confidence_integrator.h"

#include <algorithm>

namespace voxblox {

void LabelTsdfConfidenceIntegrator::integratePointCloudConfidence(
    const Transformation& T_G_C, const Pointcloud& points_C,
    const std::vector<GeometricConfidence>& geometric_confidence,
    const SegSegConfidence& label_confidence, const Colors& colors,
    const Label& label, const bool freespace_points) {
  CHECK_EQ(points_C.size(), colors.size());
  // CHECK_EQ(points_C.size(), geometric_confidence.size());
  CHECK_GE(points_C.size(), 0u);

  // Pre-compute a list of unique voxels to end on.
  // Create a hashmap: VOXEL INDEX -> index in original cloud.
  LongIndexHashMapType<AlignedVector<size_t>>::type voxel_map;
  // This is a hash map (same as above) to all the indices that need to be
  // cleared.
  LongIndexHashMapType<AlignedVector<size_t>>::type clear_map;

  std::unique_ptr<ThreadSafeIndex> index_getter(
      ThreadSafeIndexFactory::get(config_.integration_order_mode, points_C));

  bundleRays(T_G_C, points_C, freespace_points, index_getter.get(), &voxel_map,
             &clear_map);

  integrateRaysConfidence(
      T_G_C, points_C, geometric_confidence, label_confidence, colors, label,
      config_.enable_anti_grazing, false, voxel_map, clear_map);

  integrateRaysConfidence(
      T_G_C, points_C, geometric_confidence, label_confidence, colors, label,
      config_.enable_anti_grazing, true, voxel_map, clear_map);
}

void LabelTsdfConfidenceIntegrator::integrateRaysConfidence(
    const Transformation& T_G_C, const Pointcloud& points_C,
    const std::vector<GeometricConfidence>& geometric_confidence,
    const SegSegConfidence& label_confidence, const Colors& colors,
    const Label& label, const bool enable_anti_grazing, const bool clearing_ray,
    const VoxelMap& voxel_map, const VoxelMap& clear_map) {
  const Point& origin = T_G_C.getPosition();

  // if only 1 thread just do function call, otherwise spawn threads
  if (config_.integrator_threads == 1u) {
    constexpr size_t thread_idx = 0u;
    integrateVoxelsConfidence(
        T_G_C, points_C, geometric_confidence, label_confidence, colors, label,
        enable_anti_grazing, clearing_ray, voxel_map, clear_map, thread_idx);
  } 
  else {
    std::list<std::thread> integration_threads;
    for (size_t i = 0u; i < config_.integrator_threads; ++i) {
      integration_threads.emplace_back(
          &LabelTsdfConfidenceIntegrator::integrateVoxelsConfidence, this,
          T_G_C, std::cref(points_C), std::cref(geometric_confidence),
          std::cref(label_confidence), std::cref(colors), label,
          enable_anti_grazing, clearing_ray, std::cref(voxel_map),
          std::cref(clear_map), i);
    }

    for (std::thread& thread : integration_threads) {
      thread.join();
    }
  }

  updateLayerWithStoredBlocks();
  updateLabelLayerWithStoredBlocks();
}

void LabelTsdfConfidenceIntegrator::integrateVoxelsConfidence(
    const Transformation& T_G_C, const Pointcloud& points_C,
    const std::vector<GeometricConfidence>& geometric_confidence,
    const SegSegConfidence& label_confidence, const Colors& colors,
    const Label& label, const bool enable_anti_grazing, const bool clearing_ray,
    const VoxelMap& voxel_map, const VoxelMap& clear_map,
    const size_t thread_idx
  ) {
  VoxelMap::const_iterator it;
  size_t map_size;
  if (clearing_ray) {
    it = clear_map.begin();
    map_size = clear_map.size();
  } else {
    it = voxel_map.begin();
    map_size = voxel_map.size();
  }
  for (size_t i = 0u; i < map_size; ++i) {
    if (((i + thread_idx + 1) % config_.integrator_threads) == 0u) {
      integrateVoxelConfidence(
          T_G_C, points_C, geometric_confidence, label_confidence, colors,
          label, enable_anti_grazing, clearing_ray, *it, voxel_map);
    }
    ++it;
  }
}

void LabelTsdfConfidenceIntegrator::integrateVoxelConfidence(
    const Transformation& T_G_C, const Pointcloud& points_C,
    const std::vector<GeometricConfidence>& geometric_confidence,
    const SegSegConfidence& label_confidence, const Colors& colors,
    const Label& label, const bool enable_anti_grazing, const bool clearing_ray,
    const VoxelMapElement& global_voxel_idx_to_point_indices,
    const VoxelMap& voxel_map) {
  if (global_voxel_idx_to_point_indices.second.empty()) {
    return;
  }

  const Point& origin = T_G_C.getPosition();
  Color merged_color;
  Point merged_point_C = Point::Zero();
  FloatingPoint merged_weight = 0.0f;
  Label merged_label = label;
  LabelConfidence merged_label_confidence = 0;

  for (const size_t pt_idx : global_voxel_idx_to_point_indices.second) {
    const Point& point_C = points_C[pt_idx];
    const Color& color = colors[pt_idx];

    FloatingPoint point_weight = 0;
    if (use_geo_confidence_) {
      point_weight = geometric_confidence[pt_idx];
      FloatingPoint point_weight_original = getVoxelWeight(point_C);
      // LOG_EVERY_N(INFO, 10000)
      //     << "     PointConfidence: " << point_weight
      //     << " ; point_weight_original:  " << point_weight_original
      //     << "  ;  location:  " << point_C.x() << ", " << point_C.y() << ", "
      //     << point_C.z();
    } 
    else {
      point_weight = getVoxelWeight(point_C);
    }

    merged_point_C = (merged_point_C * merged_weight + point_C * point_weight) /
                     (merged_weight + point_weight);
    merged_color =
        Color::blendTwoColors(merged_color, merged_weight, color, point_weight);
    merged_weight += point_weight;
    // Assuming all the points of a segment pointcloud
    // are assigned the same label.
    if (!use_label_confidence_) {
      if (label_tsdf_config_.enable_confidence_weight_dropoff) {
        const FloatingPoint ray_distance = point_C.norm();
        merged_label_confidence = computeConfidenceWeight(ray_distance);
      } else {
        merged_label_confidence = 1;
      }
    } 
    else {
      // TODO combine with geometric confidence
      merged_label_confidence = label_confidence;
    // merged_label_confidence = 1u;
    }

    // only take first point when clearing
    if (clearing_ray)
      break;

  }

  const Point merged_point_G = T_G_C * merged_point_C;

  RayCaster ray_caster(origin, merged_point_G, clearing_ray,
                       config_.voxel_carving_enabled, config_.max_ray_length_m,
                       voxel_size_inv_, config_.default_truncation_distance);
  GlobalIndex global_voxel_idx;
  while (ray_caster.nextRayIndex(&global_voxel_idx)) {
    if (enable_anti_grazing) {
      // Check if this one is already the block hash map for this
      // insertion. Skip this to avoid grazing.
      if ((clearing_ray ||
           global_voxel_idx != global_voxel_idx_to_point_indices.first) &&
          voxel_map.find(global_voxel_idx) != voxel_map.end()) {
        continue;
      }
    }

    BlockIndex block_idx;

    Block<TsdfVoxel>::Ptr tsdf_block = nullptr;
    TsdfVoxel* tsdf_voxel = allocateStorageAndGetVoxelPtr(
        global_voxel_idx, &tsdf_block, &block_idx);

    updateTsdfVoxel(
      origin, merged_point_G, global_voxel_idx, 
      merged_color, merged_weight, tsdf_voxel
    );

    if (!config_.voxel_carving_enabled ||
        std::abs(tsdf_voxel->distance) <
            3 * config_.default_truncation_distance) {
      Block<LabelVoxel>::Ptr label_block = nullptr;
      LabelVoxel* label_voxel = allocateStorageAndGetLabelVoxelPtr(
          global_voxel_idx, &label_block, &block_idx);
      
      if (use_geo_confidence_)
        updateLabelVoxelConfidence(
          merged_point_G,
          merged_weight / global_voxel_idx_to_point_indices.second.size(),
          merged_label, label_voxel, 
          merged_label_confidence
        );
      else
        updateLabelVoxelConfidence(
          merged_point_G, 
          merged_label, label_voxel,
          merged_label_confidence
        );
    }
  }
}

void LabelTsdfConfidenceIntegrator::updateLabelVoxelConfidence(
    const Point& point_G, const Label& label, LabelVoxel* label_voxel,
    const LabelConfidence& confidence) {
  CHECK_NOTNULL(label_voxel);
  // Lookup the mutex that is responsible for this voxel and lock it.
  std::lock_guard<std::mutex> lock(mutexes_.get(
      getGridIndexFromPoint<GlobalIndex>(point_G, voxel_size_inv_)));
  CHECK_NOTNULL(label_voxel);

  // label_voxel->semantic_label = semantic_label;
  Label previous_label = label_voxel->label;
  // addVoxelLabelConfidenceSmart(label, confidence, label_voxel);
  addVoxelLabelConfidence(label, confidence, label_voxel);
  updateVoxelLabelAndConfidence(label_voxel, label);
  Label new_label = label_voxel->label;
  if (new_label != previous_label) {
    // Both of the segments corresponding to the two labels are
    // updated, one gains a voxel, one loses a voxel.
    std::lock_guard<std::mutex> lock(updated_labels_mutex_);

    updated_labels_.insert(new_label);
    changeLabelCount(new_label, 1);

    if (previous_label != 0u) {
      updated_labels_.insert(previous_label);
      changeLabelCount(previous_label, -1);
    }

    if (*highest_label_ptr_ < new_label) {
      *highest_label_ptr_ = new_label;
    }
  }
}

void LabelTsdfConfidenceIntegrator::updateLabelVoxelConfidence(
    const Point& point_G, const FloatingPoint& point_weight, const Label& label,
    LabelVoxel* label_voxel, const LabelConfidence& confidence) {
  CHECK_NOTNULL(label_voxel);
  // Lookup the mutex that is responsible for this voxel and lock it.
  std::lock_guard<std::mutex> lock(mutexes_.get(
      getGridIndexFromPoint<GlobalIndex>(point_G, voxel_size_inv_)));
  CHECK_NOTNULL(label_voxel);

  // label_voxel->semantic_label = semantic_label;
  Label previous_label = label_voxel->label;
  // addVoxelLabelConfidenceSmart(label, confidence, label_voxel);
  addVoxelLabelConfidence(label, confidence * point_weight, label_voxel);
  updateVoxelLabelAndConfidence(label_voxel, label);
  Label new_label = label_voxel->label;
  if (new_label != previous_label) {
    // Both of the segments corresponding to the two labels are
    // updated, one gains a voxel, one loses a voxel.
    std::lock_guard<std::mutex> lock(updated_labels_mutex_);

    updated_labels_.insert(new_label);
    changeLabelCount(new_label, 1);

    if (previous_label != 0u) {
      updated_labels_.insert(previous_label);
      changeLabelCount(previous_label, -1);
    }

    if (*highest_label_ptr_ < new_label) {
      *highest_label_ptr_ = new_label;
    }
  }
}

void LabelTsdfConfidenceIntegrator::addVoxelLabelConfidenceSmart(
    const Label& label, const LabelConfidence& confidence,
    LabelVoxel* label_voxel) {
  CHECK_NOTNULL(label_voxel);

  LabelCount* label_count_same = nullptr;
  LabelCount* label_count_zero = nullptr;
  LabelCount* label_count_min = nullptr;
  LabelConfidence confidence_min = 10000;

  bool updated = false;

  for (LabelCount& label_count : label_voxel->label_count) {
    if (label_count.label == label) {
      // Label already observed in this voxel.
      label_count_same = &label_count;
      break;
    }
    if (label_count.label == 0u) {
      label_count_zero = &label_count;
    }
    if (confidence_min > label_count.label_confidence) {
      confidence_min = label_count.label_confidence;
      if (confidence_min > confidence) label_count_min = &label_count;
    }
  }

  if (label_count_same) {
    label_count_same->label_confidence += confidence;
    updated = true;
  } else if (label_count_zero) {
    label_count_zero->label = label;
    label_count_zero->label_confidence = confidence;
    updated = true;
  } else if (label_count_min) {
    label_count_min->label = label;
    label_count_min->label_confidence = confidence;
    updated = true;
  }
}

// TODO-SegSegMatch currently the function is the same as original
// computeSegmentLabelCandidates, use the absolute overlap as criteria to
// determine the seg-seg match, later maybe confidence can be involved here
void LabelTsdfConfidenceIntegrator::computeSegmentLabelCandidatesConfidence(
    Segment* segment,
    std::map<Label, std::map<Segment*, SegSegConfidence>>*
        candidates_confidence,
    std::map<Segment*, std::vector<Label>>* segment_merge_candidates,
    const std::set<Label>& assigned_labels) {
  CHECK_NOTNULL(segment);
  CHECK_NOTNULL(candidates_confidence);
  CHECK_NOTNULL(segment_merge_candidates);
  // Flag to check whether there exists at least one label candidate.
  bool candidate_label_exists = false;
  const int segment_points_size = segment->points_C_.size();
  std::unordered_set<Label> merge_candidate_labels;
  // whether consider semantic consistent
  const bool consider_semantic_consistent = (data_association_ == 3);
  SemanticLabel seg_semantic = segment->semantic_label_;

  for (const Point& point_C : segment->points_C_) {
    const Point point_G = segment->T_G_C_ * point_C;
    // Get the corresponding voxel by 3D position in world frame.
    Layer<LabelVoxel>::BlockType::ConstPtr label_block_ptr =
        label_layer_->getBlockPtrByCoordinates(point_G);
    // Get the corresponding voxel by 3D position in world frame.
    Layer<TsdfVoxel>::BlockType::ConstPtr tsdf_block_ptr =
        layer_->getBlockPtrByCoordinates(point_G);

    if (label_block_ptr != nullptr) {
      const LabelVoxel& label_voxel =
          label_block_ptr->getVoxelByCoordinates(point_G);
      const TsdfVoxel& tsdf_voxel =
          tsdf_block_ptr->getVoxelByCoordinates(point_G);
      Label label = 0u;
      label = getNextUnassignedLabel(label_voxel, assigned_labels);
      if (label != 0u && std::abs(tsdf_voxel.distance) <
              label_tsdf_config_.label_propagation_td_factor * voxel_size_) {
        // Do not consider allocated but unobserved voxels with label 0.
        candidate_label_exists = true;

        // false in the args
        if (use_label_confidence_) {
          SegSegConfidence voxel_label_prob =
              label_voxel.getLabelProbability(label);
          increaseLabelConfidenceForSegment(
              segment, label, segment_points_size, voxel_label_prob,
              candidates_confidence, &merge_candidate_labels);
        } 
        else {
          // only if data association is 3
          if (consider_semantic_consistent) {
            SemanticLabel label_semantic =
                semantic_instance_label_fusion_ptr_->getSemanticLabel(label);
              
            bool is_semantic_consistent = (label_semantic == seg_semantic);
            // if semantic of segment and label is consistent, add 1.0
            if (is_semantic_consistent) {  
              increaseLabelConfidenceForSegment(
                  segment, label, segment_points_size, 1.0,
                  candidates_confidence, &merge_candidate_labels);
            } 
            // if semantic of segment and label is inconsistent, add 0.75
            else {  
              increaseLabelConfidenceForSegment(
                  segment, label, segment_points_size, 0.75,
                  candidates_confidence, &merge_candidate_labels);
            }
          } else {
            increaseLabelCountForSegment(
              segment, label, segment_points_size,
              candidates_confidence, &merge_candidate_labels);
          }
        }
      }
    }
  }

  if (label_tsdf_config_.enable_pairwise_confidence_merging) {
    std::vector<Label> merge_candidates;
    std::copy(merge_candidate_labels.begin(), merge_candidate_labels.end(),
              std::back_inserter(merge_candidates));
    (*segment_merge_candidates)[segment] = merge_candidates;
  }
  // Previously unobserved segment gets an unseen label.
  if (!candidate_label_exists) {
    Label fresh_label = getFreshLabel();
    std::map<Segment*, SegSegConfidence> map;
    map.insert(std::pair<Segment*, SegSegConfidence>(
        segment, segment->points_C_.size()));
    candidates_confidence->insert(
        std::pair<Label, std::map<Segment*, SegSegConfidence>>(fresh_label, map));
    // LOG(INFO) << "  seg size. " << segment->points_C_.size() << " - addr " <<
    // segment << "  GET NEW Label: " <<  fresh_label;
  }
}

// TODO-SegSegMatch
void LabelTsdfConfidenceIntegrator::increaseLabelCountForSegment(
    Segment* segment, const Label& label, const int segment_points_count,
    std::map<Label, std::map<Segment*, SegSegConfidence>>*
        candidates_confidence,
    std::unordered_set<Label>* merge_candidate_labels) {
  CHECK_NOTNULL(segment);
  CHECK_NOTNULL(candidates_confidence);
  CHECK_NOTNULL(merge_candidate_labels);

  auto label_it = candidates_confidence->find(label);
  if (label_it != candidates_confidence->end()) {
    auto segment_it = label_it->second.find(segment);
    if (segment_it != label_it->second.end()) {
      segment_it->second += 1.0;

      bool consider_seg_size =
          (data_association_ == 2) || (data_association_ == 3);
      bool size_large_enough =
          (!consider_seg_size) || (segment->points_C_.size() > 1000);
      if (label_tsdf_config_.enable_pairwise_confidence_merging &&
          size_large_enough) {
        checkForSegmentLabelMergeCandidate(label, segment_it->second,
                                           segment_points_count,
                                           merge_candidate_labels);
      }
    } else {
      label_it->second.emplace(segment, 1.0);
    }
  } else {
    std::map<Segment*, SegSegConfidence> segment_points_count;
    segment_points_count.emplace(segment, 1.0);
    candidates_confidence->emplace(label, segment_points_count);
  }
}


void LabelTsdfConfidenceIntegrator::increaseLabelConfidenceForSegment(
    Segment* segment, const Label& label, const int& segment_points_size,
    const SegSegConfidence confidence,
    std::map<Label, std::map<Segment*, SegSegConfidence>>*
        candidates_confidence,
    std::unordered_set<Label>* merge_candidate_labels) {
  CHECK_NOTNULL(segment);
  CHECK_NOTNULL(candidates_confidence);
  CHECK_NOTNULL(merge_candidate_labels);

  auto label_it = candidates_confidence->find(label);
  if (label_it != candidates_confidence->end()) {
    auto segment_it = label_it->second.find(segment);
    if (segment_it != label_it->second.end()) {
      segment_it->second += confidence;

      if (label_tsdf_config_.enable_pairwise_confidence_merging) {
        checkForSegmentLabelMergeCandidate(label, segment_it->second,
                                           segment_points_size,
                                           merge_candidate_labels);
      }
    } else {
      label_it->second.emplace(segment, confidence);
    }
  } else {
    std::map<Segment*, SegSegConfidence> segment_points_count;
    segment_points_count.emplace(segment, confidence);
    candidates_confidence->emplace(label, segment_points_count);
  }
}

void LabelTsdfConfidenceIntegrator::decideLabelPointCloudsConfidence(
    std::vector<Segment*>* segments_to_integrate,
    std::map<Label, std::map<Segment*, SegSegConfidence>>*
        candidates_confidence,
    std::map<Segment*, std::vector<Label>>* segment_merge_candidates
  ) {
  CHECK_NOTNULL(segments_to_integrate);
  CHECK_NOTNULL(candidates_confidence);
  CHECK_NOTNULL(segment_merge_candidates);
  std::set<Label> assigned_labels;
  std::set<Segment*, SegmentConfidence::PtrCompare> labeled_segments;
  std::pair<Segment*, Label> pair;
  std::set<InstanceLabel> assigned_instances;

  SegSegConfidence cur_pair_confidence = 0;
  while (getNextSegmentLabelPairWithConfidence(
      labeled_segments, &assigned_labels, candidates_confidence,
      segment_merge_candidates, &pair, &cur_pair_confidence)
    ) 
  {
    Segment* segment = pair.first;
    CHECK_NOTNULL(segment);
    Label& label = pair.second;
    segment->label_ = label;
    dynamic_cast<SegmentConfidence*>(segment)->seg_label_confidence_ = cur_pair_confidence;
    labeled_segments.insert(segment);
    candidates_confidence->erase(label);

    // LOG(INFO) << " Found pair - Seg size "<< segment->points_C_.size() 
    //   << " - confidence "<< (cur_pair_confidence) << "- Label " << label;
  }

  for (auto merge_candidates : *segment_merge_candidates) {
    // increasePairwiseConfidenceCountSemantics(merge_candidates.second);
    increasePairwiseConfidenceCount(merge_candidates.second);
  }
  // For every segment that didn't get a label because
  // its label counts were too few, assign it an unseen label.
  for (auto segment_it = segments_to_integrate->begin();
       segment_it != segments_to_integrate->end(); ++segment_it) {
    if (labeled_segments.find(*segment_it) == labeled_segments.end()) {
      Label fresh = getFreshLabel();
      (*segment_it)->label_ = fresh;
      dynamic_cast<SegmentConfidence*>(*segment_it)->seg_label_confidence_ = 1;
      labeled_segments.insert(*segment_it);
      // LOG(INFO) << " Add pair - Seg size "<< segment->points_C_.size() 
      //   << " - confidence "<< (cur_pair_confidence) << "- Label " << label;
    }
  }
  updateInstanceConfidence(&labeled_segments);
}

void LabelTsdfConfidenceIntegrator::updateInstanceConfidence(
    std::set<Segment*, SegmentConfidence::PtrCompare>* labeled_segments) {
  if (!label_tsdf_config_.enable_semantic_instance_segmentation)
    return;
  
  if (inst_association_ == 1) {
    // LOG(INFO) << "  updateLabelClassInstanceConfidence";
    updateLabelClassInstanceConfidence(labeled_segments);
  } 
  else if (inst_association_ == 2) {
    // LOG(INFO) << "  updateLabelInstanceClassConfidence";
    updateLabelInstanceClassConfidence(labeled_segments);
  } 
  else if (inst_association_ == 3) {
    // LOG(INFO) << "  updateSegmentGraph";
    IncreaseSegGraphConfidence(labeled_segments);
  } 
  else if (inst_association_ == 4) {
    // LOG(INFO) << "  updateLabelInstanceClassConfidence&SegmentGraph";
    updateLabelClassInstanceConfidence(labeled_segments);
    IncreaseSegGraphConfidence(labeled_segments);
  }
  else if (inst_association_ == 6 || inst_association_ == 7) {
    // update label-semantic-instance during raycast of panoptic mask
  }
  else {
    // original associaiton
    IncreaseLabelInstanceMapCount(labeled_segments);
  }
}











void LabelTsdfConfidenceIntegrator::increasePairwiseConfidenceCountSemantics(
    const std::vector<Label>& merge_candidates) {
  for (size_t i = 0u; i < merge_candidates.size(); ++i) {
    Label new_label = merge_candidates[i];
    SemanticLabel new_label_semantic =
        semantic_instance_label_fusion_ptr_->getSemanticLabel(new_label);
    for (size_t j = i + 1; j < merge_candidates.size(); ++j) {
      Label old_label = merge_candidates[j];
      SemanticLabel old_label_semantic =
          semantic_instance_label_fusion_ptr_->getSemanticLabel(old_label);
      if (new_label_semantic != old_label_semantic) continue;
      if (new_label != old_label) {
        // Pairs consist of (new_label, old_label) where new_label <
        // old_label.
        if (new_label > old_label) {
          Label tmp = old_label;
          old_label = new_label;
          new_label = tmp;
        }
        // For every pair of labels from the merge candidates
        // set or increase their pairwise confidence.
        LLMapIt new_label_it = pairwise_confidence_.find(new_label);
        if (new_label_it != pairwise_confidence_.end()) {
          LMapIt old_label_it = new_label_it->second.find(old_label);
          if (old_label_it != new_label_it->second.end()) {
            ++old_label_it->second;
          } else {
            new_label_it->second.emplace(old_label, 1);
          }
        } else {
          LMap confidence_pair;
          confidence_pair.emplace(old_label, 1);
          pairwise_confidence_.emplace(new_label, confidence_pair);
        }
      }
    }
  }
}











void LabelTsdfConfidenceIntegrator::IncreaseLabelInstanceMapCount(
    std::set<Segment*, SegmentConfidence::PtrCompare>* labeled_segments) {
  std::set<InstanceLabel> assigned_instances;
  // Instance stuff.
  // size_t labeled_seg_count=0;
  // LOG(INFO) << "  LabelInstanceMapCount: ";
  for (auto segment_it = labeled_segments->begin();
       segment_it != labeled_segments->end(); ++segment_it) {
    // LOG(INFO) << "      Add InstanceLabel Confi with Seg size. "<<
    // (*segment_it)->points_C_.size();
    // << " seg addr: " << (*segment_it);
    Label label = (*segment_it)->label_;
    InstanceLabel matched_instance_label = BackgroundLabel;

    if ((*segment_it)->points_C_.size() > 0u) {
      semantic_instance_label_fusion_ptr_->increaseLabelFramesCount(label);
    }

    // Loop through all the segments.
    if ((*segment_it)->instance_label_ != BackgroundLabel) {
      // It's a segment with a current frame instance.
      auto global_instance_it =
          current_to_global_instance_map_.find((*segment_it)->instance_label_);
      if (global_instance_it != current_to_global_instance_map_.end()) {
        // If current frame instance maps to a global instance, use it.
        matched_instance_label = global_instance_it->second;
        if (matched_instance_label != BackgroundLabel)
          LOG(INFO) << "      CurMapIncrease: Label: " << std::setfill('0')
                    << std::setw(5) << int(label)
                    << " Instance: " << std::setfill('0') << std::setw(5)
                    << int(matched_instance_label);
      } 
      else {
        // Current frame instance doesn't map to any global instance.
        // Get the global instance with max count.
        InstanceLabel instance_label =
            semantic_instance_label_fusion_ptr_->getInstanceLabel(
                label, assigned_instances);
        if (instance_label != BackgroundLabel) {
          current_to_global_instance_map_.emplace(
              (*segment_it)->instance_label_, instance_label);
          matched_instance_label = instance_label;
          if (matched_instance_label != BackgroundLabel)
            LOG(INFO) << "      GloMapIncrease: Label: " << std::setfill('0')
                      << std::setw(5) << int(label)
                      << " Instance: " << std::setfill('0') << std::setw(5)
                      << int(matched_instance_label);
        } else {
          // Create new global instance.
          InstanceLabel fresh_instance = getFreshInstance();
          current_to_global_instance_map_.emplace(
              (*segment_it)->instance_label_, fresh_instance);
          matched_instance_label = fresh_instance;
          LOG(INFO) << "      FreshIncrease: Label: " << std::setfill('0')
                    << std::setw(5) << int(label)
                    << " Instance: " << std::setfill('0') << std::setw(5)
                    << int(matched_instance_label);
        }
      }
      semantic_instance_label_fusion_ptr_->increaseLabelInstanceCount(
          label, matched_instance_label);
      assigned_instances.emplace(matched_instance_label);
      semantic_instance_label_fusion_ptr_->increaseLabelClassCount(
          label, (*segment_it)->semantic_label_);
    } else {
      // It's a segment with no instance prediction in the current frame.
      // Get the global instance it maps to, and set it as assigned.
      InstanceLabel instance_label =
          semantic_instance_label_fusion_ptr_->getInstanceLabel(label);
      if (instance_label != BackgroundLabel) {
        assigned_instances.emplace(instance_label);
      }
      matched_instance_label = instance_label;
    }
    // LOG(INFO) << "      Seg size. "<< (*segment_it)->points_C_.size() << ";
    // Label: " << (*segment_it)->label_
    //     << ";  CInstance: " << (*segment_it)->instance_label_ << ";
    //     GInstance: " << matched_instance_label;
    // labeled_seg_count++;
  }
}











void LabelTsdfConfidenceIntegrator::IncreaseSegGraphConfidence(
    std::set<Segment*, SegmentConfidence::PtrCompare>* labeled_segments) {
  // for(auto segment_it=labeled_segments->begin();
  // segment_it!=labeled_segments->end();
  //     segment_it++)
  // {
  //     LOG(INFO) << "Seg size: " << (*segment_it)->points_C_.size() << ";
  //     matched label: "
  //         << (*segment_it)->label_;
  // }

  // aggregate all observed segments of same instance and 
  // insert them at once to a seg-graph
  LSet assigned_labels;
  for (auto segment_it = labeled_segments->begin();
       segment_it != labeled_segments->end(); segment_it++) {
    const Label label = (*segment_it)->label_;
    const bool is_assigned =
        (assigned_labels.find(label) != assigned_labels.end());
    const bool is_background =
        (*segment_it)->instance_label_ == BackgroundLabel;
    const bool is_empty = ((*segment_it)->points_C_.size() <= 0u);
    if (is_assigned || is_background || is_empty)
      continue;

    assigned_labels.insert(label);

    // create a new vector for this new instance
    std::vector<Label> labels_vector;
    labels_vector.push_back(label);
    std::set<Segment*, SegmentConfidence::PtrCompare> inst_segs;
    inst_segs.insert(*segment_it);

    const InstanceLabel inst_label = (*segment_it)->instance_label_;
    const SemanticLabel semantic_label = (*segment_it)->semantic_label_;

    for (auto segment_check_it = std::next(segment_it);
          segment_check_it != labeled_segments->end(); segment_check_it++) 
    {
      const Label label_to_check = (*segment_check_it)->label_;

      const bool is_checked_label_assigned =
          (assigned_labels.find(label_to_check) != assigned_labels.end());
      const bool is_checked_label_empty =
          ((*segment_check_it)->points_C_.size() <= 0u);
      if (is_checked_label_assigned || is_checked_label_empty)
        continue;
      
      const InstanceLabel inst_label_check = (*segment_check_it)->instance_label_;
      const SemanticLabel semantic_label_check = (*segment_check_it)->semantic_label_;
      const bool is_same_inst = (inst_label_check == inst_label);
      const bool is_same_semantic = (semantic_label_check == semantic_label);
      if (is_same_inst && is_same_semantic) {
        const Label label_check = (*segment_check_it)->label_;
        labels_vector.push_back(label_check);
        assigned_labels.insert(label_check);
        inst_segs.insert(*segment_check_it);
      }
    }

    // construct fonfidence map
    LLConfidenceMap label_label_confidence_map;
    bool is_thing = dynamic_cast<SegmentConfidence*>(*(inst_segs.begin()))->is_thing_;
    ContructSegGraphInstConfidenceMap(&inst_segs, &label_label_confidence_map, is_thing);
    semantic_instance_label_fusion_ptr_->insertInstanceToSegGraph(
        labels_vector, label_label_confidence_map, semantic_label, is_thing);
    // log inserted instance
    // LOG(INFO) << "      TempInstance: " << int(inst_label) << "
    // TempSemantics: " <<
    //      int(semantic_label);
    // for(Label& label_in_inst:labels_vector)
    //     LOG(INFO)<<  "        TempLabel: " << std::setfill('0') <<
    //     std::setw(5)  << int(label_in_inst);
  }
}










void LabelTsdfConfidenceIntegrator::ContructSegGraphInstConfidenceMap(
    std::set<Segment*, SegmentConfidence::PtrCompare>* inst_segments,
    LLConfidenceMap* label_label_confidence_map, bool is_thing) {
  CHECK_GT(inst_segments->size(), 0);
  // if seg_graph_confidence_, calculate distance info
  Distance dist_sq_median;
  std::vector<Distance> dist_sq_segs;
  std::map<Label, std::map<Label, Distance>> dist_sq_map_labels;
  if (seg_graph_confidence_ == 3 && is_thing) {
    int num_segs = inst_segments->size();
    dist_sq_segs.reserve(num_segs * (num_segs - 1) / 2);

    for (auto segment_a_it = inst_segments->begin();
         segment_a_it != inst_segments->end(); segment_a_it++) 
    {
      Label label_a = (*segment_a_it)->label_;
      for (auto segment_b_it = std::next(segment_a_it);
           segment_b_it != inst_segments->end(); segment_b_it++) 
      {
        Label label_b = (*segment_b_it)->label_;

        const cv::Mat& b_box_seg_a = dynamic_cast<SegmentConfidence*>(*segment_a_it)->b_box_;
        const cv::Mat& b_box_seg_b = dynamic_cast<SegmentConfidence*>(*segment_b_it)->b_box_;
        Distance dist_sq = 1000.;

        // get dist_sq of closest points between b_box_a and b_box_b
        // by iterate through all points sampled on all edges
        for (size_t p_a_i = 0; p_a_i < b_box_seg_a.rows; p_a_i++) {
          const float& p_a_x = b_box_seg_a.at<float>(p_a_i, 0);
          const float& p_a_y = b_box_seg_a.at<float>(p_a_i, 1);
          const float& p_a_z = b_box_seg_a.at<float>(p_a_i, 2);
          for (size_t p_b_i = 0; p_b_i < b_box_seg_b.rows; p_b_i++) {
            const float& p_b_x = b_box_seg_b.at<float>(p_b_i, 0);
            const float& p_b_y = b_box_seg_b.at<float>(p_b_i, 1);
            const float& p_b_z = b_box_seg_b.at<float>(p_b_i, 2);
            float dist_sq_p_ab = pow((p_a_x - p_b_x), 2) +
              pow((p_a_y - p_b_y), 2) + pow((p_a_z - p_b_z), 2);
            
            if (dist_sq > dist_sq_p_ab) 
              dist_sq = dist_sq_p_ab;
          }
        }

        dist_sq_segs.push_back(dist_sq);
        dist_sq_map_labels[label_a][label_b] = dist_sq;
        dist_sq_map_labels[label_b][label_a] = dist_sq;
        // LOG(INFO) << "      dist_sq label " << label_a << " label " <<
        // label_b << " dist_sq: "
        //     <<  dist_sq;
      }
    }
    if (num_segs >= 2) {
      std::nth_element(dist_sq_segs.begin(),
                       dist_sq_segs.begin() + dist_sq_segs.size() / 2,
                       dist_sq_segs.end());
      dist_sq_median = dist_sq_segs[dist_sq_segs.size() / 2];

      // if(dist_sq_median < 0.05*0.05)
      // {
      //     LOG(INFO) << "      Dist_sq median: " << dist_sq_median;
      //     LOG(INFO) << "      semantic: " <<
      //     int((*inst_segments->begin())->semantic_label_); for(auto
      //     segment_a_it=inst_segments->begin();
      //     segment_a_it!=inst_segments->end();segment_a_it++)
      //     {
      //         LOG(INFO) << "          seg size: " <<
      //         (*segment_a_it)->points_C_.size();
      //     }
      //     for(Distance dist_sq:dist_sq_segs)
      //     {
      //         LOG(INFO) << "          dist_sq: " << dist_sq;
      //     }
      //     dist_sq_median = 0.05*0.05;
      // }
      // CHECK_GT(dist_sq_median, 1e-4);
    }
  }

  for (auto segment_a_it = inst_segments->begin();
       segment_a_it != inst_segments->end(); segment_a_it++) {
    // compute inner confidence
    ObjSegConfidence inner_confidence;
    if (seg_graph_confidence_ == 0)  // all edge as 1
      inner_confidence = 1.;
    else if (seg_graph_confidence_ == 1)  // use instance score
      inner_confidence =
          dynamic_cast<SegmentConfidence*>(*segment_a_it)->inst_confidence_;
    else if (seg_graph_confidence_ == 2)  // inst score and overlap ratio
      inner_confidence =
          dynamic_cast<SegmentConfidence*>(*segment_a_it)->inst_confidence_ *
          dynamic_cast<SegmentConfidence*>(*segment_a_it)->obj_seg_confidence_;
    else if (seg_graph_confidence_ == 3)  // inst score and overlap ratio
      inner_confidence =
          dynamic_cast<SegmentConfidence*>(*segment_a_it)->inst_confidence_;
    // inner_confidence =
    // dynamic_cast<SegmentConfidence*>(*segment_a_it)->inst_confidence_ *
    //     dynamic_cast<SegmentConfidence*>(*segment_a_it)->obj_seg_confidence_;
    else  // default all edge as 1
      inner_confidence = 1.;

    // add inner confidence
    CHECK_EQ(isinf(inner_confidence), false);
    Label label_a = (*segment_a_it)->label_;
    auto label_a_it = label_label_confidence_map->find(label_a);
    if (label_a_it != label_label_confidence_map->end()) {
      auto label_label_a_it = label_a_it->second.find(label_a);
      if (label_label_a_it != label_a_it->second.end())
        label_label_a_it->second += inner_confidence;
      else {
        label_a_it->second.emplace(label_a, inner_confidence);
      }
    } 
    else {
      LabelConfiMap label_confi_map;
      label_confi_map.emplace(label_a, inner_confidence);
      label_label_confidence_map->emplace(label_a, label_confi_map);
    }

    // compute external confidence
    if (is_thing) {
      for (auto segment_b_it = std::next(segment_a_it);
           segment_b_it != inst_segments->end(); segment_b_it++) 
      {
        Label label_b = (*segment_b_it)->label_;
        ObjSegConfidence external_confidence;
        if (seg_graph_confidence_ == 0) {
          external_confidence = 1.; // all edge as 1
        }
        // use instance score
        else if (seg_graph_confidence_ == 1) {
          external_confidence = 
            dynamic_cast<SegmentConfidence*>(*segment_a_it)->inst_confidence_;
        }
        // inst score and overlap ratio 
        else if (seg_graph_confidence_ == 2) {
          external_confidence = 
            dynamic_cast<SegmentConfidence*>(*segment_a_it)->inst_confidence_ 
            * dynamic_cast<SegmentConfidence*>(*segment_a_it)->obj_seg_confidence_ 
            * dynamic_cast<SegmentConfidence*>(*segment_b_it)->obj_seg_confidence_;
        } 
        // inst score, overlap ratio and dist
        else if (seg_graph_confidence_ == 3) {
          Distance dist_sq_ab = dist_sq_map_labels[label_a][label_b];
          ObjSegConfidence seg_geo_confidence = 0.;
          if (dist_sq_ab <= dist_sq_median)
            seg_geo_confidence = 1.;
          else {
            // confidence 3
            // seg_geo_confidence = dist_sq_median/dist_sq_ab;
            // confidence 4
            seg_geo_confidence = 0.5 * dist_sq_median / dist_sq_ab;
            // confidence 5
            // ObjSegConfidence dist_sq_deviation = dist_sq_ab / dist_sq_median - 1; 
            // ObjSegConfidence theta = 0.337; 
            // seg_geo_confidence = exp(-0.5 * pow(dist_sq_deviation/theta, 2) );
          }

          external_confidence = dynamic_cast<SegmentConfidence*>(*segment_a_it)->inst_confidence_ 
            * seg_geo_confidence;
        } 
        // by default all edge as 1
        else{
          external_confidence = 1.;
        }

        // add external confidence
        CHECK_EQ(isinf(inner_confidence), false);
        (*label_label_confidence_map)[label_a][label_b] = external_confidence;
        (*label_label_confidence_map)[label_b][label_a] = external_confidence;
      }
    } 
    else {
      for (auto segment_b_it = std::next(segment_a_it);
           segment_b_it != inst_segments->end(); segment_b_it++) 
      {
        Label label_b = (*segment_b_it)->label_;
        (*label_label_confidence_map)[label_a][label_b] = 0.;
        (*label_label_confidence_map)[label_b][label_a] = 0.;
      }
    }
  }
}














void LabelTsdfConfidenceIntegrator::IncreaseLabelInstanceMapConfidence2(
    std::set<Segment*, SegmentConfidence::PtrCompare>* labeled_segments
  ) {
  std::set<InstanceLabel> assigned_instances;
  // Instance stuff.
  for (auto segment_it = labeled_segments->begin();
       segment_it != labeled_segments->end(); ++segment_it) {
    Label label = (*segment_it)->label_;
    if ((*segment_it)->points_C_.size() > 0u) {
      semantic_instance_label_fusion_ptr_->increaseLabelFramesCount(label);
    }

    // Loop through all the segments.
    if ((*segment_it)->instance_label_ != BackgroundLabel) {
      // It's a segment with a current frame instance.
      auto global_instance_it =
          current_to_global_instance_map_.find((*segment_it)->instance_label_);
      if (global_instance_it != current_to_global_instance_map_.end()) {
        // If current frame instance maps to a global instance, use it.
        LabelConfidence seg_instance_confidence =
            dynamic_cast<SegmentConfidence*>(*segment_it)->seg_label_confidence_ 
            + dynamic_cast<SegmentConfidence*>(*segment_it)->obj_seg_confidence_;
        semantic_instance_label_fusion_ptr_->increaseLabelInstanceConfidence(
            label, global_instance_it->second, seg_instance_confidence);
      } 
      else {
        // Current frame instance doesn't map to any global instance.
        // Get the global instance with max count.
        InstanceLabel instance_label =
            semantic_instance_label_fusion_ptr_->getInstanceLabel(
                label, assigned_instances);

        if (instance_label != BackgroundLabel) {
          current_to_global_instance_map_.emplace(
              (*segment_it)->instance_label_, instance_label);
          LabelConfidence seg_instance_confidence =
              dynamic_cast<SegmentConfidence*>(*segment_it)->seg_label_confidence_ 
              + dynamic_cast<SegmentConfidence*>(*segment_it)->obj_seg_confidence_;
          semantic_instance_label_fusion_ptr_->increaseLabelInstanceConfidence(
              label, instance_label, seg_instance_confidence);
          assigned_instances.emplace(instance_label);
        } 
        else {
          // Create new global instance.
          InstanceLabel fresh_instance = getFreshInstance();
          current_to_global_instance_map_.emplace(
              (*segment_it)->instance_label_, fresh_instance);
          LabelConfidence seg_instance_confidence =
              dynamic_cast<SegmentConfidence*>(*segment_it)
                  ->obj_seg_confidence_;
          semantic_instance_label_fusion_ptr_->increaseLabelInstanceConfidence(
              label, fresh_instance, seg_instance_confidence);
        }
      }
      semantic_instance_label_fusion_ptr_->increaseLabelClassCount(
          label, (*segment_it)->semantic_label_);
    } 
    else {
      // It's a segment with no instance prediction in the current frame.
      // Get the global instance it maps to, and set it as assigned.
      InstanceLabel instance_label =
          semantic_instance_label_fusion_ptr_->getInstanceLabel(label);
      if (instance_label != BackgroundLabel) {
        assigned_instances.emplace(instance_label);
      }
    }
  }
}












void LabelTsdfConfidenceIntegrator::updateLabelClassInstanceConfidence(
    std::set<Segment*, SegmentConfidence::PtrCompare>* labeled_segments) {
  std::set<InstanceLabel> assigned_instances;

  // LOG(INFO) << "  labeled_segments.size(): " << labeled_segments->size();
  int segment_idx = 1;
  // Loop through all the segments with labels
  for (auto segment_it = labeled_segments->begin();
       segment_it != labeled_segments->end(); ++segment_it) {
    Label label = (*segment_it)->label_;
    InstanceLabel matched_instance_label = BackgroundLabel;

    if ((*segment_it)->points_C_.size() > 0u) {
      semantic_instance_label_fusion_ptr_->increaseLabelFramesCount(label);
    } else
      continue;
    if ((*segment_it)->instance_label_ != BackgroundLabel) {
      SemanticLabel semantic_label = (*segment_it)->semantic_label_;
      // It's a segment with a current frame instance and class prediction.
      auto global_instance_it =
          current_to_global_instance_map_.find((*segment_it)->instance_label_);
      if (global_instance_it != current_to_global_instance_map_.end()) {
        // If current frame instance maps to a global instance, use it.
        matched_instance_label = global_instance_it->second;
        // if(matched_instance_label!=BackgroundLabel)
        //     LOG(INFO) << "      CurMapIncrease: Label: " << std::setfill('0')
        //     << std::setw(5) << int(label) << " Instance: " <<
        //     std::setfill('0') << std::setw(5) <<int(matched_instance_label);
      } 
      else {
        // Current frame instance doesn't map to any global instance.
        // Get the global instance with max count.
        matched_instance_label =
            semantic_instance_label_fusion_ptr_->getInstanceLabelWithSemantic(
                label, semantic_label, assigned_instances);

        if (matched_instance_label != BackgroundLabel) {
          current_to_global_instance_map_.emplace(
              (*segment_it)->instance_label_, matched_instance_label);
          assigned_instances.emplace(matched_instance_label);
          // LOG(INFO) << "      GloMapIncrease: Label: " << std::setfill('0')
          // << std::setw(5) << int(label) << " Instance: " << std::setfill('0')
          // << std::setw(5) <<int(matched_instance_label);
        } 
        else {
          // Create new global instance.
          matched_instance_label = getFreshInstance();
          current_to_global_instance_map_.emplace(
              (*segment_it)->instance_label_, matched_instance_label);
          // LOG(INFO) << "      GloMapIncrease: Label: " << std::setfill('0')
          // << std::setw(5) << int(label) << " Instance: " << std::setfill('0')
          // << std::setw(5) <<int(matched_instance_label);
        }
      }
      // 1.0 or dynamic_cast<SegmentConfidence*>(*segment_it)->obj_seg_confidence_
      semantic_instance_label_fusion_ptr_->increaseLabelClassInstanceConfidence(
          label, semantic_label, matched_instance_label, 1.0);
      semantic_instance_label_fusion_ptr_->increaseLabelClassConfidence(
          label, semantic_label, 1.0);
      
      // LOG(INFO) << "      increase Label " << std::setfill('0') <<
      // std::setw(5) <<int(label)
      //     << "    semantic " << int(semantic_label) << "; count " <<
      //     semantic_instance_label_fusion_ptr_->label_class_count_.find(label)->second.find(semantic_label)->second;
    }
  }
}











void LabelTsdfConfidenceIntegrator::updateLabelInstanceClassConfidence(
    std::set<Segment*, SegmentConfidence::PtrCompare>* labeled_segments
  ) {
  
  std::set<InstanceLabel> assigned_instances;
  // Loop through all the segments with labels
  for (auto segment_it = labeled_segments->begin();
       segment_it != labeled_segments->end(); ++segment_it) 
  {
    Label label = (*segment_it)->label_;
    InstanceLabel matched_instance_label = BackgroundLabel;

    if ((*segment_it)->points_C_.size() > 0u) {
      semantic_instance_label_fusion_ptr_->increaseLabelFramesCount(label);
    }
    if ((*segment_it)->instance_label_ != BackgroundLabel) {
      SemanticLabel semantic_label = (*segment_it)->semantic_label_;
      // It's a segment with a current frame instance and class prediction.
      auto global_instance_it =
          current_to_global_instance_map_.find((*segment_it)->instance_label_);
      if (global_instance_it != current_to_global_instance_map_.end()) {
        // If current frame instance maps to a global instance, use it.
        matched_instance_label = global_instance_it->second;
        if (matched_instance_label != BackgroundLabel)
          LOG(INFO) << "      CurMapIncrease: Label: " << std::setfill('0')
                    << std::setw(5) << int(label)
                    << " Instance: " << std::setfill('0') << std::setw(5)
                    << int(matched_instance_label);
      } 
      else {
        // Current frame instance doesn't map to any global instance.
        // Get the global instance with max count.
        matched_instance_label = semantic_instance_label_fusion_ptr_->getInstanceLabel(
                label, assigned_instances);

        if (matched_instance_label != BackgroundLabel) {
          current_to_global_instance_map_.emplace(
              (*segment_it)->instance_label_, matched_instance_label);
          assigned_instances.emplace(matched_instance_label);

          LOG(INFO) << "      GloMapIncrease: Label: " << std::setfill('0')
                    << std::setw(5) << int(label)
                    << " Instance: " << std::setfill('0') << std::setw(5)
                    << int(matched_instance_label);
        } else {
          // Create new global instance.
          matched_instance_label = getFreshInstance();
          current_to_global_instance_map_.emplace(
              (*segment_it)->instance_label_, matched_instance_label);
          LOG(INFO) << "      FreshIncrease: Label: " << std::setfill('0')
                    << std::setw(5) << int(label)
                    << " Instance: " << std::setfill('0') << std::setw(5)
                    << int(matched_instance_label);
        }
      }
      // dynamic_cast<SegmentConfidence*>(*segment_it)->obj_seg_confidence_
      semantic_instance_label_fusion_ptr_->increaseLabelInstanceClassConfidence(
          label, matched_instance_label, semantic_label, 1.0);
      semantic_instance_label_fusion_ptr_->increaseLabelInstanceConfidence(
          label, matched_instance_label, 1.0);
    }
  }
}










bool LabelTsdfConfidenceIntegrator::getNextSegmentLabelPairWithConfidence(
    std::set<Segment*, SegmentConfidence::PtrCompare>& labeled_segments,
    std::set<Label>* assigned_labels,
    std::map<Label, std::map<Segment*, SegSegConfidence>>* candidates_confidence,
    std::map<Segment*, std::vector<Label>>* segment_merge_candidates,
    std::pair<Segment*, Label>* segment_label_pair,
    SegSegConfidence* pair_confidence
  ) {
  CHECK_NOTNULL(assigned_labels);
  CHECK_NOTNULL(candidates_confidence);
  CHECK_NOTNULL(segment_merge_candidates);
  CHECK_NOTNULL(segment_label_pair);
  CHECK_NOTNULL(pair_confidence);
  Label max_label;
  SegSegConfidence max_confidence = 0;
  Segment* max_segment;
  std::map<voxblox::Segment*, SegSegConfidence> segments_to_recompute;

  for (auto label_it = candidates_confidence->begin();
       label_it != candidates_confidence->end(); ++label_it) {
    for (auto segment_it = label_it->second.begin();
         segment_it != label_it->second.end(); segment_it++) {
      bool count_greater_than_max = segment_it->second > max_confidence;
      bool count_greater_than_min =
          segment_it->second > label_tsdf_config_.min_label_voxel_count;
      bool ratio_greater_than_min =
          segment_it->second >
          int(label_tsdf_config_.label_register_min_overlap_ratio *
              segment_it->first->points_C_.size());
      bool is_unlabelled =
          labeled_segments.find(segment_it->first) == labeled_segments.end();

      if (count_greater_than_max && count_greater_than_min && is_unlabelled) {
        max_confidence = segment_it->second;
        max_segment = segment_it->first;
        max_label = label_it->first;
        segments_to_recompute = label_it->second;
      }
    }
  }
  if (std::abs(max_confidence) < 1e-3) {
    return false;
  }
  segment_label_pair->first = max_segment;
  segment_label_pair->second = max_label;
  assigned_labels->emplace(max_label);
  *pair_confidence =
      max_confidence /
      dynamic_cast<SegmentConfidence*>(max_segment)->getSegmentSize();

  // For all segments that need to have their label
  // count recomputed, first clean their relative entries and recompute.
  for (auto segment : segments_to_recompute) {
    if (segment.first != max_segment) {
      for (auto label_it = candidates_confidence->begin();
           label_it != candidates_confidence->end(); ++label_it) {
        if (label_it->first != max_label) {
          label_it->second.erase(segment.first);
        }
      }
      computeSegmentLabelCandidatesConfidence(
          segment.first, candidates_confidence, segment_merge_candidates,
          *assigned_labels);
    }
  }
  return true;
}












// Not thread safe.
bool LabelTsdfConfidenceIntegrator::mergeLabelConfidence(
    LLSet* merges_to_publish
  ) {
  CHECK_NOTNULL(merges_to_publish);
  bool whether_merge_alias = false;

  if (label_tsdf_config_.enable_pairwise_confidence_merging) {
    Label new_label;
    Label old_label;
    while (getNextMergeConfidence(&new_label, &old_label)) {
      semantic_instance_label_fusion_ptr_->checkLabelInFrameCount();
      SemanticLabel new_label_semantic =
          semantic_instance_label_fusion_ptr_->getSemanticLabel(new_label);
      SemanticLabel old_label_semantic =
          semantic_instance_label_fusion_ptr_->getSemanticLabel(old_label);

      LOG(INFO) << "Merging labels " << int(new_label) << " and "
                << int(old_label);
      LOG(INFO) << "Before merging ";
      LOG(INFO) << "    old labels: semantic- " << int(old_label_semantic);
      LOG(INFO) << "    new labels: semantic- " << int(new_label_semantic);

      swapLabels(old_label, new_label);

      // Delete any staged segment publishing for overridden label.
      LMapIt label_age_pair_it = labels_to_publish_.find(old_label);
      if (label_age_pair_it != labels_to_publish_.end()) {
        labels_to_publish_.erase(old_label);
      }
      updated_labels_.erase(old_label);

      // Store the happened merge.
      LLSetIt label_it = merges_to_publish->find(new_label);
      if (label_it != merges_to_publish->end()) {
        // If the new_label already incorporated other labels
        // just add the just incorporated old_label to this list.
        label_it->second.emplace(old_label);
      } else {
        // If the new_label hasn't incorporated any other labels yet
        // create a new list and only add the just incorporated
        // old_label.
        std::set<Label> incorporated_labels;
        incorporated_labels.emplace(old_label);
        merges_to_publish->emplace(new_label, incorporated_labels);
      }

      updatePairwiseConfidenceAfterMerging(new_label, old_label);

      new_label_semantic =
          semantic_instance_label_fusion_ptr_->getSemanticLabel(new_label);
      old_label_semantic =
          semantic_instance_label_fusion_ptr_->getSemanticLabel(old_label);

      // update semantic_instance_label_fusion_ptr_
      semantic_instance_label_fusion_ptr_->mergeLabels(new_label, old_label);

      whether_merge_alias = true;

      semantic_instance_label_fusion_ptr_->checkLabelInFrameCount();
    }
  }
  return whether_merge_alias;
}












bool LabelTsdfConfidenceIntegrator::getNextMergeConfidence(
      Label* new_label, Label* old_label
  ) {
  CHECK_NOTNULL(new_label);
  CHECK_NOTNULL(old_label);
  // LOG(INFO) << "      Superpoint Merge Step2 ";
  for (LLMapIt confidence_map_it = pairwise_confidence_.begin();
       confidence_map_it != pairwise_confidence_.end(); ++confidence_map_it) {
    SemanticLabel new_label_semantic =
        semantic_instance_label_fusion_ptr_->getSemanticLabel(
            confidence_map_it->first);
    // LOG(INFO) << "      Pair wise new label " <<
    // int(confidence_map_it->first) << " with new sem "
    //     << int(new_label_semantic);
    for (LMapIt confidence_pair_it = confidence_map_it->second.begin();
         confidence_pair_it != confidence_map_it->second.end();
         ++confidence_pair_it) {
      SemanticLabel old_label_semantic =
          semantic_instance_label_fusion_ptr_->getSemanticLabel(
              confidence_pair_it->first);
      // LOG(INFO) << "      Pair wise old label " <<
      // int(confidence_pair_it->first) << " with new sem "
      //     << int(old_label_semantic);
      bool semantic_consistent = false;
      if (data_association_ == 1)  // only consider equal semantics merging
        semantic_consistent = (new_label_semantic == old_label_semantic);
      else if (data_association_ == 2)  // also consider background merging
        semantic_consistent = ((new_label_semantic == old_label_semantic) ||
                               (new_label_semantic == BackgroundSemLabel) ||
                               (old_label_semantic == BackgroundSemLabel));
      else if (data_association_ == 3)  // also consider background merging
        semantic_consistent = ((new_label_semantic == old_label_semantic) ||
                               (new_label_semantic == BackgroundSemLabel) ||
                               (old_label_semantic == BackgroundSemLabel));
      else
        semantic_consistent = true;

      if (confidence_pair_it->second >
              label_tsdf_config_.merging_min_frame_count &&
          semantic_consistent) {
        // If the pairwise confidence is above a threshold return
        // the two labels to merge and remove the pair
        // from the pairwise confidence counts.
        *new_label = confidence_map_it->first;
        *old_label = confidence_pair_it->first;
        confidence_pair_it =
            confidence_map_it->second.erase(confidence_pair_it);
        return true;
      }
    }
  }
  return false;
}









void LabelTsdfConfidenceIntegrator::updatePairwiseConfidenceAfterMerging(
    const Label& new_label, const Label& old_label) {
  // Add all the pairwise confidence counts of the old_label to new_label.
  // First the counts (old_label -> some_label),
  // where old_label < some_label.
  LLMapIt old_label_pc_it = pairwise_confidence_.find(old_label);
  if (old_label_pc_it != pairwise_confidence_.end()) {
    LLMapIt new_label_pc_it = pairwise_confidence_.find(new_label);
    if (new_label_pc_it != pairwise_confidence_.end()) {
      for (LMapIt old_label_pc_count_it = old_label_pc_it->second.begin();
           old_label_pc_count_it != old_label_pc_it->second.end();
           ++old_label_pc_count_it) {
        addPairwiseConfidenceCountMaximum(new_label_pc_it,
                                          old_label_pc_count_it->first,
                                          old_label_pc_count_it->second);
      }
    } else {
      LMap old_label_map(old_label_pc_it->second.begin(),
                         old_label_pc_it->second.end());
      pairwise_confidence_.emplace(new_label, old_label_map);
    }
    pairwise_confidence_.erase(old_label_pc_it);
  }

  // Next add the counts (some_label -> old_label),
  // where some_label < old_label.
  for (LLMapIt confidence_map_it = pairwise_confidence_.begin();
       confidence_map_it != pairwise_confidence_.end();
       /* no increment */) {
    for (LMapIt confidence_pair_it = confidence_map_it->second.begin();
         confidence_pair_it != confidence_map_it->second.end();
         /* no increment */) {
      if (confidence_pair_it->first == old_label) {
        if (confidence_map_it->first < new_label) {
          addPairwiseConfidenceCountMaximum(confidence_map_it, new_label,
                                            confidence_pair_it->second);
        } else {
          LLMapIt new_label_pc_it = pairwise_confidence_.find(new_label);
          if (new_label_pc_it != pairwise_confidence_.end()) {
            addPairwiseConfidenceCountMaximum(new_label_pc_it,
                                              confidence_map_it->first,
                                              confidence_pair_it->second);
          } else {
            LMap old_label_map;
            old_label_map.emplace(confidence_map_it->first,
                                  confidence_pair_it->second);
            pairwise_confidence_.emplace(new_label, old_label_map);
          }
        }
        confidence_pair_it =
            confidence_map_it->second.erase(confidence_pair_it);
      } else {
        ++confidence_pair_it;
      }
    }
    if (confidence_map_it->second.empty()) {
      confidence_map_it = pairwise_confidence_.erase(confidence_map_it);
    } else {
      ++confidence_map_it;
    }
  }
}









void LabelTsdfConfidenceIntegrator::cleanStaleLabels() {
  std::map<Label, long> label_voxel_count;
  for (auto label_it =
           semantic_instance_label_fusion_ptr_->label_frames_count_.begin();
       label_it !=
       semantic_instance_label_fusion_ptr_->label_frames_count_.end();
       label_it++) {
    label_voxel_count[label_it->first] = 0;
  }
  BlockIndexList all_label_blocks;
  label_layer_->getAllAllocatedBlocks(&all_label_blocks);

#pragma omp parallel
  {
    std::map<Label, long> label_voxel_count_local;
    for (auto label_it =
             semantic_instance_label_fusion_ptr_->label_frames_count_.begin();
         label_it !=
         semantic_instance_label_fusion_ptr_->label_frames_count_.end();
         label_it++) {
      label_voxel_count_local[label_it->first] = 0;
    }

#pragma omp for
    for (const BlockIndex& block_index : all_label_blocks) {
      Block<LabelVoxel>::Ptr label_block =
          label_layer_->getBlockPtrByIndex(block_index);
      size_t vps = label_block->voxels_per_side();
      for (size_t i = 0u; i < vps * vps * vps; i++) {
        LabelVoxel& voxel = label_block->getVoxelByLinearIndex(i);
        label_voxel_count_local[voxel.label] += 1;
      }
    }

    for (auto label_it =
             semantic_instance_label_fusion_ptr_->label_frames_count_.begin();
         label_it !=
         semantic_instance_label_fusion_ptr_->label_frames_count_.end();
         label_it++) {
#pragma omp atomatic
      label_voxel_count[label_it->first] +=
          label_voxel_count_local[label_it->first];
    }
  }

  // remove stale labels
  std::set<Label> stale_labels;
  for (auto label_it =
           semantic_instance_label_fusion_ptr_->label_frames_count_.begin();
       label_it !=
       semantic_instance_label_fusion_ptr_->label_frames_count_.end();
       label_it++) {
    if (label_voxel_count[label_it->first] < 15)
      stale_labels.insert(label_it->first);
  }
  for (Label label : stale_labels)
    semantic_instance_label_fusion_ptr_->label_frames_count_.erase(label);
}









// void LabelTsdfConfidenceIntegrator::raycastInstancePredictions(
//     const Transformation& T_G_C, const cv::Mat& panoptic_mask,
//     const cv::Mat& depth_img_scaled, const float& search_length,
//     const CameraRayGenerator* camera_ray_generaor,
//     cv::Mat* instance_map
// ) {
//   CHECK_GT(search_length, 0.0);
//   float half_search_length = 0.5 * search_length;

//   int visited_num = 0;
//   std::set<LabelVoxel*> visited_voxels_ptrs;
//   std::map<GlobalIndex, InstanceLabel> pointsIdx_inst_map;

//   omp_set_num_threads(config_.integrator_threads);
// #pragma omp parallel
//   {
//     // thread-wise variables
//     int visited_num_thread = 0;
//     std::set<LabelVoxel*> visited_voxels_ptrs_thread;
//     std::map<GlobalIndex, InstanceLabel> pointsIdx_inst_map_thread;

//     // for each pixel with predicted instance label, ray cast them
//     // to label voxels and obtain the global instance from superpoints
// #pragma omp for schedule(dynamic, 6)
//     for (int row_i = 0; row_i < panoptic_mask.rows; row_i++) {
//       for (int col_j = 0; col_j < panoptic_mask.cols; col_j++) {
//         // ignore invalid pixels (inst-id == 0) in the given mask
//         InstanceLabel inst_label = panoptic_mask.at<uint8_t>(row_i, col_j);
//         if (inst_label == BackgroundLabel) continue;

        
//         cv::Vec3f point_C_start_vec;
//         cv::Vec3f point_C_end_vec;
//         point_C_start_vec = camera_ray_generaor->getRayStart(row_i, col_j);
//         point_C_end_vec = camera_ray_generaor->getRayEnd(row_i, col_j);

//         float depth_scaled = depth_img_scaled.at<float>(row_i, col_j);
//         float min_ray_len = camera_ray_generaor->getRangeMin();
//         float max_ray_len = camera_ray_generaor->getRangeMax();
//         // make use of depth value if any
//         if (std::isfinite(depth_scaled) &&
//             depth_scaled >= min_ray_len && depth_scaled <= max_ray_len) 
//         {
//           point_C_start_vec = camera_ray_generaor->getRayWithDepth(row_i, col_j, 
//             std::max(float(depth_scaled - half_search_length), min_ray_len)
//           );
//           point_C_end_vec = camera_ray_generaor->getRayWithDepth(row_i, col_j, 
//             std::min(float(depth_scaled + half_search_length), max_ray_len)
//           );
//         }

//         Point point_C_start(point_C_start_vec[0], point_C_start_vec[1], point_C_start_vec[2]);
//         Point point_C_end(point_C_end_vec[0], point_C_end_vec[1], point_C_end_vec[2]);
//         Point point_G_start = T_G_C * point_C_start;
//         Point point_G_end = T_G_C * point_C_end;
        
//         // ray casting
//         bool clearing_ray = false;
//         bool voxel_carving_enabled = true;
//         RayCaster ray_caster(
//           point_G_start, point_G_end, clearing_ray,
//           voxel_carving_enabled, static_cast<FloatingPoint>(max_ray_len),
//           voxel_size_inv_, config_.default_truncation_distance
//         );
        
//         // get the first voxel with valid superpoint label
//         GlobalIndex global_voxel_idx;
//         int ray_march_num = 0;
//         while (ray_caster.nextRayIndex(&global_voxel_idx)) {
//           ray_march_num++;
//           LabelVoxel* hit_voxel = label_layer_->getVoxelPtrByGlobalIndex(global_voxel_idx);
//           if (hit_voxel == nullptr) continue;

//           std::set<LabelVoxel*>::iterator voxel_ptr_it =
//               visited_voxels_ptrs_thread.find(hit_voxel);
//           // assign the known instance label for visited voxel
//           if (voxel_ptr_it != visited_voxels_ptrs_thread.end()) {
//             auto global_voxel_it = pointsIdx_inst_map_thread.find(global_voxel_idx);
//             if (global_voxel_it != pointsIdx_inst_map_thread.end()) {
//               // if the global voxel index already has a label, use it
//               InstanceLabel global_inst_label = global_voxel_it->second;
//               if (global_inst_label != BackgroundLabel) {
//                 instance_map->at<uint8_t>(row_i, col_j) = global_inst_label;
//               }
//             }
//             break;
//           }
            
//           if (hit_voxel->label_confidence < 0.5)
//             continue;
//           // get superpoint label
//           Label superpoint_label = BackgroundLabel;
//           superpoint_label = hit_voxel->label;
//           if (superpoint_label == BackgroundLabel)
//             continue;

//           // find the global instance label
//           std::set<InstanceLabel> empty_set;
//           InstanceLabel global_inst_label = semantic_instance_label_fusion_ptr_->getInstanceLabel(
//             superpoint_label, empty_set);
//           visited_voxels_ptrs_thread.insert(hit_voxel);
//           pointsIdx_inst_map_thread.emplace(global_voxel_idx, global_inst_label);

//           if (global_inst_label != BackgroundLabel) {
//             instance_map->at<uint8_t>(row_i, col_j) = global_inst_label;
//           }
//           break;
//         }
//       }
//     }
// #pragma omp critical
//     {
//       for (auto voxel_ptr_ : visited_voxels_ptrs_thread)
//         visited_voxels_ptrs.insert(voxel_ptr_);

//       for (auto pointsIdx_it = pointsIdx_inst_map_thread.begin();
//           pointsIdx_it != pointsIdx_inst_map_thread.end(); pointsIdx_it++) 
//       {
//         auto global_label_it = pointsIdx_inst_map.find(pointsIdx_it->first);
//         if (global_label_it == pointsIdx_inst_map.end())
//           pointsIdx_inst_map.emplace(pointsIdx_it->first, pointsIdx_it->second);
//       }
//     }
//   }
// }


void LabelTsdfConfidenceIntegrator::raycastInstancePredictions(
    const Transformation& T_G_C, 
    const cv::Mat& panoptic_mask,
    const cv::Mat& depth_img_scaled, 
    const float& search_length,
    const CameraRayGenerator* camera_ray_generaor,
    cv::Mat* instance_map
) {
  // CHECK_GT(search_length, 0.0);
  float half_search_length = 0.5 * search_length;

  std::set<LabelVoxel*> visited_voxels_ptrs;
  std::map<Label, std::vector<GlobalIndex>> label_pointsIdx_map;
  std::map<Label, InstanceLabel> labels_insts_count;

  // for each pixel with predicted instance label, ray cast them
  // to label voxels and obtain the global instance from superpoints
  for (int row_i = 0; row_i < panoptic_mask.rows; row_i++) {
    for (int col_j = 0; col_j < panoptic_mask.cols; col_j++) {
      // ignore invalid pixels (inst-id == 0) in the given mask
      InstanceLabel inst_label = panoptic_mask.at<uint16_t>(row_i, col_j);
      if (inst_label == BackgroundLabel){
        continue;
      }

      
      cv::Vec3f point_C_start_vec;
      cv::Vec3f point_C_end_vec;
      point_C_start_vec = camera_ray_generaor->getRayStart(row_i, col_j);
      point_C_end_vec = camera_ray_generaor->getRayEnd(row_i, col_j);

      float depth_scaled = depth_img_scaled.at<float>(row_i, col_j);
      float min_ray_len = camera_ray_generaor->getRangeMin();
      float max_ray_len = camera_ray_generaor->getRangeMax();
      // make use of depth value if any
      if (std::isfinite(depth_scaled) &&
          depth_scaled >= min_ray_len && depth_scaled <= max_ray_len) 
      {
        point_C_start_vec = camera_ray_generaor->getRayWithDepth(row_i, col_j, 
          std::max(float(depth_scaled - half_search_length), min_ray_len)
        );
        point_C_end_vec = camera_ray_generaor->getRayWithDepth(row_i, col_j, 
          std::min(float(depth_scaled + half_search_length), max_ray_len)
        );
      }

      Point point_C_start(point_C_start_vec[0], point_C_start_vec[1], point_C_start_vec[2]);
      Point point_C_end(point_C_end_vec[0], point_C_end_vec[1], point_C_end_vec[2]);
      Point point_G_start = T_G_C * point_C_start;
      Point point_G_end = T_G_C * point_C_end;
      
      // ray casting
      bool clearing_ray = false;
      bool voxel_carving_enabled = true;
      RayCaster ray_caster(
        point_G_start, point_G_end, clearing_ray,
        voxel_carving_enabled, static_cast<FloatingPoint>(max_ray_len),
        voxel_size_inv_, config_.default_truncation_distance
      );
      
      // get the first hit voxel with valid superpoint label
      GlobalIndex global_voxel_idx;
      int ray_march_num = 0;
      while (ray_caster.nextRayIndex(&global_voxel_idx)) {
        ray_march_num++;
        LabelVoxel* hit_voxel = label_layer_->getVoxelPtrByGlobalIndex(global_voxel_idx);
        if (hit_voxel == nullptr) continue;

        // // assign the known instance label for visited voxel
        // std::set<LabelVoxel*>::iterator voxel_ptr_it = visited_voxels_ptrs.find(hit_voxel);
        // if (voxel_ptr_it != visited_voxels_ptrs.end()) {
        //   // find the global voxel index in the label_voxel-idx map
        //   auto sp_label_it = label_pointsIdx_map.find(hit_voxel->label);
        //   if (sp_label_it != label_pointsIdx_map.end()) {
        //     auto global_voxel_it = sp_label_it->second.begin();
        //     if (global_voxel_it != sp_label_it->second.end()) {
        //       InstanceLabel global_inst_label = labels_insts_count[hit_voxel->label];
        //       if (global_inst_label != BackgroundLabel) {
        //         instance_map->at<uint16_t>(row_i, col_j) = global_inst_label;
        //       }
        //       break;
        //     }
        //   }
        // }

        // ignore bg superpoint
        Label superpoint_label = BackgroundLabel;
        superpoint_label = hit_voxel->label;
        if (hit_voxel->label_confidence < 0.5)
          continue;
        if (superpoint_label == BackgroundLabel)
          continue;

        // find the global instance label
        std::set<InstanceLabel> empty_set;
        InstanceLabel global_inst_label = semantic_instance_label_fusion_ptr_->getInstanceLabel(
          superpoint_label, empty_set);
        if (global_inst_label != BackgroundLabel) {
          instance_map->at<uint16_t>(row_i, col_j) = global_inst_label;
        }
        // else {
        //   LOG(INFO) << "  Ignore pixel at " << row_i << ", " << col_j 
        //     << " with global inst label " << int(global_inst_label);
        // }

        visited_voxels_ptrs.insert(hit_voxel);
        auto label_pointsIdx_it = label_pointsIdx_map.find(superpoint_label);
        if (label_pointsIdx_it == label_pointsIdx_map.end()) {
          std::vector<GlobalIndex> pointsIdxs;
          pointsIdxs.emplace_back(global_voxel_idx);
          label_pointsIdx_map.emplace(superpoint_label, pointsIdxs);
        } 
        else {
          label_pointsIdx_it->second.emplace_back(global_voxel_idx);
        }

        auto labels_insts_it = labels_insts_count.find(superpoint_label);
        if (labels_insts_it == labels_insts_count.end()) {
          labels_insts_count.emplace(superpoint_label, global_inst_label);
        }
        // else {
        //   // LOG(INFO) << "  Prev inst id" << int(labels_insts_count[superpoint_label]) 
        //   //   << " current inst id " << int(global_inst_label) << " for label " << int(superpoint_label);
        // }

        break;
      }
    }
  }
}








void LabelTsdfConfidenceIntegrator::raycastPanopticPredictions(
    const Transformation& T_G_C, const cv::Mat& panoptic_mask,
    const std::map<InstanceLabel, SemanticLabel>& inst_sem_map,
    const cv::Mat& depth_img_scaled, const float& search_length,
    const CameraRayGenerator* camera_ray_generaor,
    std::map<Label, std::map<InstanceLabel, int>>& labels_instances_cout,
    float pose_confidence) 
{
  CHECK_GT(search_length, 0.0);
  float half_search_length = 0.5 * search_length;

  // LOG(INFO) << "  ray cast started on " << int(panoptic_mask.rows)
  //     << " * " << int(panoptic_mask.cols) << " with search_len " <<
  //     search_length;

  cv::Mat rayCastLabelImg(panoptic_mask.size(), CV_8U, cv::Scalar(0));

  labels_instances_cout.clear();
  int visited_num = 0;
  std::set<LabelVoxel*> visited_voxels_ptrs;
  std::map<Label, std::vector<GlobalIndex>> label_pointsIdx_map;

  omp_set_num_threads(config_.integrator_threads);
#pragma omp parallel
  {
    // thread-wise variables
    int visited_num_thread = 0;
    std::set<LabelVoxel*> visited_voxels_ptrs_thread;
    std::map<Label, std::vector<GlobalIndex>> label_pointsIdx_map_thread;
    std::map<Label, std::map<InstanceLabel, int>> labels_instances_cout_thread;

#pragma omp for schedule(dynamic, 6)
    for (int row_i = 0; row_i < panoptic_mask.rows; row_i++) {
      for (int col_j = 0; col_j < panoptic_mask.cols; col_j++) {
        // for each pixel with predicted instance label, ray cast them
        // to label voxels and assign frame-wise instance label to superpoint
        InstanceLabel inst_label = panoptic_mask.at<uint8_t>(row_i, col_j);
        if (inst_label == BackgroundLabel) continue;

        // make use of depth value if any
        cv::Vec3f point_C_start_vec;
        cv::Vec3f point_C_end_vec;
        point_C_start_vec = camera_ray_generaor->getRayStart(row_i, col_j);
        point_C_end_vec = camera_ray_generaor->getRayEnd(row_i, col_j);

        float depth_scaled = depth_img_scaled.at<float>(row_i, col_j);
        float min_ray_len = camera_ray_generaor->getRangeMin();
        float max_ray_len = camera_ray_generaor->getRangeMax();

        if (std::isfinite(depth_scaled) &&
            depth_scaled >= min_ray_len && depth_scaled <= max_ray_len) 
        {
          point_C_start_vec = camera_ray_generaor->getRayWithDepth(row_i, col_j, 
            std::max(float(depth_scaled - half_search_length), min_ray_len)
          );
          point_C_end_vec = camera_ray_generaor->getRayWithDepth(row_i, col_j, 
            std::min(float(depth_scaled + half_search_length), max_ray_len)
          );
        }

        Point point_C_start(point_C_start_vec[0], point_C_start_vec[1], point_C_start_vec[2]);
        Point point_C_end(point_C_end_vec[0], point_C_end_vec[1], point_C_end_vec[2]);

        Point point_G_start = T_G_C * point_C_start;
        Point point_G_end = T_G_C * point_C_end;

        bool clearing_ray = false;
        bool voxel_carving_enabled = true;
        RayCaster ray_caster(
          point_G_start, point_G_end, clearing_ray,
          voxel_carving_enabled, static_cast<FloatingPoint>(max_ray_len),
          voxel_size_inv_, config_.default_truncation_distance
        );

        GlobalIndex global_voxel_idx;
        int ray_march_num = 0;
        while (ray_caster.nextRayIndex(&global_voxel_idx)) {
          ray_march_num++;
          LabelVoxel* hit_voxel = label_layer_->getVoxelPtrByGlobalIndex(global_voxel_idx);
          if (hit_voxel == nullptr) continue;

          bool valid_hit = false;

          std::set<LabelVoxel*>::iterator voxel_ptr_it =
              visited_voxels_ptrs_thread.find(hit_voxel);
          if (voxel_ptr_it ==
              visited_voxels_ptrs_thread.end())  // only count unvisited voxels
          {
            if (hit_voxel->label_confidence > 0.5) {
              visited_voxels_ptrs_thread.insert(hit_voxel);
              // get superpoint label
              Label superpoint_label = BackgroundLabel;
              superpoint_label = hit_voxel->label;
              if (superpoint_label != BackgroundLabel) {
                // add label-cur_inst count
                auto label_it = labels_instances_cout_thread.find(superpoint_label);
                if (label_it != labels_instances_cout_thread.end()) {
                  auto inst_id_it = label_it->second.find(inst_label);
                  if (inst_id_it != label_it->second.end())
                    inst_id_it->second += 1;
                  else
                    label_it->second.emplace(inst_label, 1);
                } else {
                  std::map<InstanceLabel, int> inst_count_map;
                  inst_count_map.emplace(inst_label, 1);
                  labels_instances_cout_thread.emplace(
                    superpoint_label, inst_count_map);
                }
                // add label-voxel map
                auto label_pointsIdx_it = label_pointsIdx_map_thread.find(superpoint_label);
                if (label_pointsIdx_it == label_pointsIdx_map_thread.end()) {
                  std::vector<GlobalIndex> pointsIdxs;
                  pointsIdxs.emplace_back(global_voxel_idx);
                  label_pointsIdx_map_thread.emplace(superpoint_label, pointsIdxs);
                } else
                  label_pointsIdx_it->second.emplace_back(global_voxel_idx);
              }
              valid_hit = true;
            }
          } else
            valid_hit = true;

          if (valid_hit) break;
        }
      }
    }

    // aggregate labels_instances_cout_thread to global labels_instances_cout
    //          and label_pointsIdx_map_thread to global one
    visited_num_thread = visited_voxels_ptrs_thread.size();
#pragma omp critical
    {
      // to global labels_instances_cout
      for (auto label_it = labels_instances_cout_thread.begin();
           label_it != labels_instances_cout_thread.end(); label_it++) {
        for (auto inst_it = label_it->second.begin();
             inst_it != label_it->second.end(); inst_it++) {
          auto label_global_it = labels_instances_cout.find(label_it->first);
          if (label_global_it != labels_instances_cout.end()) {
            auto inst_global_it = label_global_it->second.find(inst_it->first);
            if (inst_global_it != label_global_it->second.end())
              inst_global_it->second += inst_it->second;
            else
              label_global_it->second.emplace(inst_it->first, inst_it->second);
          } else {
            std::map<InstanceLabel, int> inst_count_map;
            inst_count_map.emplace(inst_it->first, inst_it->second);
            labels_instances_cout.emplace(label_it->first, inst_count_map);
          }
        }
      }
      visited_num += visited_num_thread;
      for (auto voxel_ptr_ : visited_voxels_ptrs_thread)
        visited_voxels_ptrs.insert(voxel_ptr_);

      // to global label_pointsIdx_map_thread
      for (auto label_it = label_pointsIdx_map_thread.begin();
           label_it != label_pointsIdx_map_thread.end(); label_it++) {
        auto global_label_it = label_pointsIdx_map.find(label_it->first);
        if (global_label_it == label_pointsIdx_map.end())
          label_pointsIdx_map.emplace(label_it->first, label_it->second);
        else {
          global_label_it->second.insert(global_label_it->second.end(),
                                         label_it->second.begin(),
                                         label_it->second.end());
        }
      }
    }
  }

  // LOG(INFO) << "  total hit " << int(visited_voxels_ptrs.size()) << "voxels";
  // LOG(INFO) << "  sum of thread hit " << int(visited_num) << "voxels";
  // LOG(INFO) <<  "  Ray cast result: "; 
  
  // assign current inst label for each superpoint observation
  std::map<InstanceLabel, std::map<Label, int>> instances_labels_cout;
  int th_num_voxels = 20;
  for (auto label_it = labels_instances_cout.begin();
       label_it != labels_instances_cout.end(); label_it++) {
    InstanceLabel max_cur_inst = BackgroundLabel;
    int max_cur_inst_count = 0;

    for (auto cur_inst_count_it = label_it->second.begin();
         cur_inst_count_it != label_it->second.end(); cur_inst_count_it++) {
      // LOG(INFO)<< "    Label:" << int(label_it->first)
      // << " Inst:" << int(cur_inst_count_it->first)
      // << " Count:" << int(cur_inst_count_it->second);
      if (cur_inst_count_it->second > th_num_voxels &&
          cur_inst_count_it->second > max_cur_inst_count) {
        max_cur_inst = cur_inst_count_it->first;
        max_cur_inst_count = cur_inst_count_it->second;
      }
    }
    if (max_cur_inst != BackgroundLabel) {
      instances_labels_cout[max_cur_inst].emplace(
        label_it->first, max_cur_inst_count);
    }
  }

  // determine current-global inst association
  std::set<InstanceLabel> assigned_instances;
  for (auto inst_it = instances_labels_cout.begin();
       inst_it != instances_labels_cout.end(); inst_it++) {
    // get semantic information
    auto sem_it = inst_sem_map.find(inst_it->first);
    if (sem_it == inst_sem_map.end() || sem_it->first == BackgroundSemLabel)
      continue;
    SemanticLabel semantic_label = sem_it->second;

    // determine current-global inst association
    std::map<InstanceLabel, int> global_inst_count;
    for (auto label_it = inst_it->second.begin();
         label_it != inst_it->second.end(); label_it++) {
      // account voting for each associated global inst
      InstanceLabel label_associated_inst =
          semantic_instance_label_fusion_ptr_->getInstanceLabelWithSemantic(
              label_it->first, semantic_label, assigned_instances);
      if (label_associated_inst != BackgroundLabel) {
        auto global_inst_it = global_inst_count.find(label_associated_inst);
        if (global_inst_it != global_inst_count.end()) {
          global_inst_it->second += label_it->second;
        } else
          global_inst_count.emplace(label_associated_inst, label_it->second);
      }
    }
    int max_global_inst_count = 0;
    InstanceLabel max_global_inst = BackgroundLabel;
    for (auto global_inst_it = global_inst_count.begin();
         global_inst_it != global_inst_count.end(); global_inst_it++) {
      if (global_inst_it->second > th_num_voxels &&
          global_inst_it->second > max_global_inst_count &&
          assigned_instances.find(global_inst_it->first) ==
              assigned_instances.end()) {
        max_global_inst_count = global_inst_it->second;
        max_global_inst = global_inst_it->first;
      }
    }
    if (max_global_inst == BackgroundLabel)
      max_global_inst = getFreshInstance();

    // update semantic and instance count
    // SegmentObserveConfidence update_weight = 1.0;
    // LOG(INFO) << "      Weighting by pose confidence " << pose_confidence <<
    // std::endl;
    for (auto label_it = inst_it->second.begin();
         label_it != inst_it->second.end(); label_it++) {
      // LOG(INFO) << "  Adding Label " << int(label_it->first) << "; Sem " <<
      // int(semantic_label)
      //     << "; Inst " << int(max_global_inst) <<  " with weight " <<
      //     max_global_inst_count;
      SegmentObserveConfidence update_weight = label_it->second;
      CHECK_GT(pose_confidence, 0.0 - 1e-3);
      CHECK_LT(pose_confidence, 4.0 + 1e-3);
      update_weight *= pose_confidence;
      if (update_weight > 1e-2) {
        semantic_instance_label_fusion_ptr_
            ->increaseLabelClassInstanceConfidence(
                label_it->first, semantic_label, max_global_inst,
                update_weight);
        semantic_instance_label_fusion_ptr_->increaseLabelClassConfidence(
            label_it->first, semantic_label, update_weight);
        semantic_instance_label_fusion_ptr_->increaseLabelFramesCount(
            label_it->first);
        // update superpoint-graph information
        semantic_instance_label_fusion_ptr_->increaseSegGraphConfidence(
            label_it->first, semantic_label, label_it->first, update_weight);
      }
    }
    assigned_instances.insert(max_global_inst);

    if (semantics_metadata_ptr_->isThing(semantic_label)) {
      // form bounding box point cloud for each observed superpoint
      float sample_dist = 0.01f;
      std::map<Label, pcl::PointCloud<pcl::PointXYZ>::Ptr> label_bbox_map;
      // std::map<Label, pcl::PointCloud<pcl::PointXYZ>::Ptr > label_pcl_map;
      // #pragma omp parallel for
      for (auto label_it = inst_it->second.begin();
           label_it != inst_it->second.end(); label_it++) {
        auto label_pointsIdx_it = label_pointsIdx_map.find(label_it->first);
        if (label_pointsIdx_it == label_pointsIdx_map.end()) continue;
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_ptr(
            new pcl::PointCloud<pcl::PointXYZ>);
        // add hit points to poiintcloud
        for (GlobalIndex point_idx : label_pointsIdx_it->second) {  
          pcl_ptr->emplace_back(point_idx[0] * voxel_size_,
                                point_idx[1] * voxel_size_,
                                point_idx[2] * voxel_size_);
        }
        // down sample point cloud
        pcl::VoxelGrid<pcl::PointXYZ> down_sampler;
        down_sampler.setInputCloud(pcl_ptr);
        down_sampler.setLeafSize(0.02f, 0.02f, 0.02f);
        down_sampler.filter(*pcl_ptr);
        // filter point cloud
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> outlier_filter;
        outlier_filter.setInputCloud(pcl_ptr);
        outlier_filter.setMeanK(10);
        outlier_filter.setStddevMulThresh(2.0);
        outlier_filter.filter(*pcl_ptr);
        // add point cloud to the label map
        std::list<std::pair<int, int>> sample_pairs;
        sample_pairs.emplace_back(1, 2);
        sample_pairs.emplace_back(2, 3);
        sample_pairs.emplace_back(3, 4);
        sample_pairs.emplace_back(1, 4);
        sample_pairs.emplace_back(1, 5);
        sample_pairs.emplace_back(2, 6);
        sample_pairs.emplace_back(3, 7);
        sample_pairs.emplace_back(4, 8);
        sample_pairs.emplace_back(5, 6);
        sample_pairs.emplace_back(6, 7);
        sample_pairs.emplace_back(7, 8);
        sample_pairs.emplace_back(5, 8);
        if (pcl_ptr->size() > 10) {
          // calculate bounding box
          pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
          feature_extractor.setInputCloud(pcl_ptr);
          feature_extractor.compute();
          pcl::PointXYZ min_point_OBB;
          pcl::PointXYZ max_point_OBB;
          pcl::PointXYZ position_OBB;
          Eigen::Matrix3f rotational_matrix_OBB;
          feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB,
                                   rotational_matrix_OBB);

          // build bounding box samples
          pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_bbox_ptr(
              new pcl::PointCloud<pcl::PointXYZ>);
          pcl_bbox_ptr->emplace_back(
              0.5 * (max_point_OBB.x + min_point_OBB.x),
              0.5 * (max_point_OBB.y + min_point_OBB.y),
              0.5 * (max_point_OBB.z + min_point_OBB.z));  // box center
          pcl_bbox_ptr->emplace_back(max_point_OBB.x, max_point_OBB.y,
                                     max_point_OBB.z);  // v1
          pcl_bbox_ptr->emplace_back(min_point_OBB.x, max_point_OBB.y,
                                     max_point_OBB.z);  // v2
          pcl_bbox_ptr->emplace_back(min_point_OBB.x, max_point_OBB.y,
                                     min_point_OBB.z);  // v3
          pcl_bbox_ptr->emplace_back(max_point_OBB.x, max_point_OBB.y,
                                     min_point_OBB.z);  // v4
          pcl_bbox_ptr->emplace_back(max_point_OBB.x, min_point_OBB.y,
                                     max_point_OBB.z);  // v5
          pcl_bbox_ptr->emplace_back(min_point_OBB.x, min_point_OBB.y,
                                     max_point_OBB.z);  // v6
          pcl_bbox_ptr->emplace_back(min_point_OBB.x, min_point_OBB.y,
                                     min_point_OBB.z);  // v7
          pcl_bbox_ptr->emplace_back(max_point_OBB.x, min_point_OBB.y,
                                     min_point_OBB.z);  // v8

          // sample along edges of bbox
          for (std::pair<int, int>& sample_pair : sample_pairs) {
            pcl::PointXYZ point1 = pcl_bbox_ptr->points[sample_pair.first];
            pcl::PointXYZ point2 = pcl_bbox_ptr->points[sample_pair.second];
            pcl::PointXYZ point_diff;
            point_diff.getArray3fMap() =
                point1.getArray3fMap() - point2.getArray3fMap();
            int num_samples = std::sqrt(point_diff.x * point_diff.x +
                                        point_diff.y * point_diff.y +
                                        point_diff.z * point_diff.z) /
                              sample_dist;
            for (int sample_i = 0; sample_i < num_samples; sample_i++) {
              float lambda = sample_i * 1.0 / float(num_samples);
              pcl::PointXYZ point_sample;
              point_sample.getArray3fMap() =
                  lambda * point2.getArray3fMap() +
                  (1 - lambda) * point1.getArray3fMap();
              pcl_bbox_ptr->emplace_back(point_sample);
            }
          }
          // transform bbox to global coordinates
          Eigen::Matrix4f T_W_box = Eigen::Matrix4f::Identity();
          T_W_box.block<3, 3>(0, 0) = rotational_matrix_OBB;
          T_W_box(0, 3) = position_OBB.x;
          T_W_box(1, 3) = position_OBB.y;
          T_W_box(2, 3) = position_OBB.z;
          pcl::transformPointCloud(*pcl_bbox_ptr, *pcl_bbox_ptr, T_W_box);
          label_bbox_map.emplace(label_it->first, pcl_bbox_ptr);
          // label_pcl_map.emplace(label_it->first, pcl_ptr);
        }
      }

      // calculate spatial confidence
      for (auto label_it = label_bbox_map.begin();
           label_it != label_bbox_map.end(); label_it++) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_bbox1_ptr = label_it->second;
        for (auto neight_label_it = std::next(label_it);
             neight_label_it != label_bbox_map.end(); neight_label_it++) {
          pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_bbox2_ptr =
              neight_label_it->second;
          SegSegConfidence spatial_confidence =
              calculateSpatialConfidence(pcl_bbox1_ptr, pcl_bbox2_ptr);

          SegmentObserveConfidence update_weight =
              spatial_confidence *
              std::min(inst_it->second.find(label_it->first)->second,
                       inst_it->second.find(neight_label_it->first)->second);

          // LOG(INFO) << "  Spatial Confidence Between " <<
          // int(label_it->first) << " and "
          // << int(neight_label_it->first) << " : " <<  spatial_confidence;
          update_weight *= pose_confidence;
          if (update_weight > 1e-2) {
            semantic_instance_label_fusion_ptr_->increaseSegGraphConfidence(
                label_it->first, semantic_label, neight_label_it->first,
                update_weight);
            semantic_instance_label_fusion_ptr_->increaseSegGraphConfidence(
                neight_label_it->first, semantic_label, label_it->first,
                update_weight);
          }
        }
      }
    }
  }

  // // visualize
  // using namespace std::chrono_literals;

  // pcl::visualization::PCLVisualizer::Ptr viewer (new
  // pcl::visualization::PCLVisualizer ("3D Viewer"));
  // viewer->setBackgroundColor (0, 0, 0);
  // viewer->addCoordinateSystem (1.0);
  // viewer->initCameraParameters ();
  // for(auto label_it = label_bbox_map.begin(); label_it!=
  // label_bbox_map.end();
  //         label_it++)
  // {
  //     viewer->addPointCloud<pcl::PointXYZ>(label_it->second,
  //     std::to_string(label_it->first));
  // }
  // for(auto label_it = label_pcl_map.begin(); label_it!= label_pcl_map.end();
  //         label_it++)
  // {
  //     viewer->addPointCloud<pcl::PointXYZ>(label_it->second,
  //     std::to_string(label_it->first)+"_pcl");
  // }
  // while(!viewer->wasStopped())
  // {
  //     viewer->spinOnce (100);
  //     std::this_thread::sleep_for(100ms);
  // }
}













SegSegConfidence LabelTsdfConfidenceIntegrator::calculateSpatialConfidence(
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_1,
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_2, float th_neighbor,
    float th_cut_off) {
  float th_neighbor_sq = th_neighbor * th_neighbor;
  float th_cut_off_sq = th_cut_off * th_cut_off;
  float mini_distance_sq = 10000.0f;
  for (int p1_idx = 0; p1_idx < pcl_1->size(); p1_idx++) {
    pcl::PointXYZ p1 = pcl_1->points[p1_idx];
    for (int p2_idx = 0; p2_idx < pcl_2->size(); p2_idx++) {
      pcl::PointXYZ p2 = pcl_2->points[p2_idx];
      pcl::PointXYZ point_diff;
      point_diff.getArray3fMap() = p1.getArray3fMap() - p2.getArray3fMap();
      float dist_sq = point_diff.x * point_diff.x +
                      point_diff.y * point_diff.y + point_diff.z * point_diff.z;
      // LOG(INFO) << "      dist_sq: " << dist_sq;
      if (dist_sq < th_neighbor_sq) return 1.0;
      // else if(dist_sq > th_cut_off_sq)
      //     return 0.0;

      if (dist_sq < mini_distance_sq) mini_distance_sq = dist_sq;
    }
  }

  if (mini_distance_sq <= th_neighbor_sq)
    return 1.0;
  else if (mini_distance_sq > th_cut_off_sq)
    return 0.0;
  else {
    return th_neighbor_sq * th_cut_off_sq /
               ((th_cut_off_sq - th_neighbor_sq) * mini_distance_sq) -
           th_neighbor_sq / (th_cut_off_sq - th_neighbor_sq);
  }
}

}  // namespace voxblox