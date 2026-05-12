#ifndef GLOBAL_SEGMENT_MAP_LABEL_VOXEL_H_
#define GLOBAL_SEGMENT_MAP_LABEL_VOXEL_H_

#include <cstdint>
#include <string>

#include <voxblox/core/voxel.h>

#include "global_segment_map/common.h"

namespace voxblox {

struct LabelVoxel {
  Label label = 0u;
  LabelConfidence label_confidence = 0.0;
  LabelCount label_count[VoxelLabelArrayLen];
  LabelProbability getLabelProbability(Label label_query) const
  {
    LabelProbability label_prob = 0;
    LabelConfidence label_confidence_total = 0;
    LabelConfidence label_confidence_query = 0;
    for(int l_c_i=0; l_c_i<VoxelLabelArrayLen; l_c_i++)
    {
      // get confidence of queried label
      if(label_count[l_c_i].label == label_query)
        label_confidence_query = label_count[l_c_i].label_confidence;

      // get total confidence of labels
      if(label_count[l_c_i].label!=0u)
        label_confidence_total += label_count[l_c_i].label_confidence;
    } 
    if(label_confidence_total >= 0.99 )
      label_prob = label_confidence/label_confidence_total;

    return label_prob;
  }
};

namespace voxel_types {
const std::string kLabel = "label";
}  // namespace voxel_types

template <>
inline std::string getVoxelType<LabelVoxel>() {
  return voxel_types::kLabel;
}

}  // namespace voxblox

#endif  // GLOBAL_SEGMENT_MAP_LABEL_VOXEL_H_
