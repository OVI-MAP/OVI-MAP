#ifndef GLOBAL_SEGMENT_MAP_SEGMENT_H_
#define GLOBAL_SEGMENT_MAP_SEGMENT_H_

#include <global_segment_map/common.h>
#include <opencv2/opencv.hpp>

namespace voxblox {

class Segment {
 public:
  Segment(const pcl::PointCloud<voxblox::PointType>& point_cloud,
          const Transformation& T_G_C);

  Segment(const pcl::PointCloud<voxblox::PointLabelType>& point_cloud,
          const Transformation& T_G_C);

  Segment(
      const pcl::PointCloud<voxblox::PointSemanticInstanceType>& point_cloud,
      const Transformation& T_G_C);

    // for gms python version
  Segment(
    const cv::Mat* points, 
    // const cv::Mat* colors,
    // const cv::Mat* geometry_confidence,
    InstanceLabel instance_label,
    SemanticLabel semantic_label,
    Transformation& T_G_C
  );

  Segment() = delete;
  virtual ~Segment() = default;

  voxblox::Transformation T_G_C_;
  voxblox::Pointcloud points_C_;
  voxblox::Colors colors_;
  voxblox::Label label_;
  voxblox::SemanticLabel semantic_label_;
  voxblox::InstanceLabel instance_label_;
};
}  // namespace voxblox



#endif  // GLOBAL_SEGMENT_MAP_SEGMENT_H_
