#ifndef SEGMENT_CONFIDENCE_H_
#define SEGMENT_CONFIDENCE_H_

#include <global_segment_map/segment.h>

#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

namespace voxblox {

class SegmentConfidence: public Segment {
public:
    SegmentConfidence(
        const cv::Mat* points, // float
        // const cv::Mat* geometry_confidence, //float
        const cv::Mat* b_box, //float32
        InstanceLabel instance_label, //uint16_t
        SemanticLabel semantic_label, //uint8_t
        Transformation& T_G_C,
        ObjSegConfidence inst_confidence,
        ObjSegConfidence obj_seg_confidence,
        bool is_thing, 
        Label designated_label = BackgroundLabel
    );

    SegmentConfidence(
        const cv::Mat* points, // float
        // const cv::Mat* geometry_confidence, //float
        const cv::Mat* b_box, //float32
        InstanceLabel instance_label, //uint16_t
        SemanticLabel semantic_label, //uint8_t
        Transformation& T_G_C,
        float pose_confidence,
        ObjSegConfidence inst_confidence,
        ObjSegConfidence obj_seg_confidence,
        bool is_thing, 
        Label designated_label = BackgroundLabel
    );

    // currently, the size of segments are just number of points
    SegSegConfidence getSegmentSize(){
        return SegSegConfidence(points_C_.size());
    }

    ObjSegConfidence obj_seg_confidence_ = 0;
    SegSegConfidence inst_confidence_ = 0;
    SegSegConfidence seg_label_confidence_ = 0.;
    float pose_confidence_ = 1.0;
    
    cv::Mat b_box_;
    bool is_thing_;
    std::vector<GeometricConfidence> geometry_confidence_;

    // for set of SegmentConfidence*, use pointclouds size to sort
    struct PtrCompare
    {
        bool operator()(const Segment* leftPtr, const Segment* rightPtr) const
        {
            if(leftPtr->points_C_.size()!=rightPtr->points_C_.size())
                return leftPtr->points_C_.size()>rightPtr->points_C_.size();
            // in case the size of two segments is the same  
            else
                return leftPtr<rightPtr;

            // return leftPtr<rightPtr;
        }
    };
};

}

#endif  // SEGMENT_CONFIDENCE_H_