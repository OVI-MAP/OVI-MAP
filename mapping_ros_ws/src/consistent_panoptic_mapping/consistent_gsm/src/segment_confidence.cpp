#include "consistent_mapping/segment_confidence.h"

namespace voxblox {
    bool MY_DEBUG = false;

    SegmentConfidence::SegmentConfidence(
        const cv::Mat* points, // float
        // const cv::Mat* colors, // rgba uint8_t
        // const cv::Mat* geometry_confidence, //float32
        const cv::Mat* b_box, //float32
        InstanceLabel instance_label, //uint16_t
        SemanticLabel semantic_label, //uint8_t
        Transformation& T_G_C,
        ObjSegConfidence inst_confidence,
        ObjSegConfidence obj_seg_confidence,
        bool is_thing,
        Label designated_label
    ) : Segment(points,instance_label,semantic_label,T_G_C),
        inst_confidence_(inst_confidence),
        obj_seg_confidence_(obj_seg_confidence), 
        is_thing_(is_thing),
        b_box_(*b_box)
    {
        // std::cout << "  A segments with " << points_C_.size() << " points generated " << std::endl;
        label_ = designated_label;
    }


    SegmentConfidence::SegmentConfidence(
        const cv::Mat* points, // float
        // const cv::Mat* colors, // rgba uint8_t
        // const cv::Mat* geometry_confidence, //float32
        const cv::Mat* b_box, //float32
        InstanceLabel instance_label, //uint16_t
        SemanticLabel semantic_label, //uint8_t
        Transformation& T_G_C,
        float pose_confidence, 
        ObjSegConfidence inst_confidence,
        ObjSegConfidence obj_seg_confidence,
        bool is_thing,
        Label designated_label
    ) : Segment(points,instance_label,semantic_label,T_G_C),
        inst_confidence_(inst_confidence),
        obj_seg_confidence_(obj_seg_confidence), 
        is_thing_(is_thing),
        b_box_(*b_box),
        pose_confidence_(pose_confidence)
    {
        label_ = designated_label;
    }
}