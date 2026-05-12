# Ungraceful changes  

This file record some ungraceful modification on the source code of voxblox++.  

In order for label confidence:  

In global_segment_map/common.h:
    typedef uint16_t LabelConfidence; -> typedef float LabelConfidence;
    typedef float SegmentObserveConfidence;
    typedef std::map<SemanticLabel, int> SLMap; -> typedef std::map<SemanticLabel, SegmentObserveConfidence> SLMap;

In global_segment_map/include/segment.h:

    +:
    Segment(
    const cv::Mat* points, // use cv::Mat as input, have no idea why pybind11::array doesn't work here
    const cv::Mat* colors,
    const cv::Mat* geometry_confidence,
    InstanceLabel instance_label,
    SemanticLabel semantic_label,
    Transformation& T_G_C);

    Segment(){}
    virtual ~Segment() = default;

In global_segment_map/src/segment.cpp:

    +:
    Segment::Segment(
    const cv::Mat* points, // use cv::Mat as input, have no idea why pybind11::array doesn't work here
    const cv::Mat* colors,
    const cv::Mat* geometry_confidence,
    InstanceLabel instance_label,
    SemanticLabel semantic_label,
    Transformation& T_G_C){ ... }

In global_segment_map/include/semantic_instance_label_fusion.h:
    // for confidence
    void increaseLabelInstanceConfidence(const Label& label,
                                        const InstanceLabel& instance_label,
                                        const LabelConfidence& confidence);
In global_segment_map/src/semantic_instance_label_fusion.cpp:
    void SemanticInstanceLabelFusion::increaseLabelInstanceConfidence(const Label& label,const InstanceLabel& instance_label,const LabelConfidence& confidence) {...}