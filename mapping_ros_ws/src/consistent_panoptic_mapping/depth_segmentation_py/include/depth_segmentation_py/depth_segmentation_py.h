#ifndef DEPTH_SEGMENTATION_PY_H_
#define DEPTH_SEGMENTATION_PY_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include<opencv2/opencv.hpp>

#include "depth_segmentation/depth_segmentation.h"
#include "depth_segmentation/ros_common.h"
#include "cvnp/cvnp.h"

using namespace depth_segmentation;




class DepthSegmentation_py
{
private:
    /* config data */
    DepthCamera dep_camera;
    RgbCamera rgb_camera;
    cv::Mat K_mat_;
    Params param_seg_py;
    /* segmenter */
    std::shared_ptr<DepthSegmenter> dep_segmenter_ptr_;
    /* results */
    cv::Mat depth_map_;
    cv::Mat normal_map_;
    cv::Mat discontinuity_map_;
    cv::Mat max_distance_map_;
    cv::Mat min_convex_map_;
    cv::Mat edge_map_;
    cv::Mat label_map_;
    std::vector<depth_segmentation::Segment> segments_;
    std::vector<cv::Mat> segment_masks_;
    cv::Mat segment_mask_single_;


public:
    DepthSegmentation_py(size_t img_height,size_t img_width, 
                        int data_type, pybind11::array K);
    ~DepthSegmentation_py();

    void depthSegment(pybind11::array &depth_image_np, pybind11::array &rgb_image);

    pybind11::array get_cameraMatrix(){ return cvnp::mat_to_nparray(K_mat_, false);}
    /* get segmentation results */
    pybind11::array get_depthMap(){ return cvnp::mat_to_nparray(depth_map_, false);}
    pybind11::array get_normalMap(){ return cvnp::mat_to_nparray(normal_map_, false);}
    pybind11::array get_discontinuityMap(){ return cvnp::mat_to_nparray(discontinuity_map_, false);}
    pybind11::array get_maxDistanceMap(){ return cvnp::mat_to_nparray(max_distance_map_, false);}
    pybind11::array get_minConvexMap(){ return cvnp::mat_to_nparray(min_convex_map_, false);}
    pybind11::array get_edgeMap(){ return cvnp::mat_to_nparray(edge_map_, false);}
    pybind11::array get_labelMap(){ return cvnp::mat_to_nparray(label_map_, false);}

    pybind11::array get_segmentMasks();
    pybind11::array get_segmentMaskSingleFrame(){return cvnp::mat_to_nparray(segment_mask_single_, false);};
};

DepthSegmentation_py::DepthSegmentation_py(size_t img_height,size_t img_width, 
                                            int data_type, pybind11::array K)
{
    param_seg_py.label.display = false;
    // initialized cameras
    K_mat_ = cvnp::nparray_to_mat(K);
    dep_camera.initialize(img_height,img_width,data_type,K_mat_);
    rgb_camera.initialize(img_height,img_width,data_type,K_mat_);

    // instantiate segmenter
    dep_segmenter_ptr_ = std::make_shared<DepthSegmenter>(dep_camera,param_seg_py);
    dep_segmenter_ptr_->initialize();

}

DepthSegmentation_py::~DepthSegmentation_py()
{
}


#endif  // DEPTH_SEGMENTATION_PY_H_