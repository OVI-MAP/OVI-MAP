#include "depth_segmentation_py/depth_segmentation_py.h"

/***compute depth map with depth images
    input type: (depth_image, CV_32FC1)
    output type: (depth_map, CV_32FC3) ***/
pybind11::array DepthSegmentation_py::computeDepthMap_py( pybind11::array &depth_image_np){
    // convert numpy array to cv::mat
    cv::Mat depth_image_mat = cvnp::nparray_to_mat(depth_image_np);
    cv::Mat depth_map_mat = cv::Mat::zeros(depth_image_np.shape(0),
                                depth_image_np.shape(1), CV_32FC3);
    dep_segmenter_ptr_->computeDepthMap(depth_image_mat, &depth_map_mat);
    return cvnp::mat_to_nparray(depth_map_mat, false); // copy mat to np_array, not efficient TODO
}

/***compute normal map with depth map
    input type: (depth_map, CV_32FC3)
    output type: (normal_map, CV_32FC3) ***/
pybind11::array DepthSegmentation_py::computeNormalMap_py(pybind11::array &depth_map_np){
    // convert numpy array to cv::mat
    cv::Mat depth_map_mat = cvnp::nparray_to_mat(depth_map_np);
    cv::Mat normal_map_mat = cv::Mat::zeros(depth_map_np.shape(0),
                                depth_map_np.shape(1), CV_32FC3);
    dep_segmenter_ptr_->computeNormalMap(depth_map_mat, &normal_map_mat);
    return cvnp::mat_to_nparray(normal_map_mat, false); // copy array, not efficient TODO
}

/***compute depth discontinuity map with depth image
    input type: (depth_image, CV_32FC1)
    output type: (discontinuity_map, CV_32FC1) ***/
pybind11::array DepthSegmentation_py::computeDepthDiscontinuityMap_py(pybind11::array &depth_image_np){

    cv::Mat depth_image_mat = cvnp::nparray_to_mat(depth_image_np);
    cv::Mat discontinuity_map_mat = cv::Mat::zeros(depth_image_np.shape(0),
                            depth_image_np.shape(1), CV_32FC1);
    dep_segmenter_ptr_->computeDepthDiscontinuityMap(depth_image_mat, &discontinuity_map_mat);
    return cvnp::mat_to_nparray(discontinuity_map_mat, false); // copy array, not efficient TODO
}

/***compute max distance map with depth map
    input type: (depth_map, CV_32FC3)
    output type: (max_distance_map, CV_32FC1) ***/
pybind11::array DepthSegmentation_py::computeMaxDistanceMap_py(pybind11::array &depth_map_np){
    cv::Mat depth_map_mat = cvnp::nparray_to_mat(depth_map_np);
    cv::Mat max_dist_map_mat = cv::Mat::zeros(depth_map_np.shape(0),
                        depth_map_np.shape(1), CV_32FC1);
    dep_segmenter_ptr_->computeMaxDistanceMap(depth_map_mat, &max_dist_map_mat);
    return cvnp::mat_to_nparray(max_dist_map_mat, false); // copy array, not efficient TODO
}

/***compute min convexity map with depth map and normal map
    input type: (depth_map, CV_32FC3)
                (normal_map, CV_32FC3)
    output type: (min_convex_map, CV_32FC1) ***/
pybind11::array DepthSegmentation_py::computeMinConvexityMap_py(pybind11::array &depth_map_np,pybind11::array &normal_map_np){
    cv::Mat depth_map_mat = cvnp::nparray_to_mat(depth_map_np);
    cv::Mat normal_map_mat = cvnp::nparray_to_mat(normal_map_np);
    cv::Mat min_convex_map_mat = cv::Mat::zeros(depth_map_np.shape(0),
                    depth_map_np.shape(1), CV_32FC1);
    dep_segmenter_ptr_->computeMinConvexityMap(depth_map_mat, normal_map_mat,
                                        &min_convex_map_mat);
    return cvnp::mat_to_nparray(min_convex_map_mat, false); // copy array, not efficient TODO
}

/***compute final edge with convexity map, distance map, discontinuity map
    input type: (min_convex_map, CV_32FC1)
                (max_distance_map, CV_32FC1)
                (discontinuity_map, CV_32FC1)
    output type: (edge_map, CV_32FC1) ***/
pybind11::array DepthSegmentation_py::computeFinalEdgeMap_py(pybind11::array &min_convex_map,pybind11::array &max_distance_map,pybind11::array &discontinuity_map){
    cv::Mat min_convex_map_mat = cvnp::nparray_to_mat(min_convex_map);
    cv::Mat max_distance_map_mat = cvnp::nparray_to_mat(max_distance_map);
    cv::Mat discontinuity_map_mat = cvnp::nparray_to_mat(discontinuity_map);
    cv::Mat edge_map_mat = cv::Mat::zeros(min_convex_map.shape(0),
                min_convex_map.shape(1), CV_32FC1);
    dep_segmenter_ptr_->computeFinalEdgeMap(min_convex_map_mat, max_distance_map_mat,
                                        discontinuity_map_mat, &edge_map_mat);
    return cvnp::mat_to_nparray(edge_map_mat, false); // copy array, not efficient TODO
}

/***compute label map for depth segments with edge_map, rgb_image, depth_image, depth_map, normal_map
    input type: (edge_map, CV_32FC1)
                (rgb_image, CV_8UC3)
                (depth_image, CV_32FC1)
                (depth_map, CV_32FC3)
                (normal_map, CV_32FC3)
    output type: (label_map, CV_32FC1) ***/
void DepthSegmentation_py::labelMap_py(pybind11::array &edge_map,pybind11::array &rgb_image,pybind11::array &depth_image,pybind11::array &depth_map,pybind11::array &normal_map){
    cv::Mat edge_map_mat = cvnp::nparray_to_mat(edge_map);
    cv::Mat rgb_image_mat = cvnp::nparray_to_mat(rgb_image);
    rgb_image_mat.convertTo(rgb_image_mat, CV_8UC3); // rgb image CV_8UC3->CV_32FC3->CV_8UC3, not efficient TODO 
    cv::Mat depth_image_mat = cvnp::nparray_to_mat(depth_image);
    cv::Mat depth_map_mat = cvnp::nparray_to_mat(depth_map);
    cv::Mat normal_map_mat = cvnp::nparray_to_mat(normal_map);

    // cv::Mat label_map_mat(edge_map.shape(0),edge_map.shape(1), CV_32FC1);
    // cv::Mat label_map_mat = cvnp::nparray_to_mat(label_map);
    
    label_map_ = cv::Mat(edge_map.shape(0),edge_map.shape(1), CV_32FC1);
    segments_.clear();
    segment_masks_.clear();
    dep_segmenter_ptr_->labelMap(rgb_image_mat, depth_image_mat,
                            depth_map_mat, edge_map_mat, normal_map_mat, &label_map_,
                            &segment_masks_, &segments_);
    // return cvnp::mat_to_nparray(label_map_mat, false); // copy array, not efficient TODO
    // return cvnp::mat_to_nparray(edge_map_mat, false);
}



PYBIND11_MODULE(depth_segmentation_py, m) {
    m.doc() = "pybind11 for depth segmentation"; // optional module docstring
    
    pybind11::class_<DepthSegmentation_py>(m, "DepthSegmentation_py")
        .def(pybind11::init<size_t ,size_t ,int, pybind11::array>())
        .def("computeDepthMap_py", &DepthSegmentation_py::computeDepthMap_py, "compute depth map from depth image")
        .def("computeNormalMap_py", &DepthSegmentation_py::computeNormalMap_py)
        .def("computeDepthDiscontinuityMap_py", &DepthSegmentation_py::computeDepthDiscontinuityMap_py)
        .def("computeMaxDistanceMap_py", &DepthSegmentation_py::computeMaxDistanceMap_py)
        .def("computeMinConvexityMap_py", &DepthSegmentation_py::computeMinConvexityMap_py)
        .def("computeFinalEdgeMap_py", &DepthSegmentation_py::computeFinalEdgeMap_py)
        .def("labelMap_py", &DepthSegmentation_py::labelMap_py)
        .def("get_labelMap", &DepthSegmentation_py::get_labelMap);
}