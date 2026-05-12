#include "depth_segmentation_py/depth_segmentation_py.h"
#include <chrono> // chrono::system_clock
#include <ctime>  // localtime

void DepthSegmentation_py::depthSegment(pybind11::array &depth_image_np, pybind11::array &rgb_image)
{
    cv::Mat depth_image_mat = cvnp::nparray_to_mat(depth_image_np);
    cv::Mat rgb_image_mat = cvnp::nparray_to_mat(rgb_image);
    rgb_image_mat.convertTo(rgb_image_mat, CV_8UC3);

    auto time_original = std::chrono::system_clock::now();

    depth_map_ = cv::Mat::zeros(depth_image_np.shape(0),
                                depth_image_np.shape(1), CV_32FC3);
    dep_segmenter_ptr_->computeDepthMap(depth_image_mat, &depth_map_);

    normal_map_ = cv::Mat::zeros(depth_image_np.shape(0),
                                 depth_image_np.shape(1), CV_32FC3);
    dep_segmenter_ptr_->computeNormalMap(depth_map_, &normal_map_);

    discontinuity_map_ = cv::Mat::zeros(depth_image_np.shape(0),
                                        depth_image_np.shape(1), CV_32FC1);
    dep_segmenter_ptr_->computeDepthDiscontinuityMap(depth_image_mat, &discontinuity_map_);

    max_distance_map_ = cv::Mat::zeros(depth_image_np.shape(0),
                                       depth_image_np.shape(1), CV_32FC1);
    dep_segmenter_ptr_->computeMaxDistanceMap(depth_map_, &max_distance_map_);

    min_convex_map_ = cv::Mat::zeros(depth_image_np.shape(0),
                                     depth_image_np.shape(1), CV_32FC1);
    dep_segmenter_ptr_->computeMinConvexityMap(depth_map_, normal_map_,
                                               &min_convex_map_);

    edge_map_ = cv::Mat::zeros(depth_image_np.shape(0),
                               depth_image_np.shape(1), CV_32FC1);
    dep_segmenter_ptr_->computeFinalEdgeMap(min_convex_map_, max_distance_map_,
                                            discontinuity_map_, &edge_map_);

    cv::Mat remove_no_values =
        cv::Mat::zeros(edge_map_.size(), edge_map_.type());
    edge_map_.copyTo(remove_no_values,
                     depth_image_mat == depth_image_mat);
    edge_map_ = remove_no_values;
    label_map_ = cv::Mat(depth_image_np.shape(0), depth_image_np.shape(1), CV_32FC1);
    segments_.clear();
    segment_masks_.clear();

    // dep_segmenter_ptr_->labelMap(rgb_image_mat, depth_image_mat,
    //                         depth_map_, edge_map_, normal_map_, &label_map_,
    //                         &segment_masks_, &segments_);
    segment_mask_single_ = cv::Mat::zeros(depth_image_np.shape(0), depth_image_np.shape(1), CV_8UC1);
    // algorithm from Voxblox
    dep_segmenter_ptr_->labelMapSingleMask(rgb_image_mat, depth_image_mat,
                                           depth_map_, edge_map_, normal_map_, &label_map_,
                                           &segment_masks_, &segments_, &segment_mask_single_);

    auto time_end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration<double>(time_end - time_original).count();
    LOG(INFO) << "     Total depth segmentation cost: " << duration << " seconds";
}

pybind11::array DepthSegmentation_py::get_segmentMasks()
{
    std::vector<pybind11::array> masks_array;
    masks_array.resize(segment_masks_.size());
    for (size_t mask_i = 0; mask_i < segment_masks_.size(); mask_i++)
    {
        masks_array[mask_i] = cvnp::mat_to_nparray(segment_masks_[mask_i], false);
    }
    return pybind11::array(pybind11::cast(masks_array));
}






PYBIND11_MODULE(depth_segmentation_py, m)
{
    m.doc() = "pybind11 for depth segmentation"; // optional module docstring

    pybind11::class_<DepthSegmentation_py>(m, "DepthSegmentation_py")
        .def(pybind11::init<size_t, size_t, int, pybind11::array>())
        .def("get_cameraMatrix", &DepthSegmentation_py::get_cameraMatrix)
        .def("depthSegment", &DepthSegmentation_py::depthSegment)
        .def("get_depthMap", &DepthSegmentation_py::get_depthMap)
        .def("get_normalMap", &DepthSegmentation_py::get_normalMap)
        .def("get_discontinuityMap", &DepthSegmentation_py::get_discontinuityMap)
        .def("get_maxDistanceMap", &DepthSegmentation_py::get_maxDistanceMap)
        .def("get_minConvexMap", &DepthSegmentation_py::get_minConvexMap)
        .def("get_edgeMap", &DepthSegmentation_py::get_edgeMap)
        .def("get_labelMap", &DepthSegmentation_py::get_labelMap)
        .def("get_segmentMasks", &DepthSegmentation_py::get_segmentMasks)
        .def("get_segmentMaskSingleFrame", &DepthSegmentation_py::get_segmentMaskSingleFrame);
}