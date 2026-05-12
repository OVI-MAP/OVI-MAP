#ifndef CAMERA_RAY_GENERATOR_H_
#define CAMERA_RAY_GENERATOR_H_

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/mat.hpp>
#include <glog/logging.h>
#include <omp.h>

namespace voxblox {
class CameraRayGenerator{
public:
    CameraRayGenerator(cv::Mat K_camera, int img_height, int img_width, 
        float range_min, float range_max, int thread_num = 1):
    K_camera_(K_camera), img_height_(img_height), img_width_(img_width),
    range_min_(range_min), range_max_(range_max)
    {
        cv::Mat K_inv = K_camera_.inv();

        omp_set_num_threads(thread_num);
        #pragma omp parallel for
        for(int row_i=0; row_i < img_height; row_i++)
            for(int col_j=0; col_j < img_width; col_j++)
            {
                cv::Vec3f pt(float(col_j), float(row_i), float(1.0));
                cv::Mat pt3D_mat = K_inv * pt;
                cv::Vec3f pt3D(pt3D_mat.at<float>(0), pt3D_mat.at<float>(1), pt3D_mat.at<float>(2));
                // get start and end point for the ray
                points_C_start_.at<cv::Vec3f>(row_i, col_j) = pt3D * range_min_;
                points_C_end_.at<cv::Vec3f>(row_i, col_j) = pt3D * range_max_;
                points_C_unit_.at<cv::Vec3f>(row_i, col_j) = pt3D;
            }

        LOG(INFO) << "range_min_: " << range_min_ << " range_max_: " << range_max_;
        
    }

    const cv::Vec3f& getRayStart(const int& row_i, const int& col_j) const 
    {
        return points_C_start_.at<cv::Vec3f>(row_i, col_j);
    }
    const cv::Vec3f& getRayEnd(const int& row_i, const int& col_j) const 
    {
        return points_C_end_.at<cv::Vec3f>(row_i, col_j);
    }
    const cv::Vec3f getRayWithDepth(const int& row_i, const int& col_j, const float& depth) const 
    {
        CHECK_GT(depth, 0.0);
        return points_C_unit_.at<cv::Vec3f>(row_i, col_j) * depth;
    }
    const float getRangeMax() const
    {
        return range_max_;
    } 
    const float getRangeMin() const
    {
        return range_min_;
    } 

    cv::Mat K_camera_;
    int img_height_;
    int img_width_;
    float range_min_;
    float range_max_;
    cv::Mat points_C_start_ = cv::Mat(img_height_, img_width_, CV_32FC3);
    cv::Mat points_C_end_ = cv::Mat(img_height_, img_width_, CV_32FC3);
    cv::Mat points_C_unit_ = cv::Mat(img_height_, img_width_, CV_32FC3);
};
}

#endif  // CAMERA_RAY_GENERATOR_H_