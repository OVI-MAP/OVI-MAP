#include "global_segment_map/segment.h"
namespace voxblox
{

    Segment::Segment(
        const pcl::PointCloud<voxblox::PointType> &point_cloud,
        const Transformation &T_G_C
    ) : T_G_C_(T_G_C), semantic_label_(0u), instance_label_(0u)
    {
        points_C_.reserve(point_cloud.points.size());
        colors_.reserve(point_cloud.points.size());

        for (size_t i = 0; i < point_cloud.points.size(); ++i)
        {
            if (!std::isfinite(point_cloud.points[i].x) ||
                !std::isfinite(point_cloud.points[i].y) ||
                !std::isfinite(point_cloud.points[i].z))
            {
                continue;
            }

            points_C_.push_back(Point(point_cloud.points[i].x, point_cloud.points[i].y,
                                      point_cloud.points[i].z));

            colors_.push_back(Color(point_cloud.points[i].r, point_cloud.points[i].g,
                                    point_cloud.points[i].b, point_cloud.points[i].a));
        }
    }

    Segment::Segment(
        const pcl::PointCloud<voxblox::PointLabelType> &point_cloud,
        const Transformation &T_G_C
    ) : T_G_C_(T_G_C), 
        label_(point_cloud.points[0].label)
    {
        points_C_.reserve(point_cloud.points.size());
        colors_.reserve(point_cloud.points.size());

        for (size_t i = 0u; i < point_cloud.points.size(); ++i)
        {
            if (!std::isfinite(point_cloud.points[i].x) ||
                !std::isfinite(point_cloud.points[i].y) ||
                !std::isfinite(point_cloud.points[i].z))
            {
                continue;
            }

            points_C_.push_back(Point(point_cloud.points[i].x, point_cloud.points[i].y,
                                      point_cloud.points[i].z));

            colors_.push_back(Color(point_cloud.points[i].r, point_cloud.points[i].g,
                                    point_cloud.points[i].b, point_cloud.points[i].a));
        }
    }

    Segment::Segment(
        const pcl::PointCloud<voxblox::PointSemanticInstanceType> &point_cloud,
        const Transformation &T_G_C
    ) : T_G_C_(T_G_C),
        semantic_label_(point_cloud.points[0].semantic_label),
        instance_label_(point_cloud.points[0].instance_label)
    {
        points_C_.reserve(point_cloud.points.size());
        colors_.reserve(point_cloud.points.size());

        for (size_t i = 0u; i < point_cloud.points.size(); ++i)
        {
            if (!std::isfinite(point_cloud.points[i].x) ||
                !std::isfinite(point_cloud.points[i].y) ||
                !std::isfinite(point_cloud.points[i].z))
            {
                continue;
            }

            points_C_.push_back(Point(point_cloud.points[i].x, point_cloud.points[i].y,
                                      point_cloud.points[i].z));

            colors_.push_back(Color(point_cloud.points[i].r, point_cloud.points[i].g,
                                    point_cloud.points[i].b, point_cloud.points[i].a));
        }
    }


    // what we are using
    Segment::Segment(
        const cv::Mat *points, 
        // const cv::Mat* colors,
        // const cv::Mat* geometry_confidence,
        InstanceLabel instance_label,
        SemanticLabel semantic_label,
        Transformation &T_G_C)
        : semantic_label_(semantic_label),
          instance_label_(instance_label),
          T_G_C_(T_G_C)
    {
        points_C_.reserve(points->rows);
        colors_.reserve(points->rows);
        // geometry_confidence_towrite.reserve(points->rows);

        if (points->type() == CV_32FC1) {
            for (size_t i = 0u; i < points->rows; ++i) {
                if (!std::isfinite((points->at<float>(i, 0))) ||
                    !std::isfinite((points->at<float>(i, 1))) ||
                    !std::isfinite((points->at<float>(i, 2))))
                {
                    LOG(INFO) << "Error in initalize segments. There is NaN points in segments! ";
                }
                points_C_.push_back(Point(
                    (points->at<float>(i, 0)), (points->at<float>(i, 1)), (points->at<float>(i, 2))));
                colors_.push_back(Color(200u, 200u, 200u, 200u));
                // geometry_confidence_towrite.push_back( geometry_confidence->at<float>(i,0) );
            }
        }
        else if (points->type() == CV_32FC3) {
            for (size_t i = 0u; i < points->cols; ++i) {
                cv::Vec3f point = points->at<cv::Vec3f>(0, i);
                if (!std::isfinite(point[0]) ||
                    !std::isfinite(point[1]) ||
                    !std::isfinite(point[2]))
                {
                    LOG(INFO) << "Error in initalize segments. There is NaN points in segments! ";
                }
                points_C_.push_back(Point(point[0], point[1], point[2]));
                colors_.push_back(Color(200u, 200u, 200u, 200u));
                // geometry_confidence_towrite.push_back( geometry_confidence->at<float>(i,0) );
            }
        }
        else {
            std::cerr << " wrong data type of points !!! " << std::endl;
            exit(-1);
        }
    }

} // namespace voxblox
