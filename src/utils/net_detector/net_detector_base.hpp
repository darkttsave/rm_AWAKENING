#pragma once
#include "utils/common/image.hpp"
#include <opencv2/core/mat.hpp>
#include <optional>
namespace awakening::utils {

class NetDetectorBase {
public:
    struct OutPut {
        cv::Mat output;
        Eigen::Matrix3f transform_matrix;
        cv::Mat resized_img;
        double preprocess_ms = 0.0;
        double inference_ms = 0.0;
        double output_copy_ms = 0.0;
    };
    struct Config {
        PixelFormat target_format;
        double preprocess_scale;
        int target_w;
        int target_h;
    };
    using Ptr = std::unique_ptr<NetDetectorBase>;

    virtual OutPut detect(const cv::Mat& img, PixelFormat format) = 0;
    virtual ~NetDetectorBase() = default;
};
} // namespace awakening::utils
