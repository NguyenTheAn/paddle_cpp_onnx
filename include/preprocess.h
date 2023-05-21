#pragma once
#include "opencv2/opencv.hpp"
#include <iostream>
#include "utils.h"
#include "config.h"

namespace Preprocess{
    cv::Mat resizeImageDet(cv::Mat image, int dst_w, int dst_h);
    cv::Mat resizeImageRec(cv::Mat image, int dst_w, int dst_h);
    cv::Mat normalizeImage(cv::Mat image, cv::Scalar mean, cv::Scalar std);

    std::vector<float> preprocessDet(std::vector<cv::Mat> list_inputs, std::vector<cv::Size> &original_shapes, Config cfg);
    std::vector<float> preprocessRec(std::vector<cv::Mat> list_inputs, Config cfg);
    std::vector<float> preprocessCls(std::vector<cv::Mat> list_inputs, Config cfg);
} 
