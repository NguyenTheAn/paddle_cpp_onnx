#pragma once
#include "opencv2/opencv.hpp"
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include "config.h"
#include "preprocess.h"
#include "utils.h"
#include "bbox.h"
#include "postprocess.h"
#include <tensorrt_provider_factory.h>
#include <tensorrt_provider_options.h>
#include <chrono>

class TextDetector{

    public:
        DetPostProcess postprocess;
        Ort::Session session{nullptr};

        Config cfg;

        TextDetector(){}
        TextDetector(Config cfg);

        std::vector<std::vector<BBox>> detect(std::vector<cv::Mat> list_imgs);
};