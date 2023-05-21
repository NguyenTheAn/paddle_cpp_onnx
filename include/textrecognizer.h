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

class TextRecognizer{

    public:

        CTCLabelDecode postprocess;
        Ort::Session session{nullptr};

        Config cfg;

        TextRecognizer(){}
        TextRecognizer(Config cfg);

        std::vector<std::pair<std::string, float>> rec(std::vector<cv::Mat> list_imgs);
};