#pragma once
#include "opencv2/opencv.hpp"
#include <iostream>
#include "utils.h"
#include "config.h"
#include "bbox.h"
#include "clipper2/clipper.h"

class DetPostProcess{
    public:
        int max_candidates=1000;
        int min_size;
        cv::Mat dilation_kernel;
        Config cfg;

        explicit DetPostProcess(){};
        DetPostProcess( Config cfg, int max_candidates=1000);
                        
        std::vector<BBox> bboxes_from_bitmap(cv::Mat pred,cv::Mat _bitmap, int dest_width, int dest_height);

        BBox get_mini_boxes(std::vector<cv::Point> contour, int& minside);
        float box_score_fast(cv::Mat bitmap, BBox bbox);
        std::vector<cv::Point> unclip(BBox box);

        std::vector<std::vector<BBox>> operator()(float *output_arr, std::vector<cv::Size> original_shapes, int batch_size);
};

class BaseRecLabelDecode{
    public:
        explicit BaseRecLabelDecode(){};
        BaseRecLabelDecode( Config cfg );

        std::string begin_str = "sos";
        std::string end_str = "eos";
        std::string character_str = "";
        std::map<char, int> dict;
        Config cfg;

        std::vector<std::pair<std::string, float>> decode(std::vector<std::vector<int>> pred_idxs, std::vector<std::vector<float>> preb_probs, bool is_remove_duplicate, bool get_conf);
};

class CTCLabelDecode : public BaseRecLabelDecode{
    public:
        explicit CTCLabelDecode(){};
        CTCLabelDecode(Config cfg) : BaseRecLabelDecode(cfg){};
        
        std::vector<std::pair<std::string, float>> operator()(float *output_arr, std::vector<int64_t> outputTensorShape);
};

class ClsPostProcess{
    public:
        explicit ClsPostProcess(){};
        ClsPostProcess( Config cfg );

        std::vector<std::pair<std::string, float>> operator()(float *output_arr, std::vector<int64_t> outputTensorShape);

        std::vector<std::string> label_list{"0", "180"};
        Config cfg;
};