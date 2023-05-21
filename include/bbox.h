#pragma once
#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>

class BBox {
    public:
        cv::Point2f top_left, top_right, bottom_right, bottom_left; 
        BBox(cv::Point2f top_left, cv::Point2f top_right, cv::Point2f bottom_right, cv::Point2f bottom_left);
        BBox(std::vector<float> x, std::vector<float> y);

        std::vector<float> get_x();
        std::vector<float> get_y();
        cv::Point get_point(int index);
        std::vector<cv::Point2f> to_list();

        friend std::ostream& operator<<(std::ostream& os, const BBox& bbox) {
            os<<bbox.top_left<<" "<<bbox.top_right<<" "<<bbox.bottom_right<<" "<<bbox.bottom_left<<"\n";
            return os;
        }
};