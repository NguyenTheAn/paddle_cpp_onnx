#include "bbox.h"

BBox::BBox(cv::Point2f top_left, cv::Point2f top_right, cv::Point2f bottom_right, cv::Point2f bottom_left) {
    this->top_left = top_left;
    this->top_right = top_right;
    this->bottom_right = bottom_right;
    this->bottom_left = bottom_left;
}

BBox::BBox(std::vector<float> x, std::vector<float> y) {
    this->top_left = cv::Point2f(x[0], y[0]);
    this->top_right = cv::Point2f(x[1], y[1]);
    this->bottom_right = cv::Point2f(x[2], y[2]);
    this->bottom_left = cv::Point2f(x[3], y[3]);
}

cv::Point BBox::get_point(int index){
    if (index == 0){
        return cv::Point(this->top_left);
    }
    if (index == 1){
        return cv::Point(this->top_right);
    }
    if (index == 2){
        return cv::Point(this->bottom_right);
    }
    return cv::Point(this->bottom_left);
}

std::vector<float> BBox::get_x(){
    std::vector<float> x;
    x.push_back(this->top_left.x);
    x.push_back(this->top_right.x);
    x.push_back(this->bottom_right.x);
    x.push_back(this->bottom_left.x);

    return x;
}

std::vector<float> BBox::get_y(){
    std::vector<float> y;
    y.push_back(this->top_left.y);
    y.push_back(this->top_right.y);
    y.push_back(this->bottom_right.y);
    y.push_back(this->bottom_left.y);

    return y;
}

std::vector<cv::Point2f> BBox::to_list(){
    std::vector<cv::Point2f> list = {this->top_left, this->top_right, this->bottom_right, this->bottom_left};
    return list;
}