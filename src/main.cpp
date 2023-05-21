#include "ocr.h"
#define print std::cout<<

int main(int argc, char* argv[]){

    PP_OCR pp_ocr("../config.yaml");

    cv::Mat imageBGR1 = cv::imread("../pan-card.jpg", cv::IMREAD_COLOR);
    // cv::Mat imageBGR2 = cv::imread("../2335653_pan.png", cv::IMREAD_COLOR);
    std::vector<cv::Mat> list_imgs{imageBGR1};
    pp_ocr(list_imgs);


    return 0;
}
