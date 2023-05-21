#include "textdetector.h"
#include "textrecognizer.h"
#include "textclassifier.h"

void test_detector(Config cfg){

    TextDetector *textDetector = new TextDetector(cfg);

    // create input tensor
    cv::Mat imageBGR1 = cv::imread("../pan-card.jpg", cv::IMREAD_COLOR);
    cv::Mat imageBGR2 = cv::imread("../2335653_pan.png", cv::IMREAD_COLOR);
    std::vector<cv::Mat> list_imgs{imageBGR1, imageBGR2};

    std::vector<std::vector<BBox>> list_bboxes = textDetector->detect(list_imgs);

    // visualize
    for (int i=0; i<list_imgs.size(); i++){
        cv::Mat img = list_imgs[i];

        for (int j=0; j<list_bboxes[i].size(); j++){
            auto box = list_bboxes[i][j];

            auto rect = cv::Rect(box.get_point(0), box.get_point(2));
            auto crop = img(rect);
            cv::imwrite("../crop_images/" + std::to_string(i) + "_" + std::to_string(j) + ".jpg", crop);

            int thickness = 2;
            cv::line(img, box.get_point(0), box.get_point(1), cv::Scalar(0, 255, 0), thickness);
            cv::line(img, box.get_point(1), box.get_point(2), cv::Scalar(0, 255, 0), thickness);
            cv::line(img, box.get_point(2), box.get_point(3), cv::Scalar(0, 255, 0), thickness);
            cv::line(img, box.get_point(0), box.get_point(3), cv::Scalar(0, 255, 0), thickness);
        }
        cv::imshow(std::to_string(i+1), img);

    }
    cv::waitKey(0);
    cv::destroyAllWindows();

    if (cfg.det_useTRT) delete textDetector;
}

void test_recognizer(Config cfg){
    TextRecognizer *textrecognize = new TextRecognizer(cfg);

    cv::Mat imageBGR1 = cv::imread("../crop_images/0_6.jpg");
    cv::Mat imageBGR2 = cv::imread("../crop_images/0_7.jpg");
    
    std::vector<cv::Mat> list_imgs{imageBGR1, imageBGR2};

    std::vector<std::pair<std::string, float>> rec_results = textrecognize->rec(list_imgs);

    for (auto rec : rec_results){
        print rec.first<<" "<<rec.second<<"\n";
    }

    if (cfg.rec_useTRT) delete textrecognize;
}

void test_classifier(Config cfg){
    cv::Mat imageBGR1 = cv::imread("../crop_images/0_6.jpg");
    cv::Mat imageBGR2 = cv::imread("../crop_images/0_7.jpg");
    
    std::vector<cv::Mat> list_imgs{imageBGR1, imageBGR2};
    
    TextClassifier *textclassifier = new TextClassifier(cfg);
    std::vector<std::pair<std::string, float>> cls_results = textclassifier->cls(list_imgs);

    for (auto cls : cls_results){
        print cls.first<<" "<<cls.second<<"\n";
    }

    if (cfg.cls_useTRT) delete textclassifier;
}


int main(int argc, char* argv[]){

    Config cfg("../config.yaml");
    test_detector(cfg);
    test_recognizer(cfg);
    test_classifier(cfg);

    return 0;
}