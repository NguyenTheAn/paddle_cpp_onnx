#include "ocr.h"

PP_OCR::PP_OCR(std::string cfg_path){
    cfg = Config(cfg_path);

    textDetector = new TextDetector(cfg);
    textRecognize = new TextRecognizer(cfg);
    if (cfg.use_angle_cls) textClassifier = new TextClassifier(cfg);

}

PP_OCR::~PP_OCR(){
    if (cfg.det_useTRT){
        delete textDetector;
    }
    if (cfg.rec_useTRT){
        delete textRecognize;
    }
    if (cfg.cls_useTRT){
        delete textClassifier;
    }
}

void PP_OCR::operator()(std::vector<cv::Mat> list_imgs){
    std::vector<std::vector<BBox>> list_bboxes = textDetector->detect(list_imgs);

    for (auto bboxes : list_bboxes){
        utils::sorted_box(bboxes);
    }

}