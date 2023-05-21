#pragma once
#include "textdetector.h"
#include "textrecognizer.h"
#include "textclassifier.h"

class PP_OCR{
    public:
        explicit PP_OCR(){}
        PP_OCR(std::string cfg_path);

        TextDetector *textDetector = nullptr;
        TextRecognizer *textRecognize = nullptr;
        TextClassifier *textClassifier = nullptr;

        Config cfg;

        void operator()(std::vector<cv::Mat> list_imgs);

        ~PP_OCR();
};