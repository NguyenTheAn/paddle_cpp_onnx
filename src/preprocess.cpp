#include "preprocess.h"

cv::Mat Preprocess::resizeImageDet(cv::Mat image, int dst_w, int dst_h){
    int h = image.rows;
    int w = image.cols;

    float w_ratio =  dst_w * 1.0 / w;
    float h_ratio = dst_h * 1.0 / h;

    float ratio = std::min(w_ratio, h_ratio);

    cv::Mat resized_img;
    cv::resize(image, resized_img, cv::Size(), ratio, ratio);

    int padding_right =  std::max(dst_w - resized_img.cols, 0);
    int padding_bottom = std::max(dst_h - resized_img.rows, 0);
    cv::Mat output;
    cv::copyMakeBorder(resized_img, output, 0, padding_bottom, 0, padding_right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    
    return output;
}

cv::Mat Preprocess::resizeImageRec(cv::Mat image, int dst_w, int dst_h){
    int h = image.rows;
    int w = image.cols;

    float dst_ratio = dst_w * 1.0 / dst_h;
    float ratio = w * 1.0 / h;

    int resized_w;

    if (std::ceil(dst_h * ratio) > dst_w){
        resized_w = dst_w;
    }
    else{
        resized_w = int(std::ceil(dst_h * ratio));
    }

    cv::Mat resized_img;
    cv::resize(image, resized_img, cv::Size(resized_w, dst_h));
    
    return resized_img;
}

cv::Mat Preprocess::normalizeImage(cv::Mat image, cv::Scalar mean, cv::Scalar std){

    cv::Mat floatImage;
    image.convertTo(floatImage, CV_32FC3, 1 / 255.0);

    cv::Mat channels[3];
    cv::split(floatImage, channels);
    channels[0] = (channels[0] - mean[0]) / std[0];
    channels[1] = (channels[1] - mean[1]) / std[1];
    channels[2] = (channels[2] - mean[2]) / std[2];
    cv::merge(channels, 3, floatImage);

    return floatImage;
}

std::vector<float> Preprocess::preprocessDet(std::vector<cv::Mat> list_inputs, std::vector<cv::Size> &original_shapes, Config cfg){
    std::vector<float> batched_vector;
    for (auto image : list_inputs){

        int max_shape = std::max(image.cols, image.rows);
        original_shapes.push_back(cv::Size(max_shape, max_shape));
        
        cv::Mat resizedImage = Preprocess::resizeImageDet(image, cfg.det_imgsz, cfg.det_imgsz);
    
        cv::Scalar mean(0.485, 0.456, 0.406);
        cv::Scalar std(0.229, 0.224, 0.225);
        cv::Mat floatImage = Preprocess::normalizeImage(resizedImage, mean, std);
        
        // hwc -> chw
        cv::Mat preprocessedImage;
        cv::dnn::blobFromImage(floatImage, preprocessedImage);

        // flatten
        size_t input_size = 3 * floatImage.rows * floatImage.cols;

        std::vector<float> image_vector(input_size);
        std::copy(preprocessedImage.begin<float>(),
                    preprocessedImage.end<float>(),
                    image_vector.begin());

        batched_vector.reserve(batched_vector.size() + image_vector.size());
        std::copy(image_vector.begin(), image_vector.end(), std::back_inserter(batched_vector));
    }

    return batched_vector;
}

std::vector<float> Preprocess::preprocessRec(std::vector<cv::Mat> list_inputs, Config cfg){
    std::vector<float> batched_vector;
    for (auto image : list_inputs){

        cv::Mat resizedImage = Preprocess::resizeImageRec(image, cfg.rec_imgsz, 48);
        
        cv::Scalar mean(0.5, 0.5, 0.5);
        cv::Scalar std(0.5, 0.5, 0.5);
        cv::Mat floatImage = Preprocess::normalizeImage(resizedImage, mean, std);

        int padding_right =  std::max(cfg.rec_imgsz - floatImage.cols, 0);
        int padding_bottom = std::max(48 - floatImage.rows, 0);
        cv::copyMakeBorder(floatImage, floatImage, 0, padding_bottom, 0, padding_right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

        // hwc -> chw
        cv::Mat preprocessedImage;
        cv::dnn::blobFromImage(floatImage, preprocessedImage);

        // flatten
        size_t input_size = 3 * floatImage.rows * floatImage.cols;

        std::vector<float> image_vector(input_size);
        std::copy(preprocessedImage.begin<float>(),
                    preprocessedImage.end<float>(),
                    image_vector.begin());

        batched_vector.reserve(batched_vector.size() + image_vector.size());
        std::copy(image_vector.begin(), image_vector.end(), std::back_inserter(batched_vector));
    }

    return batched_vector;
}

std::vector<float> Preprocess::preprocessCls(std::vector<cv::Mat> list_inputs, Config cfg){
    std::vector<float> batched_vector;
    for (auto image : list_inputs){

        cv::Mat resizedImage = Preprocess::resizeImageRec(image, cfg.cls_imgsz, 48);

        cv::Scalar mean(0.5, 0.5, 0.5);
        cv::Scalar std(0.5, 0.5, 0.5);
        cv::Mat floatImage = Preprocess::normalizeImage(resizedImage, mean, std);

        int padding_right =  std::max(cfg.cls_imgsz - floatImage.cols, 0);
        int padding_bottom = std::max(48 - floatImage.rows, 0);
        cv::copyMakeBorder(floatImage, floatImage, 0, padding_bottom, 0, padding_right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        
        // hwc -> chw
        cv::Mat preprocessedImage;
        cv::dnn::blobFromImage(floatImage, preprocessedImage);

        // flatten
        size_t input_size = 3 * floatImage.rows * floatImage.cols;

        std::vector<float> image_vector(input_size);
        std::copy(preprocessedImage.begin<float>(),
                    preprocessedImage.end<float>(),
                    image_vector.begin());

        batched_vector.reserve(batched_vector.size() + image_vector.size());
        std::copy(image_vector.begin(), image_vector.end(), std::back_inserter(batched_vector));
    }

    return batched_vector;
}