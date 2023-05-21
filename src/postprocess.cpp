#include "postprocess.h"
#define print std::cout<<

DetPostProcess::DetPostProcess( Config cfg, int max_candidates){
    this->max_candidates = max_candidates;
    this->min_size = 3;
    this->cfg = cfg;

    if (cfg.use_dilation) dilation_kernel = (cv::Mat_<int>(2, 2) << 1, 1, 1, 1);

}

BBox DetPostProcess::get_mini_boxes(std::vector<cv::Point> contour, int& minside){
    cv::RotatedRect bounding_box = cv::minAreaRect(contour);
    minside = std::min(bounding_box.size.width, bounding_box.size.height);
    
    cv::Point2f corners[4];
    bounding_box.points(corners);

    std::sort(std::begin(corners), std::end(corners), [](const cv::Point2f& p1, const cv::Point2f& p2){
        return p1.x < p2.x;
    });

    int index_1 = 0; int index_2 = 1; int index_3 = 2; int index_4 = 3;

    if (corners[1].y > corners[0].y){
        index_1 = 0;
        index_4 = 1;
    }
    else{
        index_1 = 1;
        index_4 = 0;
    }

    if (corners[3].y > corners[2].y){
        index_2 = 2;
        index_3 = 3;
    }
    else{
        index_2 = 3;
        index_3 = 2;
    }

    BBox bbox(corners[index_1], corners[index_2], corners[index_3], corners[index_4]);

    return bbox;
}

std::vector<cv::Point> DetPostProcess::unclip(BBox bbox){
    std::vector<cv::Point2f> rect_points = bbox.to_list();
    double distance = cv::contourArea(rect_points) * cfg.det_db_unclip_ratio / cv::arcLength(rect_points, true);
    Clipper2Lib::ClipperOffset offset;
    Clipper2Lib::Path64 path;
    path.push_back(Clipper2Lib::Point64(rect_points[0].x, rect_points[0].y));
    path.push_back(Clipper2Lib::Point64(rect_points[1].x, rect_points[1].y));
    path.push_back(Clipper2Lib::Point64(rect_points[2].x, rect_points[2].y));
    path.push_back(Clipper2Lib::Point64(rect_points[3].x, rect_points[3].y));
    offset.AddPath(path, Clipper2Lib::JoinType::Round, Clipper2Lib::EndType::Polygon);

    Clipper2Lib::Paths64 solution;
    offset.Execute(distance, solution);
    
    std::vector<cv::Point> output;
    for (auto point : solution[0]){
        output.push_back(cv::Point(point.x, point.y));
    }
    return output;
}

float DetPostProcess::box_score_fast(cv::Mat bitmap, BBox bbox){
    float h = bitmap.rows;
    float w = bitmap.cols;

    std::vector<float> x = bbox.get_x();
    std::vector<float> y = bbox.get_y();
    float xmin = utils::clip(*(std::min_element(x.begin(), x.end())), (float) 0, w-1);
    float xmax = utils::clip(*(std::max_element(x.begin(), x.end())), (float) 0, w-1);
    float ymin = utils::clip(*(std::min_element(y.begin(), y.end())), (float) 0, h-1);
    float ymax = utils::clip(*(std::max_element(y.begin(), y.end())), (float) 0, h-1);

    cv::Mat mask = cv::Mat::zeros(cv::Size(xmax - xmin + 1, ymax - ymin + 1), CV_8UC1);
    std::transform(x.begin(), x.end(), x.begin(), [xmin](float x) { return x - xmin; });
    std::transform(y.begin(), y.end(), y.begin(), [ymin](float y) { return y - ymin; });

    std::vector<std::vector<cv::Point>> polygons; // Define the polygon(s)
    std::vector<cv::Point> polygon;
    polygon.push_back(cv::Point(std::round(x[0]), std::round(y[0])));
    polygon.push_back(cv::Point(std::round(x[1]), std::round(y[1])));
    polygon.push_back(cv::Point(std::round(x[2]), std::round(y[2])));
    polygon.push_back(cv::Point(std::round(x[3]), std::round(y[3])));
    polygons.push_back(polygon);

    cv::fillPoly(mask, polygons, 1);

    cv::Rect roi(xmin, ymin, xmax-xmin+1, ymax-ymin+1);
    auto score = cv::mean(bitmap(roi), mask)[0];
    return score;
}

std::vector<BBox> DetPostProcess::bboxes_from_bitmap(cv::Mat pred,cv::Mat bitmap, int dest_width, int dest_height){

    int width = bitmap.cols; int height = bitmap.rows;

    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::Mat gray = bitmap*255;
    gray.convertTo(gray, CV_8UC1);
    cv::findContours(gray, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

    // cv::Mat drawing = cv::Mat::zeros(bitmap.size(), CV_8UC3);
    // for (int i = 0; i < contours.size(); i++) {
    //     cv::drawContours(drawing, contours, i, cv::Scalar(0, 255, 0), 2, 8, hierarchy, 0, cv::Point());
    // }
    // cv::imshow("Contours", drawing);
    // cv::waitKey();

    int num_contours = std::min(int(contours.size()), this->max_candidates);

    std::vector<BBox> bboxes;
    // std::vector<float> scores;
    for (int i=0; i<num_contours; i++){
        std::vector<cv::Point> contour = contours[i];
        int smallside;
        BBox points = get_mini_boxes(contour, smallside);
        if (smallside <= this->min_size) continue;
        float score = box_score_fast(pred, points);
        if (score < cfg.det_db_box_thresh) continue;
        std::vector<cv::Point> list_point = unclip(points);
        BBox bbox = get_mini_boxes(list_point, smallside);
        if (smallside <= this->min_size + 2) continue;

        std::vector<float> x = bbox.get_x();
        std::vector<float> y = bbox.get_y();
        std::transform(x.begin(), x.end(), x.begin(), [width, dest_width](float x) { return (x / width) * dest_width; });
        std::transform(y.begin(), y.end(), y.begin(), [height, dest_height](float y) { return (y / height) * dest_height; });

        bboxes.push_back(BBox(x, y));
        // scores.push_back(score);
    }
    
    return bboxes;
}

std::vector<std::vector<BBox>> DetPostProcess::operator()(float *output_arr, std::vector<cv::Size> original_shapes, int batch_size){

    size_t output_tensor_size = cfg.det_imgsz * cfg.det_imgsz;
    std::vector<cv::Mat> bitmaps;
    std::vector<std::vector<BBox>> list_bboxes;
    for (int k=0; k<batch_size; k++){
        cv::Mat bitmap(cfg.det_imgsz, cfg.det_imgsz, CV_32F);
        for (int i = 0; i< cfg.det_imgsz; i++){
            for (int j=0; j<cfg.det_imgsz; j++){
                bitmap.at<float>(i, j) = (float)(output_arr[cfg.det_imgsz*i + j + output_tensor_size*k]);
            }
        }
        cv::Size ori_shape = original_shapes[k];
        
        cv::Mat mask(cfg.det_imgsz, cfg.det_imgsz, CV_8S);
        cv::threshold(bitmap, mask, cfg.det_db_thresh, 1, 3);

        if (cfg.use_dilation){
            cv::dilate(mask, mask, this->dilation_kernel);
        }
        
        std::vector<BBox> bboxes = bboxes_from_bitmap(bitmap, mask, ori_shape.width, ori_shape.height);
        list_bboxes.push_back(bboxes);
        // cv::imshow("image", mask*255);
        // cv::waitKey(0);
    }
    return list_bboxes;
}

BaseRecLabelDecode::BaseRecLabelDecode(Config cfg){
    if (cfg.rec_character_dict_path == ""){
        character_str = "0123456789abcdefghijklmnopqrstuvwxyz";
    }
    else{
        std::ifstream inputFile(cfg.rec_character_dict_path); // Open the file for reading
        std::string line;

        if (inputFile.is_open()) { // Check if the file is open
            while (std::getline(inputFile, line)){
                // print line<<"\n";
                character_str += line;
            }
        }
        if (cfg.rec_use_space_char){
            character_str += " ";
        }
    }
    
    for (int i=0; i<character_str.length(); i++){
        dict[character_str[i]] = i;
    }

    this->cfg = cfg;
}

std::vector<std::pair<std::string, float>> BaseRecLabelDecode::decode(std::vector<std::vector<int>> pred_idxs, std::vector<std::vector<float>> preb_probs, bool is_remove_duplicate, bool get_conf){
    std::vector<int> ignored_tokens{0};
    int batch_size = pred_idxs.size();
    std::vector<std::pair<std::string, float>> return_list;
    for (int i=0; i<batch_size; i++){
        std::string result = "";
        std::vector<int> pred_id = pred_idxs[i];
        // std::vector<float> preb_prob = preb_probs[i];
        float conf = 0;
        result += character_str[pred_id[0]-1];
        for (int j=1; j< pred_id.size(); j++){

            // remove ignore tokens
            bool check = false;
            for (int k=0; k<ignored_tokens.size(); k++){
                if (pred_id[j] == ignored_tokens[k]){
                    check = true;
                    break;
                }
            }
            if (check) continue;

            // remove duplicate
            if (is_remove_duplicate){
                if (pred_id[j] == pred_id[j-1]) continue;
            }

            result += character_str[pred_id[j]-1];

            if (get_conf){
                conf+=preb_probs[i][j];
            }
        }

        conf /= result.length();
        
        return_list.push_back(std::pair<std::string, float> (result, conf));
        
    }

    return return_list;
}

std::vector<std::pair<std::string, float>> CTCLabelDecode::operator()(float *output_arr, std::vector<int64_t> outputTensorShape){

    std::vector<std::vector<int>> pred_ids;
    std::vector<std::vector<float>> pred_probs;


    size_t output_tensor_size = outputTensorShape[1] * outputTensorShape[2];
    for (int k=0; k<outputTensorShape[0]; k++){
        std::vector<int> idx;
        std::vector<float> probs;
        for (int i=0; i<outputTensorShape[1]; i++){
            int max_idx = -1;
            float max_value = -1;
            for (int j=0; j<outputTensorShape[2]; j++){
                auto prob = output_arr[outputTensorShape[2]*i + j + output_tensor_size*k];
                if (prob > max_value){
                    max_value = prob;
                    max_idx = j;
                }
            }
            idx.push_back(max_idx);
            probs.push_back(max_value);
        }
        pred_ids.push_back(idx);
        pred_probs.push_back(probs);
    }

    return this->decode(pred_ids, pred_probs, true, true);

}


ClsPostProcess::ClsPostProcess(Config cfg){
    this->cfg = cfg;
}

std::vector<std::pair<std::string, float>> ClsPostProcess::operator()(float *output_arr, std::vector<int64_t> outputTensorShape){

    int idx_max[outputTensorShape[0]] = {0};
    float max_v[outputTensorShape[0]] = {-1};
    for (int k=0; k<outputTensorShape[0]; k++){
        for (int i=0; i<outputTensorShape[1]; i++){
            if (output_arr[k * outputTensorShape[1] + i] > max_v[k]){
                max_v[k] = output_arr[k * outputTensorShape[1] + i];
                idx_max[k] = i;
            }
        }
    }

    // for (int i=0; i<outputTensorShape[0]; i++) print idx_max[i]<<" "<<max_v[i]<<"\n";

    std::vector<std::pair<std::string, float>> results;
    for (int i=0; i<outputTensorShape[0]; i++){
        results.push_back(std::pair<std::string, float> (label_list[idx_max[i]], max_v[i]));
    }
    return results;
}