#pragma once
#include <iostream>
#include <vector>
#include <fstream>
#define print std::cout<<

class Config{
    public:
        explicit Config(){};
        Config(std::string config_path);
        
        std::string det_model_path;
        bool det_useCUDA = false;
        bool det_useTRT = false;
        int det_imgsz;
        int det_device_id;
        float det_db_thresh;
        float det_db_box_thresh;
        float det_db_unclip_ratio;
        bool use_dilation;
        int det_max_input_batch;

        std::string rec_model_path;
        bool rec_useCUDA = false;
        bool rec_useTRT = false;
        int rec_device_id;
        int rec_batch_size;
        int rec_imgsz;
        std::string rec_character_dict_path;
        bool rec_use_space_char;

        std::string cls_model_path;
        bool cls_useCUDA = false;
        bool cls_useTRT = false;
        int cls_device_id;
        int cls_batch_size;
        int cls_imgsz;
        bool use_angle_cls;
        float cls_thresh;

        float drop_score;
};