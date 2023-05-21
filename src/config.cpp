#include "config.h"

std::string remove_char(std::string str, char list_remove_c[], int size){
    std::string result;
    for (const auto& c : str) {
        bool check = true;
        for (int i=0; i<size; i++){
            if (c == list_remove_c[i]) {
                check = false;
                break;
            }
        }
        if (check) result += c;
    }
    return result;
}

Config::Config(std::string config_path){
    std::ifstream inputFile(config_path); // Open the file for reading
    std::string line;

    if (inputFile.is_open()) { // Check if the file is open
        while (std::getline(inputFile, line)) { // Read the file line by line
            if (line[0] == '#') { // Check if the first character is '#'
                continue;
            }
            if (line.empty()){
                continue;
            }
            // std::cout << line << "\n";
            size_t start = 0, end = 0;
            end = line.find(':', start);

            std::string key = line.substr(start, end - start);
            std::string value = line.substr(end+1);

            char list[] = {'\'', '\"', ' '};
            int size = sizeof(list) / sizeof(list[0]);
            key = remove_char(key, list, size);
            value = remove_char(value, list, size);


            if (key == "det_mode_path") this->det_model_path = value;
            if (key == "det_imgsz") this->det_imgsz = std::stoi(value);
            if (key == "det_device_id") this->det_device_id = std::stoi(value);
            if (key == "det_excution_provider"){
                if (value == "tensorrt") this->det_useTRT = true;
                if (value == "cuda") this->det_useCUDA = true;
            }
            if (key == "det_db_thresh") this->det_db_thresh = std::stof(value);
            if (key == "det_db_box_thresh") this->det_db_box_thresh = std::stof(value);
            if (key == "det_db_unclip_ratio") this->det_db_unclip_ratio = std::stof(value);
            if (key == "use_dilation") this->use_dilation = value == "true" ? true : false;
            if (key == "det_max_input_batch") this->det_max_input_batch = std::stoi(value);


            if (key == "rec_model_path") this->rec_model_path = value;
            if (key == "rec_character_dict_path") this->rec_character_dict_path = value;
            if (key == "rec_device_id") this->rec_device_id = std::stoi(value);
            if (key == "rec_batch_size") this->rec_batch_size = std::stoi(value);
            if (key == "rec_imgsz"){
                this->rec_imgsz = std::stoi(value);
            }
            if (key == "rec_excution_provider"){
                if (value == "tensorrt") this->rec_useTRT = true;
                if (value == "cuda") this->rec_useCUDA = true;
            }
            if (key == "rec_use_space_char") this->rec_use_space_char = value == "true" ? true : false;

            if (key == "cls_model_path") this->cls_model_path = value;
            if (key == "cls_device_id") this->cls_device_id = std::stoi(value);
            if (key == "cls_batch_size") this->cls_batch_size = std::stoi(value);
            if (key == "cls_imgsz"){
                this->cls_imgsz = std::stoi(value);
            }
            if (key == "cls_excution_provider"){
                if (value == "tensorrt") this->cls_useTRT = true;
                if (value == "cuda") this->cls_useCUDA = true;
            }
            if (key == "use_angle_cls") this->use_angle_cls = value == "true" ? true : false;
            if (key == "cls_thresh") this->cls_thresh = std::stof(value);

            if (key == "drop_score") this->drop_score = std::stof(value);
        }
        inputFile.close(); // Close the file
    }
}
