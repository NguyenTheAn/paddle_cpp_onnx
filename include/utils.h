#pragma once
#include <iostream>
#include <vector>
#include "bbox.h"
#define print std::cout<<

namespace utils
{
    template<typename T>
    T clip(T elem, T a_min, T a_max) {
        elem = std::max(a_min, std::min(a_max, elem));
        return elem;
    }

    std::vector<BBox> sorted_box(std::vector<BBox> bboxes);
    
}

// std::unique_ptr<OrtTensorRTProviderOptionsV2> get_default_trt_provider_options() {
//     auto tensorrt_options = std::make_unique<OrtTensorRTProviderOptionsV2>();
//     tensorrt_options->device_id = 0;
//     tensorrt_options->has_user_compute_stream = 0;
//     tensorrt_options->user_compute_stream = nullptr;
//     tensorrt_options->trt_max_partition_iterations = 1000;
//     tensorrt_options->trt_min_subgraph_size = 1;
//     tensorrt_options->trt_max_workspace_size = 1 << 30;
//     tensorrt_options->trt_fp16_enable = false;
//     tensorrt_options->trt_int8_enable = false;
//     tensorrt_options->trt_int8_calibration_table_name = "";
//     tensorrt_options->trt_int8_use_native_calibration_table = false;
//     tensorrt_options->trt_dla_enable = false;
//     tensorrt_options->trt_dla_core = 0;
//     tensorrt_options->trt_dump_subgraphs = false;
//     tensorrt_options->trt_engine_cache_enable = false;
//     tensorrt_options->trt_engine_cache_path = "";
//     tensorrt_options->trt_engine_decryption_enable = false;
//     tensorrt_options->trt_engine_decryption_lib_path = "";
//     tensorrt_options->trt_force_sequential_engine_build = false;

//     return tensorrt_options;
// }