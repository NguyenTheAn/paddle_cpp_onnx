#include "textdetector.h"

TextDetector::TextDetector(Config cfg){
    std::string instanceName{"Det model"};
    Ort::Env env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str());
    Ort::SessionOptions session_options;
    if (cfg.det_useTRT){
        const auto& api = Ort::GetApi();
        OrtTensorRTProviderOptionsV2* tensorrt_options;
        Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&tensorrt_options));
        tensorrt_options->device_id = cfg.det_device_id;
        std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype(api.ReleaseTensorRTProviderOptions)> rel_trt_options(tensorrt_options, api.ReleaseTensorRTProviderOptions);
        Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_TensorRT_V2(static_cast<OrtSessionOptions*>(session_options), rel_trt_options.get()));
    } else
    if (cfg.det_useCUDA){
        OrtCUDAProviderOptions cudaOptions;
        cudaOptions.device_id = cfg.det_device_id;
        session_options.AppendExecutionProvider_CUDA(cudaOptions);
    }
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session = Ort::Session(env, cfg.det_model_path.c_str(), session_options);
    env.release();

    // create pre and post process
    // preprocess = DetPreProcess(cfg);
    postprocess = DetPostProcess(cfg, 1000);

    // get input and ouput layer
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    std::vector<const char*> inputNames;
    inputNames.push_back(input_name.get());
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    std::vector<const char*> outputNames;
    outputNames.push_back(output_name.get());

    // init infer
    size_t input_tensor_size_single = cfg.det_imgsz * cfg.det_imgsz * 3;
    std::vector<float> input_tensor_single;
    for (unsigned int i = 0; i < input_tensor_size_single; i++) input_tensor_single.push_back((float)i / (input_tensor_size_single + 1));
    std::vector<float> input_tensor_values;


    for (int i=0; i<cfg.det_max_input_batch; i++){
        size_t input_tensor_size = input_tensor_size_single*(i+1);
        input_tensor_values.insert(input_tensor_values.end(), input_tensor_single.begin(), input_tensor_single.end());
        std::vector<int64_t> inputTensorShape {i+1, 3, cfg.det_imgsz, cfg.det_imgsz};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, inputTensorShape.data(), 4);

        session.Run(Ort::RunOptions{nullptr}, inputNames.data(), &input_tensor, 1, outputNames.data(), 1);
        // std::cout << std::to_string(i) << std::endl;
    }
    input_tensor_values.clear();
    input_tensor_single.clear();

    std::cout << "Done Init Text Detector" << std::endl;

    this->cfg = cfg;

}


std::vector<std::vector<BBox>> TextDetector::detect(std::vector<cv::Mat> list_imgs){

    auto timeStart = std::chrono::high_resolution_clock::now();

    int batch_size = list_imgs.size();
    size_t input_tensor_size = 3*cfg.det_imgsz*cfg.det_imgsz*batch_size;
    std::vector<int64_t> inputTensorShape {batch_size, 3, cfg.det_imgsz, cfg.det_imgsz};
    std::vector<cv::Size> original_shapes;
    std::vector<float> batch_input = Preprocess::preprocessDet(list_imgs, original_shapes, cfg);
    std::cout << "Done preprocess image" << std::endl;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memoryInfo, batch_input.data(), input_tensor_size, inputTensorShape.data(), 4);

    // get input and ouput layer
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    std::vector<const char*> inputNames;
    inputNames.push_back(input_name.get());
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    std::vector<const char*> outputNames;
    outputNames.push_back(output_name.get());

    std::cout << "Input name: " << inputNames[0] << std::endl;
    std::cout << "Output name: " << outputNames[0] << std::endl;

    // feed forward
    std::cout << "Start infer" << std::endl;
    std::vector<Ort::Value> output_tensors = session.Run(Ort::RunOptions{nullptr}, inputNames.data(), &input_tensor, 1, outputNames.data(), 1);
    float* output_arr = output_tensors[0].GetTensorMutableData<float>();

    // post process
    std::vector<std::vector<BBox>> list_bboxes = postprocess(output_arr, original_shapes, batch_size);

    auto timeEnd = std::chrono::high_resolution_clock::now();
    float duration = std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart).count() * 0.001;
    std::cout << "Done infer after: " << duration << std::endl;

    return list_bboxes;
}
