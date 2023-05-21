#include "textclassifier.h"

TextClassifier::TextClassifier(Config cfg){
    std::string instanceName{"Cls model"};
    Ort::Env env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str());
    Ort::SessionOptions session_options;
    if (cfg.cls_useTRT){
        const auto& api = Ort::GetApi();
        OrtTensorRTProviderOptionsV2* tensorrt_options;
        Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&tensorrt_options));
        tensorrt_options->device_id = cfg.cls_device_id;
        std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype(api.ReleaseTensorRTProviderOptions)> rel_trt_options(tensorrt_options, api.ReleaseTensorRTProviderOptions);
        Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_TensorRT_V2(static_cast<OrtSessionOptions*>(session_options), rel_trt_options.get()));
    } else
    if (cfg.cls_useCUDA){
        OrtCUDAProviderOptions cudaOptions;
        cudaOptions.device_id = cfg.cls_device_id;
        session_options.AppendExecutionProvider_CUDA(cudaOptions);
    }
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session = Ort::Session(env, cfg.cls_model_path.c_str(), session_options);
    env.release();

    // create pre and post process
    postprocess = ClsPostProcess(cfg);

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    std::vector<const char*> inputNames;
    inputNames.push_back(input_name.get());
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    std::vector<const char*> outputNames;
    outputNames.push_back(output_name.get());

    size_t input_tensor_size_single = 48 * cfg.cls_imgsz * 3;
    std::vector<float> input_tensor_single;
    for (unsigned int i = 0; i < input_tensor_size_single; i++) input_tensor_single.push_back((float)i / (input_tensor_size_single + 1));
    std::vector<float> input_tensor_values;

    for (int i=0; i<cfg.cls_batch_size; i++){
        size_t input_tensor_size = input_tensor_size_single*(i+1);
        input_tensor_values.insert(input_tensor_values.end(), input_tensor_single.begin(), input_tensor_single.end());
        std::vector<int64_t> inputTensorShape {i+1, 3, 48, cfg.cls_imgsz};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, inputTensorShape.data(), 4);
        session.Run(Ort::RunOptions{nullptr}, inputNames.data(), &input_tensor, 1, outputNames.data(), 1);
    }
    input_tensor_values.clear();
    input_tensor_single.clear();

    std::cout << "Done Init Text Classifier" << std::endl;

    this->cfg = cfg;

}

std::vector<std::pair<std::string, float>> TextClassifier::cls(std::vector<cv::Mat> list_imgs){
    int batch_size = list_imgs.size();
    size_t input_tensor_size = 3*48*cfg.cls_imgsz*batch_size;
    std::vector<int64_t> inputTensorShape {batch_size, 3, 48, cfg.cls_imgsz};
    std::vector<float> batch_input = Preprocess::preprocessCls(list_imgs, cfg);
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
    
    auto outputInfo = output_tensors[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputTensorShape;
    for (unsigned int shapeI = 0; shapeI < outputInfo.GetShape().size(); shapeI++){
            outputTensorShape.push_back(outputInfo.GetShape()[shapeI]);
    }

    std::vector<std::pair<std::string, float>> results = postprocess(output_arr, outputTensorShape);

    return results;

}