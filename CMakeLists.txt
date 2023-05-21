cmake_minimum_required(VERSION 3.13)

project(ONNX_Runtime_Examples VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OPENCV_LIB_DIR /home/ligirk/lib/opencv/lib)
set(OPENCV_INCLUDE_DIR /home/ligirk/lib/opencv/include/opencv4/)

set(ONNXRUNTIME_LIB_DIR /home/ligirk/lib/onnxruntime/lib)
set(ONNXRUNTIME_SESSION_INCLUDE_DIR /home/ligirk/lib/onnxruntime/include/onnxruntime/core/session/)
set(ONNXRUNTIME_PROVIDERS_TENSORRT_INCLUDE_DIR /home/ligirk/lib/onnxruntime/include/onnxruntime/core/providers/tensorrt/)

include_directories("./include/")

file(GLOB_RECURSE SOURCE_FILES src/**.cpp)

add_executable(main ${SOURCE_FILES})

target_link_directories(main PUBLIC "${OPENCV_LIB_DIR}")
target_link_directories(main PUBLIC "${ONNXRUNTIME_LIB_DIR}")
target_link_libraries(main PUBLIC onnxruntime opencv_imgcodecs opencv_core opencv_imgproc opencv_highgui opencv_dnn)
target_include_directories(main PUBLIC "${OPENCV_INCLUDE_DIR}" "${ONNXRUNTIME_SESSION_INCLUDE_DIR}" "${ONNXRUNTIME_PROVIDERS_TENSORRT_INCLUDE_DIR}")
