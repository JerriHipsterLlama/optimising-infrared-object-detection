cmake_minimum_required(VERSION 3.18)
project(infrared_jetson_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h PATHS /usr/include/aarch64-linux-gnu /usr/include)
find_library(TENSORRT_LIBRARY nvinfer PATHS /usr/lib/aarch64-linux-gnu /usr/lib)
find_package(CUDAToolkit REQUIRED)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT headers/library not found. Build this target on the Jetson with TensorRT installed.")
endif()

add_executable(jetson_benchmark src/main.cpp src/benchmark_runner.cpp)
target_include_directories(jetson_benchmark PRIVATE include ${TENSORRT_INCLUDE_DIR})
target_link_libraries(jetson_benchmark PRIVATE ${TENSORRT_LIBRARY} CUDA::cudart)
target_compile_options(jetson_benchmark PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)
