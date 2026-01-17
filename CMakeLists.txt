cmake_minimum_required(VERSION 3.18)
project(infrared-detection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find LibTorch
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/models)
include_directories(${PROJECT_SOURCE_DIR}/src)

# Create executable
add_executable(train_yolo 
    src/main.cpp
    src/data_loader.cpp
)

# Link libraries
target_link_libraries(train_yolo ${TORCH_LIBRARIES} ${OpenCV_LIBS})

set_property(TARGET train_yolo PROPERTY CXX_STANDARD 17)

# Small test target for data loader
enable_testing()
add_executable(test_data_loader
    tests/test_data_loader.cpp
)
target_link_libraries(test_data_loader ${TORCH_LIBRARIES} ${OpenCV_LIBS})
add_test(NAME data_loader_smoke COMMAND test_data_loader)
