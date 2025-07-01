cmake_minimum_required(VERSION 3.18)    # Update the minimum required version to support modern CUDA integration
project(GST_Project LANGUAGES CXX CUDA) # Define the project to include CUDA as a first-class language

set(CMAKE_CXX_STANDARD 17)              # Set C++ standard to C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit)               # Use the modern FindCUDAToolkit instead of FindCUDA

# Include directories
include_directories(SYSTEM
  ${CMAKE_SOURCE_DIR}/rucgraph/boost  # 可选，直接指向boost目录
)
include_directories(
  ${CMAKE_SOURCE_DIR}/rucgraph   # 添加此行，包含rucgraph及其子目录
)

# Add an executable
add_executable(GST_data GST_data.cpp)

# Link CUDA libraries (if CUDA was found and is needed)
if(CUDAToolkit_FOUND)
  target_link_libraries(GST_data PRIVATE CUDA::cudart)
endif()
