cmake_minimum_required(VERSION 3.10)
project(CUDAProfiler)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Enable CUDA language
enable_language(CUDA)

# Set CUDA architecture (adjust if needed)
set(CMAKE_CUDA_ARCHITECTURES "75" CACHE STRING "CUDA architectures")

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})

# Profiler executable
add_executable(profiler profiler.cpp)

# Link CUDA libraries
target_link_libraries(profiler 
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
)

# Platform-specific libraries
if(UNIX AND NOT APPLE)
    target_link_libraries(profiler dl)
endif()

# Set output directory
set_target_properties(profiler PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Installation (optional)
install(TARGETS profiler
    RUNTIME DESTINATION bin
)

