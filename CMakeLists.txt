cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(gpuknn LANGUAGES CXX CUDA)
  
add_library(knncuda STATIC
    gpuknn/knncuda.cu
    gpuknn/knncuda.cuh
    gpuknn/nndescent.cu
    gpuknn/nndescent.cuh
    gpuknn/result_element.cuh
    tools/distfunc.hpp
    tools/filetool.hpp
    xmuknn.h
    xmuknn.cpp
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(cuBLAS)

# set(CMAKE_VERBOSE_MAKEFILE ON)
target_compile_features(knncuda PUBLIC cxx_std_11)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_75 -Xcompiler -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -pthread -fopenmp")
add_executable(gknng main.cu)

set_property(TARGET gknng PROPERTY CUDA_SEPERABLE_COMPILATION ON)
set_property(TARGET gknng PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_property(TARGET gknng PROPERTY CUDA_ARCHITECTURES 75)

target_link_libraries(gknng PRIVATE knncuda ${CUBLAS_LIBRARIES})
 