cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(gpu LANGUAGES CXX CUDA)
  
add_library(knncuda STATIC
    gpuknn/gpudist.cu
    gpuknn/gpudist.cuh
    gpuknn/knncuda.cu
    gpuknn/knncuda.cuh
    gpuknn/nndescent.cu
    gpuknn/nndescent.cuh
    gpuknn/unittest.cu
    gpuknn/result_element.cuh
    tools/distfunc.hpp
    tools/filetool.hpp
    xmuknn.h
    xmuknn.cpp
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(cuBLAS)

# set(CMAKE_VERBOSE_MAKEFILE ON)
target_compile_features(knncuda PUBLIC cxx_std_14)
target_compile_options(knncuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -O3>) 

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

add_executable(gknng main.cu)

set_property(TARGET gknng PROPERTY CUDA_SEPERABLE_COMPILATION ON)
set_property(TARGET gknng PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_property(TARGET gknng PROPERTY CUDA_ARCHITECTURES 61)

target_link_libraries(gknng PRIVATE knncuda ${CUBLAS_LIBRARIES})
 