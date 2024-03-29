cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(gpuknn LANGUAGES CXX CUDA)
  
add_library(knncuda STATIC
    gpuknn/gen_large_knngraph.cu
    gpuknn/gen_large_knngraph.cuh
    gpuknn/knncuda.cu
    gpuknn/knncuda.cuh
    gpuknn/nndescent.cu
    gpuknn/nndescent.cuh
    gpuknn/knncuda_tools.cuh
    gpuknn/knncuda_tools.cu
    gpuknn/knnmerge.cuh
    gpuknn/knnmerge.cu
    tools/distfunc.hpp
    tools/filetool.hpp
    tools/nndescent_element.cuh
    tools/knndata_manager.hpp
    xmuknn.h
    xmuknn.cpp
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(CUDA QUIET)
# set(CMAKE_VERBOSE_MAKEFILE ON)
target_compile_features(knncuda PUBLIC cxx_std_14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_FILE_OFFSET_BITS=64 -O3 -std=c++14 -arch=sm_80 -rdc=true -Xcompiler -fopenmp -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FILE_OFFSET_BITS=64 -std=c++14 -O3 -pthread -fopenmp")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -g -pthread -fopenmp")
add_executable(gknng main.cu)

set_property(TARGET gknng PROPERTY CUDA_SEPERABLE_COMPILATION ON)
set_property(TARGET gknng PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_property(TARGET gknng PROPERTY CUDA_ARCHITECTURES 80)

target_link_libraries(gknng PRIVATE knncuda ${CUDA_cublas_LIBRARY} ${CUDA_curand_LIBRARY})
 