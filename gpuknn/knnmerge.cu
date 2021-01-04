#include <assert.h>
#include <curand.h>
#include <chrono>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "knncuda_tools.cuh"
#include "knnmerge.cuh"
#include "nndescent.cuh"

#ifdef __INTELLISENSE__
#include "../intellisense_cuda_intrinsics.h"
#endif
using namespace std;

__global__ void CopySecondHalfToKNNGraph(
    NNDElement *knngraph, const int first_half_neighb_num,
    const int second_half_neighb_num, const NNDElement *knngraph_first,
    const int knngraph_first_size, const NNDElement *knngraph_second,
    const int knngraph_second_size, const int *random_knngraph) {
  int list_id = blockIdx.x;
  int tx = threadIdx.x;
  int lane_id = tx % warpSize;
  int knngraph_base_pos = list_id * NEIGHB_NUM_PER_LIST;
  int rand_knngraph_base_pos = list_id * second_half_neighb_num;

  if (list_id < knngraph_first_size) {
    if (tx < warpSize) {
      int it_num = GetItNum(first_half_neighb_num, warpSize);
      for (int i = 0; i < it_num; i++) {
        int neighb_pos = i * warpSize + lane_id;
        if (neighb_pos >= first_half_neighb_num) break;
        knngraph[knngraph_base_pos + neighb_pos] =
            knngraph_first[knngraph_base_pos + neighb_pos];
      }
    } else {
      int it_num = GetItNum(second_half_neighb_num, warpSize);
      for (int i = 0; i < it_num; i++) {
        int neighb_pos = i * warpSize + lane_id;
        if (neighb_pos >= second_half_neighb_num) break;
        auto &elem =
            knngraph[knngraph_base_pos + first_half_neighb_num + neighb_pos];
        elem.SetDistance(1e10);
        elem.SetLabel(random_knngraph[rand_knngraph_base_pos + neighb_pos] +
                      knngraph_first_size);
      }
    }
  } else {
    int knngraph_second_base_pos =
        (list_id - knngraph_first_size) * NEIGHB_NUM_PER_LIST;
    rand_knngraph_base_pos = (list_id - knngraph_first_size) * second_half_neighb_num;
    if (tx < warpSize) {
      int it_num = GetItNum(first_half_neighb_num, warpSize);
      for (int i = 0; i < it_num; i++) {
        int neighb_pos = i * warpSize + lane_id;
        if (neighb_pos >= first_half_neighb_num) break;
        auto elem = knngraph_second[knngraph_second_base_pos + neighb_pos];
        elem.SetLabel(elem.label() + knngraph_first_size);
        knngraph[knngraph_base_pos + neighb_pos] = elem;
      }
    } else {
      int it_num = GetItNum(second_half_neighb_num, warpSize);
      for (int i = 0; i < it_num; i++) {
        int neighb_pos = i * warpSize + lane_id;
        if (neighb_pos >= second_half_neighb_num) break;
        auto &elem =
            knngraph[knngraph_base_pos + first_half_neighb_num + neighb_pos];
        elem.SetDistance(1e10);
        elem.SetLabel(random_knngraph[rand_knngraph_base_pos + neighb_pos]);
      }
    }
  }
}

void PrepareGraphForMerge(NNDElement **knngraph_dev_ptr,
                          const NNDElement *knngraph_first_dev,
                          const int knngraph_first_size,
                          const NNDElement *knngraph_second_dev,
                          const int knngraph_second_size,
                          const int *random_knngraph_dev) {
  NNDElement *&knngraph_dev = *knngraph_dev_ptr;
  int merged_graph_size = knngraph_first_size + knngraph_second_size;
  int second_half_neighb_num = NEIGHB_NUM_PER_LIST / 2;
  int first_half_neighb_num = NEIGHB_NUM_PER_LIST - second_half_neighb_num;
  cudaMalloc(&knngraph_dev, (size_t)merged_graph_size * NEIGHB_NUM_PER_LIST *
                                sizeof(NNDElement));
  CopySecondHalfToKNNGraph<<<merged_graph_size, 32 * 2>>>(
      knngraph_dev, first_half_neighb_num, second_half_neighb_num,
      knngraph_first_dev, knngraph_first_size, knngraph_second_dev,
      knngraph_second_size, random_knngraph_dev);
  cudaDeviceSynchronize();
  auto cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    cerr << cudaGetErrorString(cuda_status) << endl;
    exit(-1);
  }
}

void MergeVectors(float **vectors_dev_ptr, const float *vectors_first_dev,
                  const int vectors_first_size, const float *vectors_second_dev,
                  const int vectors_second_size) {
  float *&vectors_dev = *vectors_dev_ptr;
  int merged_size = vectors_first_size + vectors_second_size;
  cudaMalloc(&vectors_dev,
             (size_t)merged_size * VEC_DIM * sizeof(float));
  cudaMemcpy(vectors_dev, vectors_first_dev,
             (size_t)vectors_first_size * VEC_DIM * sizeof(float),
             cudaMemcpyDeviceToDevice);
  cudaMemcpy(vectors_dev + (size_t)vectors_first_size * VEC_DIM,
             vectors_second_dev,
             (size_t)vectors_second_size * VEC_DIM * sizeof(float),
             cudaMemcpyDeviceToDevice);
}

namespace gpuknn {
void KNNMerge(NNDElement **knngraph_merged_dev_ptr, float **vectors_dev_ptr,
              const float *vectors_first_dev, const int vectors_first_size,
              const NNDElement *knngraph_first_dev,
              const float *vectors_second_dev, const int vectors_second_size,
              const NNDElement *knngraph_second_dev, int *random_knngraph_dev) {
  NNDElement *&knngraph_merged_dev = *knngraph_merged_dev_ptr;
  float *&vectors_dev = *vectors_dev_ptr;
  int merged_graph_size = vectors_first_size + vectors_second_size;
  bool have_random_knngraph = random_knngraph_dev;

  auto start = chrono::steady_clock::now();
  if (!have_random_knngraph) {
    int random_knngraph_size = max(vectors_first_size, vectors_second_size);
    GenerateRandomKNNGraphIndex(&random_knngraph_dev, random_knngraph_size,
                                NEIGHB_NUM_PER_LIST / 2);
  }
  PrepareGraphForMerge(&knngraph_merged_dev, knngraph_first_dev,
                       vectors_first_size, knngraph_second_dev,
                       vectors_second_size, random_knngraph_dev);
  auto end = chrono::steady_clock::now();
  float time_cost =
      (float)chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() /
      1e6;
  cerr << "PrepareGraphForMerge costs: " << time_cost << endl;
  
  MergeVectors(&vectors_dev, vectors_first_dev, vectors_first_size,
               vectors_second_dev, vectors_second_size);
  NNDescentRefine(knngraph_merged_dev, vectors_dev, merged_graph_size, VEC_DIM);


  if (!have_random_knngraph) {
    cudaFree(random_knngraph_dev);
  }
}
}  // namespace gpuknn