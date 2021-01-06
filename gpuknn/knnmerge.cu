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
    NNDElement *knngraph, const NNDElement *knngraph_first,
    const int knngraph_first_size, const NNDElement *knngraph_second,
    const int knngraph_second_size, const int *random_knngraph) {
  int list_id = blockIdx.x;
  int tx = threadIdx.x;
  int lane_id = tx % warpSize;
  int knngraph_base_pos = list_id * NEIGHB_NUM_PER_LIST;
  int rand_knngraph_base_pos = list_id * LAST_HALF_NEIGHB_NUM;

  if (list_id < knngraph_first_size) {
    if (tx < warpSize) {
      int it_num = GetItNum(FIRST_HALF_NEIGHB_NUM, warpSize);
      for (int i = 0; i < it_num; i++) {
        int neighb_pos = i * warpSize + lane_id;
        if (neighb_pos >= FIRST_HALF_NEIGHB_NUM) break;
        knngraph[knngraph_base_pos + neighb_pos] =
            knngraph_first[knngraph_base_pos + neighb_pos];
      }
    } else {
      int it_num = GetItNum(LAST_HALF_NEIGHB_NUM, warpSize);
      for (int i = 0; i < it_num; i++) {
        int neighb_pos = i * warpSize + lane_id;
        if (neighb_pos >= LAST_HALF_NEIGHB_NUM) break;
        auto &elem =
            knngraph[knngraph_base_pos + FIRST_HALF_NEIGHB_NUM + neighb_pos];
        elem.SetDistance(1e10);
        elem.SetLabel(random_knngraph[rand_knngraph_base_pos + neighb_pos] +
                      knngraph_first_size);
      }
    }
  } else {
    int knngraph_second_base_pos =
        (list_id - knngraph_first_size) * NEIGHB_NUM_PER_LIST;
    rand_knngraph_base_pos = (list_id - knngraph_first_size) * LAST_HALF_NEIGHB_NUM;
    if (tx < warpSize) {
      int it_num = GetItNum(FIRST_HALF_NEIGHB_NUM, warpSize);
      for (int i = 0; i < it_num; i++) {
        int neighb_pos = i * warpSize + lane_id;
        if (neighb_pos >= FIRST_HALF_NEIGHB_NUM) break;
        auto elem = knngraph_second[knngraph_second_base_pos + neighb_pos];
        elem.SetLabel(elem.label() + knngraph_first_size);
        knngraph[knngraph_base_pos + neighb_pos] = elem;
      }
    } else {
      int it_num = GetItNum(LAST_HALF_NEIGHB_NUM, warpSize);
      for (int i = 0; i < it_num; i++) {
        int neighb_pos = i * warpSize + lane_id;
        if (neighb_pos >= LAST_HALF_NEIGHB_NUM) break;
        auto &elem =
            knngraph[knngraph_base_pos + FIRST_HALF_NEIGHB_NUM + neighb_pos];
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
  int LAST_HALF_NEIGHB_NUM = NEIGHB_NUM_PER_LIST / 2;
  int FIRST_HALF_NEIGHB_NUM = NEIGHB_NUM_PER_LIST - LAST_HALF_NEIGHB_NUM;
  cudaMalloc(&knngraph_dev, (size_t)merged_graph_size * NEIGHB_NUM_PER_LIST *
                                sizeof(NNDElement));
  CopySecondHalfToKNNGraph<<<merged_graph_size, 32 * 2>>>(
      knngraph_dev, knngraph_first_dev, knngraph_first_size,
      knngraph_second_dev, knngraph_second_size, random_knngraph_dev);
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
  cudaMemcpyAsync(vectors_dev, vectors_first_dev,
                  (size_t)vectors_first_size * VEC_DIM * sizeof(float),
                  cudaMemcpyDeviceToDevice);
  cudaMemcpyAsync(vectors_dev + (size_t)vectors_first_size * VEC_DIM,
                  vectors_second_dev,
                  (size_t)vectors_second_size * VEC_DIM * sizeof(float),
                  cudaMemcpyDeviceToDevice);
}

namespace gpuknn {
void KNNMerge(NNDElement **knngraph_merged_dev_ptr, float **vectors_dev_ptr,
              const float *vectors_first_dev, const int vectors_first_size,
              NNDElement *knngraph_first_dev,
              const float *vectors_second_dev, const int vectors_second_size,
              NNDElement *knngraph_second_dev, int *random_knngraph_dev) {
  NNDElement *&knngraph_merged_dev = *knngraph_merged_dev_ptr;
  float *&vectors_dev = *vectors_dev_ptr;
  int merged_graph_size = vectors_first_size + vectors_second_size;
  bool have_random_knngraph = random_knngraph_dev;
  auto start = chrono::steady_clock::now();
  MarkAllToOld<<<vectors_first_size, NEIGHB_NUM_PER_LIST>>>(knngraph_first_dev);
  MarkAllToOld<<<vectors_second_size, NEIGHB_NUM_PER_LIST>>>(knngraph_second_dev);
  cudaDeviceSynchronize();
  if (!have_random_knngraph) {
    int random_knngraph_size = max(vectors_first_size, vectors_second_size);
    GenerateRandomKNNGraphIndex(&random_knngraph_dev, random_knngraph_size,
                                NEIGHB_NUM_PER_LIST / 2);
  }
  PrepareGraphForMerge(&knngraph_merged_dev, knngraph_first_dev,
                       vectors_first_size, knngraph_second_dev,
                       vectors_second_size, random_knngraph_dev);
  MergeVectors(&vectors_dev, vectors_first_dev, vectors_first_size,
               vectors_second_dev, vectors_second_size);
  auto end = chrono::steady_clock::now();
  float time_cost =
      (float)chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() /
      1e6;
  cerr << "PrepareGraphForMerge costs: " << time_cost << endl;

  NNDescentRefine(knngraph_merged_dev, vectors_dev, merged_graph_size, VEC_DIM,
                  6);

  if (!have_random_knngraph) {
    cudaFree(random_knngraph_dev);
  }
}
}  // namespace gpuknn