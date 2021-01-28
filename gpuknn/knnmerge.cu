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

__global__ void CopySecondHalfToKNNGraph(NNDElement *knngraph,
                                         const NNDElement *knngraph_first,
                                         const int knngraph_first_size,
                                         const NNDElement *knngraph_second,
                                         const int knngraph_second_size,
                                         const int *random_knngraph) {
  int list_id = blockIdx.x;
  int tx = threadIdx.x;
  int lane_id = tx % WARP_SIZE;
  int knngraph_pos_base = list_id * NEIGHB_NUM_PER_LIST;
  int rand_knngraph_pos_base = list_id * LAST_HALF_NEIGHB_NUM;

  if (list_id < knngraph_first_size) {
    if (tx < WARP_SIZE) {
      int it_num = GetItNum(FIRST_HALF_NEIGHB_NUM, WARP_SIZE);
      for (int i = 0; i < it_num; i++) {
        int neighb_pos = i * WARP_SIZE + lane_id;
        if (neighb_pos >= FIRST_HALF_NEIGHB_NUM) break;
        knngraph[knngraph_pos_base + neighb_pos] =
            knngraph_first[knngraph_pos_base + neighb_pos];
      }
    } else {
      int it_num = GetItNum(LAST_HALF_NEIGHB_NUM, WARP_SIZE);
      for (int i = 0; i < it_num; i++) {
        int neighb_pos = i * WARP_SIZE + lane_id;
        if (neighb_pos >= LAST_HALF_NEIGHB_NUM) break;
        auto &elem =
            knngraph[knngraph_pos_base + FIRST_HALF_NEIGHB_NUM + neighb_pos];
        elem.SetDistance(1e10);
        elem.SetLabel(random_knngraph[rand_knngraph_pos_base + neighb_pos] +
                      knngraph_first_size);
      }
    }
  } else {
    int knngraph_second_pos_base =
        (list_id - knngraph_first_size) * NEIGHB_NUM_PER_LIST;
    rand_knngraph_pos_base =
        (list_id - knngraph_first_size) * LAST_HALF_NEIGHB_NUM;
    if (tx < WARP_SIZE) {
      int it_num = GetItNum(FIRST_HALF_NEIGHB_NUM, WARP_SIZE);
      for (int i = 0; i < it_num; i++) {
        int neighb_pos = i * WARP_SIZE + lane_id;
        if (neighb_pos >= FIRST_HALF_NEIGHB_NUM) break;
        auto elem = knngraph_second[knngraph_second_pos_base + neighb_pos];
        elem.SetLabel(elem.label() + knngraph_first_size);
        knngraph[knngraph_pos_base + neighb_pos] = elem;
      }
    } else {
      int it_num = GetItNum(LAST_HALF_NEIGHB_NUM, WARP_SIZE);
      for (int i = 0; i < it_num; i++) {
        int neighb_pos = i * WARP_SIZE + lane_id;
        if (neighb_pos >= LAST_HALF_NEIGHB_NUM) break;
        auto &elem =
            knngraph[knngraph_pos_base + FIRST_HALF_NEIGHB_NUM + neighb_pos];
        elem.SetDistance(1e10);
        elem.SetLabel(random_knngraph[rand_knngraph_pos_base + neighb_pos]);
      }
    }
  }
}

__global__ void InitRandomBlockedKNNGraph(NNDElement *knngraph,
                                          const NNDElement *knngraph_first,
                                          const int knngraph_first_size,
                                          const NNDElement *knngraph_second,
                                          const int knngraph_second_size) {
  __shared__ NNDElement knnlist_cache[NEIGHB_NUM_PER_LIST];
  __shared__ int blocks_size[NEIGHB_BLOCKS_NUM];
  __shared__ int current_block_id;
  int list_id = blockIdx.x;
  int global_pos_base = list_id * NEIGHB_NUM_PER_LIST;
  int merged_size = knngraph_first_size + knngraph_second_size;
  int tx = threadIdx.x;
  if (tx < NEIGHB_BLOCKS_NUM) {
    blocks_size[tx] = 0;
  }

  if (list_id < knngraph_first_size) {
    int it_num = GetItNum(FIRST_HALF_NEIGHB_NUM, WARP_SIZE);
    for (int i = 0; i < it_num; i++) {
      int pos = i * WARP_SIZE + tx;
      if (pos < FIRST_HALF_NEIGHB_NUM) {
        NNDElement elem = knngraph_first[global_pos_base + pos];
        int block_id = elem.label() % NEIGHB_BLOCKS_NUM;
        int new_pos = atomicAdd(&blocks_size[block_id], 1);
        if (new_pos >= WARP_SIZE) {
          atomicExch(&blocks_size[block_id], WARP_SIZE);
        } else {
          knnlist_cache[block_id * WARP_SIZE + new_pos] = elem;
        }
      }
    }
  } else {
    int knngraph_second_pos_base =
        (list_id - knngraph_first_size) * NEIGHB_NUM_PER_LIST;
    int it_num = GetItNum(FIRST_HALF_NEIGHB_NUM, WARP_SIZE);
    for (int i = 0; i < it_num; i++) {
      int pos = i * WARP_SIZE + tx;
      if (pos < FIRST_HALF_NEIGHB_NUM) {
        NNDElement elem = knngraph_second[knngraph_second_pos_base + pos];
        elem.SetLabel(elem.label() + knngraph_first_size);
        elem.MarkOld();
        int block_id = elem.label() % NEIGHB_BLOCKS_NUM;
        int new_pos = atomicAdd(&blocks_size[block_id], 1);
        if (new_pos >= WARP_SIZE) {
          atomicExch(&blocks_size[block_id], WARP_SIZE);
        } else {
          knnlist_cache[block_id * WARP_SIZE + new_pos] = elem;
        }
      }
    }
  }
  if (tx == 0) {
    current_block_id = 0;
  }
  int used_num = 0;
  for (int i = 0; i < NEIGHB_BLOCKS_NUM; i++) {
    int it_num = GetItNum(LAST_HALF_NEIGHB_NUM - used_num, WARP_SIZE);
    int tmp_used_num = used_num + (WARP_SIZE - blocks_size[i]);
    for (int j = 0; j < it_num; j++) {
      int pos = used_num + j * WARP_SIZE + tx;
      if (pos >= LAST_HALF_NEIGHB_NUM) break;
      int new_pos = atomicAdd(&blocks_size[i], 1);
      if (new_pos >= WARP_SIZE) {
        atomicExch(&blocks_size[i], WARP_SIZE);
        break;
      }
      NNDElement elem(1e10, 12345678);
      int new_label;
      if (list_id < knngraph_first_size) {
        int rand_knngraph_pos_base = list_id * LAST_HALF_NEIGHB_NUM;
        new_label =
            xorshift64star(rand_knngraph_pos_base) % knngraph_second_size;
        while (new_label % NEIGHB_BLOCKS_NUM != i || new_label == list_id) {
          new_label = xorshift64star(new_label) % knngraph_second_size;
        }
      } else {
        int rand_knngraph_pos_base =
            (list_id - knngraph_first_size) * LAST_HALF_NEIGHB_NUM;
        new_label =
            xorshift64star(rand_knngraph_pos_base) % knngraph_first_size;
        while (new_label % NEIGHB_BLOCKS_NUM != i || new_label == list_id) {
          new_label = xorshift64star(new_label) % knngraph_first_size;
        }
      }
      elem.SetLabel(new_label);
      knnlist_cache[i * WARP_SIZE + new_pos] = elem;
    }
    used_num = tmp_used_num;
  }
  int it_num = GetItNum(NEIGHB_NUM_PER_LIST, WARP_SIZE);
  for (int i = 0; i < it_num; i++) {
    int pos = i * WARP_SIZE + tx;
    if (pos < NEIGHB_NUM_PER_LIST)
      knngraph[global_pos_base + pos] = knnlist_cache[pos];
  }
}

void PrepareGraphForMerge(NNDElement **knngraph_dev_ptr,
                          NNDElement *knngraph_first_dev,
                          const int knngraph_first_size,
                          NNDElement *knngraph_second_dev,
                          const int knngraph_second_size,
                          const bool free_subgraph = false) {
  NNDElement *&knngraph_dev = *knngraph_dev_ptr;
  int merged_graph_size = knngraph_first_size + knngraph_second_size;
  cudaMalloc(&knngraph_dev, (size_t)merged_graph_size * NEIGHB_NUM_PER_LIST *
                                sizeof(NNDElement));
  // CopySecondHalfToKNNGraph<<<merged_graph_size, WARP_SIZE * 2>>>(
  //     knngraph_dev, knngraph_first_dev, knngraph_first_size,
  //     knngraph_second_dev, knngraph_second_size, random_knngraph_dev);
  InitRandomBlockedKNNGraph<<<merged_graph_size, WARP_SIZE>>>(
      knngraph_dev, knngraph_first_dev, knngraph_first_size,
      knngraph_second_dev, knngraph_second_size);
  cudaDeviceSynchronize();
  // vector<vector<NNDElement>> g;
  // ToHostKNNGraph(&g, knngraph_dev, merged_graph_size, NEIGHB_NUM_PER_LIST);
  // OutputHostKNNGraph(g, "/home/hwang/codes/GPU_KNNG/results/tmpg.txt");
  auto cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    cerr << cudaGetErrorString(cuda_status) << endl;
    exit(-1);
  }
  if (free_subgraph) {
    cudaFree(knngraph_first_dev);
    cudaFree(knngraph_second_dev);
  }
}

void MergeVectors(float **vectors_dev_ptr, float *vectors_first_dev,
                  const int vectors_first_size, float *vectors_second_dev,
                  const int vectors_second_size,
                  const bool free_sub_data = false) {
  float *&vectors_dev = *vectors_dev_ptr;
  int merged_size = vectors_first_size + vectors_second_size;
  cudaMalloc(&vectors_dev, (size_t)merged_size * VEC_DIM * sizeof(float));
  cudaMemcpyAsync(vectors_dev, vectors_first_dev,
                  (size_t)vectors_first_size * VEC_DIM * sizeof(float),
                  cudaMemcpyDeviceToDevice);
  if (free_sub_data) {
    cudaFree(vectors_first_dev);
  }
  cudaMemcpyAsync(vectors_dev + (size_t)vectors_first_size * VEC_DIM,
                  vectors_second_dev,
                  (size_t)vectors_second_size * VEC_DIM * sizeof(float),
                  cudaMemcpyDeviceToDevice);
  if (free_sub_data) {
    cudaFree(vectors_second_dev);
  }
}

namespace gpuknn {
void KNNMerge(NNDElement **knngraph_merged_dev_ptr, float *vectors_first_dev,
              const int vectors_first_size, NNDElement *knngraph_first_dev,
              float *vectors_second_dev, const int vectors_second_size,
              NNDElement *knngraph_second_dev, const bool free_sub_data) {
  NNDElement *&knngraph_merged_dev = *knngraph_merged_dev_ptr;
  float *vectors_dev;
  int merged_graph_size = vectors_first_size + vectors_second_size;
  auto start = chrono::steady_clock::now();
  MarkAllToOld<<<vectors_first_size, NEIGHB_NUM_PER_LIST>>>(knngraph_first_dev);
  MarkAllToOld<<<vectors_second_size, NEIGHB_NUM_PER_LIST>>>(
      knngraph_second_dev);
  cudaDeviceSynchronize();
  PrepareGraphForMerge(&knngraph_merged_dev, knngraph_first_dev,
                       vectors_first_size, knngraph_second_dev,
                       vectors_second_size, free_sub_data);
  MergeVectors(&vectors_dev, vectors_first_dev, vectors_first_size,
               vectors_second_dev, vectors_second_size, free_sub_data);
  auto end = chrono::steady_clock::now();
  float time_cost =
      (float)chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() /
      1e6;
  cerr << "PrepareGraphForMerge costs: " << time_cost << endl;
  NNDescentRefine(knngraph_merged_dev, vectors_dev, merged_graph_size, VEC_DIM,
                  3);
  cudaFree(vectors_dev);
}

void KNNMergeFromHost(NNDElement **knngraph_merged_dev_ptr,
                      const float *vectors_first, const int vectors_first_size,
                      const NNDElement *knngraph_first,
                      const float *vectors_second,
                      const int vectors_second_size,
                      const NNDElement *knngraph_second) {
  NNDElement *&knngraph_merged_dev = *knngraph_merged_dev_ptr;
  float *vectors_dev;
  int merged_graph_size = vectors_first_size + vectors_second_size;
  auto start = chrono::steady_clock::now();
  NNDElement *knngraph_first_dev, *knngraph_second_dev;
  cudaMalloc(&knngraph_first_dev, (size_t)vectors_first_size *
                                      NEIGHB_NUM_PER_LIST * sizeof(NNDElement));
  cudaMalloc(
      &knngraph_second_dev,
      (size_t)vectors_second_size * NEIGHB_NUM_PER_LIST * sizeof(NNDElement));
  cudaMemcpy(
      knngraph_first_dev, knngraph_first,
      (size_t)vectors_first_size * NEIGHB_NUM_PER_LIST * sizeof(NNDElement),
      cudaMemcpyHostToDevice);
  cudaMemcpy(
      knngraph_second_dev, knngraph_second,
      (size_t)vectors_second_size * NEIGHB_NUM_PER_LIST * sizeof(NNDElement),
      cudaMemcpyHostToDevice);

  MarkAllToOld<<<vectors_first_size, NEIGHB_NUM_PER_LIST>>>(knngraph_first_dev);
  MarkAllToOld<<<vectors_second_size, NEIGHB_NUM_PER_LIST>>>(
      knngraph_second_dev);
  cudaDeviceSynchronize();
  auto cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    cerr << "Mark failed " << cudaGetErrorString(cuda_status) << endl;
    exit(-1);
  }

  // Dev. ptrs are freed inside the function.
  PrepareGraphForMerge(&knngraph_merged_dev, knngraph_first_dev,
                       vectors_first_size, knngraph_second_dev,
                       vectors_second_size, true);

  float *vectors_first_dev, *vectors_second_dev;
  cudaMalloc(&vectors_first_dev,
             (size_t)vectors_first_size * VEC_DIM * sizeof(float));
  cudaMalloc(&vectors_second_dev,
             (size_t)vectors_second_size * VEC_DIM * sizeof(float));
  cudaMemcpy(vectors_first_dev, vectors_first,
             (size_t)vectors_first_size * VEC_DIM * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(vectors_second_dev, vectors_second,
             (size_t)vectors_second_size * VEC_DIM * sizeof(float),
             cudaMemcpyHostToDevice);

  // Dev. ptrs are freed inside the function.
  MergeVectors(&vectors_dev, vectors_first_dev, vectors_first_size,
               vectors_second_dev, vectors_second_size, true);
  auto end = chrono::steady_clock::now();
  float time_cost =
      (float)chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() /
      1e6;
  cerr << "PrepareGraphForMerge costs: " << time_cost << endl;

  NNDescentRefine(knngraph_merged_dev, vectors_dev, merged_graph_size, VEC_DIM,
                  5);
  cudaFree(vectors_dev);
}
}  // namespace gpuknn