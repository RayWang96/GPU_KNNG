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

__global__ void CopyLastHalfToKNNGraph(NNDElement *knngraph,
                                       const NNDElement *knngraph_first,
                                       const int knngraph_first_size,
                                       const NNDElement *knngraph_second,
                                       const int knngraph_second_size,
                                       const int *random_knngraph) {
  int list_id = blockIdx.x;
  int tx = threadIdx.x;
  int lane_id = tx % WARP_SIZE;
  int knngraph_base_pos = list_id * NEIGHB_NUM_PER_LIST;
  int rand_knngraph_base_pos = list_id * LAST_HALF_NEIGHB_NUM;

  if (list_id < knngraph_first_size) {
    if (tx < WARP_SIZE) {
      int it_num = GetItNum(FIRST_HALF_NEIGHB_NUM, WARP_SIZE);
      for (int i = 0; i < it_num; i++) {
        int neighb_pos = i * WARP_SIZE + lane_id;
        if (neighb_pos >= FIRST_HALF_NEIGHB_NUM) break;
        knngraph[knngraph_base_pos + neighb_pos] =
            knngraph_first[knngraph_base_pos + neighb_pos];
      }
    } else {
      int it_num = GetItNum(LAST_HALF_NEIGHB_NUM, WARP_SIZE);
      for (int i = 0; i < it_num; i++) {
        int neighb_pos = i * WARP_SIZE + lane_id;
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
    rand_knngraph_base_pos =
        (list_id - knngraph_first_size) * LAST_HALF_NEIGHB_NUM;
    if (tx < WARP_SIZE) {
      int it_num = GetItNum(FIRST_HALF_NEIGHB_NUM, WARP_SIZE);
      for (int i = 0; i < it_num; i++) {
        int neighb_pos = i * WARP_SIZE + lane_id;
        if (neighb_pos >= FIRST_HALF_NEIGHB_NUM) break;
        auto elem = knngraph_second[knngraph_second_base_pos + neighb_pos];
        elem.SetLabel(elem.label() + knngraph_first_size);
        knngraph[knngraph_base_pos + neighb_pos] = elem;
      }
    } else {
      int it_num = GetItNum(LAST_HALF_NEIGHB_NUM, WARP_SIZE);
      for (int i = 0; i < it_num; i++) {
        int neighb_pos = i * WARP_SIZE + lane_id;
        if (neighb_pos >= FIRST_HALF_NEIGHB_NUM) break;
        auto &elem =
            knngraph[knngraph_base_pos + LAST_HALF_NEIGHB_NUM + neighb_pos];
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
  cudaMalloc(&knngraph_dev, (size_t)merged_graph_size * NEIGHB_NUM_PER_LIST *
                                sizeof(NNDElement));
  CopyLastHalfToKNNGraph<<<merged_graph_size, 32 * 2>>>(
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
  cudaMalloc(&vectors_dev, (size_t)merged_size * VEC_DIM * sizeof(float));
  cudaMemcpy(vectors_dev, vectors_first_dev,
             (size_t)vectors_first_size * VEC_DIM * sizeof(float),
             cudaMemcpyDeviceToDevice);
  cudaMemcpy(vectors_dev + (size_t)vectors_first_size * VEC_DIM,
             vectors_second_dev,
             (size_t)vectors_second_size * VEC_DIM * sizeof(float),
             cudaMemcpyDeviceToDevice);
}

__global__ void CalcDistForLastHalfNeighbsKernel(NNDElement *knn_graph,
                                                 const int graph_size,
                                                 const float *vectors) {
  int list_id = blockIdx.x;
  int tx = threadIdx.x;
  int lane_id = threadIdx.x % WARP_SIZE;
  int vec_a_pos = list_id * VEC_DIM;

  for (int i = FIRST_HALF_NEIGHB_NUM; i < NEIGHB_NUM_PER_LIST; i++) {
    int it_num = GetItNum(VEC_DIM, blockDim.x);
    float sum = 0;
    int vec_b_pos =
        knn_graph[list_id * NEIGHB_NUM_PER_LIST + i].label() * VEC_DIM;
    for (int j = 0; j < it_num; j++) {
      int vec_elem_pos = j * blockDim.x + tx;
      float elem_a, elem_b, diff;
      if (vec_elem_pos < VEC_DIM) {
        elem_a = vectors[vec_a_pos + vec_elem_pos];
        elem_b = vectors[vec_b_pos + vec_elem_pos];
        diff = elem_a - elem_b;
        diff *= diff;
      } else {
        diff = 0;
      }
      for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        diff = diff + __shfl_down_sync(FULL_MASK, diff, offset);
      sum += diff;
    }
    if (lane_id == 0) {
      knn_graph[list_id * NEIGHB_NUM_PER_LIST + i].SetDistance(sum);
    }
  }
}

void CalcDistForLastHalfNeighbs(NNDElement *knngraph_dev, const int graph_size,
                                const float *vectors_dev) {
  CalcDistForLastHalfNeighbsKernel<<<graph_size, 32>>>(knngraph_dev, graph_size,
                                                       vectors_dev);
  cudaDeviceSynchronize();
  auto cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    cerr << cudaGetErrorString(cuda_status) << endl;
    cerr << "CalcDistForHalfNeighbs failed." << endl;
    exit(-1);
  }
}

__global__ void SortLastHalfNeighbsKernel(NNDElement *knn_graph,
                                          const int graph_size) {
  __shared__ NNDElement knn_list_cache[LAST_HALF_NEIGHB_NUM];
  __shared__ NNDElement sorted_elements_cache[WARP_SIZE];
  __shared__ NNDElement merged_list_cache[LAST_HALF_NEIGHB_NUM];

  int list_id = blockIdx.x;
  int global_base_pos = list_id * NEIGHB_NUM_PER_LIST;
  int tx = threadIdx.x;
  int lane_id = tx % WARP_SIZE;
  int list_size = 0;
  int it_num = GetItNum(LAST_HALF_NEIGHB_NUM, WARP_SIZE);
  for (int i = 0; i < it_num; i++) {
    int pos = i * WARP_SIZE + tx;
    if (pos >= LAST_HALF_NEIGHB_NUM) break;
    merged_list_cache[pos] = knn_list_cache[pos] = NNDElement(1e10, LARGE_INT);
  }
  for (int i = 0; i < it_num; i++) {
    NNDElement elem;
    int pos = FIRST_HALF_NEIGHB_NUM + i * WARP_SIZE + tx;
    if (pos >= NEIGHB_NUM_PER_LIST) {
      elem.SetDistance(1e10);
      elem.SetLabel(LARGE_INT);
    } else {
      elem = knn_graph[global_base_pos + pos];
    }
    BitonicSort(&elem, lane_id);
    sorted_elements_cache[lane_id] = elem;
    if (lane_id == 0) {
      list_size = MergeList(knn_list_cache, list_size, sorted_elements_cache,
                            WARP_SIZE, merged_list_cache, NEIGHB_NUM_PER_LIST);
    }
    list_size = __shfl_sync(FULL_MASK, list_size, 0);
    int copy_it_num = GetItNum(list_size, WARP_SIZE);
    for (int j = 0; j < copy_it_num; j++) {
      int pos = j * WARP_SIZE + lane_id;
      if (pos >= LAST_HALF_NEIGHB_NUM) break;
      knn_list_cache[pos] = merged_list_cache[pos];
    }
  }
  __syncthreads();
  for (int i = 0; i < it_num; i++) {
    int pos = FIRST_HALF_NEIGHB_NUM + i * WARP_SIZE + tx;
    if (pos >= NEIGHB_NUM_PER_LIST) break;
    knn_graph[global_base_pos + pos] = knn_list_cache[pos];
  }
}

void SortLastHalfNeighbs(NNDElement *knngraph_dev, const int graph_size) {
  SortLastHalfNeighbsKernel<<<graph_size, 32>>>(knngraph_dev, graph_size);
  cudaDeviceSynchronize();
  auto cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    cerr << cudaGetErrorString(cuda_status) << endl;
    cerr << "SortLastHalfNeighbs failed." << endl;
    exit(-1);
  }
}

__global__ void SampleGraphKernel(int *graph_new_dev, int *newg_list_size_dev,
                                  int *graph_old_dev, int *oldg_list_size_dev,
                                  NNDElement *knn_graph, int graph_size) {
  __shared__ int new_elements_cache[FIRST_HALF_NEIGHB_NUM];
  __shared__ int cache1_size;
  __shared__ int old_elements_cache[NEIGHB_NUM_PER_LIST];
  __shared__ int cache2_size;
  int list_id = blockIdx.x;
  int knng_base_pos = list_id * NEIGHB_NUM_PER_LIST;
  int nn_list_base_pos = list_id * (SAMPLE_NUM * 2);
  int tx = threadIdx.x;
  if (tx == 0) {
    cache1_size = cache2_size = 0;
  }
  __syncthreads();
  int it_num = GetItNum(FIRST_HALF_NEIGHB_NUM, WARP_SIZE);
  for (int i = 0; i < it_num; i++) {
    int local_pos = i * WARP_SIZE + tx;
    if (local_pos >= FIRST_HALF_NEIGHB_NUM) break;
    NNDElement elem = knn_graph[knng_base_pos + local_pos];
    int pos = atomicAdd(&cache2_size, 1);
    old_elements_cache[pos] = elem.label();
  }
  it_num = GetItNum(LAST_HALF_NEIGHB_NUM, WARP_SIZE);
  for (int i = 0; i < it_num; i++) {
    int local_pos = FIRST_HALF_NEIGHB_NUM + i * WARP_SIZE + tx;
    if (local_pos >= NEIGHB_NUM_PER_LIST) break;
    NNDElement elem = knn_graph[knng_base_pos + local_pos];
    if (elem.IsNew()) {
      int pos = atomicAdd(&cache1_size, 1);
      new_elements_cache[pos] = elem.label();
    }
  }
  __syncthreads();
  if (tx == 0) {
    cache1_size = min(cache1_size, SAMPLE_NUM);
    cache2_size = min(cache2_size, SAMPLE_NUM);
  }
  __syncthreads();
  if (tx == 0) {
    newg_list_size_dev[list_id] = cache1_size;
    oldg_list_size_dev[list_id] = cache2_size;
  }
  it_num = GetItNum(SAMPLE_NUM, WARP_SIZE);
  for (int i = 0; i < it_num; i++) {
    int local_pos = i * WARP_SIZE + tx;
    if (local_pos < cache1_size) {
      graph_new_dev[nn_list_base_pos + local_pos] =
          new_elements_cache[local_pos];
    }
    if (local_pos < cache2_size) {
      graph_old_dev[nn_list_base_pos + local_pos] =
          old_elements_cache[local_pos];
    }
  }
  __syncthreads();
}

__global__ void SampleReverseGraphKernel(
    int *graph_new_dev, int *newg_list_size_dev, int *newg_revlist_size_dev,
    int *graph_old_dev, int *oldg_list_size_dev, int *oldg_revlist_size_dev) {
  __shared__ int new_elements_cache[SAMPLE_NUM];
  __shared__ int cache1_size;
  __shared__ int old_elements_cache[SAMPLE_NUM];
  __shared__ int cache2_size;
  int tx = threadIdx.x;
  int list_id = blockIdx.x;
  int knng_base_pos = list_id * NEIGHB_NUM_PER_LIST;
  int nn_list_base_pos = list_id * (SAMPLE_NUM * 2);
  if (tx == 0) {
    cache1_size = newg_list_size_dev[list_id];
    cache2_size = oldg_list_size_dev[list_id];
  }
  __syncthreads();
  int it_num = GetItNum(SAMPLE_NUM, WARP_SIZE);
  for (int i = 0; i < it_num; i++) {
    int local_pos = i * WARP_SIZE + tx;
    if (local_pos < cache1_size) {
      new_elements_cache[local_pos] =
          graph_new_dev[nn_list_base_pos + local_pos];
    }
    if (local_pos < cache2_size) {
      old_elements_cache[local_pos] =
          graph_old_dev[nn_list_base_pos + local_pos];
    }
  }
  __syncthreads();
  it_num = GetItNum(SAMPLE_NUM, WARP_SIZE);
  for (int i = 0; i < it_num; i++) {
    int local_pos = i * WARP_SIZE + tx;
    if (local_pos < cache1_size) {
      int rev_list_id = new_elements_cache[local_pos];
      int pos = SAMPLE_NUM;
      pos += atomicAdd(&newg_revlist_size_dev[rev_list_id], 1);
      if (pos >= SAMPLE_NUM * 2)
        atomicExch(&newg_revlist_size_dev[rev_list_id], SAMPLE_NUM);
      else
        graph_new_dev[rev_list_id * (SAMPLE_NUM * 2) + pos] = list_id;
    }
    if (local_pos < cache2_size) {
      int rev_list_id = old_elements_cache[local_pos];
      int pos = SAMPLE_NUM;
      pos += atomicAdd(&oldg_revlist_size_dev[rev_list_id], 1);
      if (pos >= SAMPLE_NUM * 2)
        atomicExch(&oldg_revlist_size_dev[rev_list_id], SAMPLE_NUM);
      else
        graph_old_dev[rev_list_id * (SAMPLE_NUM * 2) + pos] = list_id;
    }
  }
}

__global__ void MarkFirstHalfToOld(NNDElement *knn_graph) {
  int list_id = blockIdx.x;
  int tx = threadIdx.x;
  int graph_base_pos = list_id * NEIGHB_NUM_PER_LIST;
  knn_graph[graph_base_pos + tx].MarkOld();
}


void PrepareForMergeUpdate(int *graph_new_dev, int *newg_list_size_dev,
                           int *newg_revlist_size_dev, int *graph_old_dev,
                           int *oldg_list_size_dev, int *oldg_revlist_size_dev,
                           NNDElement *knn_graph_dev, int graph_size) {
  cudaMemset(newg_list_size_dev, 0, graph_size * sizeof(int));
  cudaMemset(oldg_list_size_dev, 0, graph_size * sizeof(int));
  cudaMemset(newg_revlist_size_dev, 0, graph_size * sizeof(int));
  cudaMemset(oldg_revlist_size_dev, 0, graph_size * sizeof(int));
  MarkFirstHalfToOld<<<graph_size, FIRST_HALF_NEIGHB_NUM>>>(knn_graph_dev);
  cudaDeviceSynchronize();
  SampleGraphKernel<<<graph_size, WARP_SIZE>>>(
      graph_new_dev, newg_list_size_dev, graph_old_dev, oldg_list_size_dev,
      knn_graph_dev, graph_size);
  cudaDeviceSynchronize();
  SampleReverseGraphKernel<<<graph_size, WARP_SIZE>>>(
      graph_new_dev, newg_list_size_dev, newg_revlist_size_dev, graph_old_dev,
      oldg_list_size_dev, oldg_revlist_size_dev);
  cudaDeviceSynchronize();
  auto cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    cerr << cudaGetErrorString(cuda_status) << endl;
    cerr << "PrepareForMergeUpdate kernel failed." << endl;
    exit(-1);
  }
  ShrinkGraph<<<graph_size, WARP_SIZE>>>(
      graph_new_dev, newg_list_size_dev, newg_revlist_size_dev, graph_old_dev,
      oldg_list_size_dev, oldg_revlist_size_dev);
  cudaDeviceSynchronize();
  cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    cerr << cudaGetErrorString(cuda_status) << endl;
    cerr << "ShrinkGraph kernel failed." << endl;
    exit(-1);
  }
}

__device__ void InsertToLastHalfGlobalGraph(NNDElement elem,
                                            const int global_id,
                                            NNDElement *global_knn_graph,
                                            int *global_locks) {
  int tx = threadIdx.x;
  int lane_id = tx % WARP_SIZE;
  int global_pos_base = global_id * NEIGHB_NUM_PER_LIST;
  elem.distance_ = __shfl_sync(FULL_MASK, elem.distance_, 0);
  elem.label_ = __shfl_sync(FULL_MASK, elem.label_, 0);
  if (elem.label() < FIRST_HALF_NEIGHB_NUM) return;
  int loop_flag = 0;
  do {
    if (lane_id == 0)
      loop_flag = atomicCAS(&global_locks[global_id], 0, 1) == 0;
    loop_flag = __shfl_sync(FULL_MASK, loop_flag, 0);
    if (loop_flag == 1) {
      NNDElement knn_list_frag[LAST_HALF_INSERT_IT_NUM];
      for (int i = 0; i < LAST_HALF_INSERT_IT_NUM; i++) {
        int local_pos = i * WARP_SIZE + lane_id;
        int global_pos = global_pos_base + FIRST_HALF_NEIGHB_NUM + local_pos;
        if (local_pos < LAST_HALF_NEIGHB_NUM)
          knn_list_frag[i] = global_knn_graph[global_pos];
        else
          knn_list_frag[i] = NNDElement(1e10, LARGE_INT);
      }

      int pos_to_insert = -1;
      for (int i = 0; i < LAST_HALF_INSERT_IT_NUM; i++) {
        NNDElement prev_elem = __shfl_up_sync(FULL_MASK, knn_list_frag[i], 1);
        if (lane_id == 0) prev_elem = NNDElement(-1e10, -LARGE_INT);
        if (elem > prev_elem && elem < knn_list_frag[i])
          pos_to_insert = i * WARP_SIZE + lane_id;
        else if (elem == prev_elem || elem == knn_list_frag[i])
          pos_to_insert = -2;
        if (__ballot_sync(FULL_MASK, pos_to_insert == -2)) break;
        uint mask = __ballot_sync(FULL_MASK, pos_to_insert >= 0);
        if (mask) {
          uint set_lane_id = GetNthSetBitPos(mask, 1);
          pos_to_insert = __shfl_sync(FULL_MASK, pos_to_insert, set_lane_id);
          // assert(false);
          break;
        }
      }
      if (pos_to_insert >= 0) {
        pos_to_insert += FIRST_HALF_NEIGHB_NUM;
        for (int i = 0; i < LAST_HALF_INSERT_IT_NUM; i++) {
          int local_pos = FIRST_HALF_NEIGHB_NUM + i * WARP_SIZE + lane_id;
          if (local_pos > pos_to_insert) {
            local_pos++;
          } else if (local_pos == pos_to_insert) {
            global_knn_graph[global_pos_base + local_pos] = elem;
            local_pos++;
          }
          int global_pos = global_pos_base + local_pos;
          if (local_pos < NEIGHB_NUM_PER_LIST)
            global_knn_graph[global_pos] = knn_list_frag[i];
        }
      }
    }
    __threadfence();
    if (loop_flag && lane_id == 0) {
      atomicExch(&global_locks[global_id], 0);
    }
    __nanosleep(32);
  } while (!loop_flag);
}

__global__ void CompareNeighbsForMerge(
    NNDElement *knn_graph, int *global_locks, const float *vectors,
    const int *graph_new, const int *size_new, const int num_new_max,
    const int *graph_old, const int *size_old, const int num_old_max) {
  extern __shared__ char buffer[];
  __shared__ float *distances;
  __shared__ int *neighbors;

  __shared__ int gnew_base_pos, gold_base_pos, num_new, num_old;

  int tx = threadIdx.x;
  if (tx == 0) {
    distances = (float *)buffer;
    neighbors = (int *)((char *)distances +
                        (num_new_max * num_old_max) * sizeof(float));
  }
  __syncthreads();

  int list_id = blockIdx.x;
  int block_dim_x = blockDim.x;

  if (tx == 0) {
    gnew_base_pos = list_id * (SAMPLE_NUM * 2);
    gold_base_pos = list_id * (SAMPLE_NUM * 2);
  } else if (tx == 32) {
    num_new = size_new[list_id];
    num_old = size_old[list_id];
  }
  __syncthreads();
  int neighb_num = num_new + num_old;
  if (tx < num_new) {
    neighbors[tx] = graph_new[gnew_base_pos + tx];
  } else if (tx >= num_new && tx < neighb_num) {
    neighbors[tx] = graph_old[gnew_base_pos + tx - num_new];
  }
  __syncthreads();

  GetNewOldDistancesTiled(distances, vectors, neighbors, num_new,
                          neighbors + num_new, num_old);
  __syncthreads();

  int calc_num = num_new * num_old;
  // int num_it = GetItNum(NEIGHB_NUM_PER_LIST, NEIGHB_CACHE_NUM);

  // Read list to cache
  int list_size = NEIGHB_CACHE_NUM;
  int num_it3 = GetItNum(neighb_num, block_dim_x / WARP_SIZE);
  for (int j = 0; j < num_it3; j++) {
    int list_id = j * (block_dim_x / WARP_SIZE) + tx / WARP_SIZE;
    if (list_id >= neighb_num) continue;
    NNDElement min_elem(1e10, LARGE_INT);
    if (list_id < num_new) {
      min_elem =
          Min(min_elem, GetMinElement2(list_id, list_size, neighbors + num_new,
                                       num_old, distances, calc_num));
    } else {
      min_elem =
          Min(min_elem, GetMinElement3(list_id, list_size, neighbors, num_new,
                                       neighbors + num_new, num_old, distances,
                                       calc_num, vectors));
    }
    InsertToLastHalfGlobalGraph(min_elem, neighbors[list_id], knn_graph,
                                global_locks);
  }
  __syncthreads();
}

float UpdateGraphForMerge(NNDElement *origin_knn_graph_dev, const size_t g_size,
                          const float *vectors_dev, int *newg_dev,
                          int *newg_list_size_dev, int *oldg_dev,
                          int *oldg_list_size_dev, const int k) {
  float kernel_time = 0;
  cudaError_t cuda_status;

  int *global_locks_dev;
  cudaMalloc(&global_locks_dev, g_size * sizeof(int));
  cudaMemset(global_locks_dev, 0, g_size * sizeof(int));
  cuda_status = cudaGetLastError();

  if (cuda_status != cudaSuccess) {
    cerr << cudaGetErrorString(cuda_status) << endl;
    cerr << "Initiate failed" << endl;
    exit(-1);
  }

  dim3 block_size(640);
  dim3 grid_size(g_size);
  // cerr << "Start kernel." << endl;
  const int num_new_max = GetMaxListSize(newg_list_size_dev, g_size);
  const int num_old_max = GetMaxListSize(oldg_list_size_dev, g_size);
  cerr << "Num new max: " << num_new_max << endl;
  cerr << "Num old max: " << num_old_max << endl;
  size_t shared_memory_size =
      num_new_max * SKEW_DIM * sizeof(float) +
      (num_new_max * (num_new_max - 1) / 2) * sizeof(float) +
      num_new_max * sizeof(int);

  cerr << "Shmem kernel1 costs: " << shared_memory_size << endl;

  auto start = chrono::steady_clock::now();
  MarkAllToOld<<<grid_size, block_size>>>(origin_knn_graph_dev);
  cudaDeviceSynchronize();
  int neighb_num_max = num_new_max + num_old_max;
  block_size = dim3(TILE_WIDTH * TILE_WIDTH);
  shared_memory_size = (num_new_max * num_old_max) * sizeof(float) +
                       neighb_num_max * sizeof(int);
  cerr << "Shmem tiled kernel2 costs: " << shared_memory_size << endl;
  CompareNeighbsForMerge<<<grid_size, block_size, shared_memory_size>>>(
      origin_knn_graph_dev, global_locks_dev, vectors_dev, newg_dev,
      newg_list_size_dev, num_new_max, oldg_dev, oldg_list_size_dev,
      num_old_max);
  cudaDeviceSynchronize();
  auto end = chrono::steady_clock::now();
  kernel_time =
      (float)chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() /
      1e6;
  cuda_status = cudaGetLastError();

  if (cuda_status != cudaSuccess) {
    cerr << cudaGetErrorString(cuda_status) << endl;
    cerr << "Kernel failed" << endl;
    exit(-1);
  }
  // cerr << "End kernel." << endl;
  if (cuda_status != cudaSuccess) {
    cerr << cudaGetErrorString(cuda_status) << endl;
    cerr << "knn_graph cudaMemcpy failed" << endl;
    exit(-1);
  }

  cudaFree(global_locks_dev);
  return kernel_time;
}

void NNDescentForMerge(NNDElement *knngraph_dev, const float *vectors_dev,
                       const int vecs_size, const int vecs_dim,
                       const int iteration = 6) {
  int k = NEIGHB_NUM_PER_LIST;
  int *graph_new_dev, *newg_list_size_dev, *graph_old_dev, *oldg_list_size_dev;
  int *newg_revlist_size_dev, *oldg_revlist_size_dev;
  int graph_size = vecs_size;
  cudaMalloc(&graph_new_dev,
             (size_t)graph_size * (SAMPLE_NUM * 2) * sizeof(int));
  cudaMalloc(&newg_list_size_dev, (size_t)graph_size * sizeof(int));
  cudaMalloc(&newg_revlist_size_dev, (size_t)graph_size * sizeof(int));
  cudaMalloc(&graph_old_dev,
             (size_t)graph_size * (SAMPLE_NUM * 2) * sizeof(int));
  cudaMalloc(&oldg_list_size_dev, (size_t)graph_size * sizeof(int));
  cudaMalloc(&oldg_revlist_size_dev, (size_t)graph_size * sizeof(int));
  Graph result(vecs_size);
  vector<vector<NNDElement>> g(vecs_size);
  CalcDistForLastHalfNeighbs(knngraph_dev, graph_size, vectors_dev);
  SortLastHalfNeighbs(knngraph_dev, graph_size);
  auto cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    cerr << cudaGetErrorString(cuda_status) << endl;
    cerr << "Init failed" << endl;
    exit(-1);
  }
  float iteration_costs = 0;
  Graph newg, oldg;
  float get_nb_graph_time = 0;
  float kernel_costs = 0;
  auto sum_start = chrono::steady_clock::now();
  long long cmp_times = 0;
  for (int t = 0; t < iteration; t++) {
    cerr << "Start generating NBGraph." << endl;
    // Should be removed after testing.
    auto start = chrono::steady_clock::now();
    PrepareForMergeUpdate(
        graph_new_dev, newg_list_size_dev, newg_revlist_size_dev, graph_old_dev,
        oldg_list_size_dev, oldg_revlist_size_dev, knngraph_dev, graph_size);
    auto end = chrono::steady_clock::now();
    float tmp_time =
        (float)chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count() /
        1e6;
    get_nb_graph_time += tmp_time;
    cerr << "GetNBGraph costs " << tmp_time << endl;
    start = chrono::steady_clock::now();
    float tmp_kernel_costs = UpdateGraphForMerge(
        knngraph_dev, graph_size, vectors_dev, graph_new_dev,
        newg_list_size_dev, graph_old_dev, oldg_list_size_dev, k);
    kernel_costs += tmp_kernel_costs;
    end = chrono::steady_clock::now();
    float it_tmp_costs =
        (float)chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count() /
        1e6;
    iteration_costs += it_tmp_costs;
    cerr << "Kernel costs " << tmp_kernel_costs << endl;
    cerr << endl;
  }
  auto sum_end = chrono::steady_clock::now();
  float sum_costs = (float)chrono::duration_cast<std::chrono::microseconds>(
                        sum_end - sum_start)
                        .count() /
                    1e6;
  // sift10k in cpu should be 0.6s;
  cerr << "Compare times: " << cmp_times << endl;
  cerr << "FLOPS: " << cmp_times * 128 * 3 / kernel_costs / pow(1024.0, 3)
       << "G" << endl;
  cerr << "Kernel costs: " << kernel_costs << endl;
  cerr << "Update costs: " << iteration_costs << endl;
  cerr << "Get NB graph costs: " << get_nb_graph_time << endl;
  cerr << "All procedure costs: " << sum_costs << endl;
  cerr << endl;
  cudaFree(graph_new_dev);
  cudaFree(newg_list_size_dev);
  cudaFree(graph_old_dev);
  cudaFree(oldg_list_size_dev);
  cudaFree(newg_revlist_size_dev);
  cudaFree(oldg_revlist_size_dev);
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
                                LAST_HALF_NEIGHB_NUM);
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
  // NNDescentRefine(knngraph_merged_dev, vectors_dev, merged_graph_size,
  //                 VEC_DIM, 5);
  NNDescentForMerge(knngraph_merged_dev, vectors_dev, merged_graph_size,
                    VEC_DIM, 3);
  // Just need merge two half actually.
  SortKNNGraphKernel<<<merged_graph_size, WARP_SIZE>>>(knngraph_merged_dev,
                                                       merged_graph_size);
  cudaDeviceSynchronize();
  vector<vector<NNDElement>> res;
  ToHostKNNGraph(&res, knngraph_merged_dev, merged_graph_size,
                 NEIGHB_NUM_PER_LIST);
  OutputHostKNNGraph(res,
                     "/home/hwang/codes/GPU_KNNG/results/graph_half_good.txt");
  if (!have_random_knngraph) {
    cudaFree(random_knngraph_dev);
  }
}
}  // namespace gpuknn