#include <assert.h>
#include <curand.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <algorithm>
#include <bitset>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <tuple>
#include <utility>
#include <vector>

#include "../tools/distfunc.hpp"
#include "../tools/nndescent_element.cuh"
#include "../xmuknn.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "knncuda_tools.cuh"
#include "nndescent.cuh"

#ifdef __INTELLISENSE__
#include "../intellisense_cuda_intrinsics.h"
#endif

using namespace std;
using namespace xmuknn;
#define DEVICE_ID 0
#define FULL_MASK 0xffffffff
#define VERBOSE 0
// #define DONT_TILE 0
// #define INSERT_MIN_ONLY 1

__device__ int for_check = 0;

__global__ void PrepareGraph(int *graph_new_dev, int *newg_list_size_dev,
                             int *graph_old_dev, int *oldg_list_size_dev,
                             NNDElement *knn_graph, int graph_size) {
  __shared__ int new_elements_cache[NEIGHB_NUM_PER_LIST];
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
  int it_num = GetItNum(NEIGHB_NUM_PER_LIST, WARP_SIZE);
  for (int i = 0; i < it_num; i++) {
    int nth = i * WARP_SIZE + tx;
    int local_pos =
        nth % NEIGHB_BLOCKS_NUM * WARP_SIZE + nth / NEIGHB_BLOCKS_NUM;
    if (local_pos < NEIGHB_NUM_PER_LIST) {
      NNDElement elem = knn_graph[knng_base_pos + local_pos];
      if (elem.IsNew()) {
        int pos = atomicAdd(&cache1_size, 1);
        new_elements_cache[pos] = elem.label();
      } else {
        int pos = atomicAdd(&cache2_size, 1);
        old_elements_cache[pos] = elem.label();
      }
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

__global__ void PrepareGraphForMerge(int *graph_new_dev,
                                     int *newg_list_size_dev,
                                     int *graph_old_dev,
                                     int *oldg_list_size_dev,
                                     NNDElement *knn_graph, int split_pos,
                                     int graph_size) {
  __shared__ int new_elements_cache[NEIGHB_NUM_PER_LIST];
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
  int it_num = GetItNum(NEIGHB_NUM_PER_LIST, WARP_SIZE);
  for (int i = 0; i < it_num; i++) {
    int nth = i * WARP_SIZE + tx;
    int local_pos =
        nth % NEIGHB_BLOCKS_NUM * WARP_SIZE + nth / NEIGHB_BLOCKS_NUM;
    if (local_pos < NEIGHB_NUM_PER_LIST) {
      NNDElement elem = knn_graph[knng_base_pos + local_pos];
      if (elem.IsNew()) {
        int pos = atomicAdd(&cache1_size, 1);
        new_elements_cache[pos] = elem.label();
      } else {
        if (list_id < split_pos) {
          if (elem.label() >= split_pos) {
            continue;
          }
        } else if (list_id >= split_pos) {
          if (elem.label() < split_pos) {
            continue;
          }
        }
        int pos = atomicAdd(&cache2_size, 1);
        old_elements_cache[pos] = elem.label();
      }
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

template <typename T>
__device__ void Swap(T &a, T &b) {
  T c = a;
  a = b;
  b = c;
}

template <typename T>
__device__ void InsertSort(T *a, const int length) {
  for (int i = 1; i < length; i++) {
    for (int j = i - 1; j >= 0 && a[j + 1] < a[j]; j--) {
      Swap(a[j], a[j + 1]);
    }
  }
}

__device__ __forceinline__ NNDElement XorSwap(NNDElement x, int mask, int dir) {
  NNDElement y;
  y.distance_ = __shfl_xor_sync(FULL_MASK, x.distance_, mask, WARP_SIZE);
  y.label_ = __shfl_xor_sync(FULL_MASK, x.label_, mask, WARP_SIZE);
  return x < y == dir ? y : x;
}

__device__ __forceinline__ int XorSwap(int x, int mask, int dir) {
  int y;
  y = __shfl_xor_sync(FULL_MASK, x, mask, WARP_SIZE);
  return x < y == dir ? y : x;
}

__device__ __forceinline__ uint Bfe(uint lane_id, uint pos) {
  uint res;
  asm("bfe.u32 %0,%1,%2,%3;" : "=r"(res) : "r"(lane_id), "r"(pos), "r"(1));
  return res;
}

template <typename T>
__device__ __forceinline__ void BitonicSort(T *sort_element_ptr,
                                            const int lane_id) {
  auto &sort_elem = *sort_element_ptr;
  sort_elem = XorSwap(sort_elem, 0x01, Bfe(lane_id, 1) ^ Bfe(lane_id, 0));
  sort_elem = XorSwap(sort_elem, 0x02, Bfe(lane_id, 2) ^ Bfe(lane_id, 1));
  sort_elem = XorSwap(sort_elem, 0x01, Bfe(lane_id, 2) ^ Bfe(lane_id, 0));
  sort_elem = XorSwap(sort_elem, 0x04, Bfe(lane_id, 3) ^ Bfe(lane_id, 2));
  sort_elem = XorSwap(sort_elem, 0x02, Bfe(lane_id, 3) ^ Bfe(lane_id, 1));
  sort_elem = XorSwap(sort_elem, 0x01, Bfe(lane_id, 3) ^ Bfe(lane_id, 0));
  sort_elem = XorSwap(sort_elem, 0x08, Bfe(lane_id, 4) ^ Bfe(lane_id, 3));
  sort_elem = XorSwap(sort_elem, 0x04, Bfe(lane_id, 4) ^ Bfe(lane_id, 2));
  sort_elem = XorSwap(sort_elem, 0x02, Bfe(lane_id, 4) ^ Bfe(lane_id, 1));
  sort_elem = XorSwap(sort_elem, 0x01, Bfe(lane_id, 4) ^ Bfe(lane_id, 0));
  sort_elem = XorSwap(sort_elem, 0x10, Bfe(lane_id, 4));
  sort_elem = XorSwap(sort_elem, 0x08, Bfe(lane_id, 3));
  sort_elem = XorSwap(sort_elem, 0x04, Bfe(lane_id, 2));
  sort_elem = XorSwap(sort_elem, 0x02, Bfe(lane_id, 1));
  sort_elem = XorSwap(sort_elem, 0x01, Bfe(lane_id, 0));
  return;
}

template <typename T>
__device__ int MergeList(T *A, const int m, T *B, const int n, T *C) {
  int i = 0, j = 0, cnt = 0;
  while ((i < m) && (j < n)) {
    if (A[i] <= B[j]) {
      C[cnt++] = A[i++];
      if (cnt >= NEIGHB_NUM_PER_LIST) goto EXIT;
    } else {
      C[cnt++] = B[j++];
      if (cnt >= NEIGHB_NUM_PER_LIST) goto EXIT;
    }
  }

  if (i == m) {
    for (; j < n; j++) {
      C[cnt++] = B[j];
      if (cnt >= NEIGHB_NUM_PER_LIST) goto EXIT;
    }
  } else {
    for (; i < m; i++) {
      C[cnt++] = A[i];
      if (cnt >= NEIGHB_NUM_PER_LIST) goto EXIT;
    }
  }
EXIT:
  return cnt;
}

__device__ int RemoveDuplicates(int *nums, int nums_size) {
  if (nums_size < 2) return nums_size;
  int a = 0, b = 1;
  while (b < nums_size)
    if (nums[b++] > nums[a]) nums[++a] = nums[b - 1];
  return (a + 1);
}

__global__ void PrepareReverseGraph(int *graph_new_dev, int *newg_list_size_dev,
                                    int *newg_revlist_size_dev,
                                    int *graph_old_dev, int *oldg_list_size_dev,
                                    int *oldg_revlist_size_dev) {
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
      // printf("%d %d %d\n", pos, rev_list_id, list_id);
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

__global__ void PrepareReverseGraphForMerge(
    int *graph_new_dev, int *newg_list_size_dev, int *newg_revlist_size_dev,
    int *graph_old_dev, int *oldg_list_size_dev, int *oldg_revlist_size_dev,
    const int split_pos) {
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
      if (list_id < split_pos && rev_list_id >= split_pos) continue;
      int pos = SAMPLE_NUM;
      pos += atomicAdd(&newg_revlist_size_dev[rev_list_id], 1);
      // printf("%d %d %d\n", pos, rev_list_id, list_id);
      if (pos >= SAMPLE_NUM * 2)
        atomicExch(&newg_revlist_size_dev[rev_list_id], SAMPLE_NUM);
      else
        graph_new_dev[rev_list_id * (SAMPLE_NUM * 2) + pos] = list_id;
    }
    if (local_pos < cache2_size) {
      int rev_list_id = old_elements_cache[local_pos];
      if (list_id >= split_pos && rev_list_id < split_pos) continue;
      int pos = SAMPLE_NUM;
      pos += atomicAdd(&oldg_revlist_size_dev[rev_list_id], 1);
      if (pos >= SAMPLE_NUM * 2)
        atomicExch(&oldg_revlist_size_dev[rev_list_id], SAMPLE_NUM);
      else
        graph_old_dev[rev_list_id * (SAMPLE_NUM * 2) + pos] = list_id;
    }
  }
}

__global__ void ShrinkGraph(int *graph_new_dev, int *newg_list_size_dev,
                            int *newg_revlist_size_dev, int *graph_old_dev,
                            int *oldg_list_size_dev,
                            int *oldg_revlist_size_dev) {
  __shared__ int new_elements_cache[SAMPLE_NUM * 2];
  __shared__ int newg_list_size, newg_revlist_size;
  __shared__ int old_elements_cache[SAMPLE_NUM * 2];
  __shared__ int sorted_elements_cache[32];
  __shared__ int merged_list_cache[NEIGHB_NUM_PER_LIST];
  __shared__ int oldg_list_size, oldg_revlist_size;
  int tx = threadIdx.x;
  int list_id = blockIdx.x;
  int nn_list_base_pos = list_id * (SAMPLE_NUM * 2);
  int lane_id = tx % WARP_SIZE;

  if (tx == 0) {
    newg_list_size = newg_list_size_dev[list_id];
    oldg_list_size = oldg_list_size_dev[list_id];
    newg_revlist_size = newg_revlist_size_dev[list_id];
    oldg_revlist_size = oldg_revlist_size_dev[list_id];
  }
  __syncthreads();
  int it_num = GetItNum(SAMPLE_NUM * 2, WARP_SIZE);
  int list_new_size = 0, list_old_size = 0;
  for (int i = 0; i < it_num; i++) {
    int local_pos = i * WARP_SIZE + tx;

    int sort_elem = LARGE_INT;
    if ((local_pos < newg_list_size) ||
        (local_pos >= SAMPLE_NUM &&
         local_pos < SAMPLE_NUM + newg_revlist_size)) {
      sort_elem = graph_new_dev[nn_list_base_pos + local_pos];
    }
    BitonicSort(&sort_elem, lane_id);
    sorted_elements_cache[lane_id] = sort_elem;
    if (lane_id == 0) {
      list_new_size =
          MergeList(new_elements_cache, list_new_size, sorted_elements_cache,
                    WARP_SIZE, merged_list_cache);
    }
    list_new_size = __shfl_sync(FULL_MASK, list_new_size, 0);
    int copy_it_num = GetItNum(list_new_size, WARP_SIZE);
    for (int j = 0; j < copy_it_num; j++) {
      int pos = j * WARP_SIZE + lane_id;
      if (pos >= SAMPLE_NUM * 2) break;
      new_elements_cache[pos] = merged_list_cache[pos];
    }

    sort_elem = LARGE_INT;
    if ((local_pos < oldg_list_size) ||
        (local_pos >= SAMPLE_NUM &&
         local_pos < SAMPLE_NUM + oldg_revlist_size)) {
      sort_elem = graph_old_dev[nn_list_base_pos + local_pos];
    }
    BitonicSort(&sort_elem, lane_id);
    sorted_elements_cache[lane_id] = sort_elem;
    if (lane_id == 0) {
      list_old_size =
          MergeList(old_elements_cache, list_old_size, sorted_elements_cache,
                    WARP_SIZE, merged_list_cache);
    }
    list_old_size = __shfl_sync(FULL_MASK, list_old_size, 0);
    copy_it_num = GetItNum(list_old_size, WARP_SIZE);
    for (int j = 0; j < copy_it_num; j++) {
      int pos = j * WARP_SIZE + lane_id;
      if (pos >= SAMPLE_NUM * 2) break;
      old_elements_cache[pos] = merged_list_cache[pos];
    }
  }
  __syncthreads();
  if (tx == 0) {
    newg_list_size = RemoveDuplicates(new_elements_cache, list_new_size);
    newg_list_size -= (new_elements_cache[newg_list_size - 1] == LARGE_INT);
    oldg_list_size = RemoveDuplicates(old_elements_cache, list_old_size);
    oldg_list_size -= (old_elements_cache[oldg_list_size - 1] == LARGE_INT);
    for (int i = 0; i < newg_list_size; i++) {
      for (int j = 0; j < oldg_list_size; j++) {
        if (new_elements_cache[i] == old_elements_cache[j]) {
          new_elements_cache[i] = -1;
          break;
        }
      }
    }
    int pos = 0;
    for (int i = 0; i < newg_list_size; i++) {
      if (new_elements_cache[i] != -1) {
        new_elements_cache[pos++] = new_elements_cache[i];
      }
    }
    newg_list_size = pos;
  }
  __syncthreads();
  it_num = GetItNum(SAMPLE_NUM * 2, WARP_SIZE);
  for (int i = 0; i < it_num; i++) {
    int local_pos = i * WARP_SIZE + tx;
    if (local_pos < newg_list_size) {
      graph_new_dev[nn_list_base_pos + local_pos] =
          new_elements_cache[local_pos];
    }
    if (local_pos < oldg_list_size) {
      graph_old_dev[nn_list_base_pos + local_pos] =
          old_elements_cache[local_pos];
    }
  }

  newg_list_size_dev[list_id] = newg_list_size;
  oldg_list_size_dev[list_id] = oldg_list_size;
}

void PrepareForUpdate(int *graph_new_dev, int *newg_list_size_dev,
                      int *newg_revlist_size_dev, int *graph_old_dev,
                      int *oldg_list_size_dev, int *oldg_revlist_size_dev,
                      NNDElement *knn_graph_dev, int graph_size) {
  auto start = chrono::steady_clock::now();
  cudaMemset(newg_list_size_dev, 0, graph_size * sizeof(int));
  cudaMemset(oldg_list_size_dev, 0, graph_size * sizeof(int));
  cudaMemset(newg_revlist_size_dev, 0, graph_size * sizeof(int));
  cudaMemset(oldg_revlist_size_dev, 0, graph_size * sizeof(int));
  dim3 grid_size(graph_size);
  dim3 block_size(32);
  PrepareGraph<<<grid_size, block_size>>>(graph_new_dev, newg_list_size_dev,
                                          graph_old_dev, oldg_list_size_dev,
                                          knn_graph_dev, graph_size);
  cudaDeviceSynchronize();
  auto cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    cerr << cudaGetErrorString(cuda_status) << endl;
    cerr << "Prepare kernel failed." << endl;
    exit(-1);
  }
  PrepareReverseGraph<<<grid_size, block_size>>>(
      graph_new_dev, newg_list_size_dev, newg_revlist_size_dev, graph_old_dev,
      oldg_list_size_dev, oldg_revlist_size_dev);
  cudaDeviceSynchronize();
  cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    cerr << cudaGetErrorString(cuda_status) << endl;
    cerr << "PrepareReverseGraph kernel failed." << endl;
    exit(-1);
  }
  ShrinkGraph<<<grid_size, block_size>>>(
      graph_new_dev, newg_list_size_dev, newg_revlist_size_dev, graph_old_dev,
      oldg_list_size_dev, oldg_revlist_size_dev);
  cudaDeviceSynchronize();
  cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    cerr << cudaGetErrorString(cuda_status) << endl;
    cerr << "ShrinkGraph kernel failed." << endl;
    exit(-1);
  }
  auto end = chrono::steady_clock::now();
  if (VERBOSE) {
    cerr << "Prepare kernel costs: "
         << (float)chrono::duration_cast<std::chrono::microseconds>(end - start)
                    .count() /
                1e6
         << endl;
  }
}

void PrepareForUpdateForMerge(int *graph_new_dev, int *newg_list_size_dev,
                              int *newg_revlist_size_dev, int *graph_old_dev,
                              int *oldg_list_size_dev,
                              int *oldg_revlist_size_dev,
                              NNDElement *knn_graph_dev, int split_pos,
                              int graph_size) {
  auto start = chrono::steady_clock::now();
  cudaMemset(newg_list_size_dev, 0, graph_size * sizeof(int));
  cudaMemset(oldg_list_size_dev, 0, graph_size * sizeof(int));
  cudaMemset(newg_revlist_size_dev, 0, graph_size * sizeof(int));
  cudaMemset(oldg_revlist_size_dev, 0, graph_size * sizeof(int));
  dim3 grid_size(graph_size);
  dim3 block_size(32);
  PrepareGraphForMerge<<<grid_size, block_size>>>(
      graph_new_dev, newg_list_size_dev, graph_old_dev, oldg_list_size_dev,
      knn_graph_dev, split_pos, graph_size);
  cudaDeviceSynchronize();
  auto cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    cerr << cudaGetErrorString(cuda_status) << endl;
    cerr << "Prepare kernel failed." << endl;
    exit(-1);
  }
  // PrepareReverseGraphForMerge<<<grid_size, block_size>>>(
  //     graph_new_dev, newg_list_size_dev, newg_revlist_size_dev, graph_old_dev,
  //     oldg_list_size_dev, oldg_revlist_size_dev, split_pos);
  PrepareReverseGraph<<<grid_size, block_size>>>(
      graph_new_dev, newg_list_size_dev, newg_revlist_size_dev, graph_old_dev,
      oldg_list_size_dev, oldg_revlist_size_dev);
  cudaDeviceSynchronize();
  cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    cerr << cudaGetErrorString(cuda_status) << endl;
    cerr << "PrepareReverseGraph kernel failed." << endl;
    exit(-1);
  }
  ShrinkGraph<<<grid_size, block_size>>>(
      graph_new_dev, newg_list_size_dev, newg_revlist_size_dev, graph_old_dev,
      oldg_list_size_dev, oldg_revlist_size_dev);
  cudaDeviceSynchronize();
  cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    cerr << cudaGetErrorString(cuda_status) << endl;
    cerr << "ShrinkGraph kernel failed." << endl;
    exit(-1);
  }
  auto end = chrono::steady_clock::now();
  if (VERBOSE) {
    cerr << "Prepare kernel costs: "
         << (float)chrono::duration_cast<std::chrono::microseconds>(end - start)
                    .count() /
                1e6
         << endl;
  }
}

void ToDevKNNGraph(NNDElement *dev_graph, vector<vector<NNDElement>> host_graph,
                   const int k) {
  NNDElement *host_graph_tmp = new NNDElement[host_graph.size() * k];
  for (int i = 0; i < host_graph.size(); i++) {
    memcpy(&host_graph_tmp[i * k], host_graph[i].data(),
           (size_t)k * sizeof(NNDElement));
  }
  cudaMemcpy(dev_graph, host_graph_tmp,
             (size_t)host_graph.size() * k * sizeof(NNDElement),
             cudaMemcpyHostToDevice);
  auto cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    cerr << cudaGetErrorString(cuda_status) << endl;
    cerr << "ToDevKNNGraph failed." << endl;
    exit(-1);
  }
  delete[] host_graph_tmp;
}

void GetTestGraph(Graph *graph_new_ptr, Graph *graph_old_ptr,
                  int *graph_new_dev, int *newg_list_size_dev,
                  int *graph_old_dev, int *oldg_list_size_dev,
                  const int graph_size) {
  Graph &g_new = *graph_new_ptr;
  g_new.clear();
  g_new.resize(graph_size);
  Graph &g_old = *graph_old_ptr;
  g_old.clear();
  g_old.resize(graph_size);
  int *host_graph = new int[graph_size];
  int *newg_list_size = new int[graph_size];
  int *oldg_list_size = new int[graph_size];
  cudaMemcpy(newg_list_size, newg_list_size_dev,
             (size_t)graph_size * sizeof(int), cudaMemcpyDeviceToHost);
  auto cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    cerr << cudaGetErrorString(cuda_status) << endl;
    cerr << "1. Get test graph failed." << endl;
    exit(-1);
  }
  cudaMemcpy(oldg_list_size, oldg_list_size_dev,
             (size_t)graph_size * sizeof(int), cudaMemcpyDeviceToHost);
  cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    cerr << cudaGetErrorString(cuda_status) << endl;
    cerr << "2. Get test graph failed." << endl;
    exit(-1);
  }
  int *graph_new = new int[graph_size * (SAMPLE_NUM * 2)];
  int *graph_old = new int[graph_size * (SAMPLE_NUM * 2)];
  cudaMemcpy(graph_new, graph_new_dev,
             (size_t)graph_size * (SAMPLE_NUM * 2) * sizeof(int),
             cudaMemcpyDeviceToHost);
  cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    cerr << cudaGetErrorString(cuda_status) << endl;
    cerr << "3. Get test graph failed." << endl;
    exit(-1);
  }
  cudaMemcpy(graph_old, graph_old_dev,
             (size_t)graph_size * (SAMPLE_NUM * 2) * sizeof(int),
             cudaMemcpyDeviceToHost);
  cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    cerr << cudaGetErrorString(cuda_status) << endl;
    cerr << "4. Get test graph failed." << endl;
    exit(-1);
  }
  for (int i = 0; i < graph_size; i++) {
    int list_base_pos = i * (SAMPLE_NUM * 2);
    for (int j = 0; j < newg_list_size[i]; j++) {
      g_new[i].push_back(graph_new[list_base_pos + j]);
    }
  }
  for (int i = 0; i < graph_size; i++) {
    int list_base_pos = i * (SAMPLE_NUM * 2);
    for (int j = 0; j < oldg_list_size[i]; j++) {
      g_old[i].push_back(graph_old[list_base_pos + j]);
    }
  }
  // for (int i = 0; i < graph_size; i++) {
  //   sort(g_new[i].begin(), g_new[i].end());
  //   g_new[i].erase(unique(g_new[i].begin(), g_new[i].end()), g_new[i].end());

  //   sort(g_old[i].begin(), g_old[i].end());
  //   g_old[i].erase(unique(g_old[i].begin(), g_old[i].end()), g_old[i].end());
  // }
  cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    cerr << cudaGetErrorString(cuda_status) << endl;
    cerr << "Get test graph failed." << endl;
    exit(-1);
  }
  delete[] newg_list_size;
  delete[] oldg_list_size;
  delete[] graph_new;
  delete[] graph_old;
  delete[] host_graph;
}

__device__ __forceinline__ NNDElement
__shfl_down_sync(const int mask, NNDElement var, const int delta,
                 const int width = WARP_SIZE) {
  NNDElement res;
  res.distance_ = __shfl_down_sync(mask, var.distance_, delta, width);
  res.label_ = __shfl_down_sync(mask, var.label_, delta, width);
  return res;
}

__device__ __forceinline__ NNDElement
__shfl_up_sync(const int mask, NNDElement var, const int delta,
               const int width = WARP_SIZE) {
  NNDElement res;
  res.distance_ = __shfl_up_sync(mask, var.distance_, delta, width);
  res.label_ = __shfl_up_sync(mask, var.label_, delta, width);
  return res;
}

template <typename T>
__device__ __forceinline__ T Min(const T &a, const T &b) {
  return a < b ? a : b;
}

__device__ NNDElement GetMinElement(const int *neighbs_id, const int list_id,
                                    const int list_size, const float *distances,
                                    const int distances_num) {
  int head_pos = list_id * (list_id - 1) / 2;
  int tail_pos = (list_id + 1) * list_id / 2;
  int y_num = tail_pos - head_pos;

  int tx = threadIdx.x;
  int lane_id = tx % WARP_SIZE;
  NNDElement min_elem = NNDElement(1e10, LARGE_INT);

  int it_num = GetItNum(y_num, WARP_SIZE);
  for (int it = 0; it < it_num; it++) {
    NNDElement elem;
    elem.SetLabel(neighbs_id[it * WARP_SIZE + lane_id]);
    int current_pos = head_pos + it * WARP_SIZE + lane_id;
    if (current_pos < tail_pos) {
      elem.SetDistance(distances[current_pos]);
    } else {
      elem = NNDElement(1e10, LARGE_INT);
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
      elem = Min(elem, __shfl_down_sync(FULL_MASK, elem, offset));
    if (lane_id == 0) {
      min_elem = Min(elem, min_elem);
    }
  }

  head_pos = list_id * (list_id + 3) / 2;  // 0   2   5   9   14
  for (int it = 0; it < 2; it++) {
    NNDElement elem;
    int no = it * WARP_SIZE + lane_id;
    elem.SetLabel(neighbs_id[no + list_id + 1]);
    int current_pos = head_pos + no * (no + list_id * 2 + 1) / 2;
    if (current_pos < distances_num) {
      elem.SetDistance(distances[current_pos]);
    } else {
      elem = NNDElement(1e10, LARGE_INT);
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
      elem = Min(elem, __shfl_down_sync(FULL_MASK, elem, offset));
    if (lane_id == 0) {
      min_elem = Min(elem, min_elem);
    }
  }
  return min_elem;
}

__device__ NNDElement GetMinElement2(const int list_id, const int list_size,
                                     const int *old_neighbs, const int num_old,
                                     const float *distances,
                                     const int distances_num) {
  int head_pos = list_id * num_old;
  int y_num = num_old;
  int tail_pos = head_pos + num_old;

  int tx = threadIdx.x;
  int lane_id = tx % WARP_SIZE;
  NNDElement min_elem = NNDElement(1e10, LARGE_INT);

  int it_num = GetItNum(y_num, WARP_SIZE);
  for (int it = 0; it < it_num; it++) {
    NNDElement elem;
    int no = it * WARP_SIZE + lane_id;
    elem.SetLabel(old_neighbs[no]);
    int current_pos = head_pos + no;
    if (current_pos < tail_pos) {
      elem.SetDistance(distances[current_pos]);
    } else {
      elem = NNDElement(1e10, LARGE_INT);
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
      elem = Min(elem, __shfl_down_sync(FULL_MASK, elem, offset));
    if (lane_id == 0) {
      min_elem = Min(elem, min_elem);
    }
  }
  return min_elem;
}

__device__ NNDElement GetMinElement3(const int list_id, const int list_size,
                                     const int *new_neighbs, const int num_new,
                                     const int *old_neighbs, const int num_old,
                                     const float *distances,
                                     const int distances_num,
                                     const float *vectors) {
  int head_pos = list_id - num_new;
  int tx = threadIdx.x;
  int lane_id = tx % WARP_SIZE;
  NNDElement min_elem = NNDElement(1e10, LARGE_INT);

  int it_num = GetItNum(num_new, WARP_SIZE);
  for (int it = 0; it < it_num; it++) {
    NNDElement elem;
    int no = it * WARP_SIZE + lane_id;
    elem.SetLabel(new_neighbs[no]);
    int current_pos = head_pos + no * num_old;
    if (current_pos < distances_num) {
      elem.SetDistance(distances[current_pos]);
    } else {
      elem = NNDElement(1e10, LARGE_INT);
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
      elem = Min(elem, __shfl_down_sync(FULL_MASK, elem, offset));
    if (lane_id == 0) {
      min_elem = Min(elem, min_elem);
    }
  }
  return min_elem;
}

__device__ uint GetNthSetBitPos(uint mask, int nth) {
  uint res;
  asm("fns.b32 %0,%1,%2,%3;" : "=r"(res) : "r"(mask), "r"(0), "r"(nth));
  return res;
}

__device__ void InsertToGlobalGraph(NNDElement elem, const int local_id,
                                    const int global_id,
                                    NNDElement *global_knn_graph,
                                    int *global_locks) {
  int tx = threadIdx.x;
  int lane_id = tx % WARP_SIZE;
  int global_pos_base = global_id * NEIGHB_NUM_PER_LIST;
  elem.distance_ = __shfl_sync(FULL_MASK, elem.distance_, 0);
  elem.label_ = __shfl_sync(FULL_MASK, elem.label_, 0);
  int loop_flag = 0;
  do {
    int block_id = elem.label() % NEIGHB_BLOCKS_NUM;
    if (lane_id == 0)
      loop_flag =
          atomicCAS(&global_locks[global_id * NEIGHB_BLOCKS_NUM + block_id], 0,
                    1) == 0;
    loop_flag = __shfl_sync(FULL_MASK, loop_flag, 0);
    if (loop_flag == 1) {
      NNDElement knn_list_frag;
      int local_pos = block_id * WARP_SIZE + lane_id;
      int global_pos = global_pos_base + local_pos;
      if (local_pos < NEIGHB_NUM_PER_LIST)
        knn_list_frag = global_knn_graph[global_pos];
      else
        knn_list_frag = NNDElement(1e10, LARGE_INT);

      int pos_to_insert = -1;
      NNDElement prev_elem = __shfl_up_sync(FULL_MASK, knn_list_frag, 1);
      if (lane_id == 0) prev_elem = NNDElement(-1e10, -LARGE_INT);
      if (elem > prev_elem && elem < knn_list_frag)
        pos_to_insert = block_id * WARP_SIZE + lane_id;
      else if (elem == prev_elem || elem == knn_list_frag)
        pos_to_insert = -2;
      uint mask = __ballot_sync(FULL_MASK, pos_to_insert >= 0);
      if (mask) {
        uint set_lane_id = GetNthSetBitPos(mask, 1);
        pos_to_insert = __shfl_sync(FULL_MASK, pos_to_insert, set_lane_id);
        // assert(false);
      }
      if (pos_to_insert >= 0) {
        int local_pos = block_id * WARP_SIZE + lane_id;
        if (local_pos > pos_to_insert) {
          local_pos++;
        } else if (local_pos == pos_to_insert) {
          global_knn_graph[global_pos_base + local_pos] = elem;
          local_pos++;
        }
        int global_pos = global_pos_base + local_pos;
        if (local_pos < (block_id + 1) * WARP_SIZE &&
            local_pos < NEIGHB_NUM_PER_LIST)
          global_knn_graph[global_pos] = knn_list_frag;
      }
    }
    __threadfence();
    if (loop_flag && lane_id == 0) {
      atomicExch(&global_locks[global_id * NEIGHB_BLOCKS_NUM + block_id], 0);
    }
    __nanosleep(32);
  } while (!loop_flag);
}

__global__ void NewNeighborsCompareKernel(
    NNDElement *knn_graph, int *global_locks, const float *vectors,
    const int *graph_new, const int *size_new, const int num_new_max) {
  extern __shared__ char buffer[];

  __shared__ float *shared_vectors, *distances;
  __shared__ int *neighbors;
  __shared__ int gnew_base_pos, num_new;

  int tx = threadIdx.x;
  if (tx == 0) {
    shared_vectors = (float *)buffer;
    size_t offset = num_new_max * SKEW_DIM * sizeof(float);
    distances = (float *)((char *)buffer + offset);
    neighbors = (int *)((char *)distances +
                        (num_new_max * (num_new_max - 1) / 2) * sizeof(float));
  }
  __syncthreads();

  int list_id = blockIdx.x;
  int block_dim_x = blockDim.x;

  if (tx == 0) {
    gnew_base_pos = list_id * (SAMPLE_NUM * 2);
  } else if (tx == 32) {
    num_new = size_new[list_id];
  }
  __syncthreads();
  int neighb_num = num_new;
  if (tx < neighb_num) {
    neighbors[tx] = graph_new[gnew_base_pos + tx];
  }
  __syncthreads();
  int num_vec_per_it = block_dim_x / VEC_DIM;
  int num_it = GetItNum(neighb_num, num_vec_per_it);
  for (int i = 0; i < num_it; i++) {
    int x = i * num_vec_per_it + tx / VEC_DIM;
    if (x >= neighb_num) continue;
    int y = tx % VEC_DIM;
    int vec_id = neighbors[x];
    shared_vectors[x * SKEW_DIM + y] = vectors[vec_id * VEC_DIM + y];
  }
  __syncthreads();

  int calc_num = (neighb_num * (neighb_num - 1)) / 2;

  num_it = GetItNum(calc_num, block_dim_x);
  for (int i = 0; i < num_it; i++) {
    int no = i * block_dim_x + tx;
    if (no >= calc_num) continue;
    int idx = no + 1;
    int x = ceil(sqrt(2 * idx + 0.25) - 0.5);
    if (x >= neighb_num) continue;
    int y = idx - (x - 1) * x / 2 - 1;
    if (y >= neighb_num) continue;
    float sum = 0;
    int base_x = x * SKEW_DIM;
    int base_y = y * SKEW_DIM;
    for (int j = 0; j < VEC_DIM; j++) {
      float diff = shared_vectors[base_x + j] - shared_vectors[base_y + j];
      sum += diff * diff;
    }
    distances[no] = sum;
  }
  __syncthreads();
  // num_it = GetItNum(NEIGHB_NUM_PER_LIST, NEIGHB_CACHE_NUM);

  int list_size = NEIGHB_CACHE_NUM;
  int num_it3 = GetItNum(neighb_num, block_dim_x / WARP_SIZE);
  for (int j = 0; j < num_it3; j++) {
    int list_id = j * (block_dim_x / WARP_SIZE) + tx / WARP_SIZE;
    if (list_id >= neighb_num) continue;
    NNDElement min_elem =
        GetMinElement(neighbors, list_id, list_size, distances, calc_num);
    InsertToGlobalGraph(min_elem, list_id, neighbors[list_id], knn_graph,
                        global_locks);
  }
}

// blockDim.x = TILE_WIDTH * TILE_WIDTH;
__device__ void GetNewOldDistancesTiled(float *distances, const float *vectors,
                                        const int *new_neighbors,
                                        const int num_new,
                                        const int *old_neighbors,
                                        const int num_old) {
  __shared__ float nsv[TILE_WIDTH][SKEW_TILE_WIDTH];  // New shared vectors
  __shared__ float osv[TILE_WIDTH][SKEW_TILE_WIDTH];  // Old shared vectors
  const int width = VEC_DIM;

  int tx = threadIdx.x;
  int t_row = tx / TILE_WIDTH;
  int t_col = tx % TILE_WIDTH;
  int row_num = (int)(ceil(1.0 * num_new / TILE_WIDTH));
  int col_num = (int)(ceil(1.0 * num_old / TILE_WIDTH));
  int tiles_num = row_num * col_num;
  // if (threadIdx.x == 0) {
  //     printf("%d %d %d\n", row_num, col_num, tiles_num);
  // }
  for (int i = 0; i < tiles_num; i++) {
    float distance = -1.0;
    int row_new = i / col_num * TILE_WIDTH;
    int row_old = i % col_num * TILE_WIDTH;
    // Assume that the dimension of vectors larger than num of neighbors.
    for (int ph = 0; ph < ceil(width / (float)TILE_WIDTH); ph++) {
      if ((row_new + t_row < num_new) && (ph * TILE_WIDTH + t_col < VEC_DIM)) {
        nsv[t_row][t_col] = vectors[new_neighbors[row_new + t_row] * VEC_DIM +
                                    ph * TILE_WIDTH + t_col];
      } else {
        nsv[t_row][t_col] = 1e10;
      }

      if ((row_old + t_col < num_old) && (ph * TILE_WIDTH + t_row < VEC_DIM)) {
        osv[t_col][t_row] = vectors[old_neighbors[row_old + t_col] * VEC_DIM +
                                    ph * TILE_WIDTH + t_row];
      } else {
        osv[t_col][t_row] = 1e10;
      }
      __syncthreads();

      for (int k = 0; k < TILE_WIDTH; k++) {
        float a = nsv[t_row][k], b = osv[t_col][k];
        if (a > 1e9 || b > 1e9) {
        } else {
          float diff = a - b;
          if (distance == -1.0) distance = 0;
          distance += diff * diff;
        }
      }
      __syncthreads();
    }
    if (distance != -1.0) {
      distances[(row_new + t_row) * num_old + row_old + t_col] = distance;
    }
  }
}

__global__ void TiledNewOldNeighborsCompareKernel(
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
    InsertToGlobalGraph(min_elem, list_id, neighbors[list_id], knn_graph,
                        global_locks);
  }
  __syncthreads();
}

__global__ void MarkAllToOld(NNDElement *knn_graph) {
  int list_id = blockIdx.x;
  int tx = threadIdx.x;
  int graph_base_pos = list_id * NEIGHB_NUM_PER_LIST;
  knn_graph[graph_base_pos + tx].MarkOld();
}

pair<int *, int *> ReadGraphToGlobalMemory(const Graph &graph) {
  int pos = 0;
  vector<int> edges, dest;
  for (int i = 0; i < graph.size(); i++) {
    edges.push_back(pos);
    // dest.push_back(graph[i].size());
    // pos++;
    for (int j = 0; j < graph[i].size(); j++) {
      dest.push_back(graph[i][j]);
      pos++;
    }
  }
  edges.push_back(pos);

  int *edges_dev, *dest_dev;
  cudaError_t cuda_status0, cuda_status1;
  cuda_status0 = cudaMalloc(&edges_dev, edges.size() * sizeof(int));
  cuda_status1 = cudaMalloc(&dest_dev, dest.size() * sizeof(int));
  if (cuda_status0 != cudaSuccess || cuda_status1 != cudaSuccess) {
    cerr << "CudaMalloc failed" << endl;
    exit(-1);
  }

  cuda_status0 = cudaMemcpy(edges_dev, edges.data(), edges.size() * sizeof(int),
                            cudaMemcpyHostToDevice);
  cuda_status1 = cudaMemcpy(dest_dev, dest.data(), dest.size() * sizeof(int),
                            cudaMemcpyHostToDevice);
  if (cuda_status0 != cudaSuccess || cuda_status1 != cudaSuccess) {
    cerr << "CudaMemcpy failed" << endl;
    exit(-1);
  }
  return make_pair(edges_dev, dest_dev);
}

__global__ void TestKernel(NNDElement *knn_graph) {
  for (int i = 0; i < 10000 * 30; i++) {
    if (knn_graph[i].distance() == 0 && knn_graph[i].label() == 0) {
      printf("check %d %f\n", i, knn_graph[i].distance());
    }
  }
  return;
}

NNDElement *ReadKNNGraphToGlobalMemory(
    const vector<vector<NNDElement>> &knn_graph) {
  int k = knn_graph[0].size();
  NNDElement *knn_graph_dev;
  NNDElement *knn_graph_host = new NNDElement[knn_graph.size() * k];
  int idx = 0;
  for (int i = 0; i < knn_graph.size(); i++) {
    for (int j = 0; j < k; j++) {
      const auto &item = knn_graph[i][j];
      knn_graph_host[idx++] = item;
    }
  }

  auto cuda_status =
      cudaMalloc(&knn_graph_dev, knn_graph.size() * k * sizeof(NNDElement));
  if (cuda_status != cudaSuccess) {
    cerr << "knn_graph cudaMalloc failed." << endl;
    exit(-1);
  }
  cuda_status = cudaMemcpy(knn_graph_dev, knn_graph_host,
                           knn_graph.size() * k * sizeof(NNDElement),
                           cudaMemcpyHostToDevice);
  if (cuda_status != cudaSuccess) {
    cerr << cudaGetErrorString(cuda_status) << endl;
    cerr << "knn_graph cudaMemcpyHostToDevice failed." << endl;
    exit(-1);
  }
  delete[] knn_graph_host;
  return knn_graph_dev;
}

int GetMaxListSize(const Graph &g) {
  int res = 0;
  for (const auto &list : g) {
    res = max((int)list.size(), res);
  }
  return res;
}

int GetMaxListSize(int *list_size_dev, const int g_size) {
  thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast(list_size_dev);
  return *thrust::max_element(dev_ptr, dev_ptr + g_size);
}

float UpdateGraph(NNDElement *origin_knn_graph_dev, const size_t g_size,
                  const float *vectors_dev, int *newg_dev,
                  int *newg_list_size_dev, int *oldg_dev,
                  int *oldg_list_size_dev, const int k,
                  const bool calc_between_new_neighbs = true) {
  float kernel_time = 0;
  cudaError_t cuda_status;

  int *global_locks_dev;
  cudaMalloc(&global_locks_dev, g_size * sizeof(int) * NEIGHB_BLOCKS_NUM);
  cudaMemset(global_locks_dev, 0, g_size * sizeof(int) * NEIGHB_BLOCKS_NUM);
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
  if (VERBOSE) {
    cerr << "Num new max: " << num_new_max << endl;
    cerr << "Num old max: " << num_old_max << endl;
  }
  size_t shared_memory_size =
      num_new_max * SKEW_DIM * sizeof(float) +
      (num_new_max * (num_new_max - 1) / 2) * sizeof(float) +
      num_new_max * sizeof(int);

  if (VERBOSE) {
    cerr << "Shmem kernel1 costs: " << shared_memory_size << endl;
  }

  auto start = chrono::steady_clock::now();
  MarkAllToOld<<<g_size, NEIGHB_NUM_PER_LIST>>>(origin_knn_graph_dev);
  if (calc_between_new_neighbs) {
    NewNeighborsCompareKernel<<<grid_size, block_size, shared_memory_size>>>(
        origin_knn_graph_dev, global_locks_dev, vectors_dev, newg_dev,
        newg_list_size_dev, num_new_max);
  }
  int neighb_num_max = num_new_max + num_old_max;
  block_size = dim3(TILE_WIDTH * TILE_WIDTH);
  shared_memory_size = (num_new_max * num_old_max) * sizeof(float) +
                       neighb_num_max * sizeof(int);
  if (VERBOSE) {
    cerr << "Shmem tiled kernel2 costs: " << shared_memory_size << endl;
  }
  TiledNewOldNeighborsCompareKernel<<<grid_size, block_size,
                                      shared_memory_size>>>(
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

void OutputGraph(const xmuknn::Graph &g, const string &path) {
  ofstream out(path);
  for (int i = 0; i < g.size(); i++) {
    out << g[i].size() << "\t";
    for (int j = 0; j < g[i].size(); j++) {
      out << g[i][j] << "\t";
    }
    out << endl;
  }
  out.close();
}

void OutputGraph(const vector<vector<NNDElement>> &g, const string &path) {
  ofstream out(path);
  for (int i = 0; i < g.size(); i++) {
    out << g[i].size() << "\t";
    for (int j = 0; j < g[i].size(); j++) {
      out << g[i][j].label() << "\t";
    }
    out << endl;
  }
  out.close();
}

__global__ void InitKNNGraphIndexKernel(
    NNDElement *knn_graph, const int graph_size,
    const unsigned long long *random_sequence = 0) {
  int list_id = blockIdx.x;
  int tx = threadIdx.x;
  int pos = list_id * NEIGHB_NUM_PER_LIST + tx;
  int block_id = tx / WARP_SIZE;

  knn_graph[pos].SetDistance(1e10);
  int label;
  if (random_sequence == 0) {
    label = xorshift64star(pos) % (unsigned long long)graph_size;
  } else {
    label = random_sequence[pos] % (unsigned long long)graph_size;
  }
  while (label % NEIGHB_BLOCKS_NUM != block_id || label == list_id) {
    label = (label + 1) % (unsigned long long)graph_size;
  }
  knn_graph[pos].SetLabel(label);
}

__global__ void InitKNNGraphDistanceKernel(NNDElement *knn_graph,
                                           const int graph_size,
                                           const float *vectors) {
  int list_id = blockIdx.x;
  int tx = threadIdx.x;
  int lane_id = threadIdx.x % WARP_SIZE;
  int vec_a_pos = list_id * VEC_DIM;

  for (int i = 0; i < NEIGHB_NUM_PER_LIST; i++) {
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

__global__ void SortEachBlockKernel(NNDElement *knn_graph,
                                    const int graph_size) {
  int list_id = blockIdx.x;
  int it_num = GetItNum(NEIGHB_NUM_PER_LIST, WARP_SIZE);
  int global_base_pos = list_id * NEIGHB_NUM_PER_LIST;
  int tx = threadIdx.x;
  int lane_id = tx % WARP_SIZE;
  int block_id = tx / WARP_SIZE;

  NNDElement elem;
  int pos = block_id * WARP_SIZE + lane_id;
  if (pos < NEIGHB_NUM_PER_LIST) {
    elem = knn_graph[global_base_pos + pos];
  } else {
    elem.SetDistance(1e10);
    elem.SetLabel(LARGE_INT);
  }
  BitonicSort(&elem, lane_id);
  if (pos < NEIGHB_NUM_PER_LIST) {
    knn_graph[global_base_pos + pos] = elem;
  }
  return;
}

__global__ void SortKNNGraphKernel(NNDElement *knn_graph,
                                   const int graph_size) {
  __shared__ NNDElement knn_list_cache[NEIGHB_NUM_PER_LIST];
  __shared__ NNDElement sorted_elements_cache[32];
  __shared__ NNDElement merged_list_cache[NEIGHB_NUM_PER_LIST];

  int list_id = blockIdx.x;
  int it_num = GetItNum(NEIGHB_NUM_PER_LIST, WARP_SIZE);
  int global_base_pos = list_id * NEIGHB_NUM_PER_LIST;
  int tx = threadIdx.x;
  int lane_id = tx % WARP_SIZE;
  int list_size = 0;
  for (int i = 0; i < it_num; i++) {
    int pos = i * WARP_SIZE + tx;
    if (pos >= NEIGHB_NUM_PER_LIST) break;
    merged_list_cache[pos] = knn_list_cache[pos] = NNDElement(1e10, LARGE_INT);
  }
  for (int i = 0; i < it_num; i++) {
    NNDElement elem;
    int pos = i * WARP_SIZE + tx;
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
                            WARP_SIZE, merged_list_cache);
    }
    list_size = __shfl_sync(FULL_MASK, list_size, 0);
    int copy_it_num = GetItNum(list_size, WARP_SIZE);
    for (int j = 0; j < copy_it_num; j++) {
      int pos = j * WARP_SIZE + lane_id;
      if (pos >= NEIGHB_NUM_PER_LIST) break;
      knn_list_cache[pos] = merged_list_cache[pos];
    }
  }
  __syncthreads();
  for (int i = 0; i < it_num; i++) {
    int pos = i * WARP_SIZE + tx;
    if (pos >= NEIGHB_NUM_PER_LIST) break;
    knn_graph[global_base_pos + pos] = knn_list_cache[pos];
  }
}

void InitRandomKNNGraph(NNDElement *knn_graph_dev, const int graph_size,
                        const float *vectors_dev,
                        bool start_from_random_index = true,
                        bool using_thrust_random = true) {
  auto start = chrono::steady_clock().now();
  if (start_from_random_index) {
    if (using_thrust_random) {
      thrust::device_vector<unsigned long long> dev_random_sequence(
          graph_size * NEIGHB_NUM_PER_LIST);
      DevRNGLongLong(thrust::raw_pointer_cast(dev_random_sequence.data()),
                     graph_size * NEIGHB_NUM_PER_LIST);
      InitKNNGraphIndexKernel<<<graph_size, NEIGHB_NUM_PER_LIST>>>(
          knn_graph_dev, graph_size,
          thrust::raw_pointer_cast(dev_random_sequence.data()));
      cudaDeviceSynchronize();
    } else {
      InitKNNGraphIndexKernel<<<graph_size, NEIGHB_NUM_PER_LIST>>>(
          knn_graph_dev, graph_size);
      cudaDeviceSynchronize();
      auto cuda_status = cudaGetLastError();
      if (cuda_status != cudaSuccess) {
        cerr << cudaGetErrorString(cuda_status) << endl;
        cerr << "InitKNNGraphIndexKernel failed." << endl;
        exit(-1);
      }
    }
  }

  // vector<vector<NNDElement>> g(graph_size);
  // #pragma omp parallel for
  // for (int i = 0; i < graph_size; i++) {
  //   vector<int> exclusion = {i};
  //   vector<int> tmp_vec;
  //   xmuknn::GenerateRandomSequence(tmp_vec, NEIGHB_NUM_PER_LIST, graph_size,
  //                                  exclusion);
  //   for (int j = 0; j < NEIGHB_NUM_PER_LIST; j++) {
  //     int nb_id = tmp_vec[j];
  //     g[i].emplace_back(1e10, nb_id);
  //   }
  // }
  // ToDevKNNGraph(knn_graph_dev, g, NEIGHB_NUM_PER_LIST);

  InitKNNGraphDistanceKernel<<<graph_size, 32>>>(knn_graph_dev, graph_size,
                                                 vectors_dev);
  cudaDeviceSynchronize();
  auto cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    cerr << cudaGetErrorString(cuda_status) << endl;
    cerr << "InitKNNGraphDistanceKernel failed." << endl;
    exit(-1);
  }
  SortEachBlockKernel<<<graph_size, WARP_SIZE * NEIGHB_BLOCKS_NUM>>>(
      knn_graph_dev, graph_size);
  // SortKNNGraphKernel<<<graph_size, 32>>>(knn_graph_dev, graph_size);
  cudaDeviceSynchronize();
  auto end = chrono::steady_clock().now();

  cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    cerr << cudaGetErrorString(cuda_status) << endl;
    cerr << "SortKNNGraphKernel failed." << endl;
    exit(-1);
  }
  if (VERBOSE) {
    cerr << "Initiate costs: "
         << (float)chrono::duration_cast<std::chrono::microseconds>(end - start)
                    .count() /
                1e6
         << endl;
  }
}

__global__ void MergeBlocksKernel(NNDElement *knn_graph, const int graph_size) {
  __shared__ NNDElement c_cache[NEIGHB_NUM_PER_LIST];
  __shared__ NNDElement tmp_cache[NEIGHB_NUM_PER_LIST];
  int list_id = blockIdx.x;
  int global_pos_base = list_id * NEIGHB_NUM_PER_LIST;
  int tx = threadIdx.x;
  int list_size = 0;
  c_cache[tx] = knn_graph[global_pos_base + tx];
  list_size = WARP_SIZE;
  for (int i = 1; i < NEIGHB_BLOCKS_NUM; i++) {
    list_size = MergeList(c_cache, list_size,
                          &knn_graph[global_pos_base + i * WARP_SIZE],
                          WARP_SIZE, tmp_cache);
    int it_num = GetItNum(list_size, WARP_SIZE);
    for (int j = 0; j < it_num; j++) {
      int pos = j * WARP_SIZE + tx;
      if (pos < list_size) {
        c_cache[pos] = tmp_cache[pos];
      }
    }
  }
  int it_num = GetItNum(NEIGHB_NUM_PER_LIST, WARP_SIZE);
  for (int i = 0; i < it_num; i++) {
    int pos = i * WARP_SIZE + tx;
    if (pos < NEIGHB_NUM_PER_LIST) {
      knn_graph[global_pos_base + pos] = c_cache[pos];
    }
  }
}

void MergeBlocksInNNLists(NNDElement *knn_graph, const int graph_size) {
  MergeBlocksKernel<<<graph_size, WARP_SIZE>>>(knn_graph, graph_size);
  cudaDeviceSynchronize();
}

namespace gpuknn {
void NNDescentForMerge(NNDElement *knngraph_dev, const float *vectors_dev,
                       const int vecs_size, const int vecs_dim,
                       const int split_pos, const int iteration) {
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
  InitRandomKNNGraph(knngraph_dev, graph_size, vectors_dev, false, false);
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
    if (VERBOSE) {
      cerr << "Start generating NBGraph." << endl;
    }
    // Should be removed after testing.
    auto start = chrono::steady_clock::now();
    PrepareForUpdateForMerge(graph_new_dev, newg_list_size_dev,
                             newg_revlist_size_dev, graph_old_dev,
                             oldg_list_size_dev, oldg_revlist_size_dev,
                             knngraph_dev, split_pos, graph_size);
    auto end = chrono::steady_clock::now();
    float tmp_time =
        (float)chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count() /
        1e6;
    get_nb_graph_time += tmp_time;
    if (VERBOSE) {
      cerr << "GetNBGraph costs " << tmp_time << endl;
    }
    start = chrono::steady_clock::now();
    float tmp_kernel_costs = UpdateGraph(
        knngraph_dev, graph_size, vectors_dev, graph_new_dev,
        newg_list_size_dev, graph_old_dev, oldg_list_size_dev, k, false);
    kernel_costs += tmp_kernel_costs;
    end = chrono::steady_clock::now();
    float it_tmp_costs =
        (float)chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count() /
        1e6;
    iteration_costs += it_tmp_costs;
    if (VERBOSE) {
      cerr << "Kernel costs " << tmp_kernel_costs << endl;
      cerr << endl;
    }
  }
  MergeBlocksInNNLists(knngraph_dev, graph_size);
  auto sum_end = chrono::steady_clock::now();
  float sum_costs = (float)chrono::duration_cast<std::chrono::microseconds>(
                        sum_end - sum_start)
                        .count() /
                    1e6;
  // sift10k in cpu should be 0.6s;
  if (VERBOSE) {
    cerr << "Compare times: " << cmp_times << endl;
    cerr << "FLOPS: " << cmp_times * 128 * 3 / kernel_costs / pow(1024.0, 3)
         << "G" << endl;
    cerr << "Kernel costs: " << kernel_costs << endl;
    cerr << "Update costs: " << iteration_costs << endl;
    cerr << "Get NB graph costs: " << get_nb_graph_time << endl;
    cerr << "All procedure costs: " << sum_costs << endl;
    cerr << endl;
  }
  cudaFree(graph_new_dev);
  cudaFree(graph_old_dev);

  cudaFree(newg_list_size_dev);
  cudaFree(oldg_list_size_dev);
  cudaFree(newg_revlist_size_dev);
  cudaFree(oldg_revlist_size_dev);
}
void NNDescent(NNDElement **knngraph_result_ptr, const float *vectors_dev,
               const int vecs_size, const int vecs_dim, const int iteration,
               const bool store_result_in_device) {
  int k = NEIGHB_NUM_PER_LIST;
  int *graph_new_dev, *newg_list_size_dev, *graph_old_dev, *oldg_list_size_dev;
  int *newg_revlist_size_dev, *oldg_revlist_size_dev;
  int graph_size = vecs_size;
  NNDElement *&knngraph_result = *knngraph_result_ptr;
  cudaMalloc(&graph_new_dev,
             (size_t)graph_size * (SAMPLE_NUM * 2) * sizeof(int));
  cudaMalloc(&newg_list_size_dev, (size_t)graph_size * sizeof(int));
  cudaMalloc(&newg_revlist_size_dev, (size_t)graph_size * sizeof(int));
  cudaMalloc(&graph_old_dev,
             (size_t)graph_size * (SAMPLE_NUM * 2) * sizeof(int));
  cudaMalloc(&oldg_list_size_dev, (size_t)graph_size * sizeof(int));
  cudaMalloc(&oldg_revlist_size_dev, (size_t)graph_size * sizeof(int));
  cudaMalloc(&knngraph_result, (size_t)graph_size * k * sizeof(NNDElement));
  Graph result(vecs_size);
  vector<vector<NNDElement>> g(vecs_size);
  // InitRandomKNNGraph(knngraph_result, graph_size, vectors_dev);
  InitRandomKNNGraph(knngraph_result, graph_size, vectors_dev, true, false);
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
    if (VERBOSE) {
      cerr << "Start generating NBGraph." << endl;
    }
    // Should be removed after testing.
    auto start = chrono::steady_clock::now();
    PrepareForUpdate(graph_new_dev, newg_list_size_dev, newg_revlist_size_dev,
                     graph_old_dev, oldg_list_size_dev, oldg_revlist_size_dev,
                     knngraph_result, graph_size);
    auto end = chrono::steady_clock::now();
    float tmp_time =
        (float)chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count() /
        1e6;
    get_nb_graph_time += tmp_time;
    if (VERBOSE) {
      cerr << "GetNBGraph costs " << tmp_time << endl;
    }
    start = chrono::steady_clock::now();
    float tmp_kernel_costs =
        UpdateGraph(knngraph_result, graph_size, vectors_dev, graph_new_dev,
                    newg_list_size_dev, graph_old_dev, oldg_list_size_dev, k);
    kernel_costs += tmp_kernel_costs;
    end = chrono::steady_clock::now();
    float it_tmp_costs =
        (float)chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count() /
        1e6;
    iteration_costs += it_tmp_costs;
    if (VERBOSE) {
      cerr << "Kernel costs " << tmp_kernel_costs << endl;
      cerr << endl;
    }
  }
  MergeBlocksInNNLists(knngraph_result, graph_size);
  auto sum_end = chrono::steady_clock::now();
  float sum_costs = (float)chrono::duration_cast<std::chrono::microseconds>(
                        sum_end - sum_start)
                        .count() /
                    1e6;
  // sift10k in cpu should be 0.6s;
  if (VERBOSE) {
    cerr << "Compare times: " << cmp_times << endl;
    cerr << "FLOPS: " << cmp_times * 128 * 3 / kernel_costs / pow(1024.0, 3)
         << "G" << endl;
    cerr << "Kernel costs: " << kernel_costs << endl;
    cerr << "Update costs: " << iteration_costs << endl;
    cerr << "Get NB graph costs: " << get_nb_graph_time << endl;
    cerr << "All procedure costs: " << sum_costs << endl;
    cerr << endl;
  }
  cudaFree(graph_new_dev);
  cudaFree(graph_old_dev);

  cudaFree(newg_list_size_dev);
  cudaFree(oldg_list_size_dev);
  cudaFree(newg_revlist_size_dev);
  cudaFree(oldg_revlist_size_dev);

  if (!store_result_in_device) {
    NNDElement *host_graph;
    ToHostKNNGraph(&host_graph, knngraph_result, graph_size,
                   NEIGHB_NUM_PER_LIST);
    cudaFree(knngraph_result);
    knngraph_result = host_graph;
  }
}
vector<vector<NNDElement>> NNDescent(const float *vectors, const int vecs_size,
                                     const int vecs_dim, const int iteration) {
  int k = NEIGHB_NUM_PER_LIST;
  auto cuda_status = cudaSetDevice(DEVICE_ID);

  float *vectors_dev;
  cudaMalloc(&vectors_dev, (size_t)vecs_size * vecs_dim * sizeof(float));
  cudaMemcpy(vectors_dev, vectors, (size_t)vecs_size * vecs_dim * sizeof(float),
             cudaMemcpyHostToDevice);
  NNDElement *knn_graph_dev;

  auto start = chrono::steady_clock::now();
  NNDescent(&knn_graph_dev, vectors_dev, vecs_size, vecs_dim, iteration);
  auto end = chrono::steady_clock::now();
  cerr << "GPU NNDescent costs: "
       << (float)chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count() /
              1e6
       << endl;
  vector<vector<NNDElement>> g;
  ToHostKNNGraph(&g, knn_graph_dev, vecs_size, k);  // 0.6 / 6.6
  cudaFree(vectors_dev);
  cudaFree(knn_graph_dev);
  return g;
}
}  // namespace gpuknn