#ifndef XMUKNN_NNDESCENT_CU
#define XMUKNN_NNDESCENT_CU

#include <assert.h>

#include <algorithm>
#include <bitset>
#include <chrono>
#include <cstring>
#include <iostream>
#include <fstream>
#include <mutex>
#include <tuple>
#include <utility>
#include <vector>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include "../tools/distfunc.hpp"
#include "../xmuknn.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "nndescent.cuh"
#include "nndescent_element.cuh"

#ifdef __INTELLISENSE__
#include "../intellisense_cuda_intrinsics.h"
#endif

using namespace std;
using namespace xmuknn;
#define DEVICE_ID 0
#define LARGE_INT 0x3f3f3f3f
#define FULL_MASK 0xffffffff
// #define DONT_TILE 0
// #define INSERT_MIN_ONLY 1
const int VEC_DIM = 128;
const int NEIGHB_NUM_PER_LIST = 64;
const int INSERT_IT_NUM =
    NEIGHB_NUM_PER_LIST / 32 + (NEIGHB_NUM_PER_LIST % 32 != 0);
#if INSERT_MIN_ONLY
const int NEIGHB_CACHE_NUM = 1;
#else
const int NEIGHB_CACHE_NUM = 10;
#endif
const int TILE_WIDTH = 16;
const int SKEW_TILE_WIDTH = TILE_WIDTH + 1;
const int SAMPLE_NUM = 30; // assert(SAMPLE_NUM * 2 <= NEIGHB_NUM_PER_LIST);
const int SKEW_DIM = VEC_DIM + 1;

#if DONT_TILE
const int MAX_SHMEM = 49152;
#endif

__device__ int for_check = 0;

void GetNBGraph(Graph *graph_new_ptr, Graph *graph_old_ptr,
                vector<vector<NNDElement>> &knn_graph, const float *vectors,
                const int vecs_size, const int vecs_dim) {
  auto time1 = chrono::steady_clock::now();
  int sample_num = SAMPLE_NUM;
  Graph &graph_new = *graph_new_ptr;
  Graph &graph_old = *graph_old_ptr;
  Graph graph_rnew, graph_rold;
  graph_new = graph_rnew = graph_old = graph_rold = Graph(knn_graph.size());
  vector<mutex> mtx(vecs_size);

#pragma omp parallel for
  for (int i = 0; i < knn_graph.size(); i++) {
    int cnt = 0;
    for (int j = 0; j < knn_graph[i].size(); j++) {
      auto &item = knn_graph[i][j];
      if (!item.IsNew()) {
        graph_old[i].push_back(item.label());
      } else {
        if (cnt < sample_num) {
          graph_new[i].push_back(item.label());
          cnt++;
          item.MarkOld();
        }
      }
      if (cnt >= sample_num) break;
    }
  }

  for (int i = 0; i < knn_graph.size(); i++) {
    if (graph_old[i].size() > sample_num)
      graph_old[i].erase(graph_old[i].begin() + sample_num, graph_old[i].end());
  }
  // auto time2 = chrono::steady_clock::now();
  // cerr << "Mark 2: " <<
  // (float)chrono::duration_cast<std::chrono::microseconds>(time2 -
  // time1).count() / 1e6 << endl;

#pragma omp parallel for
  for (int i = 0; i < knn_graph.size(); i++) {
    for (int j = 0; j < graph_new[i].size(); j++) {
      auto &id = graph_new[i][j];
      mtx[id].lock();
      graph_rnew[id].push_back(i);
      mtx[id].unlock();
    }
    for (int j = 0; j < graph_old[i].size(); j++) {
      auto &id = graph_old[i][j];
      mtx[id].lock();
      graph_rold[id].push_back(i);
      mtx[id].unlock();
    }
  }

  // auto time3 = chrono::steady_clock::now();
  // cerr << "Mark 3: " <<
  // (float)chrono::duration_cast<std::chrono::microseconds>(time3 -
  // time1).count() / 1e6 << endl;

  // #pragma omp parallel for
  // for (int i = 0; i < knn_graph.size(); i++) {
  //     random_shuffle(graph_rnew[i].begin(), graph_rnew[i].end());
  //     random_shuffle(graph_rold[i].begin(), graph_rold[i].end());
  // }

  // auto time4 = chrono::steady_clock::now();
  // cerr << "Mark 4: " <<
  // (float)chrono::duration_cast<std::chrono::microseconds>(time4 -
  // time1).count() / 1e6 << endl;

  vector<bool> visited(vecs_size);
  // #pragma omp parallel for
  for (int i = 0; i < knn_graph.size(); i++) {
    int cnt = 0;
    for (int j = 0; j < graph_new[i].size(); j++) {
      visited[graph_new[i][j]] = true;
    }
    for (int j = 0; j < graph_old[i].size(); j++) {
      visited[graph_old[i][j]] = true;
    }
    for (int j = 0; j < graph_rnew[i].size(); j++) {
      int x = graph_rnew[i][j];
      if (!visited[x]) {
        cnt++;
        visited[x] = true;
        graph_new[i].push_back(x);
        if (cnt >= sample_num) break;
      }
    }
    cnt = 0;
    for (int j = 0; j < graph_rold[i].size(); j++) {
      int x = graph_rold[i][j];
      if (!visited[x]) {
        cnt++;
        visited[x] = true;
        graph_old[i].push_back(x);
        if (cnt >= sample_num) break;
      }
    }
    for (int j = 0; j < graph_new[i].size(); j++) {
      visited[graph_new[i][j]] = false;
    }
    for (int j = 0; j < graph_old[i].size(); j++) {
      visited[graph_old[i][j]] = false;
    }
  }

  // auto time5 = chrono::steady_clock::now();
  // cerr << "Mark 5: " <<
  // (float)chrono::duration_cast<std::chrono::microseconds>(time5 -
  // time1).count() / 1e6 << endl;

  // #pragma omp parallel for
  for (int i = 0; i < knn_graph.size(); i++) {
    sort(graph_new[i].begin(), graph_new[i].end());
    graph_new[i].erase(unique(graph_new[i].begin(), graph_new[i].end()),
                       graph_new[i].end());

    sort(graph_old[i].begin(), graph_old[i].end());
    graph_old[i].erase(unique(graph_old[i].begin(), graph_old[i].end()),
                       graph_old[i].end());
  }
  // auto time6 = chrono::steady_clock::now();
  // cerr << "Mark 6: " <<
  // (float)chrono::duration_cast<std::chrono::microseconds>(time6 -
  // time1).count() / 1e6 << endl;
  return;
}

__device__ __forceinline__ int GetItNum(const int sum_num,
                                        const int num_per_it) {
  return sum_num / num_per_it + (sum_num % num_per_it != 0);
}

__global__ void PrepareGraph(int *graph_new_dev, int *newg_list_size_dev,
                             int *graph_old_dev, int *oldg_list_size_dev,
                             NNDElement *knn_graph, int graph_size) {
  __shared__ int new_elements_cache[NEIGHB_NUM_PER_LIST];
  __shared__ int cache1_size;
  __shared__ int old_elements_cache[NEIGHB_NUM_PER_LIST];
  __shared__ int cache2_size;
  int list_id = blockIdx.x;
  int knng_base_pos = list_id * NEIGHB_NUM_PER_LIST;
  int res_g_base_pos = list_id * (SAMPLE_NUM * 2);
  int tx = threadIdx.x;
  if (tx == 0) {
    cache1_size = cache2_size = 0;
  }
  __syncthreads();
  int it_num = GetItNum(NEIGHB_NUM_PER_LIST, warpSize);
  for (int i = 0; i < it_num; i++) {
    int local_pos = i * warpSize + tx;
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
  it_num = GetItNum(SAMPLE_NUM, warpSize);
  for (int i = 0; i < it_num; i++) {
    int local_pos = i * warpSize + tx;
    if (local_pos < cache1_size) {
      graph_new_dev[res_g_base_pos + local_pos] = new_elements_cache[local_pos];
    }
    if (local_pos < cache2_size) {
      graph_old_dev[res_g_base_pos + local_pos] = old_elements_cache[local_pos];
    }
  }
  __syncthreads();
}

__device__ void Swap(int &a, int &b) {
  int c = a;
  a = b;
  b = c;
}

__device__ void InsertSort(int *a, const int length) {
  for (int i = 1; i < length; i++) {
    for (int j = i - 1; j >= 0 && a[j + 1] < a[j]; j--) {
      Swap(a[j], a[j + 1]);
    }
  }
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
  int res_g_base_pos = list_id * (SAMPLE_NUM * 2);
  if (tx == 0) {
    cache1_size = newg_list_size_dev[list_id];
    cache2_size = oldg_list_size_dev[list_id];
  }
  __syncthreads();
  int it_num = GetItNum(SAMPLE_NUM, warpSize);
  for (int i = 0; i < it_num; i++) {
    int local_pos = i * warpSize + tx;
    if (local_pos < cache1_size) {
      new_elements_cache[local_pos] = 
        graph_new_dev[res_g_base_pos + local_pos];
    }
    if (local_pos < cache2_size) {
      old_elements_cache[local_pos] = 
        graph_old_dev[res_g_base_pos + local_pos];
    }
  }
  __syncthreads();
  it_num = GetItNum(SAMPLE_NUM, warpSize);
  for (int i = 0; i < it_num; i++) {
    int local_pos = i * warpSize + tx;
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

__global__ void ShrinkGraph(int *graph_new_dev, int *newg_list_size_dev,
                            int *newg_revlist_size_dev, int *graph_old_dev,
                            int *oldg_list_size_dev,
                            int *oldg_revlist_size_dev) {
  __shared__ int new_elements_cache[SAMPLE_NUM * 2];
  __shared__ int newg_list_size, newg_revlist_size;
  __shared__ int old_elements_cache[SAMPLE_NUM * 2];
  __shared__ int oldg_list_size, oldg_revlist_size;
  int tx = threadIdx.x;
  int list_id = blockIdx.x;
  int res_g_base_pos = list_id * (SAMPLE_NUM * 2);
  if (tx == 0) {
    newg_list_size = newg_list_size_dev[list_id];
    oldg_list_size = oldg_list_size_dev[list_id];
    newg_revlist_size = newg_revlist_size_dev[list_id];
    oldg_revlist_size = oldg_revlist_size_dev[list_id];
  }
  __syncthreads();
  int it_num = GetItNum(SAMPLE_NUM * 2, warpSize);
  for (int i = 0; i < it_num; i++) {
    int local_pos = i * warpSize + tx;
    if (local_pos < SAMPLE_NUM * 2) {
      new_elements_cache[local_pos] = 
        graph_new_dev[res_g_base_pos + local_pos];
    }
    if (local_pos < SAMPLE_NUM * 2) {
      old_elements_cache[local_pos] = 
        graph_old_dev[res_g_base_pos + local_pos];
    }
  }
  __syncthreads();  

  if (tx == 0) {
    for (int i = newg_list_size; i < SAMPLE_NUM; i++) {
      new_elements_cache[i] = LARGE_INT;
    }
    for (int i = oldg_list_size; i < SAMPLE_NUM; i++) {
      old_elements_cache[i] = LARGE_INT;
    }
    for (int i = SAMPLE_NUM + newg_revlist_size; i < SAMPLE_NUM * 2; i++) {
      new_elements_cache[i] = LARGE_INT;
    }
    for (int i = SAMPLE_NUM + oldg_revlist_size; i < SAMPLE_NUM * 2; i++) {
      old_elements_cache[i] = LARGE_INT;
    }
    // Dont need LARGE_INT
    InsertSort(new_elements_cache, SAMPLE_NUM * 2);
    InsertSort(old_elements_cache, SAMPLE_NUM * 2);
    newg_list_size = RemoveDuplicates(new_elements_cache, SAMPLE_NUM * 2);
    newg_list_size -= (new_elements_cache[newg_list_size - 1] == LARGE_INT);
    oldg_list_size = RemoveDuplicates(old_elements_cache, SAMPLE_NUM * 2);
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
    // Really slow!!! But easy.
  }
  __syncthreads();
  it_num = GetItNum(SAMPLE_NUM * 2, warpSize);
  for (int i = 0; i < it_num; i++) {
    int local_pos = i * warpSize + tx;
    if (local_pos < newg_list_size) {
      graph_new_dev[res_g_base_pos + local_pos] =
        new_elements_cache[local_pos];
    }
    if (local_pos < oldg_list_size) {
      graph_old_dev[res_g_base_pos + local_pos] = 
        old_elements_cache[local_pos];
    }
  }

  newg_list_size_dev[list_id] = newg_list_size;
  oldg_list_size_dev[list_id] = oldg_list_size;
}

void PrepareForUpdate(int *graph_new_dev, int *newg_list_size_dev,
                      int *newg_revlist_size_dev,
                      int *graph_old_dev, int *oldg_list_size_dev,
                      int *oldg_revlist_size_dev,
                      NNDElement *knn_graph_dev, int graph_size) {
  auto start = chrono::steady_clock::now();
  cudaMemset(newg_list_size_dev, 0, graph_size * sizeof(int));
  cudaMemset(oldg_list_size_dev, 0, graph_size * sizeof(int));
  cudaMemset(newg_revlist_size_dev, 0, graph_size * sizeof(int));
  cudaMemset(oldg_revlist_size_dev, 0, graph_size * sizeof(int));
  dim3 grid_size(graph_size);
  dim3 block_size(32);
  PrepareGraph<<<grid_size, block_size>>>(
      graph_new_dev, newg_list_size_dev, graph_old_dev, oldg_list_size_dev,
      knn_graph_dev, graph_size);
  cudaDeviceSynchronize();
  PrepareReverseGraph<<<grid_size, block_size>>>(
      graph_new_dev, newg_list_size_dev, newg_revlist_size_dev,
      graph_old_dev, oldg_list_size_dev, oldg_revlist_size_dev);
  cudaDeviceSynchronize();
  ShrinkGraph<<<grid_size, block_size>>>(
      graph_new_dev, newg_list_size_dev, newg_revlist_size_dev,
      graph_old_dev, oldg_list_size_dev, oldg_revlist_size_dev);
  cudaDeviceSynchronize();
  auto end = chrono::steady_clock::now();
  cerr << "Prepare kernel costs: "
       << (float)chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() / 1e6
       << endl;
  auto cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    cerr << cudaGetErrorString(cuda_status) << endl;
    cerr << "Prepare kernel failed." << endl;
    exit(-1);
  }
}

void ToDevKNNGraph(NNDElement *dev_graph,
                   vector<vector<NNDElement>> host_graph,
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
  g_new.clear(); g_new.resize(graph_size);
  Graph &g_old = *graph_old_ptr;
  g_old.clear(); g_old.resize(graph_size);
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
                 const int width = warpSize) {
  NNDElement res;
  res.distance_ = __shfl_down_sync(mask, var.distance_, delta, width);
  res.label_ = __shfl_down_sync(mask, var.label_, delta, width);
  return res;
}

__device__ __forceinline__ NNDElement
__shfl_up_sync(const int mask, NNDElement var, const int delta,
               const int width = warpSize) {
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
  int lane_id = tx % warpSize;
  NNDElement min_elem = NNDElement(1e10, LARGE_INT);

  int it_num = GetItNum(y_num, warpSize);
  for (int it = 0; it < it_num; it++) {
    NNDElement elem;
    elem.SetLabel(neighbs_id[it * warpSize + lane_id]);
    int current_pos = head_pos + it * warpSize + lane_id;
    if (current_pos < tail_pos) {
      elem.SetDistance(distances[current_pos]);
    } else {
      elem = NNDElement(1e10, LARGE_INT);
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
      elem = Min(elem, __shfl_down_sync(FULL_MASK, elem, offset));
    if (lane_id == 0) {
      min_elem = Min(elem, min_elem);
    }
  }

  head_pos = list_id * (list_id + 3) / 2;  // 0   2   5   9   14
  for (int it = 0; it < 2; it++) {
    NNDElement elem;
    int no = it * warpSize + lane_id;
    elem.SetLabel(neighbs_id[no + list_id + 1]);
    int current_pos = head_pos + no * (no + list_id * 2 + 1) / 2;
    if (current_pos < distances_num) {
      elem.SetDistance(distances[current_pos]);
    } else {
      elem = NNDElement(1e10, LARGE_INT);
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
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
  int lane_id = tx % warpSize;
  NNDElement min_elem = NNDElement(1e10, LARGE_INT);

  int it_num = GetItNum(y_num, warpSize);
  for (int it = 0; it < it_num; it++) {
    NNDElement elem;
    int no = it * warpSize + lane_id;
    elem.SetLabel(old_neighbs[no]);
    int current_pos = head_pos + no;
    if (current_pos < tail_pos) {
      elem.SetDistance(distances[current_pos]);
    } else {
      elem = NNDElement(1e10, LARGE_INT);
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
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
  int lane_id = tx % warpSize;
  NNDElement min_elem = NNDElement(1e10, LARGE_INT);

  int it_num = GetItNum(num_new, warpSize);
  for (int it = 0; it < it_num; it++) {
    NNDElement elem;
    int no = it * warpSize + lane_id;
    elem.SetLabel(new_neighbs[no]);
    int current_pos = head_pos + no * num_old;
    if (current_pos < distances_num) {
      elem.SetDistance(distances[current_pos]);
    } else {
      elem = NNDElement(1e10, LARGE_INT);
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
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
  int lane_id = tx % warpSize;
  int global_pos_base = global_id * NEIGHB_NUM_PER_LIST;
  elem.distance_ = __shfl_sync(FULL_MASK, elem.distance_, 0);
  elem.label_ = __shfl_sync(FULL_MASK, elem.label_, 0);
  int loop_flag = 0;
  do {
    if (lane_id == 0)
      loop_flag = atomicCAS(&global_locks[global_id], 0, 1) == 0;
    loop_flag = __shfl_sync(FULL_MASK, loop_flag, 0);
    if (loop_flag == 1) {
      NNDElement knn_list_frag[INSERT_IT_NUM];
      for (int i = 0; i < INSERT_IT_NUM; i++) {
        int local_pos = i * warpSize + lane_id;
        int global_pos = global_pos_base + local_pos;
        if (local_pos < NEIGHB_NUM_PER_LIST)
          knn_list_frag[i] = global_knn_graph[global_pos];
        else
          knn_list_frag[i] = NNDElement(1e10, LARGE_INT);
      }

      // int flag;
      // if (lane_id == 0)
      //     flag = atomicCAS(&for_check, 0, 1);
      // flag = __shfl_sync(FULL_MASK, flag, 0);
      // if (!flag) {
      //     printf("%d %f %d %f %d\n", lane_id, knn_list_frag[0].distance,
      //            knn_list_frag[0].label, knn_list_frag[1].distance,
      //            knn_list_frag[1].label);
      // }

      int pos_to_insert = -1;
      for (int i = 0; i < INSERT_IT_NUM; i++) {
        NNDElement prev_elem = __shfl_up_sync(FULL_MASK, knn_list_frag[i], 1);
        if (lane_id == 0) prev_elem = NNDElement(-1e10, -LARGE_INT);
        if (elem > prev_elem && elem < knn_list_frag[i])
          pos_to_insert = i * warpSize + lane_id;
        else if (elem == prev_elem || elem == knn_list_frag[i])
          pos_to_insert = -2;

        // if (global_id == 0) {
        //     printf("%d %f %d %f %d %f %d %d\n", lane_id,
        //         prev_elem.distance, prev_elem.label,
        //         elem.distance, elem.label,
        //         knn_list_frag[i].distance, knn_list_frag[i].label,
        //         pos_to_insert);
        // }
        if (__ballot_sync(FULL_MASK, pos_to_insert == -2)) break;
        uint mask = __ballot_sync(FULL_MASK, pos_to_insert >= 0);
        // if (__popc(mask) != 1 && mask) {
        //     int flag;
        //     if (lane_id == 0)
        //         flag = atomicCAS(&for_check, 0, 1);
        //     flag = __shfl_sync(FULL_MASK, flag, 0);
        //     if (!flag) {
        //         if (lane_id == 0) printf("%d\n", global_id);
        //         printf("%d %f %d %f %d\n", lane_id,
        //         knn_list_frag[0].distance,
        //                knn_list_frag[0].label, knn_list_frag[1].distance,
        //                knn_list_frag[1].label);
        //         printf("%d %f %d %f %d %f %d %d\n", lane_id,
        //                 prev_elem.distance, prev_elem.label,
        //                 elem.distance, elem.label,
        //                 knn_list_frag[i].distance, knn_list_frag[i].label,
        //                 pos_to_insert);
        //     }
        // }
        if (mask) {
          uint set_lane_id = GetNthSetBitPos(mask, 1);
          pos_to_insert = __shfl_sync(FULL_MASK, pos_to_insert, set_lane_id);
          // assert(false);
          break;
        }
      }

      // int tmp_pos_to_insert = pos_to_insert;
      // if (lane_id == 0) {
      //     if (elem < global_knn_graph[global_pos_base]) {
      //         pos_to_insert = 0;
      //     }
      //     else if (elem >= global_knn_graph[global_pos_base +
      //                                       NEIGHB_NUM_PER_LIST - 1]) {
      //     }
      //     else {
      //         for (int i = 1; i < NEIGHB_NUM_PER_LIST; i++) {
      //             auto prev_elem = global_knn_graph[global_pos_base + i - 1];
      //             auto next_elem = global_knn_graph[global_pos_base + i];
      //             if (elem > prev_elem && elem < next_elem) {
      //                 pos_to_insert = i;
      //             } else if (elem == prev_elem || elem == next_elem) {
      //                 break;
      //             }
      //         }
      //     }
      //     if (pos_to_insert != tmp_pos_to_insert) {
      //         int flag = atomicCAS(&for_check, 0, 1);
      //         if (!flag) {
      //             printf("%d %d\n", pos_to_insert, tmp_pos_to_insert);
      //             printf("%d %f %d\n", global_id, elem.distance, elem.label);
      //             for (int i = 0; i < NEIGHB_NUM_PER_LIST; i++) {
      //                 printf("%d %f %d\n", i,
      //                        global_knn_graph[global_pos_base + i].distance,
      //                        global_knn_graph[global_pos_base + i].label);
      //             }
      //         }
      //     }
      // }
      // pos_to_insert = __shfl_sync(FULL_MASK, pos_to_insert, 0);

      // if (lane_id == 0 && pos_to_insert >= 0) {
      //     for (int i = NEIGHB_NUM_PER_LIST - 1; i > pos_to_insert; i--) {
      //         global_knn_graph[global_pos_base + i]
      //             = global_knn_graph[global_pos_base + i - 1];
      //     }
      //     if (pos_to_insert < NEIGHB_NUM_PER_LIST)
      //         global_knn_graph[global_pos_base + pos_to_insert] = elem;
      // }
      if (pos_to_insert >= 0) {
        for (int i = 0; i < INSERT_IT_NUM; i++) {
          int local_pos = i * warpSize + lane_id;
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
  int num_it3 = GetItNum(neighb_num, block_dim_x / warpSize);
  for (int j = 0; j < num_it3; j++) {
    int list_id = j * (block_dim_x / warpSize) + tx / warpSize;
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
  int num_it3 = GetItNum(neighb_num, block_dim_x / warpSize);
  for (int j = 0; j < num_it3; j++) {
    int list_id = j * (block_dim_x / warpSize) + tx / warpSize;
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

void ToHostKNNGraph(vector<vector<NNDElement>> *origin_knn_graph_ptr,
                    const NNDElement *knn_graph_dev, const int size,
                    const int neighb_num) {
  NNDElement *knn_graph = new NNDElement[size * neighb_num];
  cudaMemcpy(knn_graph, knn_graph_dev, (size_t)size * neighb_num * sizeof(NNDElement),
             cudaMemcpyDeviceToHost);
  auto &origin_knn_graph = *origin_knn_graph_ptr;
  vector<NNDElement> neighb_list;
  for (int i = 0; i < size; i++) {
    neighb_list.clear();
    for (int j = 0; j < neighb_num; j++) {
      neighb_list.push_back(knn_graph[i * neighb_num + j]);
    }
    origin_knn_graph[i] = neighb_list;
  }
  delete[] knn_graph;
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
                  float *vectors_dev, int *newg_dev, int *newg_list_size_dev,
                  int *oldg_dev, int *oldg_list_size_dev, const int k) {
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
  // NewNeighborsCompareKernel<<<grid_size, block_size, shared_memory_size>>>(
  //     knn_graph_dev, global_locks_dev, vectors_dev, edges_dev_new, dest_dev_new,
  //     num_new_max);
  NewNeighborsCompareKernel<<<grid_size, block_size, shared_memory_size>>>(
      origin_knn_graph_dev, global_locks_dev, vectors_dev, newg_dev,
      newg_list_size_dev, num_new_max);
  int neighb_num_max = num_new_max + num_old_max;
  block_size = dim3(TILE_WIDTH * TILE_WIDTH);
  shared_memory_size = (num_new_max * num_old_max) * sizeof(float) +
                       neighb_num_max * sizeof(int);
  cerr << "Shmem tiled kernel2 costs: " << shared_memory_size << endl;
  TiledNewOldNeighborsCompareKernel<<<grid_size, block_size,
                                      shared_memory_size>>>(
      origin_knn_graph_dev, global_locks_dev, vectors_dev, newg_dev,
      newg_list_size_dev, num_new_max, oldg_dev, oldg_list_size_dev,
      num_old_max);
  cudaDeviceSynchronize();
  block_size = dim3(NEIGHB_NUM_PER_LIST);
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

// float UpdateGraph(vector<vector<NNDElement>> *origin_knn_graph_ptr,
//                   float *vectors_dev, const Graph &newg, const Graph &oldg,
//                   const int k) {
//   float kernel_time = 0;
//   auto &origin_knn_graph = *origin_knn_graph_ptr;
//   int *edges_dev_new, *dest_dev_new;
//   tie(edges_dev_new, dest_dev_new) = ReadGraphToGlobalMemory(newg);
//   int *edges_dev_old, *dest_dev_old;
//   tie(edges_dev_old, dest_dev_old) = ReadGraphToGlobalMemory(oldg);
//   size_t g_size = newg.size();
//   cudaError_t cuda_status;
//   NNDElement *knn_graph_dev, *knn_graph = new NNDElement[g_size * k];
//   knn_graph_dev = ReadKNNGraphToGlobalMemory(origin_knn_graph);
//   int *global_locks_dev;
//   cudaMalloc(&global_locks_dev, g_size * sizeof(int));
//   vector<int> zeros(g_size);
//   cudaMemcpy(global_locks_dev, zeros.data(), g_size * sizeof(int),
//              cudaMemcpyHostToDevice);
//   cuda_status = cudaGetLastError();
//   if (cuda_status != cudaSuccess) {
//     cerr << cudaGetErrorString(cuda_status) << endl;
//     cerr << "Initiate failed" << endl;
//     exit(-1);
//   }
//   dim3 block_size(640);
//   dim3 grid_size(g_size);
//   // cerr << "Start kernel." << endl;
//   const int num_new_max = GetMaxListSize(newg);
//   const int num_old_max = GetMaxListSize(oldg);
//   cerr << "Num new max: " << num_new_max << endl;
//   cerr << "Num old max: " << num_old_max << endl;
//   size_t shared_memory_size =
//       num_new_max * SKEW_DIM * sizeof(float) +
//       (num_new_max * (num_new_max - 1) / 2) * sizeof(float) +
//       num_new_max * sizeof(int);
//   cerr << "Shmem kernel1 costs: " << shared_memory_size << endl;
//   auto start = chrono::steady_clock::now();
//   NewNeighborsCompareKernel<<<grid_size, block_size, shared_memory_size>>>(
//       knn_graph_dev, global_locks_dev, vectors_dev, edges_dev_new, dest_dev_new,
//       num_new_max);
//   int neighb_num_max = num_new_max + num_old_max;
//   block_size = dim3(TILE_WIDTH * TILE_WIDTH);
//   shared_memory_size = (num_new_max * num_old_max) * sizeof(float) +
//                        neighb_num_max * sizeof(int);
//   cerr << "Shmem tiled kernel2 costs: " << shared_memory_size << endl;
//   TiledNewOldNeighborsCompareKernel<<<grid_size, block_size,
//                                       shared_memory_size>>>(
//       knn_graph_dev, global_locks_dev, vectors_dev, edges_dev_new, dest_dev_new,
//       num_new_max, edges_dev_old, dest_dev_old, num_old_max);
//   cudaDeviceSynchronize();
//   auto end = chrono::steady_clock::now();
//   kernel_time =
//       (float)chrono::duration_cast<std::chrono::microseconds>(end - start)
//           .count() /
//       1e6;
//   cuda_status = cudaGetLastError();
//   if (cuda_status != cudaSuccess) {
//     cerr << cudaGetErrorString(cuda_status) << endl;
//     cerr << "Kernel failed" << endl;
//     exit(-1);
//   }
//   // cerr << "End kernel." << endl;
//   cuda_status =
//       cudaMemcpy(knn_graph, knn_graph_dev, g_size * k * sizeof(NNDElement),
//                  cudaMemcpyDeviceToHost);
//   if (cuda_status != cudaSuccess) {
//     cerr << cudaGetErrorString(cuda_status) << endl;
//     cerr << "knn_graph cudaMemcpy failed" << endl;
//     exit(-1);
//   }
//   ToHostGraph(&origin_knn_graph, knn_graph, g_size, k)
//   delete[] knn_graph;
//   cudaFree(edges_dev_new);
//   cudaFree(dest_dev_new);
//   cudaFree(edges_dev_old);
//   cudaFree(dest_dev_old);
//   cudaFree(knn_graph_dev);
//   return kernel_time;
// }

void OutputGraph(const xmuknn::Graph &g, const string &path) {
  ofstream out(path);
  for (int i = 0; i < g.size(); i++) {
    out << g[i].size() << "\t";
    for (int j = 0; j < g[i].size(); j++) {
      out << g[i][j] << "\t";
    } out << endl;
  }
  out.close();
}

void OutputGraph(const vector<vector<NNDElement>> &g, const string &path) {
  ofstream out(path);
  for (int i = 0; i < g.size(); i++) {
    out << g[i].size() << "\t";
    for (int j = 0; j < g[i].size(); j++) {
      out << g[i][j].label() << "\t";
    } out << endl;
  }
  out.close();
}

namespace gpuknn {
vector<vector<NNDElement>> NNDescent(const float *vectors, const int vecs_size,
                                     const int vecs_dim) {
  int k = NEIGHB_NUM_PER_LIST;
  int iteration = 6;
  auto cuda_status = cudaSetDevice(DEVICE_ID);

  float *vectors_dev;
  cudaMalloc(&vectors_dev, (size_t)vecs_size * vecs_dim * sizeof(float));
  cudaMemcpy(vectors_dev, vectors, (size_t)vecs_size * vecs_dim * sizeof(float),
             cudaMemcpyHostToDevice);
  int *graph_new_dev, *newg_list_size_dev, *graph_old_dev, *oldg_list_size_dev;
  int *newg_revlist_size_dev, *oldg_revlist_size_dev;
  NNDElement *knn_graph_dev;
  int graph_size = vecs_size;
  cudaMalloc(&graph_new_dev, (size_t)graph_size * (SAMPLE_NUM * 2) * sizeof(int));
  cudaMalloc(&newg_list_size_dev, (size_t)graph_size * sizeof(int));
  cudaMalloc(&newg_revlist_size_dev, (size_t)graph_size * sizeof(int));
  cudaMalloc(&graph_old_dev, (size_t)graph_size * (SAMPLE_NUM * 2) * sizeof(int));
  cudaMalloc(&oldg_list_size_dev, (size_t)graph_size * sizeof(int));
  cudaMalloc(&oldg_revlist_size_dev, (size_t)graph_size * sizeof(int));
  cudaMalloc(&knn_graph_dev, (size_t)graph_size * k * sizeof(NNDElement));

  cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    cerr << cudaGetErrorString(cuda_status) << endl;
    cerr << "Init failed" << endl;
    exit(-1);
  }
  Graph result(vecs_size);
  vector<vector<NNDElement>> g(vecs_size);
  vector<int> tmp_vec;

  for (int i = 0; i < vecs_size; i++) {
    vector<int> exclusion = {i};
    xmuknn::GenerateRandomSequence(tmp_vec, k, vecs_size, exclusion);
    for (int j = 0; j < k; j++) {
      int nb_id = tmp_vec[j];
      g[i].emplace_back(1e10, nb_id);
    }
  }

#pragma omp parallel for
  for (int i = 0; i < vecs_size; i++) {
    for (int j = 0; j < k; j++) {
      g[i][j].SetDistance(
          GetDistance(vectors + (size_t)i * vecs_dim,
                      vectors + (size_t)g[i][j].label() * vecs_dim, vecs_dim));
    }
  }

#pragma omp parallel for
  for (int i = 0; i < g.size(); i++) {
    sort(g[i].begin(), g[i].end());
  }

  float iteration_costs = 0;
  Graph newg, oldg;
  float get_nb_graph_time = 0;
  float kernel_costs = 0;
  ToDevKNNGraph(knn_graph_dev, g, NEIGHB_NUM_PER_LIST);
  auto sum_start = chrono::steady_clock::now();
  long long cmp_times = 0;
  for (int t = 0; t < iteration; t++) {
    cerr << "Start generating NBGraph." << endl;
    // Should be removed after testing.
    auto start = chrono::steady_clock::now();
    PrepareForUpdate(graph_new_dev, newg_list_size_dev, newg_revlist_size_dev, 
                     graph_old_dev, oldg_list_size_dev, oldg_revlist_size_dev,
                     knn_graph_dev, graph_size);
    // OutputGraph(newg, "/home/hwang/codes/GPU_KNNG/results/graph_a.txt");
    // OutputGraph(g, "/home/hwang/codes/GPU_KNNG/results/graph_origin.txt");
    // GetNBGraph(&newg, &oldg, g, vectors, vecs_size, vecs_dim);
    // OutputGraph(newg, "/home/hwang/codes/GPU_KNNG/results/graph_b.txt");
    auto end = chrono::steady_clock::now();
    float tmp_time =
        (float)chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count() /
        1e6;
    // get_nb_graph_time += tmp_time;
    // GetTestGraph(&newg, &oldg, graph_new_dev, newg_list_size_dev, graph_old_dev,
    //              oldg_list_size_dev, graph_size); // Unnecessary slow for testing.
    // for (int i = 0; i < newg.size(); i++) {
    //   cmp_times += (newg[i].size() - 1) * newg[i].size() / 2 +
    //                newg[i].size() * oldg[i].size();
    // }
    cerr << "GetNBGraph costs " << tmp_time << endl;

    start = chrono::steady_clock::now();
    // float tmp_kernel_costs = UpdateGraph(&g, vectors_dev, newg, oldg, k);
    float tmp_kernel_costs =
        UpdateGraph(knn_graph_dev, graph_size, vectors_dev, graph_new_dev,
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
  ToHostKNNGraph(&g, knn_graph_dev, graph_size, k);
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

  cudaFree(vectors_dev);
  cudaFree(graph_new_dev);
  cudaFree(newg_list_size_dev);
  cudaFree(graph_old_dev);
  cudaFree(oldg_list_size_dev);
  cudaFree(knn_graph_dev);
  cudaFree(newg_revlist_size_dev);
  cudaFree(oldg_revlist_size_dev);
  return g;
}
}  // namespace gpuknn

#endif