#ifndef XMUKNN_NNDESCENT_CU
#define XMUKNN_NNDESCENT_CU

#include <assert.h>

#include <algorithm>
#include <bitset>
#include <chrono>
#include <cstring>
#include <iostream>
#include <mutex>
#include <tuple>
#include <utility>
#include <vector>

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
#define DONT_TILE 0
#define INSERT_MIN_ONLY 1
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
const int SAMPLE_NUM = 30;
const int SKEW_DIM = VEC_DIM + 1;

#if DONT_TILE
const int MAX_SHMEM = 49152;
#endif

__device__ int for_check = 0;

void GetNBGraph(Graph *graph_new_ptr, Graph *graph_old_ptr,
                vector<vector<NNDElement>> &knn_graph,
                const float *vectors, const int vecs_size, const int vecs_dim) {
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

#if !INSERT_MIN_ONLY
__device__ __forceinline__ NNDElement XorSwap(NNDElement x, int mask, int dir) {
  NNDElement y;
  y.distance = __shfl_xor_sync(0xffffffff, x.distance, mask, warpSize);
  y.label = __shfl_xor_sync(0xffffffff, x.label, mask, warpSize);
  return x < y == dir ? y : x;
}

__device__ __forceinline__ uint Bfe(uint lane_id, uint pos) {
  uint res;
  asm("bfe.u32 %0,%1,%2,%3;" : "=r"(res) : "r"(lane_id), "r"(pos), "r"(1));
  return res;
}

__device__ __forceinline__ void BitonicSort(NNDElement *sort_element_ptr,
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

__device__ void UpdateLocalKNNLists(NNDElement *knn_list, const int *neighbs_id,
                                    const int list_id, const int list_size,
                                    const float *distances,
                                    const int distances_num) {
  int head_pos = list_id * (list_id - 1) / 2;
  int tail_pos = (list_id + 1) * list_id / 2;
  int y_num = tail_pos - head_pos;

  int tx = threadIdx.x;
  int lane_id = tx % warpSize;
  int pos_in_lists = list_id * NEIGHB_CACHE_NUM;

  int it_num = GetItNum(y_num, warpSize);
  for (int it = 0; it < it_num; it++) {
    // bitonic sort
    NNDElement sort_elem;
    sort_elem.label = neighbs_id[it * warpSize + lane_id];
    int current_pos = head_pos + it * warpSize + lane_id;
    if (current_pos < tail_pos) {
      sort_elem.distance = distances[current_pos];
    } else {
      sort_elem.distance = 1e10;
      sort_elem.label = LARGE_INT;
    }
    // printf("%d %f %d\n", lane_id, sort_elem.distance, sort_elem.label);
    BitonicSort(&sort_elem, lane_id);
    int offset;
    for (offset = 0; offset < NEIGHB_CACHE_NUM; offset++) {
      int flag = 1;
      if (lane_id == warpSize - 1) {
        if (sort_elem < knn_list[pos_in_lists + offset]) {
          flag = 0;
        }
      }
      flag = __shfl_sync(0xffffffff, flag, warpSize - 1, warpSize);
      if (!flag) break;
      NNDElement tmp;
      tmp.distance =
          __shfl_up_sync(0xffffffff, sort_elem.distance, 1, warpSize);
      tmp.label = __shfl_up_sync(0xffffffff, sort_elem.label, 1, warpSize);
      sort_elem = tmp;
    }
    if (lane_id < offset) {
      if (lane_id < NEIGHB_CACHE_NUM) {
        sort_elem = knn_list[pos_in_lists + lane_id];
      } else {
        sort_elem = NNDElement(1e10, LARGE_INT);
      }
    }
    BitonicSort(&sort_elem, lane_id);
    if (lane_id < NEIGHB_CACHE_NUM)
      knn_list[pos_in_lists + lane_id] = sort_elem;
  }

  head_pos = list_id * (list_id + 3) / 2;  // 0   2   5   9   14
  for (int it = 0; it < 2; it++) {
    NNDElement sort_elem;
    int no = it * warpSize + lane_id;
    sort_elem.label = neighbs_id[no + list_id + 1];
    int current_pos = head_pos + no * (no + list_id * 2 + 1) / 2;
    if (current_pos < distances_num) {
      sort_elem.distance = distances[current_pos];
    } else {
      sort_elem.distance = 1e10;
      sort_elem.label = LARGE_INT;
    }
    BitonicSort(&sort_elem, lane_id);
    int offset;
    for (offset = 0; offset < NEIGHB_CACHE_NUM; offset++) {
      int flag = 1;
      if (lane_id == warpSize - 1) {
        if (sort_elem < knn_list[pos_in_lists + offset]) {
          flag = 0;
        }
      }
      flag = __shfl_sync(0xffffffff, flag, warpSize - 1, warpSize);
      if (!flag) break;
      NNDElement tmp;
      tmp.distance =
          __shfl_up_sync(0xffffffff, sort_elem.distance, 1, warpSize);
      tmp.label = __shfl_up_sync(0xffffffff, sort_elem.label, 1, warpSize);
      sort_elem = tmp;
    }
    if (lane_id < offset) {
      if (lane_id < NEIGHB_CACHE_NUM) {
        sort_elem = knn_list[pos_in_lists + lane_id];
      } else {
        sort_elem = NNDElement(1e10, LARGE_INT);
      }
    }
    BitonicSort(&sort_elem, lane_id);
    if (lane_id < NEIGHB_CACHE_NUM)
      knn_list[pos_in_lists + lane_id] = sort_elem;
  }

  // printf("%d %f %d\n", lane_id, knn_list[pos_in_lists + lane_id].distance,
  // knn_list[pos_in_lists + lane_id].label);
}

__device__ void UpdateLocalNewKNNLists(NNDElement *knn_list, const int list_id,
                                       const int list_size,
                                       const int *old_neighbs,
                                       const int num_old,
                                       const float *distances,
                                       const int distances_num) {
  int head_pos = list_id * num_old;
  int y_num = num_old;
  int tail_pos = head_pos + num_old;

  int tx = threadIdx.x;
  int lane_id = tx % warpSize;
  int pos_in_lists = list_id * NEIGHB_CACHE_NUM;

  int it_num = GetItNum(y_num, warpSize);
  for (int it = 0; it < it_num; it++) {
    // bitonic sort
    NNDElement sort_elem;
    int no = it * warpSize + lane_id;
    sort_elem.label = old_neighbs[no];
    int current_pos = head_pos + no;
    if (current_pos < tail_pos) {
      sort_elem.distance = distances[current_pos];
    } else {
      sort_elem.distance = 1e10;
      sort_elem.label = LARGE_INT;
    }
    // printf("%d %f %d\n", lane_id, sort_elem.distance, sort_elem.label);
    BitonicSort(&sort_elem, lane_id);
    int offset;
    for (offset = 0; offset < NEIGHB_CACHE_NUM; offset++) {
      int flag = 1;
      if (lane_id == warpSize - 1) {
        if (sort_elem < knn_list[pos_in_lists + offset]) {
          flag = 0;
        }
      }
      flag = __shfl_sync(0xffffffff, flag, warpSize - 1, warpSize);
      if (!flag) break;
      NNDElement tmp;
      tmp.distance =
          __shfl_up_sync(0xffffffff, sort_elem.distance, 1, warpSize);
      tmp.label = __shfl_up_sync(0xffffffff, sort_elem.label, 1, warpSize);
      sort_elem = tmp;
    }
    if (lane_id < offset) {
      if (lane_id < NEIGHB_CACHE_NUM) {
        sort_elem = knn_list[pos_in_lists + lane_id];
      } else {
        sort_elem = NNDElement(1e10, LARGE_INT);
      }
    }
    BitonicSort(&sort_elem, lane_id);
    if (lane_id < NEIGHB_CACHE_NUM)
      knn_list[pos_in_lists + lane_id] = sort_elem;
  }
}

__device__ void UpdateLocalOldKNNLists(
    NNDElement *knn_list, const int list_id, const int list_size,
    const int *new_neighbs, const int num_new, const int *old_neighbs,
    const int num_old, const float *distances, const int distances_num,
    const float *vectors) {
  int head_pos = list_id - num_new;
  int tx = threadIdx.x;
  int lane_id = tx % warpSize;
  int pos_in_lists = list_id * NEIGHB_CACHE_NUM;

  int it_num = GetItNum(num_new, warpSize);
  for (int it = 0; it < it_num; it++) {
    NNDElement sort_elem;
    int no = it * warpSize + lane_id;
    sort_elem.label = new_neighbs[no];
    int current_pos = head_pos + no * num_old;
    if (current_pos < distances_num) {
      sort_elem.distance = distances[current_pos];
    } else {
      sort_elem.distance = 1e10;
      sort_elem.label = LARGE_INT;
    }
    BitonicSort(&sort_elem, lane_id);
    int offset;
    for (offset = 0; offset < NEIGHB_CACHE_NUM; offset++) {
      int flag = 1;
      if (lane_id == warpSize - 1) {
        if (sort_elem < knn_list[pos_in_lists + offset]) {
          flag = 0;
        }
      }
      flag = __shfl_sync(0xffffffff, flag, warpSize - 1, warpSize);
      if (!flag) break;
      NNDElement tmp;
      tmp.distance =
          __shfl_up_sync(0xffffffff, sort_elem.distance, 1, warpSize);
      tmp.label = __shfl_up_sync(0xffffffff, sort_elem.label, 1, warpSize);
      sort_elem = tmp;
    }
    if (lane_id < offset) {
      if (lane_id < NEIGHB_CACHE_NUM) {
        sort_elem = knn_list[pos_in_lists + lane_id];
      } else {
        sort_elem = NNDElement(1e10, LARGE_INT);
      }
    }
    BitonicSort(&sort_elem, lane_id);
    if (lane_id < NEIGHB_CACHE_NUM)
      knn_list[pos_in_lists + lane_id] = sort_elem;
  }
}
#endif

__device__ void UniqueMergeSequential(const NNDElement *A, const int m,
                                      const NNDElement *B, const int n,
                                      NNDElement *C, const int k) {
  int i = 0, j = 0, cnt = 0;
  while ((i < m) && (j < n)) {
    if (A[i] <= B[j]) {
      C[cnt++] = A[i++];
      if (cnt >= k) goto EXIT;
      while (i < m && A[i] <= C[cnt - 1]) i++;
      while (j < n && B[j] <= C[cnt - 1]) j++;
    } else {
      C[cnt++] = B[j++];
      if (cnt >= k) goto EXIT;
      while (i < m && A[i] <= C[cnt - 1]) i++;
      while (j < n && B[j] <= C[cnt - 1]) j++;
    }
  }

  if (i == m) {
    for (; j < n; j++) {
      if (B[j] > C[cnt - 1]) {
        C[cnt++] = B[j];
      }
      if (cnt >= k) goto EXIT;
    }
    for (; i < m; i++) {
      if (A[i] > C[cnt - 1]) {
        C[cnt++] = A[i];
      }
      if (cnt >= k) goto EXIT;
    }
  } else {
    for (; i < m; i++) {
      if (A[i] > C[cnt - 1]) {
        C[cnt++] = A[i];
      }
      if (cnt >= k) goto EXIT;
    }
    for (; j < n; j++) {
      if (B[j] > C[cnt - 1]) {
        C[cnt++] = B[j];
      }
      if (cnt >= k) goto EXIT;
    }
  }

EXIT:
  return;
}

__device__ void MergeLocalGraphWithGlobalGraph(
    const NNDElement *local_knn_graph, const int list_size,
    const int *neighb_ids, const int neighb_num, NNDElement *global_knn_graph,
    int *global_locks) {
  int tx = threadIdx.x;
  if (tx < neighb_num) {
    NNDElement C_cache[NEIGHB_NUM_PER_LIST];
    int neighb_id = neighb_ids[tx];
    bool loop_flag = false;
    do {
      if (loop_flag = atomicCAS(&global_locks[neighb_id], 0, 1) == 0) {
        if (local_knn_graph[tx * NEIGHB_CACHE_NUM].label() != LARGE_INT) {
          UniqueMergeSequential(
              &local_knn_graph[tx * NEIGHB_CACHE_NUM], NEIGHB_CACHE_NUM,
              &global_knn_graph[neighb_id * NEIGHB_NUM_PER_LIST],
              NEIGHB_NUM_PER_LIST, C_cache, NEIGHB_NUM_PER_LIST);
          for (int i = 0; i < NEIGHB_NUM_PER_LIST; i++) {
            global_knn_graph[neighb_id * NEIGHB_NUM_PER_LIST + i] = C_cache[i];
          }
        }
      }
      __threadfence();
      if (loop_flag) {
        atomicExch(&global_locks[neighb_id], 0);
      }
      __nanosleep(32);
    } while (!loop_flag);
  }
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
        if (lane_id == 0) prev_elem = NNDElement(-1e10, -1);
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
    const int *edges_new, const int *dest_new, const int num_new_max) {
  extern __shared__ char buffer[];

  __shared__ float *shared_vectors, *distances;
  __shared__ int *neighbors;
  __shared__ int pos_gnew, num_new;

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
    int next_pos = edges_new[list_id + 1];
    int now_pos = edges_new[list_id];
    num_new = next_pos - now_pos;
    pos_gnew = now_pos;
  }
  __syncthreads();
  int neighb_num = num_new;
  if (tx < neighb_num) {
    neighbors[tx] = dest_new[pos_gnew + tx];
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

#if !INSERT_MIN_ONLY
  int num_it2 = GetItNum(neighb_num * NEIGHB_CACHE_NUM, block_dim_x);
  for (int j = 0; j < num_it2; j++) {
    int pos = j * block_dim_x + tx;
    if (pos < neighb_num * NEIGHB_CACHE_NUM)
      knn_graph_cache[pos] = NNDElement(1e10, LARGE_INT);
  }
#endif
  int list_size = NEIGHB_CACHE_NUM;
  int num_it3 = GetItNum(neighb_num, block_dim_x / warpSize);
  for (int j = 0; j < num_it3; j++) {
    int list_id = j * (block_dim_x / warpSize) + tx / warpSize;
    if (list_id >= neighb_num) continue;
#if INSERT_MIN_ONLY
    NNDElement min_elem =
        GetMinElement(neighbors, list_id, list_size, distances, calc_num);
    InsertToGlobalGraph(min_elem, list_id, neighbors[list_id], knn_graph,
                        global_locks);
#else
    UpdateLocalKNNLists(knn_graph_cache, neighbors, list_id, list_size,
                        distances, calc_num);
#endif
  }
}

__global__ void NewOldNeighborsCompareKernel(
    NNDElement *knn_graph, int *global_locks, const float *vectors,
    const int *edges_new, const int *dest_new, const int num_new_max,
    const int *edges_old, const int *dest_old, const int num_old_max) {
  extern __shared__ char buffer[];

  __shared__ float *shared_vectors, *distances;
  __shared__ int *neighbors;
  __shared__ NNDElement *knn_graph_cache;

  __shared__ int pos_gnew, pos_gold, num_new, num_old;

  int neighb_num_max = num_new_max + num_old_max;
  int tx = threadIdx.x;
  if (tx == 0) {
    shared_vectors = (float *)buffer;
    knn_graph_cache = (NNDElement *)buffer;
    size_t offset = max(neighb_num_max * SKEW_DIM * sizeof(float),
                        neighb_num_max * NEIGHB_CACHE_NUM * sizeof(NNDElement));
    distances = (float *)((char *)buffer + offset);
    neighbors = (int *)((char *)distances +
                        (num_new_max * num_old_max) * sizeof(float));
  }
  __syncthreads();

  int list_id = blockIdx.x;
  int block_dim_x = blockDim.x;

  if (tx == 0) {
    int next_pos = edges_new[list_id + 1];
    int now_pos = edges_new[list_id];
    num_new = next_pos - now_pos;
    pos_gnew = now_pos;
  } else if (tx == 32) {
    int next_pos = edges_old[list_id + 1];
    int now_pos = edges_old[list_id];
    num_old = next_pos - now_pos;
    pos_gold = now_pos;
  }
  __syncthreads();
  int neighb_num = num_new + num_old;
  if (tx < num_new) {
    neighbors[tx] = dest_new[pos_gnew + tx];
  } else if (tx >= num_new && tx < neighb_num) {
    neighbors[tx] = dest_old[pos_gold + tx - num_new];
  }
  __syncthreads();

  int num_vec_per_it = block_dim_x / VEC_DIM;
  int num_it = GetItNum(neighb_num, num_vec_per_it);
  for (int i = 0; i < num_it; i++) {
    int row = i * num_vec_per_it + tx / VEC_DIM;
    if (row >= neighb_num) continue;
    int col = tx % VEC_DIM;
    int vec_id = neighbors[row];
    shared_vectors[row * SKEW_DIM + col] = vectors[vec_id * VEC_DIM + col];
  }
  __syncthreads();

  int calc_num = num_new * num_old;
  num_it = GetItNum(calc_num, block_dim_x);
  for (int i = 0; i < num_it; i++) {
    int idx = i * block_dim_x + tx;
    if (idx >= calc_num) continue;
    int x = idx / num_old;
    int y = idx % num_old + num_new;
    float sum = 0;
    int base_x = x * SKEW_DIM;
    int base_y = y * SKEW_DIM;
    for (int j = 0; j < VEC_DIM; j++) {
      float diff = shared_vectors[base_x + j] - shared_vectors[base_y + j];
      sum += diff * diff;
    }
    distances[idx] = sum;
  }
  __syncthreads();

  // Read list to cache
  int num_it2 = GetItNum(neighb_num * NEIGHB_CACHE_NUM, block_dim_x);
  for (int j = 0; j < num_it2; j++) {
    int pos = j * block_dim_x + tx;
    if (pos < neighb_num * NEIGHB_CACHE_NUM)
      knn_graph_cache[pos] = NNDElement(1e10, LARGE_INT);
  }
  int list_size = NEIGHB_CACHE_NUM;
  int num_it3 = GetItNum(neighb_num, block_dim_x / warpSize);
  for (int j = 0; j < num_it3; j++) {
    int list_id = j * (block_dim_x / warpSize) + tx / warpSize;
    if (list_id >= neighb_num) continue;
#if INSERT_MIN_ONLY
#else
    if (list_id < num_new) {
      UpdateLocalNewKNNLists(knn_graph_cache, list_id, list_size,
                             neighbors + num_new, num_old, distances, calc_num);
    } else {
      UpdateLocalOldKNNLists(knn_graph_cache, list_id, list_size, neighbors,
                             num_new, neighbors + num_new, num_old, distances,
                             calc_num, vectors);
    }
#endif
  }
  __syncthreads();
  MergeLocalGraphWithGlobalGraph(knn_graph_cache, list_size, neighbors,
                                 neighb_num, knn_graph, global_locks);
  __syncthreads();
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
    const int *edges_new, const int *dest_new, const int num_new_max,
    const int *edges_old, const int *dest_old, const int num_old_max) {
  extern __shared__ char buffer[];

  __shared__ float *distances;
  __shared__ int *neighbors;

  __shared__ int pos_gnew, pos_gold, num_new, num_old;

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
    int next_pos = edges_new[list_id + 1];
    int now_pos = edges_new[list_id];
    num_new = next_pos - now_pos;
    pos_gnew = now_pos;
  } else if (tx == 32) {
    int next_pos = edges_old[list_id + 1];
    int now_pos = edges_old[list_id];
    num_old = next_pos - now_pos;
    pos_gold = now_pos;
  }
  __syncthreads();
  int neighb_num = num_new + num_old;
  if (tx < num_new) {
    neighbors[tx] = dest_new[pos_gnew + tx];
  } else if (tx >= num_new && tx < neighb_num) {
    neighbors[tx] = dest_old[pos_gold + tx - num_new];
  }
  __syncthreads();

  GetNewOldDistancesTiled(distances, vectors, neighbors, num_new,
                          neighbors + num_new, num_old);
  __syncthreads();

  int calc_num = num_new * num_old;
// int num_it = GetItNum(NEIGHB_NUM_PER_LIST, NEIGHB_CACHE_NUM);

// Read list to cache
#if !INSERT_MIN_ONLY
  int num_it2 = GetItNum(neighb_num * NEIGHB_CACHE_NUM, block_dim_x);
  for (int j = 0; j < num_it2; j++) {
    int pos = j * block_dim_x + tx;
    if (pos < neighb_num * NEIGHB_CACHE_NUM)
      knn_graph_cache[pos] = NNDElement(1e10, LARGE_INT);
  }
#endif
  int list_size = NEIGHB_CACHE_NUM;
  int num_it3 = GetItNum(neighb_num, block_dim_x / warpSize);
  for (int j = 0; j < num_it3; j++) {
    int list_id = j * (block_dim_x / warpSize) + tx / warpSize;
    if (list_id >= neighb_num) continue;
#if INSERT_MIN_ONLY
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
#else
    if (list_id < num_new) {
      UpdateLocalNewKNNLists(knn_graph_cache, list_id, list_size,
                             neighbors + num_new, num_old, distances, calc_num);
    } else {
      UpdateLocalOldKNNLists(knn_graph_cache, list_id, list_size, neighbors,
                             num_new, neighbors + num_new, num_old, distances,
                             calc_num, vectors);
    }
#endif
  }
  __syncthreads();
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

void ToHostGraph(vector<vector<NNDElement>> *origin_knn_graph_ptr,
                 const NNDElement *knn_graph, const int size,
                 const int neighb_num) {
  auto &origin_knn_graph = *origin_knn_graph_ptr;
  vector<NNDElement> neighb_list;
  for (int i = 0; i < size; i++) {
    neighb_list.clear();
    for (int j = 0; j < neighb_num; j++) {
      neighb_list.push_back(knn_graph[i * neighb_num + j]);
    }
    for (int j = 0; j < neighb_num; j++) {
      for (int k = 0; k < neighb_num; k++) {
        if (neighb_list[j] == origin_knn_graph[i][k]) {
          neighb_list[j].MarkOld();
          break;
        }
      }
    }
    origin_knn_graph[i] = neighb_list;
  }
}

int GetMaxListSize(const Graph &g) {
  int res = 0;
  for (const auto &list : g) {
    res = max((int)list.size(), res);
  }
  return res;
}

float UpdateGraph(vector<vector<NNDElement>> *origin_knn_graph_ptr,
                  float *vectors_dev, const Graph &newg, const Graph &oldg,
                  const int k) {
  float kernel_time = 0;
  auto &origin_knn_graph = *origin_knn_graph_ptr;

  int *edges_dev_new, *dest_dev_new;
  tie(edges_dev_new, dest_dev_new) = ReadGraphToGlobalMemory(newg);

  int *edges_dev_old, *dest_dev_old;
  tie(edges_dev_old, dest_dev_old) = ReadGraphToGlobalMemory(oldg);

  size_t g_size = newg.size();

  cudaError_t cuda_status;
  NNDElement *knn_graph_dev, *knn_graph = new NNDElement[g_size * k];
  knn_graph_dev = ReadKNNGraphToGlobalMemory(origin_knn_graph);

  int *global_locks_dev;
  cudaMalloc(&global_locks_dev, g_size * sizeof(int));
  vector<int> zeros(g_size);
  cudaMemcpy(global_locks_dev, zeros.data(), g_size * sizeof(int),
             cudaMemcpyHostToDevice);
  cuda_status = cudaGetLastError();

  if (cuda_status != cudaSuccess) {
    cerr << cudaGetErrorString(cuda_status) << endl;
    cerr << "Initiate failed" << endl;
    exit(-1);
  }

  dim3 block_size(640);
  dim3 grid_size(g_size);
  // cerr << "Start kernel." << endl;
  const int num_new_max = GetMaxListSize(newg);
  const int num_old_max = GetMaxListSize(oldg);
  cerr << "Num new max: " << num_new_max << endl;
  cerr << "Num old max: " << num_old_max << endl;
  size_t shared_memory_size =
      num_new_max * SKEW_DIM * sizeof(float) +
      (num_new_max * (num_new_max - 1) / 2) * sizeof(float) +
      num_new_max * sizeof(int);

  cerr << "Shmem kernel1 costs: " << shared_memory_size << endl;

  auto start = chrono::steady_clock::now();
  NewNeighborsCompareKernel<<<grid_size, block_size, shared_memory_size>>>(
      knn_graph_dev, global_locks_dev, vectors_dev, edges_dev_new, dest_dev_new,
      num_new_max);
  int neighb_num_max = num_new_max + num_old_max;

#if DONT_TILE
  shared_memory_size =
      max(neighb_num_max * SKEW_DIM * sizeof(float),
          neighb_num_max * NEIGHB_CACHE_NUM * sizeof(NNDElement)) +
      num_new_max * num_old_max * sizeof(float) + neighb_num_max * sizeof(int);
  if (shared_memory_size < MAX_SHMEM) {
    cerr << "Shmem kernel2 costs: " << shared_memory_size << endl;
    block_size = dim3(640);
    NewOldNeighborsCompareKernel<<<grid_size, block_size, shared_memory_size>>>(
        knn_graph_dev, global_locks_dev, vectors_dev, edges_dev_new,
        dest_dev_new, num_new_max, edges_dev_old, dest_dev_old, num_old_max);
  } else {
    block_size = dim3(TILE_WIDTH * TILE_WIDTH);
    int neighb_num_max = num_new_max + num_old_max;
    shared_memory_size = (num_new_max * num_old_max) * sizeof(float) +
                         neighb_num_max * sizeof(int) +
                         neighb_num_max * NEIGHB_CACHE_NUM * sizeof(NNDElement);
    cerr << "Shmem tiled kernel2 costs: " << shared_memory_size << endl;
    TiledNewOldNeighborsCompareKernel<<<grid_size, block_size,
                                        shared_memory_size>>>(
        knn_graph_dev, global_locks_dev, vectors_dev, edges_dev_new,
        dest_dev_new, num_new_max, edges_dev_old, dest_dev_old, num_old_max);
  }
#else
  block_size = dim3(TILE_WIDTH * TILE_WIDTH);
  shared_memory_size = (num_new_max * num_old_max) * sizeof(float) +
                       neighb_num_max * sizeof(int);
  cerr << "Shmem tiled kernel2 costs: " << shared_memory_size << endl;
  TiledNewOldNeighborsCompareKernel<<<grid_size, block_size,
                                      shared_memory_size>>>(
      knn_graph_dev, global_locks_dev, vectors_dev, edges_dev_new, dest_dev_new,
      num_new_max, edges_dev_old, dest_dev_old, num_old_max);
#endif

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
  cuda_status =
      cudaMemcpy(knn_graph, knn_graph_dev, g_size * k * sizeof(NNDElement),
                 cudaMemcpyDeviceToHost);
  if (cuda_status != cudaSuccess) {
    cerr << cudaGetErrorString(cuda_status) << endl;
    cerr << "knn_graph cudaMemcpy failed" << endl;
    exit(-1);
  }

  ToHostGraph(&origin_knn_graph, knn_graph, g_size, k);

  delete[] knn_graph;
  cudaFree(edges_dev_new);
  cudaFree(dest_dev_new);
  cudaFree(edges_dev_old);
  cudaFree(dest_dev_old);
  cudaFree(knn_graph_dev);
  return kernel_time;
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

  if (cuda_status != cudaSuccess) {
    cerr << cudaGetErrorString(cuda_status) << endl;
    cerr << "cudaSetDevice failed" << endl;
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
  auto sum_start = chrono::steady_clock::now();
  long long cmp_times = 0;
  for (int t = 0; t < iteration; t++) {
    cerr << "Start generating NBGraph." << endl;
    auto start = chrono::steady_clock::now();
    GetNBGraph(&newg, &oldg, g, vectors, vecs_size, vecs_dim);
    auto end = chrono::steady_clock::now();
    float tmp_time =
        (float)chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count() /
        1e6;
    get_nb_graph_time += tmp_time;
    for (int i = 0; i < newg.size(); i++) {
      cmp_times += (newg[i].size() - 1) * newg[i].size() / 2 +
                   newg[i].size() * oldg[i].size();
    }
    cerr << "GetNBGraph costs " << tmp_time << endl;

    start = chrono::steady_clock::now();
    vector<pair<float, int>> tmp_result;
    // long long update_times = 0;
    float tmp_kernel_costs = UpdateGraph(&g, vectors_dev, newg, oldg, k);
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
  return g;
}
}  // namespace gpuknn

#endif