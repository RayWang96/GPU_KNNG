#ifndef XMUKNN_KNNCUDA_TOOLS_CUH
#define XMUKNN_KNNCUDA_TOOLS_CUH
#include<vector>
#include "nndescent_element.cuh"
#include "../xmuknn.h"
using namespace std;
using namespace xmuknn;
#define FULL_MASK 0xffffffff
const int WARP_SIZE = 32;

void DevRNGLongLong(unsigned long long *dev_data, int n);
void GenerateRandomKNNGraphIndex(int **knn_graph_index, const int graph_size,
                                 const int neighb_num);
__device__ int GetItNum(const int sum_num, const int num_per_it);
void ToHostKNNGraph(vector<vector<NNDElement>> *origin_knn_graph_ptr,
                    const NNDElement *knn_graph_dev, const int size,
                    const int neighb_num);
void OutputHostKNNGraph(const vector<vector<NNDElement>> &knn_graph,
                        const string &out_path);
__device__ int RemoveDuplicates(int *nums, int nums_size);
int GetMaxListSize(const Graph &g);
int GetMaxListSize(int *list_size_dev, const int g_size);
__device__ NNDElement __shfl_down_sync(const int mask, NNDElement var,
                                       const int delta,
                                       const int width = WARP_SIZE);
__device__ NNDElement __shfl_up_sync(const int mask, NNDElement var,
                                     const int delta,
                                     const int width = WARP_SIZE);
__device__ uint GetNthSetBitPos(uint mask, int nth);

template <typename T>
__device__ __forceinline__ T Min(const T &a, const T &b) {
  return a < b ? a : b;
}

__device__ __forceinline__ NNDElement XorSwap(NNDElement x, int mask, int dir) {
  NNDElement y;
  y.distance_ = __shfl_xor_sync(FULL_MASK, x.distance_, mask, warpSize);
  y.label_ = __shfl_xor_sync(FULL_MASK, x.label_, mask, warpSize);
  return x < y == dir ? y : x;
}

__device__ __forceinline__ int XorSwap(int x, int mask, int dir) {
  int y;
  y = __shfl_xor_sync(FULL_MASK, x, mask, warpSize);
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

template <typename T>
__device__ int MergeList(T *A, const int m, T *B, const int n, T *C,
                         const int max_size) {
  int i = 0, j = 0, cnt = 0;
  while ((i < m) && (j < n)) {
    if (A[i] <= B[j]) {
      C[cnt++] = A[i++];
      if (cnt >= max_size) goto EXIT;
    } else {
      C[cnt++] = B[j++];
      if (cnt >= max_size) goto EXIT;
    }
  }

  if (i == m) {
    for (; j < n; j++) {
      C[cnt++] = B[j];
      if (cnt >= max_size) goto EXIT;
    }
  } else {
    for (; i < m; i++) {
      C[cnt++] = A[i];
      if (cnt >= max_size) goto EXIT;
    }
  }
EXIT:
  return cnt;
}
#endif