#ifndef XMUKNN_NNDESCENT_CU
#define XMUKNN_NNDESCENT_CU

#include <vector>
#include <iostream>
#include <assert.h>
#include <bitset>
#include <algorithm>
#include <cstring>
#include <tuple>
#include <utility>
#include <chrono>
#include <mutex> 
#include <mma.h>

#include "result_element.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "nndescent.cuh"
#include "../xmuknn.h"
#include "../tools/distfunc.hpp"

#ifdef __INTELLISENSE__
#include "../intellisense_cuda_intrinsics.h"
#endif

using namespace std;
using namespace xmuknn;
#define DEVICE_ID 0
#define LARGE_INT 0x3f3f3f3f
const int VEC_DIM = 128;
const int NEIGHB_NUM_PER_LIST = 40;
const int NEIGHB_CACHE_NUM = 16;
const int TILE_WIDTH = 16;
const int THREADS_PER_LIST = 32;
const int SAMPLE_NUM = 30;
__device__ int for_check = 0;

void GetNBGraph(Graph *graph_new_ptr,
                Graph *graph_old_ptr,
                vector<vector<gpuknn::NNDItem>>& knn_graph, 
                const float *vectors, const int vecs_size, 
                const int vecs_dim) {
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
            auto& item = knn_graph[i][j];
            if (item.visited) {
                graph_old[i].push_back(item.id);
            }
            else {
                if (cnt < sample_num) {
                    graph_new[i].push_back(item.id);
                    cnt++;
                    item.visited = true;
                }
            }
            if (cnt >= sample_num) break;
        }
    }
    // auto time2 = chrono::steady_clock::now();
    // cerr << "Mark 2: " << (float)chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count() / 1e6 << endl;

    #pragma omp parallel for
    for (int i = 0; i < knn_graph.size(); i++) {
        for (int j = 0; j < graph_new[i].size(); j++) {
            auto& id = graph_new[i][j];
            mtx[id].lock();
            graph_rnew[id].push_back(i);
            mtx[id].unlock();
        }
        for (int j = 0; j < graph_old[i].size(); j++) {
            auto& id = graph_old[i][j];
            mtx[id].lock();
            graph_rold[id].push_back(i);
            mtx[id].unlock();
        }
    }

    // auto time3 = chrono::steady_clock::now();
    // cerr << "Mark 3: " << (float)chrono::duration_cast<std::chrono::microseconds>(time3 - time1).count() / 1e6 << endl;

    // #pragma omp parallel for
    // for (int i = 0; i < knn_graph.size(); i++) {
    //     random_shuffle(graph_rnew[i].begin(), graph_rnew[i].end());
    //     random_shuffle(graph_rold[i].begin(), graph_rold[i].end());
    // }

    // auto time4 = chrono::steady_clock::now();
    // cerr << "Mark 4: " << (float)chrono::duration_cast<std::chrono::microseconds>(time4 - time1).count() / 1e6 << endl;

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
    // cerr << "Mark 5: " << (float)chrono::duration_cast<std::chrono::microseconds>(time5 - time1).count() / 1e6 << endl;

    // #pragma omp parallel for
    for (int i = 0; i < knn_graph.size(); i++) {
        sort(graph_new[i].begin(), graph_new[i].end());
        graph_new[i].erase(unique(graph_new[i].begin(), 
                                  graph_new[i].end()), graph_new[i].end());

        sort(graph_old[i].begin(), graph_old[i].end());
        graph_old[i].erase(unique(graph_old[i].begin(), 
                                  graph_old[i].end()), graph_old[i].end());
    }
    // auto time6 = chrono::steady_clock::now();
    // cerr << "Mark 6: " << (float)chrono::duration_cast<std::chrono::microseconds>(time6 - time1).count() / 1e6 << endl;
    return;
}

__device__ int GetItNum(const int sum_num, const int num_per_it) {
    return sum_num / num_per_it + (sum_num % num_per_it != 0);
}

__device__ void Swap(int &a, int &b) {
    int c = a;
    a = b;
    b = c;
}

__device__ __forceinline__ ResultElement XorSwap(ResultElement x, int mask, int dir) {
    ResultElement y;
    y.distance = __shfl_xor_sync(0xffffffff, x.distance, mask, THREADS_PER_LIST);
    y.label = __shfl_xor_sync(0xffffffff, x.label, mask, THREADS_PER_LIST);
    return x < y == dir ? y : x;
}

__device__ __forceinline__ int Bfe(int lane_id, int pos) {
    int res;
    asm("bfe.u32 %0,%1,%2,%3;"
        : "=r"(res) : "r"(lane_id), "r"(pos),"r"(1));
    return res;
}

__device__ void BitonicSort(ResultElement *sort_element_ptr, const int lane_id) {
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
    sort_elem = XorSwap(sort_elem, 0x10,                   Bfe(lane_id, 4));
    sort_elem = XorSwap(sort_elem, 0x08,                   Bfe(lane_id, 3));
    sort_elem = XorSwap(sort_elem, 0x04,                   Bfe(lane_id, 2));
    sort_elem = XorSwap(sort_elem, 0x02,                   Bfe(lane_id, 1));
    sort_elem = XorSwap(sort_elem, 0x01,                   Bfe(lane_id, 0));
    return;
}

__device__ void UpdateLocalKNNLists(ResultElement *knn_list,
                                    const int *neighbs_id,
                                    const int list_id,
                                    const int list_size,
                                    const float *distances,
                                    const int distances_num) {
    int head_pos = list_id * (list_id - 1) / 2;
    int tail_pos = (list_id + 1) * list_id / 2;
    int y_num = tail_pos - head_pos;

    int tx = threadIdx.x;
    int lane_id = tx % THREADS_PER_LIST;
    int pos_in_lists = list_id * NEIGHB_CACHE_NUM;

    int it_num = GetItNum(y_num, THREADS_PER_LIST);
    for (int it = 0; it < it_num; it++) {
        // bitonic sort
        ResultElement sort_elem;
        sort_elem.label = neighbs_id[it * THREADS_PER_LIST + lane_id];
        int current_pos = head_pos + it * THREADS_PER_LIST + lane_id;
        if (current_pos < tail_pos) {
            sort_elem.distance = distances[current_pos];
        } else {
            sort_elem.distance = 1e10;
            sort_elem.label = 87654321;
        }
        // printf("%d %f %d\n", lane_id, sort_elem.distance, sort_elem.label);
        BitonicSort(&sort_elem, lane_id);
        int offset;
        for (offset = 0; offset < NEIGHB_CACHE_NUM; offset++) {
            int flag = 1;
            if (lane_id == THREADS_PER_LIST - 1) {
                if (sort_elem < knn_list[pos_in_lists + offset]) {
                    flag = 0;
                }
            }
            flag = __shfl_sync(0xffffffff, flag, THREADS_PER_LIST - 1, 
                               THREADS_PER_LIST);
            if (!flag) break;
            ResultElement tmp;
            tmp.distance = __shfl_up_sync(0xffffffff, sort_elem.distance, 
                                          1, THREADS_PER_LIST);
            tmp.label = __shfl_up_sync(0xffffffff, sort_elem.label, 
                                       1, THREADS_PER_LIST);
            sort_elem = tmp;
        }
        if (lane_id < offset) {
            if (lane_id < NEIGHB_CACHE_NUM) {
                sort_elem = knn_list[pos_in_lists + lane_id];
            } else {
                sort_elem = ResultElement(1e10, 12345678);
            }        
        }
        BitonicSort(&sort_elem, lane_id);
        if (lane_id < NEIGHB_CACHE_NUM)
            knn_list[pos_in_lists + lane_id] = sort_elem;
    }

    head_pos = list_id * (list_id + 3) / 2; // 0   2   5   9   14
    for (int it = 0; it < 2; it++) {
        ResultElement sort_elem;
        int no = it * THREADS_PER_LIST + lane_id;
        sort_elem.label = neighbs_id[no + list_id + 1];
        int current_pos = head_pos + no * (no + list_id * 2 + 1) / 2;
        if (current_pos < distances_num) {
            sort_elem.distance = distances[current_pos];
        } else {
            sort_elem.distance = 1e10;
            sort_elem.label = 99999999;
        }
        BitonicSort(&sort_elem, lane_id);
        int offset;
        for (offset = 0; offset < NEIGHB_CACHE_NUM; offset++) {
            int flag = 1;
            if (lane_id == THREADS_PER_LIST - 1) {
                if (sort_elem < knn_list[pos_in_lists + offset]) {
                    flag = 0;
                }
            }
            flag = __shfl_sync(0xffffffff, flag, THREADS_PER_LIST - 1, 
                               THREADS_PER_LIST);
            if (!flag) break;
            ResultElement tmp;
            tmp.distance = __shfl_up_sync(0xffffffff, sort_elem.distance, 
                                          1, THREADS_PER_LIST);
            tmp.label = __shfl_up_sync(0xffffffff, sort_elem.label, 
                                       1, THREADS_PER_LIST);
            sort_elem = tmp;
        }
        if (lane_id < offset) {
            if (lane_id < NEIGHB_CACHE_NUM) {
                sort_elem = knn_list[pos_in_lists + lane_id];
            } else {
                sort_elem = ResultElement(1e10, 12345678);
            }
        }
        BitonicSort(&sort_elem, lane_id);
        if (lane_id < NEIGHB_CACHE_NUM)
            knn_list[pos_in_lists + lane_id] = sort_elem;
    }

    // printf("%d %f %d\n", lane_id, knn_list[pos_in_lists + lane_id].distance, knn_list[pos_in_lists + lane_id].label);
}

__device__ void UpdateLocalNewKNNLists(ResultElement *knn_list,
                                       const int list_id,
                                       const int list_size,
                                       const int *old_neighbs,
                                       const int num_old,
                                       const float *distances,
                                       const int distances_num) {
    int head_pos = list_id * num_old;
    int y_num = num_old;
    int tail_pos = head_pos + num_old;

    int tx = threadIdx.x;
    int lane_id = tx % THREADS_PER_LIST;
    int pos_in_lists = list_id * NEIGHB_CACHE_NUM;

    int it_num = GetItNum(y_num, THREADS_PER_LIST);
    for (int it = 0; it < it_num; it++) {
        // bitonic sort
        ResultElement sort_elem;
        int no = it * THREADS_PER_LIST + lane_id;
        sort_elem.label = old_neighbs[no];
        int current_pos = head_pos + no;
        if (current_pos < tail_pos) {
            sort_elem.distance = distances[current_pos];
        } else {
            sort_elem.distance = 1e10;
            sort_elem.label = 87654321;
        }
        // printf("%d %f %d\n", lane_id, sort_elem.distance, sort_elem.label);
        BitonicSort(&sort_elem, lane_id);
        int offset;
        for (offset = 0; offset < NEIGHB_CACHE_NUM; offset++) {
            int flag = 1;
            if (lane_id == THREADS_PER_LIST - 1) {
                if (sort_elem < knn_list[pos_in_lists + offset]) {
                    flag = 0;
                }
            }
            flag = __shfl_sync(0xffffffff, flag, THREADS_PER_LIST - 1, 
                               THREADS_PER_LIST);
            if (!flag) break;
            ResultElement tmp;
            tmp.distance = __shfl_up_sync(0xffffffff, sort_elem.distance, 
                                          1, THREADS_PER_LIST);
            tmp.label = __shfl_up_sync(0xffffffff, sort_elem.label, 
                                       1, THREADS_PER_LIST);
            sort_elem = tmp;
        }
        if (lane_id < offset) {
            if (lane_id < NEIGHB_CACHE_NUM) {
                sort_elem = knn_list[pos_in_lists + lane_id];
            } else {
                sort_elem = ResultElement(1e10, 12345678);
            }
        }
        BitonicSort(&sort_elem, lane_id);
        if (lane_id < NEIGHB_CACHE_NUM)
            knn_list[pos_in_lists + lane_id] = sort_elem;
    }
}

__device__ void UpdateLocalOldKNNLists(ResultElement *knn_list,
                                       const int list_id,
                                       const int list_size,
                                       const int *new_neighbs,
                                       const int num_new,
                                       const int *old_neighbs,
                                       const int num_old,
                                       const float *distances,
                                       const int distances_num,
                                       const float *vectors) {
    int head_pos = list_id - num_new;
    int tx = threadIdx.x;
    int lane_id = tx % THREADS_PER_LIST;
    int pos_in_lists = list_id * NEIGHB_CACHE_NUM;

    int it_num = GetItNum(num_new, THREADS_PER_LIST);
    for (int it = 0; it < it_num; it++) {
        ResultElement sort_elem;
        int no = it * THREADS_PER_LIST + lane_id;
        sort_elem.label = new_neighbs[no];
        int current_pos = head_pos + no * num_old;
        if (current_pos < distances_num) {
            sort_elem.distance = distances[current_pos];
        } else {
            sort_elem.distance = 1e10;
            sort_elem.label = 55555555;
        }
        BitonicSort(&sort_elem, lane_id);
        int offset;
        for (offset = 0; offset < NEIGHB_CACHE_NUM; offset++) {
            int flag = 1;
            if (lane_id == THREADS_PER_LIST - 1) {
                if (sort_elem < knn_list[pos_in_lists + offset]) {
                    flag = 0;
                }
            }
            flag = __shfl_sync(0xffffffff, flag, THREADS_PER_LIST - 1, 
                               THREADS_PER_LIST);
            if (!flag) break;
            ResultElement tmp;
            tmp.distance = __shfl_up_sync(0xffffffff, sort_elem.distance, 
                                          1, THREADS_PER_LIST);
            tmp.label = __shfl_up_sync(0xffffffff, sort_elem.label, 
                                       1, THREADS_PER_LIST);
            sort_elem = tmp;
        }
        if (lane_id < offset) {
            if (lane_id < NEIGHB_CACHE_NUM) {
                sort_elem = knn_list[pos_in_lists + lane_id];
            } else {
                sort_elem = ResultElement(1e10, 12345678);
            }
        }
        BitonicSort(&sort_elem, lane_id);
        if (lane_id < NEIGHB_CACHE_NUM)
            knn_list[pos_in_lists + lane_id] = sort_elem;
    }
}

__device__ int InsertToLocalKNNList(ResultElement *knn_list, 
                                    const int list_size,
                                    const ResultElement &element,
                                    int *local_lock_ptr) {
    int &local_lock = *local_lock_ptr;
    int pos = -1;
    bool loop_flag = false;
    do {
        if (loop_flag = atomicCAS(&local_lock, 0, 1) == 0) {
            if (element >= knn_list[list_size-1]) ;
            else {
                int i = 0;
                while (i < list_size && knn_list[i] < element) {
                    i++;
                }
                if (knn_list[i] != element) {
                    for (int j = list_size - 1; j > i && j > 0; j--) {
                        knn_list[j] = knn_list[j-1];
                    }
                    knn_list[i] = element;
                    pos = i;
                }
            }
        }
        __threadfence();
        if (loop_flag) atomicExch(&local_lock, 0);
    } while (!loop_flag);
    return pos;
}

__device__ void UniqueMergeSequential(const ResultElement* A, const int m,
                                      const ResultElement* B, const int n,
                                      ResultElement* C, const int k) {
    int i = 0, j = 0, cnt = 0;
    while ((i < m) && (j < n)) {
        if (A[i] <= B[j]) {
            C[cnt++] = A[i++];
            if (cnt >= k) goto EXIT;
            while (i < m && A[i] <= C[cnt-1]) i++;
            while (j < n && B[j] <= C[cnt-1]) j++;
        } else {
            C[cnt++] = B[j++];
            if (cnt >= k) goto EXIT;
            while (i < m && A[i] <= C[cnt-1]) i++;
            while (j < n && B[j] <= C[cnt-1]) j++;
        }
    }

    if (i == m) {
        for (; j < n; j++) {
            if (B[j] > C[cnt-1]) {
                C[cnt++] = B[j];
            }
            if (cnt >= k) goto EXIT;
        }
        for (; i < m; i++) {
            if (A[i] > C[cnt-1]) {
                C[cnt++] = A[i];
            }
            if (cnt >= k) goto EXIT;
        }
    } else {
        for (; i < m; i++) {
            if (A[i] > C[cnt-1]) {
                C[cnt++] = A[i];
            }
            if (cnt >= k) goto EXIT;
        }
        for (; j < n; j++) {
            if (B[j] > C[cnt-1]) {
                C[cnt++] = B[j];
            }
            if (cnt >= k) goto EXIT;
        }
    }

EXIT:
    if (cnt != k) {
        int flag = atomicCAS(&for_check, 0, 1);
        if (!flag) {
            printf("%d %d\n", cnt, k);
            for (int i = 0; i < m; i++) {
                printf("%f ", A[i].distance);
            } printf("\n\n");
            for (int i = 0; i < n; i++) {
                printf("%f ", B[i].distance);
            } printf("\n\n");
            for (int i = 0; i < k; i++) {
                printf("%f ", C[i].distance);
            } printf("\n\n");

            for (int i = 0; i < m; i++) {
                printf("%d ", A[i].label);
            } printf("\n\n");
            for (int i = 0; i < n; i++) {
                printf("%d ", B[i].label);
            } printf("\n\n");
            for (int i = 0; i < k; i++) {
                printf("%d ", C[i].label);
            } printf("\n\n");
            printf("%d %d\n", cnt, k);
            assert(cnt == k);
        }
    }
    return;
}

__device__ void MergeLocalGraphWithGlobalGraph(const ResultElement* local_knn_graph,
                                               const int list_size, const int* neighb_ids,
                                               const int neighb_num,
                                               ResultElement* global_knn_graph,
                                               int* global_locks) {
    int tx = threadIdx.x;
    if (tx < neighb_num) {
        ResultElement C_cache[NEIGHB_NUM_PER_LIST];
        int neighb_id = neighb_ids[tx];
        bool loop_flag = false;
        do {
            __nanosleep(8);
            if (loop_flag = atomicCAS(&global_locks[neighb_id], 0, 1) == 0) {
                UniqueMergeSequential(&local_knn_graph[tx * NEIGHB_CACHE_NUM], 
                                      NEIGHB_CACHE_NUM, 
                                      &global_knn_graph[neighb_id * NEIGHB_NUM_PER_LIST],
                                      NEIGHB_NUM_PER_LIST, C_cache, NEIGHB_NUM_PER_LIST);
                for (int i = 0; i < NEIGHB_NUM_PER_LIST; i++) {
                    global_knn_graph[neighb_id * NEIGHB_NUM_PER_LIST + i]
                        = C_cache[i];
                }
            }
            __threadfence();
            if (loop_flag) atomicExch(&global_locks[neighb_id], 0);
        } while (!loop_flag);
    }
}

__global__ void NewNeighborsCompareKernel(ResultElement *knn_graph, int *global_locks,
                                          const float *vectors,
                                          const int *edges_new, const int *dest_new,
                                          const int num_new_max) {
    extern __shared__ char buffer[];

    __shared__ float *shared_vectors, *distances;
    __shared__ int *neighbors, *local_locks;
    __shared__ ResultElement *knn_graph_cache;
    __shared__ int pos_gnew, num_new;

    int tx = threadIdx.x;
    if (tx == 0) {
        shared_vectors = (float *)buffer;
        distances = 
            (float *)((char *)buffer + num_new_max * VEC_DIM * sizeof(float));
        neighbors = 
            (int *)((char *)distances + (num_new_max * (num_new_max - 1) / 2) * sizeof(float));
        local_locks = (int *)((char *)neighbors + num_new_max * sizeof(int));
        knn_graph_cache = 
            (ResultElement *)((char *)local_locks + num_new_max * sizeof(int));
    }
    __syncthreads();

    int list_id = blockIdx.x;
    int block_dim_x = blockDim.x;

    if (tx < num_new_max) {
        local_locks[tx] = 0;
    }

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
        shared_vectors[x * VEC_DIM + y] = vectors[vec_id * VEC_DIM + y];
    }

    int calc_num = (neighb_num * (neighb_num - 1)) / 2;

    num_it = GetItNum(calc_num, block_dim_x);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("check calc. num. %d %d\n", neighb_num, calc_num);
    }
    for (int i = 0; i < num_it; i++) {
        int x = i * block_dim_x + tx;
        if (x < calc_num) {
            distances[x] = 0;
        }
    }
    __syncthreads();
    for (int i = 0; i < num_it; i++) {
        int no = i * block_dim_x + tx;
        if (no >= calc_num) continue;
        int idx = no + 1;
        int x = ceil(sqrt(2 * idx + 0.25) - 0.5);
        int y = idx - (x - 1) * x / 2 - 1;
        if (x >= neighb_num || y >= neighb_num) continue;
        float sum = 0;
        for (int j = 0; j < VEC_DIM; j++) {
            float diff = shared_vectors[x * VEC_DIM + j] - 
                         shared_vectors[y * VEC_DIM + j];
            sum += diff * diff;
        }
        distances[no] = sum;
    }
    __syncthreads();
    // num_it = GetItNum(NEIGHB_NUM_PER_LIST, NEIGHB_CACHE_NUM);
    num_it = 1;
    for (int i = 0; i < num_it; i++) {
        int num_it2 = GetItNum(neighb_num * NEIGHB_CACHE_NUM, block_dim_x);
        for (int j = 0; j < num_it2; j++) {
            int pos = j * block_dim_x + tx;
            if (pos < neighb_num * NEIGHB_CACHE_NUM)
                knn_graph_cache[pos] = ResultElement(1e10, 77777777);
        }
        int list_size = NEIGHB_CACHE_NUM;
        int num_it3 = GetItNum(neighb_num, block_dim_x / THREADS_PER_LIST);
        for (int j = 0; j < num_it3; j++) {
            int list_id = j * (block_dim_x / THREADS_PER_LIST) + tx / THREADS_PER_LIST;
            if (list_id >= neighb_num) continue;
            UpdateLocalKNNLists(knn_graph_cache, neighbors, 
                                list_id, list_size, distances, calc_num);
        }
        __syncthreads();
        MergeLocalGraphWithGlobalGraph(knn_graph_cache, NEIGHB_CACHE_NUM, neighbors,
                                       neighb_num, knn_graph, global_locks);
        __syncthreads();
    }
}


// blockDim.x = TILE_WIDTH * TILE_WIDTH;
__device__ void GetNewOldDistances(float *distances, const float *vectors,
                                   const int *new_neighbors, const int num_new,
                                   const int *old_neighbors, const int num_old) {
    __shared__ float nsv[TILE_WIDTH][TILE_WIDTH]; //New shared vectors
    __shared__ float osv[TILE_WIDTH][TILE_WIDTH]; //Old shared vectors
    const int tile_size = TILE_WIDTH * TILE_WIDTH;
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
                nsv[t_row][t_col] = 
                    vectors[new_neighbors[row_new + t_row] * VEC_DIM +
                            ph * TILE_WIDTH + t_col];
            } else {
                nsv[t_row][t_col] = 1e10;
            }

            if ((row_old + t_col < num_old) && (ph * TILE_WIDTH + t_row < VEC_DIM)) {
                osv[t_col][t_row] = 
                    vectors[old_neighbors[row_old + t_col] * VEC_DIM +
                            ph * TILE_WIDTH + t_row];
            } else {
                osv[t_col][t_row] = 1e10;
            }
            __syncthreads();

            for (int k = 0; k < TILE_WIDTH; k++) {
                float a = nsv[t_row][k], b = osv[t_col][k];
                if (a > 1e9 || b > 1e9) {
                }
                else {
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

const int M = 16;
const int N = 16;
const int K = 16;

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

const int BLOCK_COL_WARPS = 1;
const int BLOCK_ROW_WARPS = 4;

const int ROWS_PER_IT = BLOCK_ROW_WARPS * WMMA_M;
const int COLS_PER_IT = BLOCK_COL_WARPS * WMMA_K;

using namespace nvcuda;
// threads = 4 * 32 = 128
__device__ void GetDistancesWMMA(float *distances, 
                                 const float *vectors,
                                 const int *new_neighbors, const int num_new,
                                 const int *old_neighbors, const int num_old,
                                 half shmem_a[][WMMA_K], 
                                 half shmem_b[][WMMA_K]) {
    __shared__ float distances_cache[4 * 4 * 16 * 16];
    __shared__ float squa_suma_cache[64];
    __shared__ float squa_sumb_cache[64];

    const int tx = threadIdx.x;
    const int warp_id = tx / warpSize;
    const int lane_id = tx % warpSize;
    int arow_it = GetItNum(num_new, ROWS_PER_IT);
    int col_it = GetItNum(VEC_DIM, COLS_PER_IT);
    int brow_it = GetItNum(num_old, ROWS_PER_IT);
    for (int i = 0; i < arow_it; i++) {
        int local_base_ay = warp_id * WMMA_M;
        int global_base_ay = i * ROWS_PER_IT + local_base_ay;
        for (int j = 0; j < brow_it; j++) {
            int local_base_by = warp_id * WMMA_N;
            int global_base_by = j * ROWS_PER_IT + local_base_by;
            wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major>
                a_frag;
            wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major>
                b_frag;
            wmma::fragment<wmma::accumulator, M, N, K, float> 
                acc_frag[BLOCK_ROW_WARPS];
            for (int k = 0; k < ROWS_PER_IT; k++) {
                wmma::fill_fragment(acc_frag[k], 0.0f);
            } 
            float squa_suma = 0, squa_sumb = 0;
            for (int k = 0; k < col_it; k++) {
                int local_x = lane_id % WMMA_K;
                int global_x = k * COLS_PER_IT + local_x; 
                if (lane_id < 16) {
                    for (int t = 0; t < WMMA_M; t++) {
                        int global_ay = global_base_ay + t;
                        int local_ay = local_base_ay + t;

                        int global_by = global_base_by + t;
                        int local_by = local_base_by + t;
                        if (global_ay < num_new && global_x < VEC_DIM) {
                            int pos = new_neighbors[global_ay];
                            float val = vectors[pos * VEC_DIM + global_x];
                            shmem_a[local_ay][local_x] = (half)val;
                        } else {
                            shmem_a[local_ay][local_x] = (half)0.0;
                        }
                        if (global_by < num_old && global_x < VEC_DIM) {
                            int pos = old_neighbors[global_by];
                            float val = vectors[pos * VEC_DIM + global_x];
                            shmem_b[local_by][local_x] = (half)val;
                        } else {
                            shmem_b[local_by][local_x] = (half)0.0;
                        }
                    }
                } 
                __syncthreads();
                if (lane_id < 16) {
                    for (int t = 0; t < WMMA_K; t++) {
                        float val = (float)shmem_a[warp_id * 16 + lane_id][t];
                        squa_suma += val * val;
                        val = (float)shmem_b[warp_id * 16 + lane_id][t];
                        squa_sumb += val * val;
                    }
                }
                // if (num_old > 0 && k == 0) {
                //     int flag = atomicCAS(&for_check, 0, 1);
                //     if (!flag) {
                //         for (int ii = 0; ii < num_old; ii++) {
                //             for (int jj = 0; jj < 16; jj++) {
                //                 int x = j * ROWS_PER_IT + ii;
                //                 int y = k * COLS_PER_IT + jj;
                //                 printf("%f ", vectors[old_neighbors[x] * VEC_DIM + y]);
                //             } printf("\n");
                //         } printf("\n\n");
                //         for (int i = 0; i < num_old; i++) {
                //             for (int j = 0; j < 16; j++) {
                //                 printf("%f ", (float)shmem_b[i][j]);
                //             } printf("\n");
                //         } printf("\n\n");
                //     }
                // }
                // __syncthreads();
                // if (num_new > 0 && k == 0) {
                //     int flag = atomicCAS(&for_check, 0, 1);
                //     if (!flag) {
                //         for (int ii = 0; ii < num_new; ii++) {
                //             for (int jj = 0; jj < 16; jj++) {
                //                 int x = i * ROWS_PER_IT + ii;
                //                 int y = k * COLS_PER_IT + jj;
                //                 printf("%f ", vectors[new_neighbors[x] * VEC_DIM + y]);
                //             } printf("\n");
                //         } printf("\n\n");
                //         for (int i = 0; i < num_new; i++) {
                //             for (int j = 0; j < 16; j++) {
                //                 printf("%f ", (float)shmem_a[i][j]);
                //             }printf("\n");
                //         } printf("\n\n");
                //     }
                // }
                // __syncthreads();
                wmma::load_matrix_sync(a_frag, 
                                       &shmem_a[warp_id * M][0],
                                       COLS_PER_IT);
                for (int t = 0; t < BLOCK_ROW_WARPS; t++) {
                    wmma::load_matrix_sync(b_frag, 
                                           &shmem_b[t * N][0],
                                           COLS_PER_IT);
                    wmma::mma_sync(acc_frag[t], a_frag, b_frag, acc_frag[t]);
                }
                // if (num_new > 0 && j == 0 && num_old > 0 && global_base_by == 0) {
                //     int flag = atomicCAS(&for_check, 0, 1);
                //     if (!flag) {
                //         printf("check %d %d\n", num_new, num_old);
                //         auto *a_ptr = &shmem_a[warp_idy * M][warp_idx * N];
                //         for (int i = 0; i < 16; i++) {
                //             for (int j = 0; j < 16; j++) {
                //                 printf("%f ", (float)a_ptr[i * COLS_PER_IT + j]);
                //             } printf("\n");
                //         } printf("\n\n");
                //         for (int i = 0; i < a_frag.num_elements; i++) {
                //             printf("%f ", (float)a_frag.x[i]);
                //         } printf("\n\n");
                //         auto *b_ptr = &shmem_b[warp_idy * M][warp_idx * N];
                //         for (int i = 0; i < 16; i++) {
                //             for (int j = 0; j < 16; j++) {
                //                 printf("%f ", (float)b_ptr[i * COLS_PER_IT + j]);
                //             } printf("\n");
                //         } printf("\n\n");
                //         for (int i = 0; i < b_frag.num_elements; i++) {
                //             printf("%f ", (float)b_frag.x[i]);
                //         } printf("\n\n");
                //         for (int i = 0; i < acc_frag.num_elements; i++) {
                //             printf("%f ", (float)acc_frag.x[i]);
                //         } printf("\n\n");
                //     }
                // }
                // __syncthreads();
            }
            // __syncthreads();
            for (int k = 0; k < BLOCK_ROW_WARPS; k++) {
                for (int t = 0; t < acc_frag[k].num_elements; t++) {
                    acc_frag[k].x[t] *= -2.0f;
                }
            }
            for (int k = 0; k < BLOCK_ROW_WARPS; k++) {
                wmma::store_matrix_sync(&distances_cache[warp_id * BLOCK_ROW_WARPS * WMMA_M * WMMA_N + k * WMMA_N], 
                                        acc_frag[k], BLOCK_ROW_WARPS * WMMA_M, wmma::mem_row_major);
            }
            if (lane_id < 16) {
                squa_suma_cache[warp_id * 16 + lane_id] = squa_suma;
                squa_sumb_cache[warp_id * 16 + lane_id] = squa_sumb;
            }
            __syncthreads();
            // if (warp_id == 0 && global_base_by == 0) {
            //     int flag = atomicCAS(&for_check, 0, 1);
            //     if (!flag) {
            //         for (int i = 0; i < 64; i++) {
            //             printf("%f ", squa_suma_cache[i]);
            //         } printf("\n\n");
            //         for (int i = 0; i < 64; i++) {
            //             printf("%f ", squa_sumb_cache[i]);
            //         } printf("\n\n");
            //         int x = new_neighbors[0];
            //         float suma = 0;
            //         for (int i = 0; i < VEC_DIM; i++) {
            //             float val = vectors[x * VEC_DIM + i];
            //             suma += val * val;
            //         }
            //         printf("check %f\n", suma);
            //     }
            // }
            // back conflict!!!
            if (lane_id < 16) {
                int local_base_dcy = warp_id * WMMA_M;
                for (int k = 0; k < BLOCK_ROW_WARPS * WMMA_N; k++) {
                    int local_x = k;
                    int local_dcy = local_base_dcy + lane_id;
                    int global_dy = i * BLOCK_ROW_WARPS * WMMA_M + local_dcy;
                    int global_dx = j * BLOCK_ROW_WARPS * WMMA_N + local_x;
                    // if (warp_id >= 2) {
                    //     printf("check %d %d\n", global_dy, global_dx);
                    //     assert(warp_id < 2);
                    // }
                    if (global_dy < num_new && global_dx < num_old) {
                        // assert(warp_id < 2);
                        distances[global_dy * num_old + global_dx] = 
                            distances_cache[local_dcy * BLOCK_ROW_WARPS * WMMA_M + local_x] + 
                            squa_suma_cache[local_dcy] + squa_sumb_cache[local_x];
                    }
                }
            }
            // if (tx == 0) {
            //     for (int i = 0; i < 64; i++) {
            //         for (int j = 0; j < 64; j++) {
            //             int x = i, y = j;
            //             if (x < num_new && y < num_old) {
            //                 distances[x * num_old + y] = 
            //                     distances_cache[x * 64 + y];
            //             }
            //         }
            //     }
            // }
            __syncthreads();
            // __syncthreads();
            // if (global_base_ay < 16 && global_base_by < 16) {
            //     int flag = atomicCAS(&for_check, 0, 1);
            //     if (!flag) {
            //         printf("ffff %d %d %d\n", warp_id, global_base_ay, global_base_by);
            //         for (int i = 0; i < acc_frag[0].num_elements; i++) {
            //             printf("%f ", acc_frag[0].x[i]);
            //         } printf("\n\n");
            //         for (int i = 0; i < min((int)16, num_new); i++) {
            //             printf("%d ", new_neighbors[i]);
            //         } printf("\n\n");
            //         for (int i = 0; i < num_old; i++) {
            //             printf("%d ", old_neighbors[i]);
            //         } printf("\n\n");
            //         for (int i = 0; i < num_new; i++) {
            //             int x = new_neighbors[i];
            //             int y = old_neighbors[0];
            //             float suma = 0;
            //             half sumb = (half)0.0;
            //             for (int i = 0; i < VEC_DIM; i++) {
            //                 float a = vectors[x * VEC_DIM + i];
            //                 float b = vectors[y * VEC_DIM + i];
            //                 suma += a * b;
            //                 half aa = (half)a; half bb = (half)b;
            //                 sumb += aa * bb;
            //             }
            //             printf("Sum %d %f %f\n", i, suma, (float)sumb);
            //         }
            //         for (int i = 0; i < 64; i++) {
            //             for (int j = 0; j < 64; j++) {
            //                 printf("%f ", distances_cache[i * 64 + j]);
            //             } printf("\n");
            //         } printf("\n\n");
            //     }
            // }
            // __syncthreads();
            // if (lane_id == 0) {
            //     float *start = &distances_cache[(warp_id / 3) * 48 * 16 + (warp_id % 3) * 16];
            //     for (int i = 0; i < 16; i++) {
            //         for (int j = 0; j < 16; j++) {
            //             if (global_base_ay + i < num_new && global_base_by + j < num_old) {
            //                 distances[(global_base_ay + i) * num_old + global_base_by + j] = start[i * 48 + j];
            //             }
            //         }
            //     }
            // }
            // global_base_ay * num_old + global_base_by
        }
    }
}

__global__ void NewOldNeighborsCompareKernel(ResultElement *knn_graph, int *global_locks, 
                                             const float *vectors,
                                             const int *edges_new, const int *dest_new,
                                             const int num_new_max, 
                                             const int *edges_old, const int *dest_old,
                                             const int num_old_max) {
    extern __shared__ char buffer[];

    __shared__ float *distances;
    __shared__ int *neighbors, *local_locks;
    __shared__ ResultElement *knn_graph_cache;

    __shared__ int pos_gnew, pos_gold, num_new, num_old;

    __shared__ half shmem_a[BLOCK_ROW_WARPS * WMMA_M][BLOCK_COL_WARPS * WMMA_K];
    __shared__ half shmem_b[BLOCK_ROW_WARPS * WMMA_N][BLOCK_COL_WARPS * WMMA_K];

    int neighb_num_max = num_new_max + num_old_max;
    int tx = threadIdx.x;
    if (tx == 0) {
        distances = (float *)buffer;
        neighbors = 
            (int *)((char *)buffer + (num_new_max * num_old_max) * sizeof(float));
        local_locks = 
            (int *)((char *)neighbors + neighb_num_max * sizeof(int));
        knn_graph_cache =
            (ResultElement *)((char *)local_locks + neighb_num_max * sizeof(int));
    }
    __syncthreads();

    int list_id = blockIdx.x;
    int block_dim_x = blockDim.x;

    if (tx < neighb_num_max) {
        local_locks[tx] = 0;
    }

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

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("new-old calc. num. %d %d %d\n", num_new, num_old, neighb_num);
    } __syncthreads();

    GetDistancesWMMA(distances, vectors, 
                     neighbors, num_new, neighbors + num_new, num_old,
                     shmem_a, shmem_b);
    __syncthreads();
    // if (num_old >= 2) {
    //     int flag = atomicCAS(&for_check, 0, 1);
    //     if (!flag) {
    //         printf("\nCheck distance %d %d: \n", num_new, num_old);
    //         for (int i = 0; i < num_new; i++) {
    //             for (int j = 0; j < num_old; j++) {
    //                 float distance = 0;
    //                 int x = *(neighbors + i);
    //                 int y = *(neighbors + j + num_new);
    //                 for (int k = 0; k < VEC_DIM; k++) {
    //                     float diff = vectors[x * VEC_DIM + k] - 
    //                                  vectors[y * VEC_DIM + k];
    //                     distance += diff * diff;
    //                 }
    //                 printf("%f ", distance);
    //             } 
    //         } printf("\n\n");
    //         for (int i = 0; i < num_new * num_old; i++) {
    //             printf("%f ", distances[i]);
    //         } printf("\n\n");
    //     }
    // }
    // return;
    int calc_num = num_new * num_old;
    // int num_it = GetItNum(NEIGHB_NUM_PER_LIST, NEIGHB_CACHE_NUM);
    const int num_it = 1;
    for (int i = 0; i < num_it; i++) {
        // Read list to cache
        int num_it2 = GetItNum(neighb_num * NEIGHB_CACHE_NUM, block_dim_x);
        for (int j = 0; j < num_it2; j++) {
            int pos = j * block_dim_x + tx;
            if (pos < neighb_num * NEIGHB_CACHE_NUM)
                knn_graph_cache[pos] = ResultElement(1e10, 33333333);
        }
        int list_size = NEIGHB_CACHE_NUM;
        int num_it3 = GetItNum(neighb_num, block_dim_x / THREADS_PER_LIST);
        for (int j = 0; j < num_it3; j++) {
            int list_id = j * (block_dim_x / THREADS_PER_LIST) + 
                          tx / THREADS_PER_LIST;
            if (list_id >= neighb_num) continue;
            if (list_id < num_new) {
                UpdateLocalNewKNNLists(knn_graph_cache, list_id, list_size, 
                                       neighbors + num_new, num_old, 
                                       distances, calc_num);
            } else {
                UpdateLocalOldKNNLists(knn_graph_cache, list_id, list_size, 
                                       neighbors, num_new, 
                                       neighbors + num_new, num_old, 
                                       distances, calc_num, vectors);
            }
       }
        __syncthreads();
        MergeLocalGraphWithGlobalGraph(knn_graph_cache, list_size, neighbors,
                                       neighb_num, knn_graph, global_locks);
        __syncthreads();
    }
}

pair<int*, int*> ReadGraphToGlobalMemory(const Graph& graph) {
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

    cuda_status0 = cudaMemcpy(edges_dev, edges.data(), 
                              edges.size() * sizeof(int), cudaMemcpyHostToDevice);
    cuda_status1 = cudaMemcpy(dest_dev, dest.data(), 
                              dest.size() * sizeof(int), cudaMemcpyHostToDevice);
    if (cuda_status0 != cudaSuccess || cuda_status1 != cudaSuccess) {
        cerr << "CudaMemcpy failed" << endl;
        exit(-1);
    }
    return make_pair(edges_dev, dest_dev);
}

__global__ void TestKernel(ResultElement* knn_graph) {
    for (int i = 0; i < 10000 * 30; i++) {
        if (knn_graph[i].distance == 0 && knn_graph[i].label == 0) {
            printf("check %d %f\n", i, knn_graph[i].distance);
        }
    }
    return;
}

ResultElement* ReadKNNGraphToGlobalMemory(const vector<vector<gpuknn::NNDItem>> &knn_graph) {
    int k = knn_graph[0].size();
    ResultElement *knn_graph_dev;
    ResultElement *knn_graph_host = new ResultElement[knn_graph.size() * k];
    int idx = 0;
    for (int i = 0; i < knn_graph.size(); i++) {
        for (int j = 0; j < k; j++) {
            const auto &item = knn_graph[i][j];
            knn_graph_host[idx++] = ResultElement(item.distance, item.id);
        }
    }

    auto cuda_status = cudaMalloc(&knn_graph_dev, 
                                  knn_graph.size() * k * sizeof(ResultElement));
    if (cuda_status != cudaSuccess) {
        cerr << "knn_graph cudaMalloc failed." << endl;
        exit(-1);
    }
    cuda_status = cudaMemcpy(knn_graph_dev, knn_graph_host, 
                             knn_graph.size() * k * sizeof(ResultElement), 
                             cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        cerr << cudaGetErrorString(cuda_status) << endl;
        cerr << "knn_graph cudaMemcpyHostToDevice failed." << endl;
        exit(-1);
    }
    delete [] knn_graph_host;
    return knn_graph_dev;
}

void ToHostGraph(vector<vector<gpuknn::NNDItem>> *origin_knn_graph_ptr,
                 const ResultElement *knn_graph, const int size, const int neighb_num) {
    auto &origin_knn_graph = *origin_knn_graph_ptr;
    vector<gpuknn::NNDItem> neighb_list;
    for (int i = 0; i < size; i++) {
        neighb_list.clear();
        for (int j = 0; j < neighb_num; j++) {
            ResultElement tmp = knn_graph[i * neighb_num + j];
            neighb_list.emplace_back(tmp.label, false, tmp.distance);
        }
        for (int j = 0; j < neighb_num; j++) {
            for (int k = 0; k < neighb_num; k++) {
                if (neighb_list[j].id == origin_knn_graph[i][k].id) {
                    neighb_list[j].visited = true;
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

void UpdateGraph(vector<vector<gpuknn::NNDItem>> *origin_knn_graph_ptr, 
                 float* vectors_dev, 
                 const Graph& newg, const Graph& oldg, const int k) {
    auto &origin_knn_graph = *origin_knn_graph_ptr;

    int *edges_dev_new, *dest_dev_new;
    tie(edges_dev_new, dest_dev_new) = ReadGraphToGlobalMemory(newg);

    int *edges_dev_old, *dest_dev_old;
    tie(edges_dev_old, dest_dev_old) = ReadGraphToGlobalMemory(oldg);

    size_t g_size = newg.size();

    cudaError_t cuda_status;
    ResultElement *knn_graph_dev, *knn_graph = new ResultElement[g_size * k];
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

    dim3 block_size(512);
    dim3 grid_size(g_size);
    // cerr << "Start kernel." << endl;
    const int num_new_max = GetMaxListSize(newg);
    const int num_old_max = GetMaxListSize(oldg);
    size_t shared_memory_size = 
        num_new_max * VEC_DIM * sizeof(float) + 
        (num_new_max * (num_new_max - 1) / 2) * sizeof(float) +
        num_new_max * 2 * sizeof(int) + 
        num_new_max * NEIGHB_CACHE_NUM * sizeof(ResultElement);

    NewNeighborsCompareKernel<<<grid_size, block_size, shared_memory_size>>>
        (knn_graph_dev, global_locks_dev, vectors_dev,
         edges_dev_new, dest_dev_new, num_new_max);
    cudaDeviceSynchronize();

    block_size = dim3(128);
    int neighb_num_max = num_new_max + num_old_max;
    shared_memory_size = (num_new_max * num_old_max) * sizeof(float) + 
                         neighb_num_max * 2 * sizeof(int) + 
                         neighb_num_max * NEIGHB_CACHE_NUM * sizeof(ResultElement);
    cerr << "shmem kernel2 size: " << shared_memory_size << endl;
    auto start = chrono::steady_clock::now();
    NewOldNeighborsCompareKernel<<<grid_size, block_size, shared_memory_size>>>
        (knn_graph_dev, global_locks_dev, vectors_dev, 
         edges_dev_new, dest_dev_new, num_new_max,
         edges_dev_old, dest_dev_old, num_old_max);

    // LocalDistCompareKernel<<<grid_size, block_size>>>(knn_graph_dev, 
    //                                                   global_locks_dev,
    //                                                   vectors_dev,
    //                                                   edges_dev_new, dest_dev_new, 
    //                                                   edges_dev_old, dest_dev_old);
    cudaDeviceSynchronize();
    auto end = chrono::steady_clock::now();
    cerr << "Kernel 2 costs: "
         << (float)chrono::duration_cast<chrono::microseconds>(end - start).count() / 1e6
         << endl;

    cuda_status = cudaGetLastError();

    if (cuda_status != cudaSuccess) {
        cerr << cudaGetErrorString(cuda_status) << endl;
        cerr << "Kernel failed" << endl;
        exit(-1);
    }
    // cerr << "End kernel." << endl;
    cuda_status = cudaMemcpy(knn_graph, knn_graph_dev, 
                             g_size * k * sizeof(ResultElement), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        cerr << cudaGetErrorString(cuda_status) << endl;
        cerr << "knn_graph cudaMemcpy failed" << endl;
        exit(-1);
    }

    ToHostGraph(&origin_knn_graph, knn_graph, g_size, k);

    delete [] knn_graph;
    cudaFree(edges_dev_new);
    cudaFree(dest_dev_new);
    cudaFree(edges_dev_old);
    cudaFree(dest_dev_old);
    cudaFree(knn_graph_dev);
}

namespace gpuknn {
    vector<vector<NNDItem>> NNDescent(const float* vectors, const int vecs_size, const int vecs_dim) {
        int k = NEIGHB_NUM_PER_LIST;
        int iteration = 6;
        auto cuda_status = cudaSetDevice(DEVICE_ID);

        float* vectors_dev;
        cudaMalloc(&vectors_dev, (size_t)vecs_size * vecs_dim * sizeof(float));
        cudaMemcpy(vectors_dev, vectors, 
                   (size_t)vecs_size * vecs_dim * sizeof(float),
                   cudaMemcpyHostToDevice);

        if (cuda_status != cudaSuccess) {
            cerr << cudaGetErrorString(cuda_status) << endl;
            cerr << "cudaSetDevice failed" << endl;
            exit(-1);
        }
        Graph result(vecs_size);
        vector<vector<NNDItem>> g(vecs_size);
        vector<int> tmp_vec;

        for (int i = 0; i < vecs_size; i++) {
            vector<int> exclusion = {i};
            xmuknn::GenerateRandomSequence(tmp_vec, k, vecs_size, exclusion);
            for (int j = 0; j < k; j++) {
                int nb_id = tmp_vec[j];
                g[i].emplace_back(nb_id, false, 1e10);
            }
        }

        #pragma omp parallel for
        for (int i = 0; i < vecs_size; i++) {
            for (int j = 0; j < k; j++) {
                g[i][j].distance = 
                    GetDistance(vectors + (size_t)i * vecs_dim, 
                                vectors + (size_t)g[i][j].id * vecs_dim,
                                vecs_dim);
            }
        }

        #pragma omp parallel for
        for (int i = 0; i < g.size(); i++) {
            sort(g[i].begin(), g[i].end(), [](NNDItem a, NNDItem b) {
                    if (fabs(a.distance - b.distance) < 1e-10) return a.id < b.id;
                    return a.distance < b.distance;
                 });
        }

        float kernel_costs = 0;
        Graph newg, oldg;
        float get_nb_graph_time = 0;
        auto sum_start = chrono::steady_clock::now();
        for (int t = 0; t < iteration; t++) {
            cerr << "Start generating NBGraph." << endl;
            auto start = chrono::steady_clock::now();
            GetNBGraph(&newg, &oldg, g, vectors, vecs_size, vecs_dim);
            auto end = chrono::steady_clock::now();
            float tmp_time = 
                (float)chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e6;
            get_nb_graph_time += tmp_time;
            cerr << "GetNBGraph costs "
                 << tmp_time
                 << endl;

            start = chrono::steady_clock::now();
            vector<pair<float, int>> tmp_result;
            // long long update_times = 0;
            UpdateGraph(&g, vectors_dev, newg, oldg, k);
            end = chrono::steady_clock::now();
            float kernel_tmp_costs = (float)chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e6;
            kernel_costs += kernel_tmp_costs;
            cerr << "Kernel costs "
                 << kernel_tmp_costs
                 << endl;
            cerr << endl;
        }
        auto sum_end = chrono::steady_clock::now();
        float sum_costs = (float)chrono::duration_cast<std::chrono::microseconds>(sum_end - sum_start).count() / 1e6;
        // sift10k in cpu should be 0.6s;
        cerr << "All kernel costs: " << kernel_costs << endl;
        cerr << "Get NB graph costs: " << get_nb_graph_time << endl; 
        cerr << "All procedure costs: " << sum_costs << endl;
        return g;
    }
}

#endif