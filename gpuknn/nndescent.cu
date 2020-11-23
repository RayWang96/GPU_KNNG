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

pair<Graph, Graph> GetNBGraph(vector<vector<gpuknn::NNDItem>>& knn_graph, 
                              const float *vectors, const int vecs_size, 
                              const int vecs_dim) {
    int sample_num = 30;
    Graph graph_new, graph_rnew, graph_old, graph_rold;
    graph_new = graph_rnew = graph_old = graph_rold = Graph(knn_graph.size());
    for (int i = 0; i < knn_graph.size(); i++) {
        int cnt = 0;
        int last_cnt = 0;
        while (cnt < sample_num) {
            for (int j = 0; j < knn_graph[i].size(); j++) {
                auto& item = knn_graph[i][j];
                if (item.id >= LARGE_INT) continue;
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
            if (last_cnt == cnt) break;
            last_cnt = cnt;
        }
    }

    for (int i = 0; i < knn_graph.size(); i++) {
        for (int j = 0; j < graph_new[i].size(); j++) {
            auto& id = graph_new[i][j];
            graph_rnew[id].push_back(i);
        }
        for (int j = 0; j < graph_old[i].size(); j++) {
            auto& id = graph_old[i][j];
            graph_rold[id].push_back(i);
        }
    }

    for (int i = 0; i < knn_graph.size(); i++) {
        random_shuffle(graph_rnew[i].begin(), graph_rnew[i].end());
        random_shuffle(graph_rold[i].begin(), graph_rold[i].end());
    }

    for (int i = 0; i < knn_graph.size(); i++) {
        int cnt = 0;
        int last_cnt = 0;
        while (cnt < sample_num) {
            for (int j = 0; j < graph_rnew[i].size(); j++) {
                int x = graph_rnew[i][j];
                if (x >= LARGE_INT) continue;
                cnt++;
                graph_new[i].push_back(x);
                if (cnt >= sample_num) break;
            }
            if (cnt == last_cnt) break;
            last_cnt = cnt;
        }
        cnt = 0;
        last_cnt = 0;
        while (cnt < sample_num) {
            for (int j = 0; j < graph_rold[i].size(); j++) {
                int x = graph_rold[i][j];
                if (x >= LARGE_INT) continue;
                cnt++;
                graph_old[i].push_back(x);
                if (cnt >= sample_num) break;
            }
            if (cnt == last_cnt) break;
            last_cnt = cnt;
        }
    }
    
    for (int i = 0; i < knn_graph.size(); i++) {
        sort(graph_new[i].begin(), graph_new[i].end());
        graph_new[i].erase(unique(graph_new[i].begin(), 
                                  graph_new[i].end()), graph_new[i].end());

        sort(graph_old[i].begin(), graph_old[i].end());
        graph_old[i].erase(unique(graph_old[i].begin(), 
                                  graph_old[i].end()), graph_old[i].end());
    }
    return make_pair(graph_new, graph_old);
}

__device__ int GetItNum(const int sum_num, const int num_per_it) {
    return sum_num / num_per_it + (sum_num % num_per_it != 0);
}

__device__ void Swap(int &a, int &b) {
    int c = a;
    a = b;
    b = c;
}

const int VEC_DIM = 128;
const int NEIGHB_NUM_PER_LIST = 30;
const int VECS_NUM_PER_BLOCK = NEIGHB_NUM_PER_LIST * 2;
const int MAX_CALC_NUM = (VECS_NUM_PER_BLOCK * (VECS_NUM_PER_BLOCK-1)) / 2;
const int NEIGHB_CACHE_NUM = 16;
const int TILE_WIDTH = 16;
// const int HALF_NEIGHB_CACHE_NUM = NEIGHB_CACHE_NUM / 2;

// __device__ int InsertToLocalKNNList(ResultElement *knn_list, 
//                                     const int list_size,
//                                     const ResultElement &element,
//                                     int *local_lock_ptr, 
//                                     int *position_to_insert_ptr) {
//     int tx = threadIdx.x;
//     int t_pos = tx % HALF_NEIGHB_CACHE_NUM * 2;
//     int threads_id = tx / HALF_NEIGHB_CACHE_NUM;
//     if (threads_id >= VECS_NUM_PER_BLOCK) return -1;
//     int &local_lock = *local_lock_ptr;
//     int &position_to_insert = *position_to_insert_ptr;
//     int position_result = -1;
//     if (element >= knn_list[NEIGHB_CACHE_NUM - 1]) return position_result;
//     for (int i = 0; i < 2; i++) {
//         int loop_flag = -1;
//         // spin lock;
//         do {
//             bool is_first_thread = (tx % HALF_NEIGHB_CACHE_NUM == 0);
//             if (is_first_thread) {
//                 atomicCAS(&local_lock, -1, threads_id);
//             }
//             loop_flag = local_lock;
//             if (loop_flag == threads_id) {
//                 if (element < knn_list[0]) {
//                     position_to_insert = 0;
//                 } else if (element == knn_list[0]) {
//                     position_to_insert = -1;
//                 }
//                 int offset = i & 1;
//                 if (t_pos + 1 + offset >= list_size) ;
//                 else if (element == knn_list[t_pos + offset] ||
//                          element == knn_list[t_pos + 1 + offset]) {
//                     position_to_insert = -1;
//                 }
//                 else if (element > knn_list[t_pos + offset] && 
//                          element < knn_list[t_pos + 1 + offset]) {
//                     // Insert the element before this position.
//                     position_to_insert = t_pos + 1 + offset;
//                 }
//                 // It's neccesarry when HALF_NEIGHB_CACHE_NUM > the size of warp
//                 __threadfence_block(); 

//                 if (position_to_insert != -1) {
//                     ResultElement even_pos_element, front_element;
//                     if (t_pos >= position_to_insert && t_pos < list_size) {
//                         if (t_pos != 0)
//                             front_element = knn_list[t_pos-1];
//                         else 
//                             front_element = knn_list[0];
//                         even_pos_element = knn_list[t_pos];
//                         __threadfence_block(); 
//                         knn_list[t_pos] = front_element;
//                     }
//                     if (t_pos < position_to_insert && 
//                         t_pos + 1 > position_to_insert &&
//                         t_pos < list_size) {
//                         even_pos_element = knn_list[t_pos];
//                     }
//                     if (t_pos + 1 > position_to_insert && t_pos + 1 < list_size) {
//                         knn_list[t_pos+1] = even_pos_element;
//                     }
//                     knn_list[position_to_insert] = element;
//                 }
//                 position_result = position_to_insert;
//                 position_to_insert = -1;
//                 __threadfence_block();
//             }
//             if (loop_flag != -1) atomicExch(&local_lock, -1);
//         } while (loop_flag == -1);
//         if (position_result != -1) {
//             break;
//         }
//     }
//     return position_to_insert;
// }

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
            if (cnt >= k) return;
            while (i < m && A[i] <= C[cnt-1]) i++;
            while (j < n && B[j] <= C[cnt-1]) j++;
        } else {
            C[cnt++] = B[j++];
            if (cnt >= k) return;
            while (i < m && A[i] <= C[cnt-1]) i++;
            while (j < n && B[j] <= C[cnt-1]) j++;
        }
    }

    if (i == m) {
        for (; j < n; j++) {
            if (B[j] > C[cnt-1]) {
                C[cnt++] = B[j];
            }
            if (cnt >= k) return;
        }
        for (; i < m; i++) {
            if (A[i] > C[cnt-1]) {
                C[cnt++] = A[i];
            }
            if (cnt >= k) return;
        }
    } else {
        for (; i < m; i++) {
            if (A[i] > C[cnt-1]) {
                C[cnt++] = A[i];
            }
            if (cnt >= k) return;
        }
        for (; j < n; j++) {
            if (B[j] > C[cnt-1]) {
                C[cnt++] = B[j];
            }
            if (cnt >= k) return;
        }
    }
}

__device__ void MergeLocalGraphWithGlobalGraph(const ResultElement* local_knn_graph,
                                               const int list_size, const int* neighb_ids,
                                               const int neighb_num,
                                               ResultElement* global_knn_graph,
                                               int* global_locks) {
    int tx = threadIdx.x;
    if (tx < neighb_num) {
        ResultElement C_cache[NEIGHB_NUM_PER_LIST + NEIGHB_CACHE_NUM];
        int neighb_id = neighb_ids[tx];
        bool loop_flag = false;
        do {
            if (loop_flag = atomicCAS(&global_locks[neighb_id], 0, 1) == 0) {
                UniqueMergeSequential(&local_knn_graph[tx * NEIGHB_CACHE_NUM], 
                                      NEIGHB_CACHE_NUM, 
                                      &global_knn_graph[neighb_ids[tx] * NEIGHB_NUM_PER_LIST],
                                      NEIGHB_NUM_PER_LIST, C_cache, NEIGHB_NUM_PER_LIST);
                for (int i = 0; i < NEIGHB_NUM_PER_LIST; i++) {
                    global_knn_graph[neighb_ids[tx] * NEIGHB_NUM_PER_LIST + i]
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

    num_it = GetItNum(NEIGHB_NUM_PER_LIST, NEIGHB_CACHE_NUM);
    for (int i = 0; i < num_it; i++) {
        int num_it2 = GetItNum(neighb_num * NEIGHB_CACHE_NUM, block_dim_x);
        for (int j = 0; j < num_it2; j++) {
            int pos = j * block_dim_x + tx;
            if (pos < neighb_num * NEIGHB_CACHE_NUM)
                knn_graph_cache[pos] = ResultElement(1e10, 0x3f3f3f3f);
        }
        int list_size = 
            i == num_it - 1 ? NEIGHB_NUM_PER_LIST
                 % NEIGHB_CACHE_NUM : NEIGHB_CACHE_NUM;
        int no = tx / NEIGHB_CACHE_NUM;
        if (no >= neighb_num) continue;
        __syncthreads();
        int lists_per_it = block_dim_x;
        num_it2 = GetItNum(calc_num, lists_per_it); 
        for (int j = 0; j < num_it2; j++) {
            no = j * lists_per_it + tx;
            if (no >= calc_num) continue;
            int idx = no + 1;
            int x = ceil(sqrt(2 * idx + 0.25) - 0.5);
            int y = idx - (x - 1) * x / 2 - 1;
            if (x >= neighb_num || y >= neighb_num) continue;
            Swap(x, y);
            if (neighbors[x] == neighbors[y]) continue;
            ResultElement *list_x = &knn_graph_cache[x * NEIGHB_CACHE_NUM];
            ResultElement *list_y = &knn_graph_cache[y * NEIGHB_CACHE_NUM];

            ResultElement re_xy = ResultElement(distances[no], neighbors[y]);
            ResultElement re_yx = ResultElement(distances[no], neighbors[x]);

            InsertToLocalKNNList(list_x, list_size, re_xy, &local_locks[x]);
            InsertToLocalKNNList(list_y, list_size, re_yx, &local_locks[y]);
            __syncthreads();
        }
        MergeLocalGraphWithGlobalGraph(knn_graph_cache, list_size, neighbors,
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
        // if ((row_new + t_row) * num_old + row_old + t_col == 35) {
        //     printf("%d %d %f\n", tx, i, distance);
        // }
        if (distance != -1.0)
            distances[(row_new + t_row) * num_old + row_old + t_col] = distance;
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
    }

    GetNewOldDistances(distances, vectors, 
                       neighbors, num_new, neighbors + num_new, num_old);
    __syncthreads();

    int calc_num = num_new * num_old;
    int num_it = GetItNum(NEIGHB_NUM_PER_LIST, NEIGHB_CACHE_NUM);
    for (int i = 0; i < num_it; i++) {
        // Read list to cache
        int num_it2 = GetItNum(neighb_num * NEIGHB_CACHE_NUM, block_dim_x);
        for (int j = 0; j < num_it2; j++) {
            int pos = j * block_dim_x + tx;
            if (pos < neighb_num * NEIGHB_CACHE_NUM)
                knn_graph_cache[pos] = ResultElement(1e10, 0x3f3f3f3f);
        }
        int list_size = i == num_it - 1 ? 
                        NEIGHB_NUM_PER_LIST % NEIGHB_CACHE_NUM : NEIGHB_CACHE_NUM;
        int no = tx / NEIGHB_CACHE_NUM;
        if (no >= neighb_num) continue;
        __syncthreads();
        //Update the partial list
        int lists_per_it = block_dim_x;
        num_it2 = GetItNum(calc_num, lists_per_it); 
        // 1520, 1024 = 2
        for (int j = 0; j < num_it2; j++) {
            no = j * lists_per_it + tx;
            if (no >= calc_num) continue;
            int idx = no;
            int x = idx / num_old;
            int y = idx % num_old + num_new;
            if (x >= neighb_num || y >= neighb_num) continue;
            Swap(x, y); // Reduce threads confliction
            if (neighbors[x] == neighbors[y]) continue;
            ResultElement *list_x = &knn_graph_cache[x * NEIGHB_CACHE_NUM];
            ResultElement *list_y = &knn_graph_cache[y * NEIGHB_CACHE_NUM];

            ResultElement re_xy = ResultElement(distances[no], neighbors[y]);
            ResultElement re_yx = ResultElement(distances[no], neighbors[x]);
            InsertToLocalKNNList(list_x, list_size, re_xy, &local_locks[x]);
            InsertToLocalKNNList(list_y, list_size, re_yx, &local_locks[y]);
            __syncthreads();
        }
        MergeLocalGraphWithGlobalGraph(knn_graph_cache, list_size, neighbors,
                                       neighb_num, knn_graph, global_locks);
        __syncthreads();
    }
}

// __global__ void LocalDistCompareKernel(ResultElement *knn_graph, int *global_locks,
//                                        const float* vectors,
//                                        const int *edges_new, const int *dest_new, 
//                                        const int *edges_old, const int *dest_old) {
//     //shared memory limit is 96KB
//     __shared__ float shared_vectors[VECS_NUM_PER_BLOCK][VEC_DIM]; 
//     // 60 * 128 * 4 = 30720B
//     __shared__ int pos_gnew, pos_gold; 
//     // 4B
//     __shared__ int neighbors[VECS_NUM_PER_BLOCK]; 
//     // 60 * 4 = 240B
//     __shared__ int num_new, num_old; 
//     // 8B
//     __shared__ float distances[MAX_CALC_NUM]; 
//     // 1770 * 4 = 7080B
//     __shared__ ResultElement knn_graph_cache[VECS_NUM_PER_BLOCK * NEIGHB_CACHE_NUM]; 
//     // 60 * 15 * 8 = 7200B
//     __shared__ int local_locks[VECS_NUM_PER_BLOCK];
//     // 60 * 4 = 240B
//     // 30720 + 4 + 240 + 8 + 7080 + 7200 + 240 = 45732;

//     // Initiate 
//     int list_id = blockIdx.x;
//     int tx = threadIdx.x; // blockDim.x == 1024

//     if (tx < VECS_NUM_PER_BLOCK) {
//         local_locks[tx] = 0;
//     }

//     if (tx == 0) {
//         int next_pos = edges_new[list_id + 1];
//         int now_pos = edges_new[list_id];
//         num_new = next_pos - now_pos;
//         pos_gnew = now_pos;
//     } else if (tx == 32) {
//         int next_pos = edges_old[list_id + 1];
//         int now_pos = edges_old[list_id];
//         num_old = next_pos - now_pos;
//         pos_gold = now_pos;
//     }
//     __syncthreads();
//     int neighb_num = num_new + num_old;
//     // Read all neighbors to shared memory.
//     if (tx < num_new) {
//         neighbors[tx] = dest_new[pos_gnew + tx];
//     } else if (tx >= num_new && tx < neighb_num) {
//         neighbors[tx] = dest_old[pos_gold + tx - num_new];
//     }
//     __syncthreads();

//     // Read needed vectors to shared memory.
//     //operation per iteration
//     int num_vec_per_it = blockDim.x / VEC_DIM; // 1024 / 128 = 8
//     int num_it = GetItNum(neighb_num, num_vec_per_it); // 60, 8 = 8
//     for (int i = 0; i < num_it; i++) {
//         int x = i * num_vec_per_it + tx / VEC_DIM;
//         if (x >= neighb_num) continue;
//         int y = tx % VEC_DIM;
//         int vec_id = neighbors[x];
//         shared_vectors[x][y] = vectors[vec_id * VEC_DIM + y];
//     }

//     // Calculate distances.
//     int calc_new_num = (num_new * (num_new - 1)) / 2;
//     int calc_new_old_num = num_new * num_old;
//     int calc_num = calc_new_num + calc_new_old_num;
//     num_it = GetItNum(calc_num, blockDim.x);
//     if (blockIdx.x == 0 && threadIdx.x == 0) {
//         printf("check calc. num. %d %d %d %d %d %d\n", 
//                num_new, num_old, neighb_num, calc_new_num, calc_new_old_num, calc_num);
//     }
//     for (int i = 0; i < num_it; i++) {
//         int x = i * blockDim.x + tx;
//         if (x < calc_num) {
//             distances[x] = 0;
//         }
//     }
//     __syncthreads();
//     num_vec_per_it = blockDim.x; // 1024 / 128 = 8
//     num_it = GetItNum(calc_num, num_vec_per_it); // 1520 / 8 = 190
//     for (int i = 0; i < num_it; i++) {
//         int no = i * num_vec_per_it + tx; // xth distance calculation
//         if (no >= calc_num) continue;
//         int x, y;
//         if (no < calc_new_num) {
//             int idx = no + 1; // To fit the following formula
//             x = ceil(sqrt(2 * idx + 0.25) - 0.5);
//             y = idx - (x - 1) * x / 2 - 1;
//         } else {
//             int idx = no - calc_new_num;
//             x = idx / num_old;
//             y = idx % num_old + num_new;
//         }
//         if (x >= neighb_num || y >= neighb_num) continue;
//         float sum = 0;
//         for (int j = 0; j < VEC_DIM; j++) {
//             float diff = shared_vectors[x][j] - shared_vectors[y][j];
//             sum += diff * diff;
//         }
//         distances[no] = sum;
//     }
//     __syncthreads();
//     // Update the local graph.
//     num_it = GetItNum(NEIGHB_NUM_PER_LIST, NEIGHB_CACHE_NUM);
//     for (int i = 0; i < num_it; i++) {
//         // Read list to cache
//         int num_it2 = GetItNum(neighb_num * NEIGHB_CACHE_NUM, blockDim.x);
//         for (int j = 0; j < num_it2; j++) {
//             int pos = j * blockDim.x + tx;
//             if (pos < neighb_num * NEIGHB_CACHE_NUM)
//                 knn_graph_cache[pos] = ResultElement(1e10, 0x3f3f3f3f);
//         }
//         int list_size = 
//             i == num_it - 1 ? NEIGHB_NUM_PER_LIST
//                  % NEIGHB_CACHE_NUM : NEIGHB_CACHE_NUM;
//         int no = tx / NEIGHB_CACHE_NUM;
//         if (no >= neighb_num) continue;
//         __syncthreads();
//         //Update the partial list
//         int lists_per_it = blockDim.x;
//         num_it2 = GetItNum(calc_num, lists_per_it); 
//         // 1520, 1024 = 2
//         for (int j = 0; j < num_it2; j++) {
//             __syncthreads();
//             no = j * lists_per_it + tx;
//             if (no >= calc_num) continue;
//             int x, y;
//             if (no < calc_new_num) {
//                 int idx = no + 1; // To fit the following formula
//                 x = ceil(sqrt(2 * idx + 0.25) - 0.5);
//                 y = idx - (x - 1) * x / 2 - 1;
//             } else {
//                 int idx = no - calc_new_num;
//                 x = idx / num_old;
//                 y = idx % num_old + num_new;
//             }
//             if (x >= neighb_num || y >= neighb_num) continue;
//             Swap(x, y); // Reduce threads confliction
//             if (neighbors[x] == neighbors[y]) continue;
//             ResultElement *list_x = &knn_graph_cache[x * NEIGHB_CACHE_NUM];
//             ResultElement *list_y = &knn_graph_cache[y * NEIGHB_CACHE_NUM];

//             ResultElement re_xy = ResultElement(distances[no], neighbors[y]);
//             ResultElement re_yx = ResultElement(distances[no], neighbors[x]);

//             InsertToLocalKNNList(list_x, list_size, re_xy, &local_locks[x]);
//             InsertToLocalKNNList(list_y, list_size, re_yx, &local_locks[y]);
//         }
//         MergeLocalGraphWithGlobalGraph(knn_graph_cache, list_size, neighbors,
//                                        neighb_num, knn_graph, global_locks);
//         __syncthreads();
//     }
// }

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

    dim3 block_size(1024);
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

    block_size = dim3(TILE_WIDTH * TILE_WIDTH);
    int neighb_num_max = num_new_max + num_old_max;
    shared_memory_size = (num_new_max * num_old_max) * sizeof(float) + 
                         neighb_num_max * 2 * sizeof(int) + 
                         neighb_num_max * NEIGHB_CACHE_NUM * sizeof(ResultElement);

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
            xmuknn::GenerateRandomSequence(tmp_vec, k, vecs_size);
            for (int j = 0; j < k; j++) {
                int nb_id = tmp_vec[j];
                if (nb_id == i) {
                    int flag = 1;
                    while (flag) {
                        flag = 0;
                        nb_id++;
                        nb_id %= vecs_size;
                        for (int x : tmp_vec) {
                            if (x == nb_id) {
                                flag = 1;
                                break;
                            }
                        }
                        if (!flag) {
                            tmp_vec[j] = nb_id;
                        }
                    }
                }
                g[i].emplace_back(nb_id, false, 
                                  GetDistance(vectors + (size_t)i * vecs_dim, 
                                              vectors + (size_t)nb_id * vecs_dim,
                                              vecs_dim));
            }
        }
        for (int i = 0; i < g.size(); i++) {
            sort(g[i].begin(), g[i].end(), [](NNDItem a, NNDItem b) {
                    if (fabs(a.distance - b.distance) < 1e10) return a.id < b.id;
                    return a.distance < b.distance;
                });
        }

        Graph newg, oldg;
        float get_nb_graph_time = 0;
        for (int t = 0; t < iteration; t++) {
            cerr << "Start generating NBGraph." << endl;
            auto start = chrono::steady_clock::now();
            tie(newg, oldg) = GetNBGraph(g, vectors, vecs_size, vecs_dim);
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
            cerr << "Kernel costs "
                 << (float)chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e6
                 << endl;
            cerr << endl;
        }
        // sift10k in cpu should be 0.6s;
        cerr << "Get NB graph costs: " <<  get_nb_graph_time << endl; 
        return g;
    }
}

#endif