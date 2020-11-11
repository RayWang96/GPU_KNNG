#ifndef XMUKNN_NNDESCENT_CU
#define XMUKNN_NNDESCENT_CU

#include <vector>
#include <device_functions.h>
#include <iostream>
#include <assert.h>
#include <bitset>
#include <algorithm>
#include <cstring>
#include <tuple>
#include <utility>

#include "result_element.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gpudist.cuh"
#include "nndescent.cuh"
#include "../xmuknn.h"
#include "../tools/distfunc.hpp"

#ifdef __INTELLISENSE__
#include "../intellisense_cuda_intrinsics.h"
#endif

using namespace std;
using namespace xmuknn;
bitset<100000> visited;
#define DEVICE_ID 5

pair<Graph, Graph> GetNBGraph(vector<vector<gpuknn::NNDItem>>& knn_graph, 
                              const float *vectors, const int vecs_size, 
                              const int vecs_dim) {
    int sample_num = 32;
    Graph graph_new, graph_rnew, graph_old, graph_rold;
    graph_new = graph_rnew = graph_old = graph_rold = Graph(knn_graph.size());
    for (int i = 0; i < knn_graph.size(); i++) {
        visited.reset();
        int cnt = 0;
        int last_cnt = 0;
        while (cnt < sample_num) {
            for (int j = 0; j < knn_graph[i].size(); j++) {
                auto& item = knn_graph[i][j];
                if (visited[item.id]) continue;
                visited[item.id] = 1;
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
        visited.reset();
        int cnt = 0;
        int last_cnt = 0;
        while (cnt < sample_num) {
            for (int j = 0; j < graph_rnew[i].size(); j++) {
                int x = graph_rnew[i][j];
                if (visited[x]) continue;
                visited[x] = 1;
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
                if (visited[x]) continue;
                visited[x] = 1;
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

int UpdateNBs(vector<gpuknn::NNDItem> *nbs_ptr, 
              const pair<float, int> &p, const int k) {
    auto& nbs = *nbs_ptr;
    if (p.first > (*nbs.rbegin()).distance) return 0;
    int i = 0;
    while (i < nbs.size() && nbs[i].distance < p.first) {
        i++;
    }
    if (i >= k) return 0;
    if (i < nbs.size() && p.second == nbs[i].id) return 0;
    nbs.insert(nbs.begin() + i, gpuknn::NNDItem(p.second, false, p.first));
    nbs.pop_back();
    return 1;
}

long long UpdateNBGraph(vector<vector<gpuknn::NNDItem>>* graph_ptr, 
                        const vector<tuple<int, int, float>> &dist_info, int k) {
    auto& graph = *graph_ptr;
    long long sum = 0;
    for (const auto &dist_tuple : dist_info) {
        int list_id, nb_id;
        float dist;
        tie(list_id, nb_id, dist) = dist_tuple;
        UpdateNBs(&graph[list_id], make_pair(nb_id, dist), k);
        UpdateNBs(&graph[nb_id], make_pair(list_id, dist), k);
    }
    //cerr << index << " " << sum << endl;
    return sum;
}

__device__ int GetItNum(const int sum_num, const int num_per_it) {
    return sum_num / num_per_it + (sum_num % num_per_it != 0);
}

__device__ void Swap(int &a, int &b) {
    a ^= b;
    b ^= b;
    a ^= b;
}

const int VEC_DIM = 128;
const int VECS_PER_BLOCK = 64;
const int MAX_CALC_NUM = (32 * 31) / 2  + 32 * 32; // 1520
const int NEIGHB_NUM_PER_LIST = 32;
const int NEIGHB_CACHE_NUM = 16;
const int HALF_NEIGHB_CACHE_NUM = NEIGHB_CACHE_NUM / 2;

__device__ int InsertToLocalKNNList(ResultElement *knn_list, 
                                    const ResultElement &element,
                                    int *local_lock_ptr, 
                                    int *position_to_insert_ptr) {
    int tx = threadIdx.x;
    int t_pos = tx % HALF_NEIGHB_CACHE_NUM * 2;
    int threads_id = tx / HALF_NEIGHB_CACHE_NUM;
    int &local_lock = *local_lock_ptr;
    int &position_to_insert = *position_to_insert_ptr;
    int position_result = -1;
    if (element >= knn_list[NEIGHB_CACHE_NUM - 1]) return position_result;
    for (int i = 0; i < 2; i++) {
        int loop_flag = -1;
        // spin lock;
        do {
            bool is_first_thread = (tx % HALF_NEIGHB_CACHE_NUM == 0);
            if (is_first_thread) {
                atomicCAS(&local_lock, -1, threads_id);
            }
            loop_flag = local_lock;
            if (loop_flag == threads_id) {
                if (element <= knn_list[0]) {
                    position_to_insert = 0;
                }
                int offset = i & 1;
                if (t_pos + 1 + offset >= NEIGHB_CACHE_NUM) ;
                else if (element == knn_list[t_pos + offset] ||
                         element == knn_list[t_pos + 1 + offset]) {
                    position_to_insert = -1;
                }
                else if (element > knn_list[t_pos + offset] && 
                         element < knn_list[t_pos + 1 + offset]) {
                    // Insert the element before this position.
                    position_to_insert = t_pos + 1 + offset;
                }
                // It's neccesarry when HALF_NEIGHB_CACHE_NUM > the size of warp
                __threadfence_block(); 
                // printf("check %d %d\n", t_pos, position_to_insert);
                if (position_to_insert != -1) {
                    ResultElement even_pos_element, front_element;
                    if (t_pos >= position_to_insert) {
                        if (t_pos != 0)
                            front_element = knn_list[t_pos-1];
                        else 
                            front_element = knn_list[0];
                        even_pos_element = knn_list[t_pos];
                        __threadfence_block(); 
                        knn_list[t_pos] = front_element;
                    }
                    if (t_pos < position_to_insert && t_pos + 1 > position_to_insert) {
                        even_pos_element = knn_list[t_pos];
                    }
                    if (t_pos + 1 > position_to_insert) {
                        knn_list[t_pos+1] = even_pos_element;
                    }
                    knn_list[position_to_insert] = element;
                }
                position_result = position_to_insert;
                position_to_insert = -1;
                __threadfence_block();
            }
            if (loop_flag != -1) atomicExch(&local_lock, -1);
        } while (loop_flag == -1);
        if (position_result != -1) {
            break;
        }
    }
    return position_to_insert;
}

__device__ void UniqueMergeSequential(const ResultElement* A, const int m,
                                      const ResultElement* B, const int n,
                                      ResultElement* C, const int k) {
    int i = 0, j = 0, cnt = 0;
    while ((i < m) && (j < n)) {
        if (A[i] <= B[j]) {
            if (A[i] != C[cnt-1]) {
                C[cnt++] = A[i];
            }
            if (cnt >= k) return;
            i++;
        } else {
            if (B[j] != C[cnt-1]) {
                C[cnt++] = B[j];
            }
            if (cnt >= k) return;
            j++;
        }
    }

    if (i == m) {
        for (; j < n; j++) {
            if (B[j] != C[cnt-1]) {
                C[cnt++] = B[j];
            }
            if (cnt >= k) return;
        }
        for (; i < m; i++) {
            if (A[i] != C[cnt-1]) {
                C[cnt++] = A[i];
            }
            if (cnt >= k) return;
        }
    } else {
        for (; i < m; i++) {
            if (A[i] != C[cnt-1]) {
                C[cnt++] = A[i];
            }
            if (cnt >= k) return;
        }
        for (; j < n; j++) {
            if (B[j] != C[cnt-1]) {
                C[cnt++] = B[j];
            }
            if (cnt >= k) return;
        }
    }
}

__device__ void MergeLocalGraphWithGlobalGraph(const ResultElement* local_knn_graph,
                                               const int* neighb_ids,
                                               ResultElement* global_knn_graph,
                                               int* global_locks) {
    int tx = threadIdx.x;
    if (tx < VECS_PER_BLOCK) {
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

__global__ void LocalDistCompareKernel(ResultElement *knn_graph, int *global_locks,
                                       const float* vectors,
                                       const int *edges_new, const int *dest_new, 
                                       const int *edges_old, const int *dest_old) {
    //shared memory limit is 96KB
    __shared__ float shared_vectors[VECS_PER_BLOCK][VEC_DIM]; 
    // 64 * 128 * 4 = 32768B
    __shared__ int pos_gnew, pos_gold; 
    // 4B
    __shared__ int neighbors[VECS_PER_BLOCK]; 
    // 64 * 4 = 256B
    __shared__ int num_new, num_old; 
    // 8B
    __shared__ float distances[MAX_CALC_NUM]; 
    // 1520 * 4 = 6080B
    __shared__ ResultElement knn_graph_cache[VECS_PER_BLOCK * NEIGHB_CACHE_NUM]; 
    // 64 * 16 * 8 = 8192B
    __shared__ int local_locks[VECS_PER_BLOCK];
    // 64 * 4 = 256B
    __shared__ int positions_to_insert[VECS_PER_BLOCK];
    // 64 * 4 = 256B
    // 32768 + 4 + 256 + 8 + 6080 + 8192 + 256 + 256 = 47820;

    // Initiate 
    int list_id = blockIdx.x;
    int tx = threadIdx.x; // blockDim.x == 1024

    if (tx < VECS_PER_BLOCK) {
        local_locks[tx] = -1;
        positions_to_insert[tx] = -1;
    }

    if (tx == 0) {
        int next_pos = edges_new[list_id + 1];
        int now_pos = edges_new[list_id];
        num_new = next_pos - now_pos;
        pos_gnew = now_pos;
    } else if (tx == 1) {
        int next_pos = edges_old[list_id + 1];
        int now_pos = edges_old[list_id];
        num_old = next_pos - now_pos;
        pos_gold = now_pos;
    }
    __syncthreads();

    // Read all neighbors to shared memory.
    if (tx < num_new) {
        neighbors[tx] = dest_new[pos_gnew + tx];
    } else if (tx < num_old && tx < VECS_PER_BLOCK) {
        neighbors[tx] = dest_old[pos_gold + tx - num_new];
    }
    __syncthreads();

    // Read needed vectors to shared memory.
    //operation per iteration
    int num_vec_per_it = blockDim.x / VEC_DIM; // 1024 / 128 = 8
    int num_it = VECS_PER_BLOCK / num_vec_per_it; // 64 / 8 = 8
    if (tx < num_vec_per_it * VEC_DIM) {
        for (int i = 0; i < num_it; i++) {
            int x = i * num_it + tx / VEC_DIM;
            int y = tx % VEC_DIM;
            int vec_id = neighbors[x];
            shared_vectors[x][y] = vectors[vec_id * VEC_DIM + y];
        }
    }
    __syncthreads();

    // Calculate distances.
    int calc_new_num = (num_new * (num_new - 1)) / 2;
    int calc_new_old_num = num_new * num_old;
    int calc_num = calc_new_num + calc_new_old_num;
    // (32 * 31) / 2  + 32 * 32 = 1520
    num_it = GetItNum(calc_num, blockDim.x);
    for (int i = 0; i < num_it; i++) {
        int x = i * num_it + tx;
        if (x < calc_num) {
            distances[x] = 0;
        }
    }
    num_vec_per_it = blockDim.x / VEC_DIM; // 1024 / 128 = 8
    num_it = GetItNum(calc_num, num_vec_per_it); // 1520 / 8 = 190
    if (tx < num_vec_per_it * VEC_DIM) {
        for (int i = 0; i < num_it; i++) {
            int no = i * num_it + tx / VEC_DIM; // xth distance calculation
            int x, y;
            if (no < calc_new_num) {
                int idx = no + 1; // To fit the following formula
                x = ceil(sqrt(2 * idx + 0.25) - 0.5);
                y = idx - (x - 1) * x / 2 - 1;
            } else {
                int idx = no - calc_new_num;
                x = idx / num_old;
                y = idx % num_old + num_new;
            }
            int vec_pos = tx % VEC_DIM;
            float diff = shared_vectors[x][vec_pos] - shared_vectors[y][vec_pos];
            distances[no] += diff * diff;
        }
    }

    // Update the local graph.
    num_it = GetItNum(NEIGHB_NUM_PER_LIST, NEIGHB_CACHE_NUM);
    // blockDim.x / NEIGHB_CACHE_NUM = 1024 / 16 = 64
    for (int i = 0; i < num_it; i++) {
        // Read list to cache
        int no = tx / NEIGHB_CACHE_NUM;
        if (no >= num_new + num_old) continue;
        int neighb_id = neighbors[no];
        knn_graph_cache[no * NEIGHB_CACHE_NUM + tx % NEIGHB_CACHE_NUM]
            = knn_graph[neighb_id * NEIGHB_NUM_PER_LIST + 
                        tx % NEIGHB_CACHE_NUM + i * NEIGHB_CACHE_NUM];
        __syncthreads();

        //Update the partial list
        int lists_per_it = blockDim.x / HALF_NEIGHB_CACHE_NUM;
        int num_it2 = GetItNum(calc_num, lists_per_it); 
        // 1520, 128 = 12
        for (int j = 0; j < num_it2; j++) {
            __syncthreads();
            no = j * lists_per_it + tx / HALF_NEIGHB_CACHE_NUM;
            int x, y;
            if (no < calc_new_num) {
                int idx = no + 1; // To fit the following formula
                x = ceil(sqrt(2 * idx + 0.25) - 0.5);
                y = idx - (x - 1) * x / 2 - 1;
            } else {
                int idx = no - calc_new_num;
                x = idx / num_old;
                y = idx % num_old + num_new;
            }
            Swap(x, y); // Reduce threads confliction
            ResultElement *list_x = &knn_graph_cache[x * NEIGHB_CACHE_NUM];
            ResultElement *list_y = &knn_graph_cache[y * NEIGHB_CACHE_NUM];

            ResultElement re_xy = ResultElement(distances[no], y);
            ResultElement re_yx = ResultElement(distances[no], x);
            InsertToLocalKNNList(list_x, re_xy, &local_locks[x], 
                                 &positions_to_insert[x]);
            InsertToLocalKNNList(list_y, re_yx, &local_locks[y], 
                                 &positions_to_insert[y]);
        }
        MergeLocalGraphWithGlobalGraph(knn_graph_cache, neighbors,
                                       knn_graph, global_locks);
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

__global__ void TestKernel(ResultElement *knn_graph) {
    for (int i = 0; i < NEIGHB_NUM_PER_LIST; i++) {
        printf("check %f\n", knn_graph[i].distance);
    }
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
    TestKernel<<<dim3(1), dim3(1)>>> (knn_graph_dev);
    cudaDeviceSynchronize();
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
    cudaMalloc(&global_locks_dev, g_size);
    vector<int> zeros(g_size);
    cudaMemcpy(global_locks_dev, zeros.data(), g_size * sizeof(int),
               cudaMemcpyHostToDevice);

    dim3 block_size(1024);
    dim3 grid_size(g_size);
    // cerr << "Start kernel." << endl;
    LocalDistCompareKernel<<<grid_size, block_size>>>(knn_graph_dev, 
                                                      global_locks_dev,
                                                      vectors_dev,
                                                      edges_dev_new, dest_dev_new, 
                                                      edges_dev_old, dest_dev_old);
    cudaDeviceSynchronize();

    cuda_status = cudaGetLastError();

    if (cuda_status != cudaSuccess) {
        runtime_error("Kernel failed.");
    }

    // cerr << "End kernel." << endl;
    cuda_status = cudaMemcpy(knn_graph_dev, knn_graph, 
                             g_size * k * sizeof(int), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        runtime_error("knn_graph cudaMemcpy failed.");
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
    Graph NNDescent(const float* vectors, const int vecs_size, const int vecs_dim) {
        int k = NEIGHB_NUM_PER_LIST;
        int iteration = 3;
        auto cuda_status = cudaSetDevice(DEVICE_ID);

        float* vectors_dev;
        cudaMalloc(&vectors_dev, (size_t)vecs_size * sizeof(float));
        cudaMemcpy(vectors_dev, vectors, (size_t)vecs_size * sizeof(float),
                   cudaMemcpyHostToDevice);

        if (cuda_status != cudaSuccess) {
            runtime_error("cudaSetDevice failed.");
        }
        Graph result(vecs_size);
        vector<vector<NNDItem>> g(vecs_size);
        vector<int> tmp_vec;

        for (int i = 0; i < vecs_size; i++) {
            xmuknn::GenerateRandomSequence(tmp_vec, k, vecs_size);
            for (int j = 0; j < k; j++) {
                int nb_id = tmp_vec[j];
                if (nb_id == i) continue;
                g[i].emplace_back(nb_id, false, 
                                  GetDistance(vectors + (size_t)i * vecs_dim, 
                                              vectors + (size_t)nb_id * vecs_dim,
                                              vecs_dim));
            }
        }
        for (int i = 0; i < g.size(); i++) {
            sort(g[i].begin(), g[i].end(), [](NNDItem a, NNDItem b) {
                    return a.distance < b.distance;
                });
        }

        Graph newg, oldg;
        for (int t = 0; t < iteration; t++) {
            auto start = clock();

            tie(newg, oldg) = GetNBGraph(g, vectors, vecs_size, vecs_dim);
            auto end = clock();
            cerr << "GetNBGraph costs " << (1.0 * end - start) / CLOCKS_PER_SEC 
                                        << endl;

            start = clock();
            vector<pair<float, int>> tmp_result;
            long long update_times = 0;
            auto ss = clock();
            UpdateGraph(&g, vectors_dev, newg, oldg, k);
            time_sum += clock() - ss;
            end = clock();
            // cerr << "update_times: " << update_times << endl;
            cerr << "Iteration costs " << (1.0 * end - start) / CLOCKS_PER_SEC 
                                       << endl;
            cerr << "Kernel costs " << (1.0 * time_sum) / CLOCKS_PER_SEC << endl;
            cerr << endl;
            time_sum = 0;
        }

    Error:
        for (int i = 0; i < g.size(); i++) {
            for (auto x : g[i]) {
                result[i].push_back(x.id);
            }
        }
        return result;
    }
}

#endif