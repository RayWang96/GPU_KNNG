#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gpudist.cuh"
#include "gpuknnsearch.cuh"
#include "devpq.cuh"
#include "devq.cuh"
#include "rsltelmt.cuh"
#include "../xmuknn.h"

#include <cuda.h>
#include <math_functions.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <cmath>
#include <algorithm>
#include <device_functions.h>
#include <iostream>
#include <vector>
#include <assert.h>
#include <ctime>
#include <queue>

#ifdef __INTELLISENSE__
#include "../intellisense_cuda_intrinsics.h"
#endif
using namespace std;
using namespace xmuknn;

void xmuknn::GPUSearch(vector<int>* knn_result_ptr, float* result_dev, int* index_dev, CUDAData<float> query,
    const int& dimension, CUDAData<float> vecs,
    const Graph& graph, const int& k,
    const int& iterations, const int& candidates_size) {
    auto& knn_result = *knn_result_ptr;
    knn_result.clear();
    vector<bool> visited(graph.size());
    vector<bool> in_candis(graph.size());
    priority_queue<pair<float, int>> result_pq;
    auto& re = rand_engine;
    vector<pair<float, int>> tmp_candidates;

    for (int i = 0; i < iterations || result_pq.size() < k; i++) {
        vector<pair<float, int>> candidates;
        vector<int> neighbors;
        vector<int> start_points;
        GenerateRandomSequence(start_points, candidates_size, graph.size());
        CUDAData<int> index(start_points.data(), index_dev, start_points.size(), true);
        GetOneToNDistanceByIndex(&candidates, result_dev, vecs, query, dimension, index);
        for (auto p : start_points) {
            in_candis[p] = true;
        }
        sort(candidates.begin(), candidates.end());

        int j = 0;
        while (j < candidates.size()) {
            int j = 0;
            while (j < candidates.size() && visited[candidates[j].second]) {
                j++;
            }
            if (j == candidates.size()) break;
            int point = candidates[j].second;
            visited[point] = true;
            neighbors.clear();
            for (auto y : graph[point]) {
                if (in_candis[y]) continue;
                neighbors.push_back(y);
                in_candis[y] = true;
            }
            if (neighbors.size() == 0) continue;
            index.data_ptr = neighbors.data();
            index.if_need_copy = true;
            index.size = neighbors.size();
            GetOneToNDistanceByIndex(&tmp_candidates, result_dev, vecs, query, dimension, index);
            candidates.insert(candidates.end(), tmp_candidates.begin(), tmp_candidates.end());
            sort(candidates.begin(), candidates.end());
            if (candidates.size() > candidates_size)
                candidates.erase(candidates.begin() + candidates_size, candidates.end());
        }
        for (auto p : candidates) {
            result_pq.push(p);
            if (result_pq.size() > k) result_pq.pop();
        }
    }
    knn_result = vector<int>(k);
    int i = k - 1;
    while (!result_pq.empty()) {
        knn_result[i--] = result_pq.top().second;
        result_pq.pop();
    }
    assert(i == -1);
}


void GetLinearMemoryTexObj(cudaTextureObject_t* tex_obj, vector<int> data) {
    int* dev_data;
    cudaMalloc(&dev_data, data.size() * sizeof(int));
    cudaMemcpy(dev_data, data.data(), data.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);

    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeLinear;
    res_desc.res.linear.devPtr = dev_data;
    res_desc.res.linear.desc = channelDesc;
    res_desc.res.linear.sizeInBytes = data.size() * sizeof(int);

    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.readMode = cudaReadModeElementType;

    cudaCreateTextureObject(tex_obj, &res_desc, &tex_desc, NULL);
}

void xmuknn::ReadGraphToTextureMemory(cudaTextureObject_t* tex_edges, cudaTextureObject_t* tex_dest,
    const Graph& graph) {
    int pos = 0;
    vector<int> edges, dest;
    for (int i = 0; i < graph.size(); i++) {
        edges.push_back(pos);
        dest.push_back(graph[i].size());
        pos++;
        for (int j = 0; j < graph[i].size(); j++) {
            dest.push_back(graph[i][j]);
            pos++;
        }
    }
    GetLinearMemoryTexObj(tex_edges, edges);
    GetLinearMemoryTexObj(tex_dest, dest);
}

__global__ void PrepareKernel(DevPriorityQueue** candi_pq_ptrs,
                              DevQueue** expanded_node_dev_ptr,
                              const int k, const int candidate_pq_num, const int candidate_pq_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("%d %d %d %d\n", i, candidate_pq_num, blockIdx.x, gridDim.x);
    if (i >= candidate_pq_num) return;
    if (i == 0) {
        expanded_node_dev_ptr[0] = new DevQueue(candidate_pq_num * 10); //要改，现在先固定。
    }
    //printf("you %d\n", i);
    candi_pq_ptrs[i] = new DevPriorityQueue(candidate_pq_size);
    return;
}

GPUSearchDevParam xmuknn::PrepareSearch(const int k, const int candidate_pq_num, const int candidate_pq_size,
                                       const Graph& graph, const float* vectors, const int vecs_num, const int vecs_dim) {

    DevQueue** expanded_node_dev_ptr, **expanded_node_ptr;
    cudaMalloc(&expanded_node_dev_ptr, sizeof(DevQueue*));
    expanded_node_ptr = (DevQueue**)malloc(sizeof(DevQueue*));

    float* query_dev_ptr;
    cudaMalloc(&query_dev_ptr, (size_t)vecs_dim * sizeof(float));

    DevPriorityQueue** candi_pq_dev_ptrs;
    cudaMalloc(&candi_pq_dev_ptrs, (size_t)candidate_pq_num * sizeof(DevPriorityQueue*));

    float* vectors_dev_ptr;
    cudaMalloc(&vectors_dev_ptr, (size_t)vecs_num * vecs_dim * sizeof(float));
    cudaMemcpy(vectors_dev_ptr, vectors, (size_t)vecs_num * vecs_dim * sizeof(float), cudaMemcpyHostToDevice);
    //candi_pq_ptrs = (DevPriorityQueue**)malloc((size_t)candidate_pq_num * sizeof(DevPriorityQueue*));

    dim3 block_size(1024);
    dim3 grid_size(ceil(1.0 * candidate_pq_num / block_size.x));
    PrepareKernel << <grid_size, block_size >> > (candi_pq_dev_ptrs, expanded_node_dev_ptr, k, candidate_pq_num, candidate_pq_size);
    cudaDeviceSynchronize();
    auto cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "PrepareKernel launch failed: %s\n", cudaGetErrorString(cuda_status));
    }

    cuda_status = cudaMemcpy(expanded_node_ptr, expanded_node_dev_ptr, sizeof(DevQueue*), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "MemCpy failed: %s\n", cudaGetErrorString(cuda_status));
    }
    cudaTextureObject_t tex_edges, tex_dest;
    ReadGraphToTextureMemory(&tex_edges, &tex_dest, graph);

    int* dev_visited;
    vector<int> visited(vecs_num);
    cudaMalloc(&dev_visited, vecs_num * sizeof(int));
    cudaMemcpy(dev_visited, visited.data(), vecs_num * sizeof(int), cudaMemcpyHostToDevice);

    return GPUSearchDevParam(candi_pq_dev_ptrs, candidate_pq_num, 
                             *expanded_node_ptr, query_dev_ptr, vectors_dev_ptr, 
                             tex_edges, tex_dest, dev_visited);
}

__global__ void InitPQKernel(DevPriorityQueue** candi_pq_ptrs, const int candidate_pq_num, 
                             const float* query, const float* vecs, const int dim,
                             int *visited, DevQueue* dev_q_ptr) {
    int pq_id = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    if (pq_id >= candidate_pq_num) return;
    auto& pq = *candi_pq_ptrs[pq_id];
    float dist = 0;

    __shared__ float vec1[128]; //要改，现在先设好再说。
    if (tx < dim) {
        vec1[tx] = query[tx];
    }
    __syncthreads();
    const float* vec2 = vecs + pq_id * dim;
    for (int i = 0; i < dim; i++) {
        float diff = vec1[i] - vec2[i];
        dist += diff * diff;
    }
    dist = sqrt(dist);
    atomicExch(&visited[pq_id], 1);
    ResultElement tmp(dist, pq_id);
    dev_q_ptr->AtomicPush(tmp);
    pq.Push(tmp);
}

//要改，以后再说
#define EXTEND_NUM 96
#define NODES_PER_PQ 10
#define TOP_K 10
#define PT_RSLT_SZ 32
#define MAX_FLOAT 1e9

//线程数要大于维度
__global__ void ExtendKernel(cudaTextureObject_t tex_edges, cudaTextureObject_t tex_dest,
                             DevPriorityQueue** candi_pq_ptrs, const int candidate_pq_num, 
                             const float* query, const float* vecs, const int dim,
                             DevQueue* expanded_nodes, int* visited) {
    int pq_id = blockIdx.x;
    int tx = threadIdx.x;
    auto& pq = *candi_pq_ptrs[pq_id];
    auto& global_expanded_nodes = *expanded_nodes;
    __shared__ ResultElement node_to_extend;
    __shared__ ResultElement topk[EXTEND_NUM];
    __shared__ int topk_size;
    __shared__ int position;
    __shared__ int nb_num;
    __shared__ float vec1[128]; //要改，现在先设好再说。
    __shared__ int swap_cnt;
    __shared__ int last_cnt;

    if (tx == 0) {
        topk_size = 0;
        node_to_extend = pq.PopMin();
        last_cnt = -1;
        position = tex1Dfetch<int>(tex_edges, node_to_extend.label);
        nb_num = tex1Dfetch<int>(tex_dest, position);
    }
    for (int j = 0; j < int(ceil((double)dim / (double)blockDim.x)); j++) {
        int pos = tx + j * blockDim.x;
        if (pos > dim) break;
        vec1[pos] = query[pos];
    }
    __syncthreads();

    int node_id = node_to_extend.label;
    if (tx >= nb_num) return;

    int nb_label = tex1Dfetch<int>(tex_dest, position + tx + 1);
    int was_visited = atomicExch(&visited[nb_label], 1);

    if (!was_visited) {
        const float* vec2 = vecs + nb_label * dim;
        float dist = 0;
        for (int j = 0; j < dim; j++) {
            float diff = vec1[j] - vec2[j];
            dist += diff * diff;
        }
        dist = sqrt(dist);

        int topk_pos = atomicAdd(&topk_size, 1);
        topk[topk_pos] = ResultElement(dist, nb_label);
    }
    __syncthreads();

    for (int i = 0; i < NODES_PER_PQ; i++) {
        int begin_pos = i;
        ResultElement* topk_tmp = topk + i;
        for (int stride = (topk_size - i) / 2; stride >= 1; stride >>= 1) {
            __syncthreads();
            if (tx < stride) {
                if (topk_tmp[tx].distance > topk_tmp[tx + stride].distance) {
                    ResultElement tmp(topk_tmp[tx]);
                    topk_tmp[tx] = topk_tmp[tx + stride];
                    topk_tmp[tx + stride] = tmp;
                }
            }
        }
        __syncthreads();
    }

    //if (tx == 0) {
    //    for (int i = 0; i < topk_size; i++) {
    //        printf("(%f, %d), ", topk[i].distance, topk[i].label);
    //    } printf("\n\n");
    //}
   
    if (tx < NODES_PER_PQ && tx < topk_size) {
        //printf("%d\n", topk[tx].label);
        global_expanded_nodes.AtomicPush(topk[tx]);
    }
}

__global__ void ClearArray(int* array, const int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    array[i] = 0;
}

#define NUM_PER_BLOCK 256
__global__ void GetResultAndClearKernel(int* dev_result, DevQueue* dev_nodes_ptr, const int k,
                                        DevPriorityQueue** candi_pq_ptrs, const int candidate_pq_num) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //if ((i - 1) * NUM_PER_BLOCK < graph_size) {
    //    dim3 block_size(NUM_PER_BLOCK);
    //    dim3 grid_size(ceil((double)graph_size / (double)NUM_PER_BLOCK));
    //    int size = NUM_PER_BLOCK;
    //    if (i * NUM_PER_BLOCK > graph_size) {
    //        size = graph_size - (i - 1) * NUM_PER_BLOCK;
    //    }
    //    ClearArray<<<grid_size, block_size>>>(visited + i * NUM_PER_BLOCK, size);
    //}

    auto& dev_nodes = *dev_nodes_ptr;
    int size = min(dev_nodes.Size(), k);

    if (i < size) {
        dev_result[i] = dev_nodes.GetElements()[i].label;
    }

    if (i == 0) dev_nodes.Clear();
    if (i >= candidate_pq_num) return;
    candi_pq_ptrs[i]->Clear();
}

void GetResultAndClear(vector<int>* knn_result_ptr, const int k, DevQueue* dev_nodes_ptr, GPUSearchDevParam param) {
    auto& res = *knn_result_ptr;
    res.resize(k);
    int* dev_result;
    cudaMalloc(&dev_result, k * sizeof(int));

    dim3 block_size(128);
    dim3 grid_size(ceil(1.0 * param.candidate_pq_num / block_size.x));
    GetResultAndClearKernel<<<grid_size, block_size>>>(dev_result, dev_nodes_ptr, k, param.candidate_pq_ptrs, param.candidate_pq_num);
    cudaDeviceSynchronize();

    cudaMemcpy(res.data(), dev_result, k * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_result);
    //printf("%d\n", res[0]);
}

__global__ void NodeAssignKernel(DevQueue* nodes_q_ptr, DevPriorityQueue** pqs_ptr, const int pq_num, const int k) {
    int pq_id = blockIdx.x * blockDim.x + threadIdx.x;
    ResultElement* nodes = nodes_q_ptr->GetElements() + k;
    int nodes_num = nodes_q_ptr->Size();
    if (pq_id >= pq_num) return;

    int nodes_per_pq = ceil((double)nodes_num / (double)pq_num);
    for (int i = 0; i < nodes_per_pq; i++) {
        int node_pos = i * pq_num + pq_id;
        if (node_pos >= nodes_num) break;
        //nodes[node_pos];
        //printf("check %d %d %d\n", pq_id, node_pos, nodes_num);
        pqs_ptr[pq_id]->Push(nodes[node_pos]);
    }
}

//节点队列会被resize到k。

__device__ int locked = 0;

__global__ void KSelectionKernel(DevQueue* expanded_nodes_ptr, int* dev_improved, const int k, const int top_i) {
    int tx = threadIdx.x;
    __shared__ int shared_nodes_num;
    __shared__ ResultElement partial_result[PT_RSLT_SZ];
    __shared__ int ids[PT_RSLT_SZ];

    auto nodes = expanded_nodes_ptr->GetElements() + top_i;

    if (top_i == 0 && tx == 0 && blockIdx.x == 0) {
        atomicExch((int*)dev_improved, 0);
    }

    if (tx == 0) {
        shared_nodes_num = expanded_nodes_ptr->Size();
    }

    __syncthreads();
    int nodes_num = shared_nodes_num - top_i;
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= nodes_num) {
        partial_result[tx] = ResultElement(MAX_FLOAT, -1);
    } else {
        partial_result[tx] = nodes[pos];
    }
    ids[tx] = pos;
    for (int stride = PT_RSLT_SZ / 2; stride >= 1; stride >>= 1) {
        __syncthreads();
        if (tx < stride) {
            if (partial_result[tx].distance > partial_result[tx + stride].distance) {
                partial_result[tx] = partial_result[tx + stride];
                ids[tx] = ids[tx + stride];
            }
        }
    }
    __syncthreads();
    __threadfence_system();
    if (tx == 0) {
        while (atomicCAS(&locked, 0, 1) != 0) {}
        if (nodes[0].distance > partial_result[0].distance) {
            nodes[ids[0]] = nodes[0];
            nodes[0] = partial_result[0];
            atomicExch((int*)dev_improved, 1);
        }
        atomicExch(&locked, 0);
        __threadfence_system();
    }

    if (tx == 0 && k == top_i + 1) {
        expanded_nodes_ptr->Resize(k);
    }

}

__global__ void Test(DevQueue* Q) {
    for (int i = 0; i < Q->Size(); i++) {
        printf("(%f, %d), ", Q->GetElements()[i].distance, Q->GetElements()[i].label);
    } printf("\n\n\n");
}

bool KSelection(DevQueue* expanded_nodes_ptr, int nodes_num, int k) {
    int* dev_improved;
    cudaMalloc(&dev_improved, sizeof(int));

    dim3 block_size(PT_RSLT_SZ);
    dim3 grid_size(ceil((double)nodes_num / (double)block_size.x));

    for (int i = 0; i < k; i++) {
        KSelectionKernel << <grid_size, block_size >> > (expanded_nodes_ptr, dev_improved, k, i);
        //cudaDeviceSynchronize();
    }
    auto cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "Sort failed: %s\n", cudaGetErrorString(cuda_status));
        return 0;
    }
    //Test << <dim3(1), dim3(1) >> > (expanded_nodes_ptr);
    int improved = 0;
    cudaMemcpy(&improved, dev_improved, sizeof(int), cudaMemcpyDeviceToHost);

    return improved;
}

void xmuknn::GPUSearch(vector<int>* knn_result_ptr, const float* query, const Graph& graph, const int k, int dim, GPUSearchDevParam dev_param) {
    auto& knn_result = *knn_result_ptr;
    knn_result.clear();
    dim3 block_size(128);
    dim3 grid_size(ceil(1.0 * dev_param.candidate_pq_num / block_size.x));
    cudaMemcpy(dev_param.query_ptr, query, (size_t)dim * sizeof(float), cudaMemcpyHostToDevice);
    InitPQKernel <<<grid_size, block_size>>> (dev_param.candidate_pq_ptrs, dev_param.candidate_pq_num, 
                                              dev_param.query_ptr, dev_param.vectors_ptr, dim, dev_param.visited,
                                              dev_param.expanded_nodes_ptr);
    cudaDeviceSynchronize();

    bool improved = 1; 

    //Test << <dim3(1), dim3(1) >> > (dev_param.expanded_nodes_ptr);
    //return;

    auto ss = clock();
    KSelection(dev_param.expanded_nodes_ptr, NODES_PER_PQ * dev_param.candidate_pq_num, k);
    time_sum += clock() - ss;
    //return;
    //cudaFree(dev_improved);
    auto cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "InitPQ launch failed: %s\n", cudaGetErrorString(cuda_status));
        goto Error;
    }

    improved = 1;
    while (improved) {
        improved = 0;
        block_size = dim3(EXTEND_NUM);
        grid_size = dim3(dev_param.candidate_pq_num);
        ExtendKernel << <grid_size, block_size >> > (dev_param.tex_edges, dev_param.tex_dest,
                                                     dev_param.candidate_pq_ptrs, dev_param.candidate_pq_num,
                                                     dev_param.query_ptr, dev_param.vectors_ptr, dim,
                                                     dev_param.expanded_nodes_ptr, dev_param.visited);
        cudaDeviceSynchronize();
        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "ExtendKernel launch failed: %s\n", cudaGetErrorString(cuda_status));
            goto Error;
        }

        block_size = dim3(128);
        grid_size = dim3(ceil(1.0 * dev_param.candidate_pq_num / block_size.x));
        NodeAssignKernel << <grid_size, block_size >> > (dev_param.expanded_nodes_ptr, dev_param.candidate_pq_ptrs, dev_param.candidate_pq_num, k);
        cudaDeviceSynchronize();
        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "NodeAssignKernel launch failed: %s\n", cudaGetErrorString(cuda_status));
            goto Error;
        }
        ss = clock();

        //同步
        improved = KSelection(dev_param.expanded_nodes_ptr, NODES_PER_PQ * dev_param.candidate_pq_num, k);
        time_sum += clock() - ss;

        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "ClearNodesKernel launch failed: %s\n", cudaGetErrorString(cuda_status));
            goto Error;
        }
    }


Error:

    GetResultAndClear(knn_result_ptr, k, dev_param.expanded_nodes_ptr, dev_param);
    ClearArray<<<dim3(ceil(graph.size() / 128.0)), dim3(128)>>>(dev_param.visited, graph.size());
    ///vector<int> visited(graph.size());
    ///cudaMemcpy(dev_param.visited, visited.data(), graph.size() * sizeof(int), cudaMemcpyHostToDevice);

    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "ClearKernel launch failed: %s\n", cudaGetErrorString(cuda_status));
    }
}


__global__ void EndSearchKernel(DevPriorityQueue** candi_pq_ptrs, const int candidate_pq_num) {
    int i = blockIdx.x * gridDim.x + threadIdx.x;
    if (i >= candidate_pq_num) return;
    delete candi_pq_ptrs[i];
}

void xmuknn::EndSearch(GPUSearchDevParam ptrs) {
    dim3 block_size(1024);
    dim3 grid_size(ceil(1.0 * ptrs.candidate_pq_num / block_size.x));
    EndSearchKernel << <block_size, grid_size >> > (ptrs.candidate_pq_ptrs, ptrs.candidate_pq_num);
    cudaDeviceSynchronize();
    auto cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "ClearKernel launch failed: %s\n", cudaGetErrorString(cuda_status));
    }
    cudaDestroyTextureObject(ptrs.tex_dest);
    cudaDestroyTextureObject(ptrs.tex_edges);
    cudaFree(ptrs.candidate_pq_ptrs);
    cudaFree(ptrs.vectors_ptr);
    cudaFree(ptrs.query_ptr);
}