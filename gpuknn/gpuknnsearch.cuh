#ifndef XMUKNN_GPUKNNSEARCH_CUH
#define XMUKNN_GPUKNNSEARCH_CUH
#include <vector>
#include "../xmuknn.h"
#include "gpudist.cuh"
#include "devpq.cuh"
#include "rsltelmt.cuh"
#include "devq.cuh"
using namespace std;
namespace xmuknn {
    struct GPUSearchDevParam {
        DevPriorityQueue** candidate_pq_ptrs;
        int candidate_pq_num;
        DevQueue* expanded_nodes_ptr;
        cudaTextureObject_t tex_edges;
        cudaTextureObject_t tex_dest;
        float* query_ptr;
        float* vectors_ptr;
        int* visited;
        GPUSearchDevParam(DevPriorityQueue** candidate_pq_ptrs, const int candidate_pq_num,
                          DevQueue* expanded_nodes_ptr,
                          float* query_ptr, float* vectors_ptr,
                          cudaTextureObject_t tex_edges, cudaTextureObject_t tex_dest,
                          int* visited) {
            this->candidate_pq_ptrs = candidate_pq_ptrs;
            this->candidate_pq_num = candidate_pq_num;
            this->expanded_nodes_ptr = expanded_nodes_ptr;
            this->tex_edges = tex_edges;
            this->tex_dest = tex_dest;
            this->query_ptr = query_ptr;
            this->vectors_ptr = vectors_ptr;
            this->visited = visited;
        }
    };

    GPUSearchDevParam PrepareSearch(const int k, const int candidate_num, const int candidate_size,
                                   const Graph& graph, const float* vectors, const int vecs_num, const int vecs_dim);
    void EndSearch(GPUSearchDevParam ptrs);

    void ReadGraphToTextureMemory(cudaTextureObject_t* tex_edges, cudaTextureObject_t* tex_dest,
                                  const Graph& graph);

    void GPUSearch(vector<int>* knn_result_ptr, float *result_dev, int *index_dev, CUDAData<float> query,
                   const int& dimension, CUDAData<float> vecs,
                   const Graph& graph, const int& k,
                   const int& iterations = 1, const int& candidates_size = 15);

    void GPUSearch(vector<int>* knn_result_ptr, const float* query, const Graph& graph, const int k, int dim, GPUSearchDevParam ptrs);
}
#endif