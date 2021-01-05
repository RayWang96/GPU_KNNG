#ifndef XMUKNN_NNDESCENT_CUH
#define XMUKNN_NNDESCENT_CUH
#include "../xmuknn.h"
#include "nndescent_element.cuh"
using namespace std;
using namespace xmuknn;

#define LARGE_INT 0x3f3f3f3f
#define FULL_MASK 0xffffffff

const int VEC_DIM = 128;
const int NEIGHB_NUM_PER_LIST = 64;
const int TILE_WIDTH = 16;
const int SAMPLE_NUM = 32;  // assert(SAMPLE_NUM * 2 <= NEIGHB_NUM_PER_LIST);

const int INSERT_IT_NUM =
    NEIGHB_NUM_PER_LIST / 32 + (NEIGHB_NUM_PER_LIST % 32 != 0);
const int LAST_HALF_NEIGHB_NUM = NEIGHB_NUM_PER_LIST / 2;
const int FIRST_HALF_NEIGHB_NUM = NEIGHB_NUM_PER_LIST - NEIGHB_NUM_PER_LIST / 2;
const int LAST_HALF_INSERT_IT_NUM = 
    LAST_HALF_NEIGHB_NUM / 32 + (LAST_HALF_NEIGHB_NUM % 32 != 0);
const int NEIGHB_CACHE_NUM = 1;
const int SKEW_TILE_WIDTH = TILE_WIDTH + 1;
const int SKEW_DIM = VEC_DIM + 1;

__global__ void ShrinkGraph(int *graph_new_dev, int *newg_list_size_dev,
                            int *newg_revlist_size_dev, int *graph_old_dev,
                            int *oldg_list_size_dev,
                            int *oldg_revlist_size_dev);
__device__ void GetNewOldDistancesTiled(float *distances, const float *vectors,
                                        const int *new_neighbors,
                                        const int num_new,
                                        const int *old_neighbors,
                                        const int num_old);
__device__ NNDElement GetMinElement2(const int list_id, const int list_size,
                                     const int *old_neighbs, const int num_old,
                                     const float *distances,
                                     const int distances_num);
__device__ NNDElement GetMinElement3(const int list_id, const int list_size,
                                     const int *new_neighbs, const int num_new,
                                     const int *old_neighbs, const int num_old,
                                     const float *distances,
                                     const int distances_num,
                                     const float *vectors);
__global__ void MarkAllToOld(NNDElement *knn_graph);
__global__ void SortKNNGraphKernel(NNDElement *knn_graph, const int graph_size);
namespace gpuknn {
void NNDescentRefine(NNDElement *knngraph_result_dev_ptr,
                     const float *vectors_dev, const int vecs_size,
                     const int vecs_dim, const int iteration = 6);
void NNDescent(NNDElement **knngraph_result_dev_ptr, const float *vectors_dev,
               const int vecs_size, const int vecs_dim,
               const int iteration = 6);
vector<vector<NNDElement>> NNDescent(const float *vectors, const int vecs_size,
                                     const int vecs_dim,
                                     const int iteration = 6);
}  // namespace gpuknn
#endif
