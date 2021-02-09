#ifndef XMUKNN_NNDESCENT_CUH
#define XMUKNN_NNDESCENT_CUH
#include "../xmuknn.h"
#include "../tools/nndescent_element.cuh"
using namespace std;
using namespace xmuknn;

#define LARGE_INT 0x3f3f3f3f
const int VEC_DIM = 128;
const int NEIGHB_NUM_PER_LIST = 64;
const int NND_ITERATION = 6;
const int SAMPLE_NUM = 32;  // assert(SAMPLE_NUM * 2 <= NEIGHB_NUM_PER_LIST);
const int MERGE_SAMPLE_NUM = 10;
const int MERGE_ITERATION = 8;

const int WARP_SIZE = 32;
const int NEIGHB_BLOCKS_NUM =
    NEIGHB_NUM_PER_LIST / 32 + (NEIGHB_NUM_PER_LIST % 32 != 0);
// const int INSERT_IT_NUM =
//     NEIGHB_NUM_PER_LIST / 32 + (NEIGHB_NUM_PER_LIST % 32 != 0);
const int NEIGHB_CACHE_NUM = 1;
const int TILE_WIDTH = 16;
const int SKEW_TILE_WIDTH = TILE_WIDTH + 1;
const int SKEW_DIM = VEC_DIM + 1;
const int LAST_HALF_NEIGHB_NUM = NEIGHB_NUM_PER_LIST / 2;
const int FIRST_HALF_NEIGHB_NUM =
    NEIGHB_NUM_PER_LIST - NEIGHB_NUM_PER_LIST / 2;
__global__ void MarkAllToOld(NNDElement *knn_graph);
namespace gpuknn {
void NNDescentForMerge(NNDElement *knngraph_result_dev_ptr,
                       const float *vectors_dev, const int vecs_size,
                       const int vecs_dim, const int split_pos, const int iteration = 6);
void NNDescent(NNDElement **knngraph_result_ptr, const float *vectors_dev,
               const int vecs_size, const int vecs_dim, const int iteration = 6,
               const bool store_result_in_device = true);
vector<vector<NNDElement>> NNDescent(const float *vectors, const int vecs_size,
                                     const int vecs_dim,
                                     const int iteration = 6);
}  // namespace gpuknn
#endif
