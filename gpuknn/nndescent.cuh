#ifndef XMUKNN_NNDESCENT_CUH
#define XMUKNN_NNDESCENT_CUH
#include "../xmuknn.h"
#include "nndescent_element.cuh"
#define LARGE_INT 0x3f3f3f3f
using namespace std;
using namespace xmuknn;

namespace gpuknn {
void NNDescent(NNDElement **knngraph_result_dev_ptr, const float *vectors_dev,
               const int vecs_size, const int vecs_dim);
vector<vector<NNDElement>> NNDescent(const float *vectors, const int vecs_size,
                                     const int vecs_dim);
}  // namespace gpuknn
#endif
