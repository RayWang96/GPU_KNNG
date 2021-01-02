#ifndef XMUKNN_KNNMERGE_CUH
#define XMUKNN_KNNMERGE_CUH
#include "nndescent_element.cuh"
namespace gpuknn {
void KNNMerge(int *knngraph_result_dev, const float *vectors_first_dev,
              const int *knngraph_first_dev, const float *vectors_second_dev,
              const int *knngraph_second_dev);
}
#endif