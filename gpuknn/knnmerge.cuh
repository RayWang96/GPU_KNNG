#ifndef XMUKNN_KNNMERGE_CUH
#define XMUKNN_KNNMERGE_CUH
#include <vector>

#include "../tools/nndescent_element.cuh"
namespace gpuknn {
void KNNMerge(NNDElement **knngraph_merged_dev_ptr,
              float *vectors_first_dev, const int vectors_first_size,
              NNDElement *knngraph_first_dev,
              float *vectors_second_dev, const int vectors_second_size,
              NNDElement *knngraph_second_dev, const bool free_sub_data = false);

void KNNMergeFromHost(NNDElement **knngraph_merged_dev_ptr,
                      const float *vectors_first, const int vectors_first_size,
                      const NNDElement *knngraph_first, const float *vectors_second,
                      const int vectors_second_size,
                      const NNDElement *knngraph_second);
                      
std::vector<std::vector<NNDElement>> KNNMerge(
    const float *vectors_first, const int vectors_first_size,
    const std::vector<std::vector<NNDElement>> &knngraph_first,
    const float *vectors_second, const int vectors_second_size,
    const std::vector<std::vector<NNDElement>> &knngraph_second);

void KNNJMerge(NNDElement **knngraph_merged_dev_ptr, float *vectors_first_dev,
               const int vectors_first_size, NNDElement *knngraph_first_dev,
               float *vectors_second_dev, const int vectors_second_size,
               NNDElement *knngraph_second_dev);
}  // namespace gpuknn
#endif