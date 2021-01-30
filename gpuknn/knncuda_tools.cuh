#ifndef XMUKNN_KNNCUDA_TOOLS_CUH
#define XMUKNN_KNNCUDA_TOOLS_CUH
#include <vector>
#include <string>

#include "../tools/nndescent_element.cuh"
using namespace std;
void DevRNGLongLong(unsigned long long *dev_data, int n);
void GenerateRandomKNNGraphIndex(int **knn_graph_index, const int graph_size,
                                 const int neighb_num);
__device__ int GetItNum(const int sum_num, const int num_per_it);
void ToHostKNNGraph(vector<vector<NNDElement>> *origin_knn_graph_ptr,
                    const NNDElement *knn_graph_dev, const int size,
                    const int neighb_num);
void ToHostKNNGraph(NNDElement **host_knn_graph_ptr,
                    const NNDElement *knn_graph_dev, const int size,
                    const int neighb_num);
void OutputHostKNNGraph(const vector<vector<NNDElement>> &knn_graph,
                        const string &out_path,
                        const bool output_distance = false);
size_t PredPeakGPUMemory(const int vecs_num, const int vecs_dim, const int k,
                         const int sample_num, const bool thrust_random = true);
__host__ __device__ uint64_t xorshift64star(uint64_t x); 
#endif