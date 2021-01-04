#include <curand.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <assert.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "knncuda_tools.cuh"
using namespace std;
void DevRNGLongLong(unsigned long long *dev_data, int n) {
  curandGenerator_t gen;
  curandCreateGenerator(&gen,
                        curandRngType_t::CURAND_RNG_QUASI_SCRAMBLED_SOBOL64);
  curandSetPseudoRandomGeneratorSeed(gen, clock());
  curandGenerateLongLong(gen, dev_data, n);
}

__device__ int GetItNum(const int sum_num,
                                        const int num_per_it) {
  return sum_num / num_per_it + (sum_num % num_per_it != 0);
}

void ToHostKNNGraph(vector<vector<NNDElement>> *origin_knn_graph_ptr,
                    const NNDElement *knn_graph_dev, const int size,
                    const int neighb_num) {
  NNDElement *knn_graph = new NNDElement[size * neighb_num];
  cudaMemcpy(knn_graph, knn_graph_dev,
             (size_t)size * neighb_num * sizeof(NNDElement),
             cudaMemcpyDeviceToHost);
  auto &origin_knn_graph = *origin_knn_graph_ptr;
  vector<NNDElement> neighb_list;
  origin_knn_graph = vector<vector<NNDElement>>(size);
  for (int i = 0; i < size; i++) {
    neighb_list.clear();
    for (int j = 0; j < neighb_num; j++) {
      neighb_list.push_back(knn_graph[i * neighb_num + j]);
    }
    origin_knn_graph[i] = neighb_list;
  }
  delete[] knn_graph;
}

void OutputHostKNNGraph(const vector<vector<NNDElement>> &knn_graph, const string &out_path) {
  auto out = ofstream(out_path);
  if (!out.is_open()) {
    cerr << "Output file is not opened!" << endl;
    return;
  }
  out << knn_graph.size() << " " << knn_graph[0].size() << endl;
  for (int i = 0; i < knn_graph.size(); i++) {
    const auto &x = knn_graph[i];
    out << i << " " << x.size() << " ";
    for (auto y : x) {
      // out << y.distance() << " " << y.label() << "\t";
      // assert(y.label() != i);
      out << y.label() << "\t";
    }
    out << endl;
  }
  out.close();
}

__global__ void GenRandKNNGraphIndexKernel(
    int *knn_graph_index, const int graph_size, const int neighb_num,
    const unsigned long long *random_sequence) {
  int list_id = blockIdx.x;
  int tx = threadIdx.x;
  int pos = list_id * neighb_num + tx;
  knn_graph_index[pos] = random_sequence[pos] % (unsigned long long)graph_size;
}

void GenerateRandomKNNGraphIndex(int **knn_graph_index_ptr, const int graph_size,
                                 const int neighb_num) {
  unsigned long long *random_sequence_dev;
  int *&knn_graph_index = *knn_graph_index_ptr;
  cudaMalloc(&random_sequence_dev,
             graph_size * neighb_num * sizeof(unsigned long long));
  DevRNGLongLong(random_sequence_dev, graph_size * neighb_num);
  cudaMalloc(&knn_graph_index, graph_size * neighb_num * sizeof(int));
  GenRandKNNGraphIndexKernel<<<graph_size, neighb_num>>>(
      knn_graph_index, graph_size, neighb_num, random_sequence_dev); 
  cudaDeviceSynchronize();
  cudaFree(random_sequence_dev);
  auto cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    cerr << "GenRandomKNNGraph failed: " << cudaGetErrorString(cuda_status)
         << endl;
    exit(-1);
  }
}