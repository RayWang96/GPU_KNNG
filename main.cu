#include <assert.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <istream>
#include <vector>

#include "gpuknn/knncuda_tools.cuh"
#include "gpuknn/knnmerge.cuh"
#include "gpuknn/nndescent.cuh"
#include "tools/distfunc.hpp"
#include "tools/filetool.hpp"
#include "xmuknn.h"

using namespace std;
using namespace xmuknn;

void Evaluate(const string &data_path, const string &ground_truth_path) {
  string cmd = "python3 -u \"/home/hwang/codes/GPU_KNNG/tools/evaluate.py\"";
  cmd += " ";
  cmd += data_path;
  cmd += " ";
  cmd += ground_truth_path;
  int re = system(cmd.c_str());
}

void TestCUDANNDescent() {
  int k = 30;
  // string out_path = FileTool::GetOutPath();

  // string base_path
  //     = "/home/hwang//data/sift10k/sift10k.txt";
  // string out_path
  //     = "/home/hwang/data/result/sift10k_knng_k64.txt";
  // string ground_truth_path
  //     = "/home/hwang//data/sift10k/sift10k_groundtruth_self.txt";

  // string base_path = "/home/hwang//data/sift100k/sift100k.txt";
  // string out_path = "/home/hwang/data/result/sift100k_knng_k64.txt";
  // string ground_truth_path =
  //     "/home/hwang//data/sift100k/sift100k_groundtruth_self.txt";

  // string base_path = "/home/hwang/data/glove1m/glove1m_norm_base.txt";
  // string out_path = "/home/hwang/data/result/glove1m_knng_k64.txt";
  // string ground_truth_path =
  //     "/home/hwang/data/glove1m/glove1m_gold_knn40.txt";

  // string base_path = "/home/hwang//data/sift1m/sift1m.txt";
  // string out_path = "/home/hwang/data/result/sift1m_knng_k64.txt";
  // string ground_truth_path =
  //     "/home/hwang//data/sift1m/sift1m_gold_knn40_sorted.txt";

  // string base_path = "/home/hwang//data/sift10m/sift10m.txt";
  // string out_path = "/home/hwang/data/result/sift10m_knng_k64.txt";
  // string ground_truth_path =
  //     "/home/hwang/data/sift10m/sift10m_gold_knn40.txt";

  // string base_path = "/home/hwang/data/glove100k/glove100k_norm_base.txt";
  // string out_path = "/home/hwang/data/result/glove100k_knng_k64.txt";
  // string ground_truth_path =
  //     "/home/hwang/data/glove100k/glove100k_self_ground_truth.txt";

  string base_path = "/home/hwang/data/sift1m/sift_base.fvecs";
  string out_path = "/home/hwang/data/result/sift1m_knng_k64.txt";
  string ground_truth_path =
      "/home/hwang//data/sift1m/sift1m_gold_knn40_sorted.txt";

  // string base_path = "/home/hwang/data/sift5m/sift5m.fvecs";
  // string out_path = "/home/hwang/data/result/sift5m_knng_k64.txt";
  // string ground_truth_path =
  //     "/home/hwang/data/sift1m/sift1m_gold_knn40_sorted.txt";

  auto out = ofstream(out_path);
  if (!out.is_open()) {
    cerr << "Output file is not opened!" << endl;
    return;
  }

  float *vectors;
  int vecs_size, vecs_dim;
  // FileTool::ReadVecs(vectors, vecs_size, vecs_dim, base_path);
  FileTool::ReadFVecs(base_path, &vectors, &vecs_size, &vecs_dim);

  auto knn_graph = gpuknn::NNDescent(vectors, vecs_size, vecs_dim, 6);

  // out << knn_graph.size() << " " << k << endl;
  // for (int i = 0; i < knn_graph.size(); i++) {
  //   const auto &x = knn_graph[i];
  //   out << i << " " << x.size() << " ";
  //   for (auto y : x) {
  //     // out << y.distance() << " " << y.label() << "\t";
  //     assert(y.label() != i);
  //     out << y.label() << "\t";
  //   }
  //   out << endl;
  // }
  // out.close();
  // Evaluate(out_path, ground_truth_path);
  delete[] vectors;
  return;
}

void DivideData(float *vectors, const int dim, float **vectors_first_ptr,
                const int vectors_first_size, float **vectors_second_ptr,
                const int vectors_second_size) {
  float *&vectors_first = *vectors_first_ptr;
  float *&vectors_second = *vectors_second_ptr;
  vectors_first = new float[vectors_first_size * dim];
  vectors_second = new float[vectors_second_size * dim];
  for (int i = 0; i < vectors_first_size; i++) {
    for (int j = 0; j < dim; j++) {
      vectors_first[i * dim + j] = vectors[i * dim + j];
    }
  }
  for (int i = 0; i < vectors_second_size; i++) {
    for (int j = 0; j < dim; j++) {
      vectors_second[i * dim + j] = vectors[(i + vectors_first_size) * dim + j];
    }
  }
  return;
}

void TestCUDAMerge() {
  string base_path = "/home/hwang/data/sift100k/sift100k.txt";
  string out_path = "/home/hwang/data/result/sift100k_knng_k64_merged.txt";
  string ground_truth_path =
      "/home/hwang//data/sift100k/sift100k_groundtruth_self.txt";
  // string base_path = "/home/hwang/data/glove100k/glove100k_norm_base.txt";
  // string out_path = "/home/hwang/data/result/glove100k_knng_k64_merged.txt";
  // string ground_truth_path =
  //     "/home/hwang/data/glove100k/glove100k_self_ground_truth.txt";
  // string base_path = "/home/hwang/data/sift1m/sift1m.txt";
  // string out_path = "/home/hwang/data/result/sift1m_knng_k64_merged.txt";
  // string ground_truth_path =
  //     "/home/hwang/data/sift1m/sift1m_gold_knn40_sorted.txt";
  float *vectors;
  int vecs_size, vecs_dim;
  FileTool::ReadVecs(vectors, vecs_size, vecs_dim, base_path);
  int vectors_first_size = vecs_size / 2;
  int vectors_second_size = vecs_size - vectors_first_size;
  float *vectors_first, *vectors_second;
  DivideData(vectors, vecs_dim, &vectors_first, vectors_first_size,
             &vectors_second, vectors_second_size);
  float *vectors_first_dev, *vectors_second_dev;
  cudaMalloc(&vectors_first_dev,
             (size_t)vectors_first_size * vecs_dim * sizeof(float));
  cudaMalloc(&vectors_second_dev,
             (size_t)vectors_second_size * vecs_dim * sizeof(float));
  cudaMemcpy(vectors_first_dev, vectors_first,
             (size_t)vectors_first_size * vecs_dim * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(vectors_second_dev, vectors_second,
             (size_t)vectors_second_size * vecs_dim * sizeof(float),
             cudaMemcpyHostToDevice);

  NNDElement *knngraph_first_dev, *knngraph_second_dev;
  gpuknn::NNDescent(&knngraph_first_dev, vectors_first_dev, vectors_first_size,
                    vecs_dim);
  gpuknn::NNDescent(&knngraph_second_dev, vectors_second_dev,
                    vectors_second_size, vecs_dim);

  NNDElement *knngraph_merged_dev;
  float *vectors_merged_dev;
  gpuknn::KNNMerge(&knngraph_merged_dev, &vectors_merged_dev, vectors_first_dev,
                   vectors_first_size, knngraph_first_dev, vectors_second_dev,
                   vectors_second_size, knngraph_second_dev);

  vector<vector<NNDElement>> knngraph_host;
  // ToHostKNNGraph(&knngraph_host, knngraph_first_dev, vectors_first_size,
  //                NEIGHB_NUM_PER_LIST);
  // OutputHostKNNGraph(knngraph_host,
  //                    "/home/hwang/codes/GPU_KNNG/results/graph_a.txt");
  // ToHostKNNGraph(&knngraph_host, knngraph_second_dev, vectors_second_size,
  //                NEIGHB_NUM_PER_LIST);
  // OutputHostKNNGraph(knngraph_host,
  //                    "/home/hwang/codes/GPU_KNNG/results/graph_b.txt");
  ToHostKNNGraph(&knngraph_host, knngraph_merged_dev,
                 vectors_first_size + vectors_second_size, NEIGHB_NUM_PER_LIST);
  OutputHostKNNGraph(knngraph_host, out_path);
  Evaluate(out_path, ground_truth_path);
  cudaFree(vectors_merged_dev);
  cudaFree(knngraph_merged_dev);
  cudaFree(knngraph_first_dev);
  cudaFree(knngraph_second_dev);
  delete[] vectors;
  delete[] vectors_first;
  delete[] vectors_second;
}

void TestFileTools() {
  // float *vectors;
  // int num;
  // int dim;
  // auto start = chrono::steady_clock::now();
  // FileTool::ReadFVecs("/home/hwang/data/sift1m/sift_base.fvecs", &vectors, &num,
  //                     &dim);
  // auto end = chrono::steady_clock::now();
  // float time_cost =
  //     (float)chrono::duration_cast<std::chrono::microseconds>(end - start)
  //         .count() /
  //     1e6;
  // cerr << time_cost << endl;
  // float *gt_vectors;
  // int gt_num;
  // int gt_dim;
  // FileTool::ReadVecs(gt_vectors, gt_num, gt_dim,
  //                    "/home/hwang/data/sift1m/sift1m.txt");
  // for (int i = 0; i < num; i++) {
  //   for (int j = 0; j < dim; j++) {
  //     int pos = i * dim + j;
  //     assert(vectors[pos] == gt_vectors[pos]);
  //   }
  // }
  // delete[] vectors;
  // delete[] gt_vectors;

  // float *vectors;
  // int num;
  // int dim;
  // auto start = chrono::steady_clock::now();
  // FileTool::ReadFVecs("/media/hwang_data/deep1b/base_00", &vectors, &num,
  //                     &dim);
  // auto end = chrono::steady_clock::now();
  // float time_cost =
  //     (float)chrono::duration_cast<std::chrono::microseconds>(end - start)
  //         .count() /
  //     1e6;
  // FileTool::WriteFVecs("./test.fvecs", vectors, num, dim);
  // delete[] vectors;

  // int *vectors;
  // int num;
  // int dim;
  // FileTool::ReadIVecs("/home/hwang/data/sift1m/sift_groundtruth.ivecs",
  //                     &vectors, &num, &dim);
  // cerr << num << " " << dim << endl;
  // for (int i = 0; i < 200; i++) {
  //   cerr << vectors[i] << " ";
  // } cerr << endl;
  
  // float *vectors;
  // int num;
  // int dim;
  // auto start = chrono::steady_clock::now();
  // FileTool::ReadFVecs("/home/hwang/data/sift10m/sift10m.fvecs", &vectors, &num,
  //                     &dim, 5000000);
  // auto end = chrono::steady_clock::now();
  // float time_cost =
  //     (float)chrono::duration_cast<std::chrono::microseconds>(end - start)
  //         .count() /
  //     1e6;
  // FileTool::WriteFVecs("/home/hwang/data/sift5m/sift5m.fvecs", vectors, num, dim);
  // delete[] vectors;

  float *vectors;
  int start_pos = 900000;
  int num = 100000;
  int dim;
  auto start = chrono::steady_clock::now();
  FileTool::ReadFVecs("/home/hwang/data/sift1m/sift_base.fvecs", &vectors,
                      &dim, start_pos, num);
  auto end = chrono::steady_clock::now();
  float time_cost =
      (float)chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count() /
      1e6;
  float *gt_vectors;
  int gt_num, gt_dim;
  FileTool::ReadVecs(gt_vectors, gt_num, gt_dim,
                     "/home/hwang/data/sift1m/sift1m.txt");
  cerr << num << " " << dim << endl;
  for (int i = start_pos; i < start_pos + num; i++) {
    for (int j = 0; j < dim; j++) {
      assert(vectors[(i-start_pos) * dim + j] == gt_vectors[i * dim + j]);
    }
  }
  delete[] vectors;
}

void TestMemoryManager() {
  PredPeakGPUMemory(5000000, 128, 64, 32);
}

int main() {
  // UnitTest();
  // TestKNNAlgorithm();
  TestCUDANNDescent();
  // TestFileTools();
  // TestMemoryManager();
  // TestCUDAMerge();
  // TestTiledDistanceCompare();
  // TestCUDADistance();
  // TestCUDASearch();
  // TestCUDANewSearch();
  // TestCUDAPriorityQueue();
  // GetRGraph();
}