#include <assert.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <istream>
#include <vector>

#include "gpuknn/gen_large_knngraph.cuh"
#include "gpuknn/knncuda_tools.cuh"
#include "gpuknn/knnmerge.cuh"
#include "gpuknn/nndescent.cuh"
#include "tools/distfunc.hpp"
#include "tools/filetool.hpp"
#include "tools/knndata_manager.hpp"
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

  // string base_path = "/home/hwang//data/sift10m/sift10m.fvecs";
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
  FileTool::ReadBinaryVecs(base_path, &vectors, &vecs_size, &vecs_dim);
  auto knn_graph = gpuknn::NNDescent(vectors, vecs_size, vecs_dim, 6);

  // vector<float> vectors_vec(16000000ul * 128ul);
  // float *vectors;
  // int vecs_size, vecs_dim;
  // // FileTool::ReadVecs(vectors, vecs_size, vecs_dim, base_path);
  // FileTool::ReadBinaryVecs(base_path, &vectors, &vecs_size, &vecs_dim);
  // for (size_t i = 0; i < 16000000ul * 128ul; i++) {
  //   vectors_vec[i] = vectors[i % 10000000];
  // }
  // vecs_size = 16000000;
  // auto knn_graph =
  //     gpuknn::NNDescent(vectors_vec.data(), vecs_size, vecs_dim, 6);

  out << knn_graph.size() << " " << k << endl;
  for (int i = 0; i < knn_graph.size(); i++) {
    const auto &x = knn_graph[i];
    out << i << " " << x.size() << " ";
    for (auto y : x) {
      // out << y.distance() << ", " << y.label() << "\t";
      // assert(y.label() != i);
      out << y.label() << "\t";
    }
    out << endl;
  }
  out.close();
  Evaluate(out_path, ground_truth_path);
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
  // string base_path = "/home/hwang/data/sift100k/sift100k.txt";
  // string out_path = "/home/hwang/data/result/sift100k_knng_k64_merged.txt";
  // string ground_truth_path =
  //     "/home/hwang//data/sift100k/sift100k_groundtruth_self.txt";
  // string base_path = "/home/hwang/data/glove100k/glove100k_norm_base.txt";
  // string out_path = "/home/hwang/data/result/glove100k_knng_k64_merged.txt";
  // string ground_truth_path =
  //     "/home/hwang/data/glove100k/glove100k_self_ground_truth.txt";
  // string base_path = "/home/hwang/data/sift1m/sift1m.fvecs";
  // string out_path = "/home/hwang/data/result/sift1m_knng_k64_merged.txt";
  // string ground_truth_path =
  //     "/home/hwang/data/sift1m/sift1m_gold_knn40_sorted.txt";
  string base_path = "/home/hwang//data/sift10m/sift10m.fvecs";
  string out_path = "/home/hwang/data/result/sift10m_knng_k64.txt";
  string ground_truth_path =
      "/home/hwang/data/sift10m/sift10m_gold_knn40.txt";
  float *vectors;
  int vecs_size, vecs_dim;
  FileTool::ReadBinaryVecs(base_path, &vectors, &vecs_size, &vecs_dim);
  // FileTool::ReadVecs(vectors, vecs_size, vecs_dim, base_path);
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
  gpuknn::KNNMerge(&knngraph_merged_dev, vectors_first_dev, vectors_first_size,
                   knngraph_first_dev, vectors_second_dev, vectors_second_size,
                   knngraph_second_dev, true);

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
  cudaFree(knngraph_merged_dev);
  cudaFree(knngraph_first_dev);
  cudaFree(knngraph_second_dev);
  delete[] vectors;
  delete[] vectors_first;
  delete[] vectors_second;
}

void TestFileTools() {
  // float *vectors;
  // int num, dim;
  // FileTool::ReadBinaryVecs("/home/hwang/data/sift10m/sift10m.fvecs", &vectors, &num,
  //                          &dim);
  // cerr << num << endl;
  // for (int i = 0; i < 512; i++) {
  //   if (i % 128 == 0) puts("\n");
  //   printf("%.0f ", vectors[i]);
  // } puts("");
  // delete[] vectors;

  // float *vectors;
  // int num = 50000000, dim;
  // FileTool::ReadBinaryVecs("/home/hwang/data/deep1b/deep1b.fvecs", &vectors,
  //                          &dim, 50000000, num);
  // for (int i = 0; i < 128; i++) {
  //   if (i % 128 == 0) puts("\n");
  //   printf("%f ", vectors[i]);
  // } puts("");
  // FileTool::WriteBinaryVecs("/home/hwang/data/deep100m/deep100m_2.fvecs", vectors,
  //                           num, dim);
  // delete[] vectors;

  float *vectors;
  int num, dim;
  FileTool::ReadVecs(vectors, num, dim,
                     "/home/hwang/data/new_yfcc1m/yfcc1m_txt.txt");
  FileTool::WriteBinaryVecs("/home/hwang/data/new_yfcc1m/yfcc1m.fvecs", vectors,
                            num, dim);
}

void TestMemoryManager() {
  PredPeakGPUMemory(16000000, 128, 32, 32, false);
}

void TestDataManager() {
  KNNDataManager data_manager("/home/hwang/data/deep1b/deep1b");
  data_manager.CheckStatus();
  string cmd;
  while (cin >> cmd) {
    auto start = chrono::steady_clock::now();
    if (cmd == "add") {
      int id;
      cin >> id;
      data_manager.ActivateShard(id);
    } else if (cmd == "del"){
      int id;
      cin >> id;
      data_manager.DiscardShard(id);
    } else if (cmd == "qry") {
      int id;
      cin >> id;
      for (int i = 0; i < 96; i++) {
        cerr << data_manager.GetVectors(id)[i] << " ";
      } cerr << endl;
    } 
    else {
      cout << "Unknown command" << endl;
    }
    auto end = chrono::steady_clock::now();
    float time_cost =
        (float)chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count() /
        1e6;
    cout << "Time costs: " << time_cost << endl;
    data_manager.OutPutActiveIds();
  }
}

void TestConstructLargeKNNGraph() {
  GenLargeKNNGraph("/home/hwang/data/sift1m/sift1m",
                   "/home/hwang/data/result/sift1m.kgraph", 64);
}

void CheckKNNGraph() {
  NNDElement *knn_graph;
  int num, k;
  FileTool::ReadBinaryVecs("/home/hwang/data/sift10m/sift10m.kgraph", &knn_graph,
                           &num, &k);
  cout << num << " " << k << endl;
  int id;
  while (cin >> id) {
    for (int i = 0; i < k; i++) {
      printf("(%f, %d) ", knn_graph[id * k + i].label(),
             knn_graph[id * k + i].distance());
    } puts("");
  }
  delete[] knn_graph;
}

void TxtToIVecs() {
  string in_path = "/home/hwang/data/sift1m/sift1m_gold_knn40_sorted.txt";
  string out_path = "/home/hwang/data/sift1m/sift1m_knngraph_k40.ivecs";
  ifstream in(in_path);
  int graph_size, dim;
  in >> graph_size >> dim;
  int *knn_graph = new int[graph_size * dim];
  for (int i = 0; i < graph_size; i++) {
    int id, neighb_num;
    in >> id >> neighb_num;
    for (int j = 0; j < neighb_num; j++) {
      int nb_id;
      in >> nb_id;
      knn_graph[i * neighb_num + j] = nb_id;
    }
  }
  FileTool::WriteBinaryVecs(out_path, knn_graph, graph_size, dim);
  delete[] knn_graph;
}

int main() {
  // UnitTest();
  // TestKNNAlgorithm();
  // TestCUDANNDescent();
  // TestDataManager();
  TestConstructLargeKNNGraph();
  // TxtToIVecs();
  // TestFileTools();
  // TestMemoryManager();
  // TestCUDAMerge();
  // TestTiledDistanceCompare();
  // TestCUDADistance();
  // TestCUDASearch();
  // TestCUDANewSearch();
  // TestCUDAPriorityQueue();
  // GetRGraph();
  // CheckKNNGraph();
}