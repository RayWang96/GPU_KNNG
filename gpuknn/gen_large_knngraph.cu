#include <thread>
#include <algorithm>

#include "cuda_runtime.h"
#include "gen_large_knngraph.cuh"
#include "knnmerge.cuh"
#include "nndescent.cuh"
#include "knncuda_tools.cuh"
#include "../tools/nndescent_element.cuh"
#include "../tools/knndata_manager.hpp"
#include "../tools/timer.hpp"
using namespace std;

void ReadGraph(const string &graph_path, NNDElement **knn_graph_ptr,
               const int read_pos, const int read_num) {
  NNDElement *&knn_graph = *knn_graph_ptr;
  int dim;
  FileTool::ReadBinaryVecs(graph_path, &knn_graph, &dim, read_pos, read_num);
}

void WriteGraph(const string &graph_path, const NNDElement *knn_graph,
                const int graph_size, const int k, const int write_pos) {
  FileTool::WriteBinaryVecs(graph_path, knn_graph, write_pos, graph_size, k);
}

void WriteTXTGraph(const string &graph_path, const NNDElement *knn_graph,
                   const int graph_size, const int k, const int write_pos) {
  ofstream out(graph_path);
  for (int i = 0; i < graph_size; i++) {
    out << k << "\t";
    for (int j = 0; j < k; j++) {
      auto elem = knn_graph[i * k + j];
      out << elem.distance() << "\t" << elem.label() << "\t";
    } out << endl;
  }
  out.close();
}

void BuildEachShard(KNNDataManager &data_manager, const string &out_data_path) {
  Timer knn_timer;
  FileTool::CreateBlankKNNGraph(data_manager.GetGraphDataPath(),
                                data_manager.GetVecsNum(), data_manager.GetK());
  FileTool::CreateBlankKNNGraph(out_data_path, data_manager.GetVecsNum(),
                                data_manager.GetK());
  for (int i = 0; i < data_manager.GetShardsNum(); i++) {
    NNDElement *knn_graph;
    data_manager.ActivateShard(i);
    float *vectors_dev;
    cudaMalloc(&vectors_dev, (size_t)data_manager.GetVecsNum(i) *
                                 data_manager.GetDim() * sizeof(float));
    cudaMemcpy(vectors_dev, data_manager.GetVectors(i),
               (size_t)data_manager.GetVecsNum(i) * data_manager.GetDim() *
                   sizeof(float),
               cudaMemcpyHostToDevice);
    cout << "Building No. " << i << endl;
    knn_timer.start();
    gpuknn::NNDescent(&knn_graph, vectors_dev,
                      data_manager.GetVecsNum(i), data_manager.GetDim(), 6,
                      false);
    cout << "End building No." << i << " in " << knn_timer.end() << " seconds"
         << endl;
    WriteGraph(data_manager.GetGraphDataPath(), knn_graph,
               data_manager.GetVecsNum(i), data_manager.GetK(),
               data_manager.GetBeginPosition(i));
    for (int j = 0; j < data_manager.GetVecsNum(i) * data_manager.GetK(); j++) {
      knn_graph[j].SetLabel(knn_graph[j].label() +
                            data_manager.GetBeginPosition(i));
    }
    WriteGraph(out_data_path, knn_graph, data_manager.GetVecsNum(i),
               data_manager.GetK(), data_manager.GetBeginPosition(i));
    data_manager.DiscardShard(i);
    cudaFree(vectors_dev);
  }
}

int MergeList(const NNDElement *A, const int m, const NNDElement *B,
              const int n, NNDElement *C, const int max_size) {
  int i = 0, j = 0, cnt = 0;
  while ((i < m) && (j < n)) {
    if (A[i] <= B[j]) {
      C[cnt++] = A[i++];
      if (cnt >= max_size) goto EXIT;
    } else {
      C[cnt++] = B[j++];
      if (cnt >= max_size) goto EXIT;
    }
  }

  if (i == m) {
    for (; j < n; j++) {
      C[cnt++] = B[j];
      if (cnt >= max_size) goto EXIT;
    }
  } else {
    for (; i < m; i++) {
      C[cnt++] = A[i];
      if (cnt >= max_size) goto EXIT;
    }
  }
EXIT:
  return cnt;
}

void UpdateKNNGraph(NNDElement **old_graph_ptr, const NNDElement *new_graph,
                    const int graph_size, const int k) {
  NNDElement *&old_graph = *old_graph_ptr;
  NNDElement *tmp_list = new NNDElement[k * 2];
  for (int i = 0; i < graph_size; i++) {
    MergeList(&old_graph[i * k], k, &new_graph[i * k], k, tmp_list, k * 2);
    unique(tmp_list, tmp_list + k * 2);
    for (int j = 0; j < k; j++) {
      old_graph[i * k + j] = tmp_list[j];
    }
  }
  delete[] tmp_list;
}

void PreProcID(NNDElement *result_knn_graph_host, const int first_graph_size,
               const int graph_size, const int k, const int offset_a,
               const int offset_b) {
  for (int i = 0; i < graph_size; i++) {
    for (int j = 0; j < k; j++) {
      auto &elem = result_knn_graph_host[i * k + j];
      if (elem.label() >= first_graph_size) {
        elem.SetLabel(elem.label() - first_graph_size + offset_b);
      } else {
        elem.SetLabel(elem.label() + offset_a);
      }
    }
  }
}

void GenLargeKNNGraph(const string &vecs_data_path,
                      const string &out_data_path,
                      const int k) {
  KNNDataManager data_manager(vecs_data_path);
  data_manager.CheckStatus();
  BuildEachShard(data_manager, out_data_path);
  int shards_num = data_manager.GetShardsNum();
  for (int i = 0; i < shards_num - 1; i++) {
    data_manager.ActivateShard(i);
    data_manager.ActivateShard(i + 1);
    NNDElement *result_first, *result_second;
    ReadGraph(out_data_path, &result_first, data_manager.GetBeginPosition(i),
              data_manager.GetVecsNum(i));
    ReadGraph(out_data_path, &result_second, data_manager.GetBeginPosition(i + 1),
              data_manager.GetVecsNum(i + 1));
    for (int j = i + 1; j < shards_num; j++) {
      NNDElement *result_knn_graph_dev;
      thread th1([&result_knn_graph_dev, &data_manager, i, j]() {
        Timer timer;
        timer.start();
        gpuknn::KNNMergeFromHost(
            &result_knn_graph_dev, data_manager.GetVectors(i),
            data_manager.GetVecsNum(i), data_manager.GetKNNGraph(i),
            data_manager.GetVectors(j), data_manager.GetVecsNum(j),
            data_manager.GetKNNGraph(j));
        cout << "Merge costs: " << timer.end() << endl;
      });
      th1.join();
      thread th2([&data_manager, &shards_num, &i, &j]() {
        if (j + 1 < shards_num) {
          data_manager.ActivateShard(j + 1);
        }
      });
      th2.join();
      NNDElement *result_knn_graph_host;
      ToHostKNNGraph(&result_knn_graph_host, result_knn_graph_dev,
                     data_manager.GetVecsNum(i) + data_manager.GetVecsNum(j),
                     data_manager.GetK());
      thread th3([&result_first, &result_second, &result_knn_graph_host,
                  &data_manager, &i, &j, &out_data_path]() {
        Timer timer;
        timer.start();
        PreProcID(result_knn_graph_host, data_manager.GetVecsNum(i),
                  data_manager.GetVecsNum(i) + data_manager.GetVecsNum(j),
                  data_manager.GetK(), data_manager.GetBeginPosition(i),
                  data_manager.GetBeginPosition(j));
        // WriteTXTGraph(string("/home/hwang/codes/GPU_KNNG/tmp/") + to_string(i) +
        //                   to_string(j) + ".txt",
        //               result_knn_graph_host,
        //               data_manager.GetVecsNum(i) + data_manager.GetVecsNum(j),
        //               data_manager.GetK(), 0);
        UpdateKNNGraph(&result_first,
                       &result_knn_graph_host[0],
                       data_manager.GetVecsNum(i), data_manager.GetK());
        UpdateKNNGraph(
            &result_second,
            &result_knn_graph_host[(size_t)data_manager.GetVecsNum(i) *
                                   data_manager.GetK()],
            data_manager.GetVecsNum(j), data_manager.GetK());
        WriteGraph(out_data_path, result_second, data_manager.GetVecsNum(j),
                   data_manager.GetK(), data_manager.GetBeginPosition(j));
        data_manager.DiscardShard(j); 
        cout << "Updating KNN graph costs: " << timer.end() << endl;
      });
      th3.join();
      // printf("The result of %d: ", data_manager.GetBeginPosition(i));
      // for (int k = 0; k < 64; k++) {
      //   printf("(%f, %d) ", result_first[k].distance(),
      //          result_first[k].label());
      // } puts("");
    }
    data_manager.DiscardShard(i);
  }
  return;
}