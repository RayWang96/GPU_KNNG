#ifndef XMUKNN_KNNDATA_MANAGER_HPP
#define XMUKNN_KNNDATA_MANAGER_HPP
#include <memory>
#include <set>
#include <map>
#include <iostream>
#include <mutex>

#include "filetool.hpp"
#include "nndescent_element.cuh"
using namespace std;
class KNNDataManager {
 private:
  map<int, unique_ptr<float[]>> vecs_data_;
  map<int, unique_ptr<NNDElement[]>> knngs_data_;
  set<int> active_ids_;

  mutex mtx_;
  string data_path_;
  int dim_;
  int shards_num_;
  int min_shards_num_;
  int max_vecs_num_per_shard_;
  int vecs_num_per_shard_;
  int vecs_num_last_shard_;
  int vecs_num_;
  int k_;
 public:
  KNNDataManager(const string &data_path,
                 const int k = 64,
                 const int min_shards_num = 3,
                 const int max_vecs_num_per_shard = 8500000)
      : data_path_(data_path),
        min_shards_num_(min_shards_num),
        max_vecs_num_per_shard_(max_vecs_num_per_shard),
        k_(k) {
    int vecs_num = FileTool::GetFVecsNum(GetVecsDataPath());
    dim_ = FileTool::GetFVecsDim(GetVecsDataPath());
    vecs_num_ = vecs_num;
    // int knn_lists_num = FileTool::GetKNNListsNum(GetGraphDataPath());
    // if (knn_lists_num != vecs_num) {
    //   FileTool::CreateBlankKNNGraph(GetGraphDataPath(), vecs_num, k_);
    // }
    if (vecs_num < max_vecs_num_per_shard_ * min_shards_num) {
      shards_num_ = min_shards_num_;
    } else {
      shards_num_ = vecs_num / max_vecs_num_per_shard_ +
                    (vecs_num % max_vecs_num_per_shard != 0);
    }
    vecs_num_per_shard_ = vecs_num / shards_num_;
    vecs_num_last_shard_ = vecs_num - vecs_num_per_shard_ * (shards_num_ - 1);
  }
  void OutPutActiveIds() {
    if (active_ids_.empty()) {
      cout << "There is no active ID." << endl;
    } else {
      cout << "There are " << active_ids_.size() << " active ids: ";
      for (auto id : active_ids_) {
        cout << id << " ";
      } cout << endl;
    }
  }
  void CheckStatus() {
    cout << "Data path: " << data_path_ << endl;
    cout << "Total vecs num: " << vecs_num_ << endl;
    cout << "Shards num: " << shards_num_ << endl;
    cout << "Max vecs num. per shard: " << max_vecs_num_per_shard_ << endl;
    cout << "Vecs num. per shard: " << vecs_num_per_shard_ << endl;
    cout << "Vecs num. of last shard: " << vecs_num_last_shard_ << endl;
    OutPutActiveIds();
  }
  string GetVecsDataPath() {
    return data_path_ + ".fvecs";
  }
  string GetGraphDataPath() {
    return data_path_ + ".kgraph";
  }
  int GetK() {
    return k_;
  }
  int GetDim() {
    return dim_;
  }
  int GetBeginPosition(const int id) {
    if (id < shards_num_) {
      return id * vecs_num_per_shard_;
    } else {
      cerr << "GetBeginPosition ID: " << id << "exceed the max. num of shards."
           << endl;
      exit(-1);
    }
  }
  int GetVecsNum() {
    return vecs_num_;
  }
  int GetVecsNum(const int id) {
    if (id < shards_num_ - 1) {
      return vecs_num_per_shard_;
    } else if (id == shards_num_ - 1) {
      return vecs_num_last_shard_;
    } else {
      cerr << "GetVecsNum ID: " << id << "exceed the max. num of shards."
           << endl;
      exit(-1);
    }
  }
  int GetShardsNum() {
    return shards_num_;
  }
  bool IsActive(const int id) {
    lock_guard<mutex> local_lock(mtx_);
    if (active_ids_.find(id) != active_ids_.end()) {
      return true;
    } else {
      return false;
    }
  }
  void ActivateShard(const int id) {
    if (id >= shards_num_) {
      cerr << "No such id in the data manager." << endl;
      exit(-1);
    }
    if (IsActive(id)) {
      cerr << "id: " << id << " shard is already active."
           << endl;
      return;
    }
    lock_guard<mutex> local_lock(mtx_);
    active_ids_.insert(id);
    int begin_pos = id * vecs_num_per_shard_;
    int read_num = vecs_num_per_shard_;
    if (id == shards_num_ - 1) {
      read_num = vecs_num_last_shard_;
    }
    float *vectors;
    int dim = 0;
    FileTool::ReadBinaryVecs(GetVecsDataPath(), &vectors, &dim, begin_pos,
                             read_num);
    vecs_data_.insert(make_pair(id, unique_ptr<float[]>(vectors)));
    return;
  }
  void DiscardShard(const int id) {
    lock_guard<mutex> local_lock(mtx_);
    active_ids_.erase(id);
    vecs_data_.erase(id);
    knngs_data_.erase(id);
  }
  const float *GetVectors(const int id) {
    if (!IsActive(id)) {
      cerr << "GetVectors failed: ID " << id << " is not active" << endl;
      exit(-1);
    }
    lock_guard<mutex> local_lock(mtx_);
    return vecs_data_[id].get();
  }
  const NNDElement *GetKNNGraph(const int id) {
    if (!IsActive(id)) {
      cerr << "GetVectors failed: ID " << id << " is not active" << endl;
      exit(-1);
    }
    lock_guard<mutex> local_lock(mtx_);
    if (knngs_data_.find(id) == knngs_data_.end()) {
      NNDElement *knn_graph;
      int k = 0;
      int begin_pos = GetBeginPosition(id);
      int read_num = GetVecsNum(id);
      FileTool::ReadBinaryVecs(GetGraphDataPath(), &knn_graph, &k, begin_pos,
                               read_num);
      knngs_data_.insert(make_pair(id, unique_ptr<NNDElement[]>(knn_graph)));
    }
    return knngs_data_[id].get();
  }
};
#endif