#include <vector>
#include <iostream>
#include <istream>
#include <algorithm>
#include <chrono>

#include "xmuknn.h"
#include "tools/filetool.hpp"
#include "tools/distfunc.hpp"

#include "gpuknn/nndescent.cuh"
#include "gpuknn/unittest.cu"

using namespace std;
using namespace xmuknn;

void evaluate(const string &data_path, const string &ground_truth_path) {
    string cmd = "python3 -u \"/home/hwang/codes/GPU_KNNG/tools/evaluate.py\"";
    cmd += " "; cmd += data_path;
    cmd += " "; cmd += ground_truth_path;
    system(cmd.c_str());
}

struct KNNItem {
    int id;
    bool visited = false;
    KNNItem(int id, bool visited) :id(id), visited(visited) {}
};

void TestCUDANNDescent() {
    int k = 30;
    // string out_path = FileTool::GetOutPath();

    // string base_path 
    //     = "/mnt/d/cache/hwang_data/data/sift10k/sift10k.txt";
    // string out_path 
    //     = "/mnt/d/cache/hwang_data/data/result/sift10k_knng_k32.txt";
    // string ground_truth_path 
    //     = "/mnt/d/cache/hwang_data/data/sift10k/sift10k_groundtruth_self.txt";

    string base_path 
        = "/mnt/d/cache/hwang_data/data/sift100k/sift100k.txt";
    string out_path 
        = "/mnt/d/cache/hwang_data/data/result/sift100k_knng_k30.txt";
    string ground_truth_path 
        = "/mnt/d/cache/hwang_data/data/sift100k/sift100k_groundtruth_self.txt";

    // string base_path 
    //     = "/mnt/d/cache/hwang_data/data/sift1m/sift1m.txt";
    // string out_path 
    //     = "/mnt/d/cache/hwang_data/data/result/sift1m_knng_k30.txt";
    // string ground_truth_path 
    //     = "/mnt/d/cache/hwang_data/data/sift1m/sift1m_gold_knn40_sorted.txt";

    auto out = ofstream(out_path);
    if (!out.is_open()) {
        cerr << "Output file is not opened!" << endl;
        return;
    }

    float* vectors;
    int vecs_size, vecs_dim;
    FileTool::ReadVecs(vectors, vecs_size, vecs_dim, base_path);

    auto start = chrono::steady_clock::now();
    auto knn_graph = gpuknn::NNDescent(vectors, vecs_size, vecs_dim);
    auto end = chrono::steady_clock::now();

    out << knn_graph.size() << " " << k << endl;
    for (int i = 0; i < knn_graph.size(); i++) {
        const auto &x = knn_graph[i];
        out << i << " " << x.size() << " ";
        for (auto y : x) {
            // out << y.distance << " " << y.id << "\t";
            out << y.id << "\t";
        } out << endl;
    }
    out.close();
    cerr << "GPU NNDescent costs: "
         << (float)chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e6
         << endl;
    evaluate(out_path, ground_truth_path);
}

void TestCUDAASM() {
    int x, pos;
    while (cin >> x) {
        cerr << "start" << endl;
        for (int i = 0; i < 32; i++) {
            TestASMKernel<<<dim3(1), dim3(1)>>> (x, i);
            cudaDeviceSynchronize();
        } cerr << endl;
        cerr << "end" << endl;
    }
}

void UnitTest() {
    // TestKNNListInsert();
    // TestCUDAASM();
    TestLocalListUpdate();
}

int main() {
    UnitTest();
    //TestKNNAlgorithm();
    // TestCUDANNDescent();
    // TestTiledDistanceCompare();
    //TestCUDADistance();
    //TestCUDASearch();
    //TestCUDANewSearch();
    //TestCUDAPriorityQueue();
    //GetRGraph();
}