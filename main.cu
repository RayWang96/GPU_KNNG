#include <vector>
#include <iostream>
#include <istream>
#include <algorithm>

#include "xmuknn.h"
#include "tools/filetool.hpp"
#include "tools/distfunc.hpp"

#include "gpuknn/gpudist.cuh"
#include "gpuknn/nndescent.cuh"
#include "gpuknn/unittest.cu"

using namespace std;
using namespace xmuknn;

void evaluate(const string &data_path, const string &ground_truth_path) {
    string cmd = "python3 -u \"/media/data4/huiwang/codes/gpuknng/tools/evaluate.py\"";
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
    int k = 32;
    // string out_path = FileTool::GetOutPath();
    string base_path 
        = "/media/data4/huiwang/data/sift10k/sift10k.txt";
    string out_path 
        = "/media/data4/huiwang/data/result/sift10k_knng_k32.txt";
    string ground_truth_path 
        = "/media/data4/huiwang/data/sift10k/sift10k_groundtruth_self.txt";

    auto out = ofstream(out_path);
    if (!out.is_open()) {
        cerr << "Output file is not opened!" << endl;
        return;
    }

    float* vectors;
    int vecs_size, vecs_dim;
    FileTool::ReadVecs(vectors, vecs_size, vecs_dim, base_path);

    auto start = clock();
    Graph knn_graph = gpuknn::NNDescent(vectors, vecs_size, vecs_dim);
    auto end = clock();

    out << knn_graph.size() << " " << k << endl;
    for (int i = 0; i < knn_graph.size(); i++) {
        const auto &x = knn_graph[i];
        out << i << " " << x.size() << " ";
        for (auto y : x) {
            out << y << " ";
        } out << endl;
    }
    out.close();
    cerr << "GPU NNDescent costs " << (1.0 * end - start) / CLOCKS_PER_SEC << " seconds" << endl;
    evaluate(out_path, ground_truth_path);
}

void UnitTest() {
    TestKNNListInsert();
}

int main() {
    // UnitTest();
    //TestKNNAlgorithm();
    TestCUDANNDescent();
    //TestCUDADistance();
    //TestCUDASearch();
    //TestCUDANewSearch();
    //TestCUDAPriorityQueue();
    //GetRGraph();
}