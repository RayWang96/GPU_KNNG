#include <vector>
#include <iostream>
#include <istream>
#include <algorithm>

#include "xmuknn.h"
#include "tools/filetool.hpp"
#include "tools/distfunc.hpp"

#include "gpuknn/gpudist.cuh"
#include "gpuknn/nndescent.cuh"

using namespace std;
using namespace xmuknn;

void evaluate(const string &data_path, const string &ground_truth_path) {
    string cmd = "python3 -u \"/media/data4/huiwang/codes/gpuknng/tools/evaluate.py\"";
    cmd += " "; cmd += data_path;
    cmd += " "; cmd += ground_truth_path;
    system(cmd.c_str());
}

void MergeGraphWithRGraph(Graph& graph, const float* vectors, const int& dim, const float& ratio1, const float& ratio2) {
    auto start = clock();
    Graph rgraph(graph.size());
    for (int i = 0; i < graph.size(); i++) {
        for (auto x : graph[i]) {
            rgraph[x].push_back(i);
        }
    }
    // for (int i = 0; i < rgraph.size(); i++) {
    //     sort(rgraph[i].begin(), rgraph[i].end(), [&vectors, &i, &dim](int a, int b) {
    //         float dist_a = GetDistance(vectors + i * dim, vectors + a * dim, dim);
    //         float dist_b = GetDistance(vectors + i * dim, vectors + b * dim, dim);
    //         return dist_a < dist_b;
    //     });
    // }
    Graph new_graph(graph.size());
    for (int i = 0; i < graph.size(); i++) {
        for (int j = 0; j < graph[i].size() * ratio1; j++) {
            new_graph[i].push_back(graph[i][j]);
        }
        for (int j = 0; j < rgraph[i].size() * ratio2; j++) {
            new_graph[i].push_back(rgraph[i][j]);
        }
    }
     for (int i = 0; i < graph.size(); i++) {
         sort(graph[i].begin(), graph[i].end(), [&vectors, &i, &dim](int a, int b) {
             float dist_a = GetDistance(vectors + i * dim, vectors + a * dim, dim);
             float dist_b = GetDistance(vectors + i * dim, vectors + b * dim, dim);
             return dist_a < dist_b;
         });
         auto it = unique(graph[i].begin(), graph[i].end());
         graph[i].erase(it, graph[i].end());
     }
    graph = move(new_graph);
    auto end = clock();
    //cerr << "Build graph spends:" << 1.0 * (end - start) / CLOCKS_PER_SEC << endl;
    return;
}

struct KNNItem {
    int id;
    bool visited = false;
    KNNItem(int id, bool visited) :id(id), visited(visited) {}
};

void TestCUDANNDescent() {
    int k = 100;
    // string out_path = FileTool::GetOutPath();
    string base_path = "/media/data4/huiwang/data/sift100k/sift100k.txt";
    string out_path = "/media/data4/huiwang/data/result/sift100k_knng_k100.txt";
    string ground_truth_path = "/media/data4/huiwang/data/sift100k/sift100k_groundtruth_self.txt";

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

void GetRGraph() {
    Graph graph;
    FileTool::Read2DVector(graph, FileTool::GetGraphPath());

    int vecs_num, vecs_dim;
    float* vectors;
    FileTool::ReadVecs(vectors, vecs_num, vecs_dim, FileTool::GetFilePath());
    MergeGraphWithRGraph(graph, vectors, vecs_dim, 1, 1);
    auto out = ofstream("D:/KNNDatasets/rand10k/rand10k_knn_graph_r.txt");
    out << graph.size() << " " << vecs_dim << endl;
    for (int i = 0; i < graph.size(); i++) {
        out << i << " " << graph[i].size() << " ";
        for (auto x : graph[i]) {
            out << x << " ";
        } out << endl;
    }
}

int main() {
    //TestKNNAlgorithm();
    TestCUDANNDescent();
    //TestCUDADistance();
    //TestCUDASearch();
    //TestCUDANewSearch();
    //TestCUDAPriorityQueue();
    //GetRGraph();
}