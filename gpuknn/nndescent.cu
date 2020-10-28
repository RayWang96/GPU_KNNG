#include <vector>
#include <device_functions.h>
#include <iostream>
#include <assert.h>
#include <bitset>
#include <algorithm>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gpudist.cuh"
#include "nndescent.cuh"
#include "../xmuknn.h"
#include "../tools/distfunc.hpp"

#ifdef __INTELLISENSE__
#include "../intellisense_cuda_intrinsics.h"
#endif

using namespace std;
using namespace xmuknn;
bitset<100000> visited;

/*
 * @param graph             A k-NN graph.
 */
pair<Graph, Graph> GetNBGraph(vector<vector<gpuknn::NNDItem>>& kNN_graph, const float *vectors, const int vecs_size, const int vecs_dim) {
    int sample_num = 30;
    Graph graph_new, graph_rnew, graph_old, graph_rold;
    graph_new = graph_rnew = graph_old = graph_rold = Graph(kNN_graph.size());
    for (int i = 0; i < kNN_graph.size(); i++) {
        visited.reset();
        int cnt = 0;
        int last_cnt = 0;
        while (cnt < sample_num) {
            for (int j = 0; j < kNN_graph[i].size(); j++) {
                auto& item = kNN_graph[i][j];
                if (visited[item.id]) continue;
                visited[item.id] = 1;
                if (item.visited) {
                    graph_old[i].push_back(item.id);
                }
                else {
                    if (cnt < sample_num) {
                        graph_new[i].push_back(item.id);
                        cnt++;
                        item.visited = true;
                    }
                }
                if (cnt >= sample_num) break;
            }
            if (last_cnt == cnt) break;
            last_cnt = cnt;
        }
    }
    for (int i = 0; i < kNN_graph.size(); i++) {
        for (int j = 0; j < graph_new[i].size(); j++) {
            auto& id = graph_new[i][j];
            graph_rnew[id].push_back(i);
        }
        for (int j = 0; j < graph_old[i].size(); j++) {
            auto& id = graph_old[i][j];
            graph_rold[id].push_back(i);
        }
    }

    for (int i = 0; i < kNN_graph.size(); i++) {
        visited.reset();
        int cnt = 0;
        int last_cnt = 0;
        while (cnt < sample_num) {
            for (int j = 0; j < graph_rnew[i].size(); j++) {
                int x = graph_rnew[i][j];
                if (visited[x]) continue;
                visited[x] = 1;
                cnt++;
                graph_new[i].push_back(x);
                if (cnt >= sample_num) break;
            }
            if (cnt == last_cnt) break;
            last_cnt = cnt;
        }
        cnt = 0;
        last_cnt = 0;
        while (cnt < sample_num) {
            for (int j = 0; j < graph_rold[i].size(); j++) {
                int x = graph_rold[i][j];
                if (visited[x]) continue;
                visited[x] = 1;
                cnt++;
                graph_old[i].push_back(x);
                if (cnt >= sample_num) break;
            }
            if (cnt == last_cnt) break;
            last_cnt = cnt;
        }
    }
    
    for (int i = 0; i < kNN_graph.size(); i++) {
        sort(graph_new[i].begin(), graph_new[i].end());
        graph_new[i].erase(unique(graph_new[i].begin(), graph_new[i].end()), graph_new[i].end());

        sort(graph_old[i].begin(), graph_old[i].end());
        graph_old[i].erase(unique(graph_old[i].begin(), graph_old[i].end()), graph_old[i].end());
    }
    return make_pair(graph_new, graph_old);
}

vector<vector<int>> GetQueryMat(const Graph &graph_new, const Graph &graph_old, const int nbs_per_nb) {
    vector<vector<int>> result(graph_new.size());
    for (int i = 0; i < graph_new.size(); i++) {
        for (int j = 0; j < graph_new[i].size(); j++) {
            auto a = graph_new[i][j];
            for (int k = j + 1; k < graph_new[i].size(); k++) {
                auto b = graph_new[i][k];
                if (result[a].size() < 1500)
                    result[a].push_back(b);
            }
            for (int k = 0; k < graph_old[i].size(); k++) {
                auto b = graph_old[i][k];
                if (result[a].size() < 1500)
                    result[a].push_back(b);
            }
        }
    }
    for (auto& nbs : result) {
        sort(nbs.begin(), nbs.end());
        nbs.erase(unique(nbs.begin(), nbs.end()), nbs.end());
    }
    return result;
}

int UpdateNBs(vector<gpuknn::NNDItem> *nbs_ptr, const pair<float, int> &p, const int k) {
    auto& nbs = *nbs_ptr;
    if (p.first > (*nbs.rbegin()).distance) return 0;
    int i = 0;
    while (i < nbs.size() && nbs[i].distance < p.first) {
        i++;
    }
    if (i >= k) return 0;
    if (i < nbs.size() && p.second == nbs[i].id) return 0;
    nbs.insert(nbs.begin() + i, gpuknn::NNDItem(p.second, false, p.first));
    nbs.pop_back();
    return 1;
}

long long UpdateNBGraph(vector<vector<gpuknn::NNDItem>>* graph_ptr, const int index, const vector<pair<float, int>> &new_nbs, int k) {
    auto& graph = *graph_ptr;
    long long sum = 0;
    for (const auto& p : new_nbs) {
        if (index == p.second) continue;
        sum += UpdateNBs(&graph[index], p, k);
        sum += UpdateNBs(&graph[p.second], make_pair(p.first, index), k);
    }
    //cerr << index << " " << sum << endl;
    return sum;
}

namespace gpuknn {
    Graph NNDescent(const float* vectors, const int vecs_size, const int vecs_dim) {
        int k = 40;
        int iteration = 6;

        Graph result(vecs_size);
        vector<vector<NNDItem>> g(vecs_size);
        vector<int> tmp_vec;

        for (int i = 0; i < vecs_size; i++) {
            xmuknn::GenerateRandomSequence(tmp_vec, k, vecs_size);
            for (int j = 0; j < k; j++) {
                int nb_id = tmp_vec[j];
                if (nb_id == i) continue;
                g[i].emplace_back(nb_id, false, GetDistance(vectors + (size_t)i * vecs_dim, 
                                                            vectors + (size_t)nb_id * vecs_dim,
                                                            vecs_dim));
            }
        }
        for (int i = 0; i < g.size(); i++) {
            sort(g[i].begin(), g[i].end(), [](NNDItem a, NNDItem b) {
                    return a.distance < b.distance;
                });
        }
        auto cuda_status = cudaSuccess;
        int* nbsnbs_dev, *index_dev; //It's array contains neighbor's neighbors.
        float* result_dev; float* vecs_dev;
        int* nbsnbs = NULL; //Neighbor's neighbors for query
        long long cmp_cnt = 0;
        cuda_status = cudaMalloc(&nbsnbs_dev, (size_t)k * k * sizeof(int));
        cuda_status = cudaMalloc(&index_dev, (size_t)k * k * sizeof(int));
        cuda_status = cudaMalloc(&result_dev, (size_t)k * k * sizeof(float));
        cuda_status = cudaMalloc(&vecs_dev, (size_t)vecs_size * vecs_dim * sizeof(float));
        
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cuda_status));
            goto Error;
        }
        cuda_status = cudaMemcpy(vecs_dev, vectors, (size_t)vecs_size * vecs_dim * sizeof(float), cudaMemcpyHostToDevice);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cuda_status));
            goto Error;
        }

        cudaMallocHost(&nbsnbs, (size_t)k * k * sizeof(int));
        for (int t = 0; t < iteration; t++) {
            auto start = clock();
            auto new_and_old_graph = GetNBGraph(g, vectors, vecs_size, vecs_dim);
            auto query_mat = GetQueryMat(new_and_old_graph.first, new_and_old_graph.second, k);
            auto end = clock();
            cerr << "GetNBGraph costs " << (1.0 * end - start) / CLOCKS_PER_SEC << endl;

            start = clock();
            vector<pair<float, int>> tmp_result;
            long long update_times = 0;
            for (int i = 0; i < vecs_size; i++) {
                //cerr << "\r" << i;
                int index_size = query_mat[i].size();
                for (int j = 0; j < index_size; j++) {
                    nbsnbs[j] = query_mat[i][j];
                }
                if (index_size == 0) continue;
                cmp_cnt += index_size;
                cuda_status = cudaMemcpy(nbsnbs_dev, nbsnbs, (size_t)index_size * sizeof(int), cudaMemcpyHostToDevice);
                if (cuda_status != cudaSuccess) {
                    cerr << index_size << endl;
                    fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cuda_status));
                    goto Error;
                }
                GetOneToNDistanceAllByIndex(&tmp_result, result_dev, vecs_dev, vecs_size, vecs_dim, i, nbsnbs, nbsnbs_dev, index_size);
                auto ss = clock();
                update_times += UpdateNBGraph(&g, i, tmp_result, k);
                time_sum += clock() - ss;
            } //cerr << endl;
            end = clock();
            cerr << "update_times: " << update_times << endl;
            cerr << "cmp_cnt: " << cmp_cnt << endl; cmp_cnt = 0;
            cerr << "Iteration costs " << (1.0 * end - start) / CLOCKS_PER_SEC << endl;
            cerr << "Kernel costs " << (1.0 * time_sum) / CLOCKS_PER_SEC << endl;
            cerr << endl;
            time_sum = 0;
        }
    Error:
        cudaFreeHost(nbsnbs);
        cudaFree(nbsnbs_dev);
        cudaFree(index_dev);
        cudaFree(result_dev);
        cudaFree(vecs_dev);
        for (int i = 0; i < g.size(); i++) {
            for (auto x : g[i]) {
                result[i].push_back(x.id);
            }
        }
        return result;
    }
}