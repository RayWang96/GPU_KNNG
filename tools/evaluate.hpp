#ifndef XMUKNN_EVALUATE_HPP
#define XMUKNN_EVALUATE_HPP
#include <algorithm>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "filetool.hpp"
#include "nndescent_element.cuh"
using namespace std;

float Evaluate(const string &result_path, const string &ground_truth_path,
               const int recall_at, int cmp_list_cnt = -1) {
  NNDElement *result_graph;
  int graph_size, k;
  FileTool::ReadBinaryVecs(result_path, &result_graph, &graph_size, &k);
  int *gt_graph;
  int gt_graph_size, gt_k;
  FileTool::ReadBinaryVecs(ground_truth_path, &gt_graph, &gt_graph_size, &gt_k);
  int true_positive = 0, false_negative = 0;
  vector<int> a, b, c;

  if (cmp_list_cnt == -1) {
    cmp_list_cnt = min(gt_graph_size, graph_size);
  }
  for (int i = 0; i < cmp_list_cnt; i++) {
    a.clear();
    b.clear();
    for (int j = 0; j < recall_at; j++) {
      a.push_back(result_graph[i * k + j].label());
    }
    for (int j = 0; j < recall_at; j++) {
      b.push_back(gt_graph[i * gt_k + j]);
    }
    c.resize(a.size() + b.size());
    int cnt =
        set_intersection(a.begin(), a.end(), b.begin(), b.end(), c.begin()) -
        c.begin();
    true_positive += cnt;
    false_negative += recall_at - cnt;
  }
  float recall = true_positive / (1.0 * true_positive + false_negative);

  delete[] result_graph;
  delete[] gt_graph;
  return recall;
}

float EvaluateHead(const string &result_path, const string &ground_truth_path,
                   const int recall_at) {
  int *gt_graph;
  int gt_graph_size, gt_k;
  FileTool::ReadBinaryVecs(ground_truth_path, &gt_graph, &gt_graph_size, &gt_k);
  
  NNDElement *result_graph;
  int graph_size = gt_graph_size, k;
  FileTool::ReadBinaryVecs(result_path, &result_graph, &k, 0, gt_graph_size);

  int true_positive = 0, false_negative = 0;
  vector<int> a, b, c;
  for (int i = 0; i < gt_graph_size; i++) {
    a.clear();
    b.clear();
    for (int j = 0; j < recall_at; j++) {
      a.push_back(result_graph[i * k + j].label());
    }
    for (int j = 0; j < recall_at; j++) {
      b.push_back(gt_graph[i * gt_k + j]);
    }
    c.resize(a.size() + b.size());
    sort(a.begin(), a.end());
    sort(b.begin(), b.end());
    int cnt =
        set_intersection(a.begin(), a.end(), b.begin(), b.end(), c.begin()) -
        c.begin();
    true_positive += cnt;
    false_negative += recall_at - cnt;
  }
  float recall = true_positive / (1.0 * true_positive + false_negative);

  delete[] result_graph;
  delete[] gt_graph;
  return recall;
}

#endif