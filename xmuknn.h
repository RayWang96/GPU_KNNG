#ifndef XMUKNN_XMUKNN_HPP
#define XMUKNN_XMUKNN_HPP
#include <ctime>
#include <list>
#include <random>
#include <string>
#include <vector>

namespace xmuknn {
extern size_t time_sum;
typedef std::vector<std::vector<int>> Graph;
typedef std::pair<int, std::list<int>> GraphElement;
enum AlgType { NSW_t = 0, HNSW_t };
enum DistFuncType { l2f = 0 };
class KNNAlgorithm {
 public:
  virtual void SearchKNN(std::vector<int> &result, const float *query_vec,
                         const int &k, int iterations = 1) = 0;
  virtual void SaveIndex(const std::string &save_path) const = 0;
  virtual void LoadIndex(const std::string &load_path, AlgType index_type) = 0;
  virtual void test() = 0;
};

extern long long dist_calc_count;
extern std::uniform_real_distribution<float> unif;
extern std::default_random_engine rand_engine;
static void GenerateRandomSequence(std::vector<int> &result,
                                   const int &need_number, const int max_value,
                                   const std::vector<int> &exclusion) {
  std::vector<bool> visited(max_value);
  for (auto x : exclusion) {
    visited[x] = true;
  }
  result = std::vector<int>(need_number);
  for (int i = 0; i < need_number; i++) {
    int num = rand_engine() % max_value;
    while (visited[num]) {
      num = rand_engine() % max_value;
    }
    visited[num] = true;
    result[i] = num;
  }
}
}  // namespace xmuknn

#endif