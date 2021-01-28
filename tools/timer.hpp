#ifndef XMUKNN_TIMER_HPP
#define XMUKNN_TIMER_HPP

#include <chrono>
using namespace std;
class Timer {
  chrono::_V2::steady_clock::time_point start_;

 public:
  void start() { start_ = chrono::steady_clock::now(); };
  float end() {
    auto end = chrono::steady_clock::now();
    float tmp_time =
        (float)chrono::duration_cast<std::chrono::microseconds>(end - start_)
            .count() /
        1e6;
    return tmp_time;
  }
};

#endif