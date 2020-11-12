#ifndef MYKNN_DISTFUNC_HPP
#define MYKNN_DISTFUNC_HPP
#include <vector>
#include <cmath>
namespace xmuknn {

    static float GetDistance(const std::vector<int> &a, const std::vector<int> &b) {
        float sum = 0;
        for (int i = 0; i < a.size(); i++) {
            sum += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return sum;
    }

    static float GetDistance(const float* vec1_start, const float* vec1_end, const float* vec2_start, const float* vec2_end) {
        float sum = 0;
        for (int i = 0; vec1_start + i != vec1_end; i++) {
            float diff = vec1_start[i] - vec2_start[i];
            sum += diff * diff;
        }
        return sum;
    }

    static float GetDistance(const float* vec1_start, const float* vec2_start, const int dim) {
        //extern long long dist_calc_count;
        float sum = 0;
        for (int i = 0; i < dim; i++) {
            float diff = vec1_start[i] - vec2_start[i];
            sum += diff * diff;
        }
        return sum;
    }
}
#endif