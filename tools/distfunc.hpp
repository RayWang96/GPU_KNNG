#ifndef MYKNN_DISTFUNC_HPP
#define MYKNN_DISTFUNC_HPP
#include <vector>
#include <cmath>
#include <xmmintrin.h>
#include <ammintrin.h>

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

    static float GetDistance(const float* vec1_start, const float* vec2_start, const unsigned int dim) {
        #define PORTABLE_ALIGN32 __attribute__((aligned(32)))
        size_t qty4 = dim >> 2;
        size_t qty16 = dim >> 4;
        const float *pVect1 = vec1_start;
        const float *pVect2 = vec2_start;
        const float* pEnd1 = pVect1 + (qty16 << 4);
        const float* pEnd2 = pVect1 + (qty4 << 2);
        const float* pEnd3 = pVect1 + dim;

        float delta, res;
        float PORTABLE_ALIGN32 TmpRes[8];

        __m128  diff, v1, v2;
        __m128  sum = _mm_set1_ps(0);

        while (pVect1 < pEnd1)
        {
            _mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }

        while (pVect1 < pEnd2)
        {
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }

        _mm_store_ps(TmpRes, sum);
        res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

        while (pVect1 < pEnd3)
        {
            delta = *pVect1++ - *pVect2++;
            res += delta * delta;
        }

        return res;
    }
}
#endif