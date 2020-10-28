#ifndef XMUKNN_RSLTELMT_CUH
#define XMUKNN_RSLTELMT_CUH

#include "cuda_runtime.h"

#include <cuda.h>

struct ResultElement {
    float distance;
    int label;
    __host__ __device__ ResultElement() { distance = 0, label = 0; }
    __host__ __device__ ResultElement(float distance, int label) : distance(distance), label(label) {}
    __host__ __device__ bool operator < (const ResultElement& other) const {
        if (this->distance == other.distance) return this->label < other.label;
        return this->distance < other.distance;
    }
};
#endif