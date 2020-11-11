#ifndef XMUKNN_RESULT_ELEMENT_CUH
#define XMUKNN_RESULT_ELEMENT_CUH

#include "cuda_runtime.h"

#include <cuda.h>

#define RE_EPS 1e-10
struct ResultElement {
    float distance;
    int label;
    __host__ __device__ ResultElement() { distance = 0, label = 0; }
    __host__ __device__ ResultElement(float distance, int label) : distance(distance), label(label) {}
    __host__ __device__ bool operator < (const ResultElement& other) const {
        if (fabs(this->distance - other.distance) < RE_EPS) return this->label < other.label;
        return this->distance < other.distance;
    }
    __host__ __device__ bool operator == (const ResultElement& other) const {
        return (fabs(this->distance - other.distance) < RE_EPS) && (this->label == other.label);
    }
    __host__ __device__ bool operator >= (const ResultElement& other) const {
        return !(*this < other);
    }
    __host__ __device__ bool operator <= (const ResultElement& other) const {
        return (*this == other) || (*this < other);
    }
    __host__ __device__ bool operator > (const ResultElement& other) const {
        return !(*this <= other);
    }
    __host__ __device__ bool operator != (const ResultElement& other) const {
        return !(*this == other);
    } 
};
#endif