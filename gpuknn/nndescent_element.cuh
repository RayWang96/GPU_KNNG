#ifndef XMUKNN_NNDESCENT_ELEMENT_CUH
#define XMUKNN_NNDESCENT_ELEMENT_CUH

#include <cuda.h>

#include "cuda_runtime.h"

#define RE_EPS 1e-10
struct NNDElement {
  float distance;
  int label;
  int is_new;
  __host__ __device__ NNDElement() {
    distance = 1e10, label = 0x3f3f3f3f, is_new = 1;
  }
  __host__ __device__ NNDElement(float distance, int label)
      : distance(distance), label(label), is_new(1) {}
  __host__ __device__ NNDElement(float distance, int label, int is_new)
      : distance(distance), label(label), is_new(is_new) {}

  __host__ __device__ bool operator<(const NNDElement& other) const {
    if (fabs(this->distance - other.distance) < RE_EPS)
      return this->label < other.label;
    return this->distance < other.distance;
  }
  __host__ __device__ bool operator==(const NNDElement& other) const {
    return (this->label == other.label) && (this->distance == other.distance);
  }
  __host__ __device__ bool operator>=(const NNDElement& other) const {
    return !(*this < other);
  }
  __host__ __device__ bool operator<=(const NNDElement& other) const {
    return (*this == other) || (*this < other);
  }
  __host__ __device__ bool operator>(const NNDElement& other) const {
    return !(*this <= other);
  }
  __host__ __device__ bool operator!=(const NNDElement& other) const {
    return !(*this == other);
  }
};
#endif