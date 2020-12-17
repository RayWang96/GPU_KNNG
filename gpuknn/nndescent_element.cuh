#ifndef XMUKNN_NNDESCENT_ELEMENT_CUH
#define XMUKNN_NNDESCENT_ELEMENT_CUH

#include <cuda.h>

#include "cuda_runtime.h"

#define RE_EPS 1e-10
struct NNDElement {
  float distance_;
  int label_;
  __host__ __device__ NNDElement() { distance_ = 1e10, label_ = 0x3f3f3f3f; }
  __host__ __device__ NNDElement(float distance, int label, bool is_new = true)
      : distance_(distance), label_(label) {
    if (!is_new) {
      label_ = -label - 1;
    }
  }
  __host__ __device__ bool IsNew() const { return label_ >= 0; }
  __host__ __device__ void SetLabel(const int new_label) {
    this->label_ = new_label;
  }
  __host__ __device__ void SetDistance(const int new_distance) {
    this->distance_ = new_distance;
  }
  __host__ __device__ int label() const {
    if (this->IsNew()) return label_;
    return -label_ - 1;
  }
  __host__ __device__ int distance() const {
    return distance_;
  }
  __host__ __device__ void MarkOld() { 
    if (label_ >= 0)
      label_ = -label_ - 1; 
  }    
  __host__ __device__ bool operator<(const NNDElement& other) const {
    if (fabs(this->distance_ - other.distance_) < RE_EPS)
      return this->label() < other.label();
    return this->distance_ < other.distance_;
  }
  __host__ __device__ bool operator==(const NNDElement& other) const {
    return this->label() == other.label();
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