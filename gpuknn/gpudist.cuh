#ifndef XMUKNN_GPUDIST_CUH
#define XMUKNN_GPUDIST_CUH
#include <vector>
namespace xmuknn {
	template<typename T>
	class CUDAData
	{
	public:
		T* data_ptr = nullptr;
		T* data_dev_ptr = nullptr;
		int size = 0;
		bool if_need_copy = true;
		CUDAData(T* data_ptr, T* data_dev_ptr, int size, bool if_need_copy) : 
		data_ptr(data_ptr), data_dev_ptr(data_dev_ptr), size(size), if_need_copy(if_need_copy)
		{};
		CUDAData() {};
	};

	//float GPUGetDistance(const float* vec1_start, const float* vec2_start, const int dim);
	std::vector<float> GetOneToNDistance(float* vec, float* vecs, const int vecs_size, 
										 const int dim, const bool vecs_on_gpu = true);
	float* GetNToNDistance(const float* vecs1, const long long vecs1_size, const float* vecs2, const long long vecs2_size,
													const int dim);
	void GetOneToNDistanceAllByIndex(std::vector<std::pair<float, int>> *result_ptr, float* result_dev, const float* vecs_dev, const int vecs_size,
																const int dim, const int vec_index, const int* index, const int* index_dev, const int index_size);

	void GetOneToNDistanceByIndex(std::vector<std::pair<float, int>>* result_ptr, float* result_dev, CUDAData<float> vecs,
								  CUDAData<float> query, const int dim, CUDAData<int> index);
}
#endif