#include <cuda.h>
#include <cmath>
#include <algorithm>
#include <device_functions.h>
#include <iostream>
#include <vector>
#include <assert.h>
#include <ctime>
#include <cstring>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gpudist.cuh"
#include "../tools/distfunc.hpp"

#ifdef __INTELLISENSE__
#include "../intellisense_cuda_intrinsics.h"
#endif

//通过修改query_per_thread可以减少block的数量，以减少对vec的global读入
//但是每个block需要的执行时间变长
//由于每个gpu同时执行的block数量有限，延长每个block执行时间也是有意义的（它们本来也不并行）
//现在只支持dim < 1024
//__global__ void GetOneToNDistanceKernel(float* results, const float* vec, const float* vecs, const int size, const int dim,
//										const int query_per_thread = 1) {
//	//一方面减少同一warp里线程的竞争，另一方面可以并行化数组求和
//	const int partial_result_size = 32;
//	//在保证shared memory足够的情况下，减少block数目，这样可以减少访问global memory的次数。
//	//将待查询的向量vec和结果数组放入共享内存中以加速。
//	extern __shared__ float shared_array[];
//	float* vec1 = shared_array;
//	float* partial_result = &shared_array[dim]; //别忘了前面有个vec1的空间dim
//	//CUDA只能动态分配一次share memory，所以这样分配。
//
//	int t_x = threadIdx.x;
//	if (t_x < dim) {
//		vec1[t_x] = vec[t_x];
//	}
//
//	int rslt_loc = t_x % partial_result_size;
//
//	for (int qi = 0; qi < query_per_thread; qi++) {
//		if (t_x < partial_result_size) {
//			partial_result[t_x] = 0;
//		}
//		__syncthreads();
//		int id_vec2 = blockIdx.x  * query_per_thread + qi;
//		if (id_vec2 >= size) {
//			continue;
//		}
//		int i = t_x;
//		int j = id_vec2 * dim + threadIdx.x;
//		float diff = vec1[i] - vecs[j];
//		diff *= diff;
//
//		//原子操作对不是同一warp里的线程性能影响不大
//		atomicAdd(&partial_result[rslt_loc], diff);
//
//		//以下是并行求数组和的方法，该方法保证了合并访存。
//		for (int stride = partial_result_size / 2; stride >= 1; stride >>= 1) {
//			__syncthreads();
//			if (t_x < stride) {
//				partial_result[t_x] += partial_result[t_x + stride];
//			}
//		}
//		if (t_x == 0 && id_vec2 < size) {
//			//不会访问同一个结果，所以可以不用原子加法
//			results[id_vec2] += partial_result[0];
//		}
//	}
//}

__global__ void GetOneToNDistanceKernel(float* results, const float* vec, const float* vecs, const int size, const int dim,
	const int query_per_thread = 1) {
	//一方面减少同一warp里线程的竞争，另一方面可以并行化数组求和
	const int partial_result_size = 32;
	//在保证shared memory足够的情况下，减少block数目，这样可以减少访问global memory的次数。
	//将待查询的向量vec和结果数组放入共享内存中以加速。
	extern __shared__ float shared_array[];
	float* vec1 = shared_array;
	float* partial_result = &shared_array[dim]; //别忘了前面有个vec1的空间dim
	//CUDA只能动态分配一次share memory，所以这样分配。
	int t_x = threadIdx.x;
	if (t_x < dim) {
		//printf("%d\n", t_x);
		vec1[t_x] = vec[t_x];
	}
	__syncthreads();
	int id_vec2 = blockDim.x * blockIdx.x + threadIdx.x;
	if (id_vec2 > size) {
		return;
	}
	//printf("%d\n", id_vec2);

	float ans = 0;
	for (int i = 0; i < dim; i++) {
		float diff = vecs[id_vec2 * dim + i] - vec1[i];
		//printf("%d %f %f %f\n", i, vecs[id_vec2 * dim + i], vec1[i], diff);
		ans += diff * diff;
	}
	results[id_vec2] = ans;
}

//__global__ void GetOneToNDistanceByIndexKernel(float* results, const float* vecs, const int size, const int dim,
//											   const int vec_index, const int* index, const int index_size,
//											   const int query_per_thread = 1) {
//	//一方面减少同一warp里线程的竞争，另一方面可以并行化数组求和
//	const int partial_result_size = 32;
//	//在保证shared memory足够的情况下，减少block数目，这样可以减少访问global memory的次数。
//	//将待查询的向量vec和结果数组放入共享内存中以加速。
//	extern __shared__ float shared_array[];
//	float* vec1 = shared_array;
//	float* partial_result = &shared_array[dim]; //别忘了前面有个vec1的空间dim
//	//CUDA只能动态分配一次share memory，所以这样分配。
//	const float* vec = &vecs[vec_index * dim];
//
//	int t_x = threadIdx.x;
//	if (t_x < dim) {
//		vec1[t_x] = vec[t_x];
//	}
//	
//	for (int qi = 0; qi < query_per_thread; qi++) {
//		if (t_x < partial_result_size) {
//			partial_result[t_x] = 0;
//		}
//		__syncthreads();
//		int loc = blockIdx.x * query_per_thread + qi;
//		if (loc > index_size) {
//			continue;
//		}
//		int id_vec2 = index[loc];
//		if (id_vec2 >= size) {
//			continue;
//		}
//		int i = t_x;
//		int j = id_vec2 * dim + threadIdx.x;
//		float diff = vec1[i] - vecs[j];
//		diff *= diff;
//
//		//原子操作对不是同一warp里的线程性能影响不大
//		atomicAdd(&partial_result[t_x % partial_result_size], diff);
//
//		//以下是并行求数组和的方法，该方法保证了合并访存。
//		for (int stride = partial_result_size / 2; stride >= 1; stride >>= 1) {
//			__syncthreads();
//			if (t_x < stride) {
//				partial_result[t_x] += partial_result[t_x + stride];
//			}
//		}
//		if (t_x == 0 && id_vec2 < size) {
//			results[loc] = partial_result[0];
//		}
//	}
//}

__global__ void GetOneToNDistanceAllByIndexKernel(float* results, const float* vecs, const int size, const int dim,
											   const int vec_index, const int* index, const int index_size,
											   const int query_per_thread = 1) {
	//在保证shared memory足够的情况下，减少block数目，这样可以减少访问global memory的次数。
	//将待查询的向量vec和结果数组放入共享内存中以加速。
	extern __shared__ float shared_array[];
	float* vec1 = shared_array;
	//CUDA只能动态分配一次share memory，所以这样分配。
	const float* vec = &vecs[vec_index * dim];
	int t_x = threadIdx.x;
	if (t_x < dim) {
		//printf("%d\n", t_x);
		vec1[t_x] = vec[t_x];
	}
	__syncthreads();
	if (blockDim.x * blockIdx.x + threadIdx.x > index_size) return;
	int id_vec2 = index[blockDim.x * blockIdx.x + threadIdx.x];
	if (id_vec2 > size) {
		return;
	}
	//printf("%d\n", id_vec2);

	float ans = 0;
	for (int i = 0; i < dim; i++) {
		float diff = vecs[id_vec2 * dim + i] - vec1[i];
		//printf("%d %f %f %f\n", i, vecs[id_vec2 * dim + i], vec1[i], diff);
		ans += diff * diff;
	}
	//printf("%d %f\n", id_vec2, ans);
	results[blockDim.x * blockIdx.x + threadIdx.x] = ans;
}

__global__ void GetOneToNDistanceByIndexKernel(float* results, const float* vecs, const int size, const int dim,
											   const float* query, const int* index, const int index_size,
											   const int query_per_thread = 1) {
	extern __shared__ float shared_array[];
	float* vec1 = shared_array;
	const float* vec = query;
	int t_x = threadIdx.x;
	if (t_x < dim) {
		vec1[t_x] = vec[t_x];
	}
	__syncthreads();
	if (blockDim.x * blockIdx.x + threadIdx.x > index_size) return;
	int id_vec2 = index[blockDim.x * blockIdx.x + threadIdx.x];
	if (id_vec2 > size) {
		return;
	}

	float ans = 0;
	for (int i = 0; i < dim; i++) {
		float diff = vecs[id_vec2 * dim + i] - vec1[i];
		ans += diff * diff;
	}
	results[blockDim.x * blockIdx.x + threadIdx.x] = ans;
}

__global__ void GetNToNDistanceKernel(float* results, const float* vecs1, const int vecs1_size,
									  const float* vecs2, const int vecs2_size, const int dim) {
	const int size_n = 4, size_m = 4;
	const int partial_result_size = 32;

	extern __shared__ float shared_array[];
	int t_x = threadIdx.x;
	int vec1_base_id = blockIdx.x * size_n;
	int vec2_base_id = blockIdx.y * size_m;
	//printf("%d %d %d %d %d %d\n", gridDim.x, gridDim.y, blockIdx.x, blockIdx.y, vec1_base_id, vec2_base_id);
	float* shared_vecs1 = shared_array;
	float* shared_vecs2 = &shared_array[size_n * dim];
	float* partial_result = &shared_array[(size_m + size_n) * dim];

	if (vec1_base_id >= vecs1_size || vec2_base_id >= vecs2_size) {
		return;
	}

	for (int i = 0; i < size_n; i++) {
		shared_vecs1[i * dim + t_x] = vecs1[(vec1_base_id + i) * dim + t_x];
	}
	__syncthreads(); //试试看同步有没有加成
	for (int i = 0; i < size_m; i++) {
		shared_vecs2[i * dim + t_x] = vecs2[(vec2_base_id + i) * dim + t_x];
	}
	__syncthreads(); 

	for (int i = 0; i < size_n; i++) {
		int id_vec1 = vec1_base_id + i;
		for (int j = 0; j < size_m; j++) {
			if (t_x < partial_result_size) {
				partial_result[t_x] = 0;
			}
			__syncthreads();
			int id_vec2 = vec2_base_id + j;
			if (id_vec1 >= vecs1_size || id_vec2 >= vecs2_size) {
				break;
			}
			float diff = shared_vecs1[i * dim + t_x] - shared_vecs2[j * dim + t_x];
			diff *= diff;
			//原子操作对不是同一warp里的线程性能影响不大
			atomicAdd(&partial_result[t_x % partial_result_size], diff);

			//以下是并行求数组和的方法，该方法保证了合并访存。
			for (int stride = partial_result_size / 2; stride >= 1; stride >>= 1) {
				__syncthreads();
				if (t_x < stride) {
					partial_result[t_x] += partial_result[t_x + stride];
				}
			}
			__syncthreads();
			if (t_x == 0) {
				//不会访问同一个结果，所以可以不用原子加法
				results[id_vec1 * vecs2_size + id_vec2] = sqrt(partial_result[0]);
			}
		}
	}
}

namespace xmuknn {
	//float GPUGetDistance(const float* vec1_start, const float* vec2_start, const int dim) {
	//	//获取两个向量之间的距离
	//	//返回一个float
	//	//维度太小的话发挥不出。128维要慢cpu几千倍
	//	float* dev_result;
	//	float* dev_vec1, * dev_vec2;
	//	cudaSetDevice(0);
	//	cudaMalloc(&dev_result, sizeof(float));
	//	cudaMalloc(&dev_vec1, dim * sizeof(float));
	//	cudaMalloc(&dev_vec2, dim * sizeof(float));

	//	float result = 0;

	//	cudaMemcpy(dev_result, &result, sizeof(float), cudaMemcpyHostToDevice);
	//	cudaMemcpy(dev_vec1, vec1_start, dim * sizeof(float), cudaMemcpyHostToDevice);
	//	cudaMemcpy(dev_vec2, vec2_start, dim * sizeof(float), cudaMemcpyHostToDevice);

	//	dim3 block_size(128);
	//	dim3 grid_size(ceil(1.0 * dim / block_size.x));
	//	GetDistanceKernel <<<grid_size, block_size>>> (dev_result, dev_vec1, dev_vec2, dim);
	//	cudaDeviceSynchronize();

	//	cudaMemcpy(&result, dev_result, sizeof(float), cudaMemcpyDeviceToHost);

	//	cudaFree(dev_vec1);
	//	cudaFree(dev_vec2);
	//	//std::cerr << sqrt(result) << std::endl;
	//	return sqrt(result);
	//}

	//获取n个向量和m个向量两两距离
	//返回二维矩阵
	//维度和查询量太小的话发挥不出
	//目前维度需要小于等于1024

	std::vector<float> GetOneToNDistance(float* vec, float* vecs, const int vecs_size, 
										 const int dim, const bool vecs_on_gpu) {
		float* dev_vec, * dev_vecs;
		auto result = std::vector<float>();
		if (dim > 1024) {
			std::cerr << "现在还不支持维度大于1024的向量" << std::endl;
			return result;
		}
		if (vecs_on_gpu) {
			dev_vecs = vecs;
		}
		else {
			cudaMalloc(&dev_vecs, vecs_size * dim * sizeof(float));
			cudaMemcpy(dev_vecs, vecs, vecs_size * dim * sizeof(float), cudaMemcpyHostToDevice);
		}
		cudaMalloc(&dev_vec, dim * sizeof(float));
		cudaMemcpy(dev_vec, vec, dim * sizeof(float), cudaMemcpyHostToDevice);

		float *dev_tmp_result;
		float *tmp_result = new float[vecs_size];
		cudaMalloc(&dev_tmp_result, vecs_size * sizeof(float));

		//增加每个线程查询的数量可以减少block的数量，使得block中的shared memory可以重复利用。
		//但每个线程查询次数过多会导致并行度下降，需要权衡。
		const int query_per_thread = 20;

		////每个block只要一个查询
		dim3 block_size(1024); 
		////grid_size最好保持整数，这样可以减少浪费
		dim3 grid_size(ceil(1.0 * vecs_size / block_size.x));

		//每个block只要一个查询
		//dim3 block_size(dim); 
		//grid_size最好保持整数，这样可以减少浪费
		//dim3 grid_size(ceil(1.0 * vecs_size / query_per_thread));

		memset(tmp_result, 0, vecs_size * sizeof(float));
		auto cuda_status = cudaMemcpy(dev_tmp_result, tmp_result, 0, cudaMemcpyHostToDevice);
		if (cuda_status != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cuda_status));
			delete[] tmp_result;
			cudaFree(dev_tmp_result);
			cudaFree(dev_vec);
			if (!vecs_on_gpu)
				cudaFree(dev_vecs);
			return result;
		}

		//32是指分配给答案的内存，以减少同一warp中原子操作的竞争。
		const int shared_memory_size = (dim + 32) * sizeof(float);
		GetOneToNDistanceKernel <<<grid_size, block_size, shared_memory_size>>> (dev_tmp_result, dev_vec, dev_vecs, vecs_size, 
																				 dim, query_per_thread);
		cudaDeviceSynchronize();
		cuda_status = cudaGetLastError();
		if (cuda_status != cudaSuccess) {
			fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
			goto Error;
		}
		cuda_status = cudaMemcpy(tmp_result, dev_tmp_result, vecs_size * sizeof(float), cudaMemcpyDeviceToHost);
		if (cuda_status != cudaSuccess) {
			std::cerr << cuda_status << std::endl;
			fprintf(stderr, "Memcpy from device failed: %s\n", cudaGetErrorString(cuda_status));
			goto Error;
		}

		for (int j = 0; j < vecs_size; j++) {
			//std::cerr << sqrt(tmp_result[j]) << std::endl;
			result.push_back(sqrt(tmp_result[j]));
		}		

	Error:
		delete[] tmp_result;
		cudaFree(dev_tmp_result);
		cudaFree(dev_vec);
		if (!vecs_on_gpu)
			cudaFree(dev_vecs);
		return result;
	}

	float* GetNToNDistance(const float* vecs1, const long long vecs1_size,
						   const float* vecs2, const long long vecs2_size, const int dim) {
		float* dev_vecs1, * dev_vecs2;
		if (dim > 1024) {
			std::cerr << "现在还不支持维度大于1024的向量" << std::endl;
			return NULL;
		}
		cudaMalloc(&dev_vecs1, vecs1_size * dim * sizeof(float));
		cudaMemcpy(dev_vecs1, vecs1, vecs1_size * dim * sizeof(float), cudaMemcpyHostToDevice);

		cudaMalloc(&dev_vecs2, vecs2_size * dim * sizeof(float));
		cudaMemcpy(dev_vecs2, vecs2, vecs2_size * dim * sizeof(float), cudaMemcpyHostToDevice);

		//一次性读入到shared_memory的向量的个数，在保证shared_memory足够的情况下越大越好。
		//不传参到内核里了，这样性能好，但是要保证和host的参数一致。
		int size_n = 4, size_m = 4; 
		float* result = new float[vecs1_size * vecs2_size * sizeof(float)];
		float* dev_result;
		cudaMalloc(&dev_result, vecs1_size * vecs2_size * sizeof(float));
		auto cuda_status = cudaGetLastError();
		if (cuda_status != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cuda_status));
			cudaFree(dev_vecs1);
			cudaFree(dev_vecs2);
			cudaFree(dev_result);
			return result;
		}
		//每个block只要一个查询
		dim3 block_size(dim);
		//grid_size最好保持整数，这样可以减少浪费
		dim3 grid_size(ceil(1.0 * vecs1_size / size_n), ceil(1.0 * vecs2_size / size_m));

		//32是指分配给答案的内存，以减少同一warp中原子操作的竞争。
		const int shared_memory_size = ((size_n + size_m) * dim + 32) * sizeof(float);
		GetNToNDistanceKernel <<<grid_size, block_size, shared_memory_size>>> (dev_result, dev_vecs1, vecs1_size, 
																			   dev_vecs2, vecs2_size, dim);
		cudaDeviceSynchronize();
		cuda_status = cudaGetLastError();
		if (cuda_status != cudaSuccess) {
			fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
			goto Error;
		}
		cuda_status = cudaMemcpy(result, dev_result, vecs1_size * vecs2_size * sizeof(float), cudaMemcpyDeviceToHost);
		if (cuda_status != cudaSuccess) {
			std::cerr << cuda_status << std::endl;
			fprintf(stderr, "Memcpy from device failed: %s\n", cudaGetErrorString(cuda_status));
			goto Error;
		}

	Error:
		cudaFree(dev_vecs1);
		cudaFree(dev_vecs2);
		cudaFree(dev_result);
		return result;
	}

	//It's a function need device pointer, please load the memory to device first.
	void GetOneToNDistanceAllByIndex(std::vector<std::pair<float, int>> *result_ptr, float* result_dev, const float* vecs_dev, const int vecs_size,
																const int dim, const int vec_index, const int* index, const int* index_dev, const int index_size) {
		auto& result = *result_ptr;
		if (dim > 1024) {
			std::cerr << "现在还不支持维度大于1024的向量" << std::endl;
			return;
		}
		//增加每个线程查询的数量可以减少block的数量，使得block中的shared memory可以重复利用。
		//但每个线程查询次数过多会导致并行度下降，需要权衡。
		const int query_per_thread = 20;

		////每个block只要一个查询
		//dim3 block_size(dim);
		////grid_size最好保持整数，这样可以减少浪费
		//dim3 grid_size(ceil(1.0 * index_size / query_per_thread));
		//每个block只要一个查询
		dim3 block_size(256);
		//grid_size最好保持整数，这样可以减少浪费
		dim3 grid_size(ceil(1.0 * index_size / block_size.x));
		auto cuda_status = cudaSuccess;

		//32是指分配给答案的内存，以减少同一warp中原子操作的竞争。
		const int shared_memory_size = (dim + 32) * sizeof(float);
		GetOneToNDistanceAllByIndexKernel <<<grid_size, block_size, shared_memory_size>>> (result_dev, vecs_dev, vecs_size,
																						dim, vec_index, index_dev, index_size, query_per_thread);
		cudaDeviceSynchronize();
		cuda_status = cudaGetLastError();
		if (cuda_status != cudaSuccess) {
			fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
			return;
		}
		float *tmp_result = new float[index_size];
		cuda_status = cudaMemcpy(tmp_result, result_dev, index_size * sizeof(float), cudaMemcpyDeviceToHost);
		if (cuda_status != cudaSuccess) {
			std::cerr << cuda_status << std::endl;
			fprintf(stderr, "Memcpy from device failed: %s\n", cudaGetErrorString(cuda_status));
			goto Error;
		}
		result.clear();
		for (int i = 0; i < index_size; i++) {
			//std::cerr << sqrt(tmp_result[i]) << std::endl;
			result.push_back(std::make_pair(sqrt(tmp_result[i]), index[i]));
		}
	Error:
		delete[] tmp_result;
		return;
	}


	void GetOneToNDistanceByIndex(std::vector<std::pair<float, int>>* result_ptr, float* result_dev, CUDAData<float> vecs,
		CUDAData<float> query, const int dim, CUDAData<int> index) {
		auto& result = *result_ptr;
		result.clear();
		if (dim > 1024) {
			std::cerr << "现在还不支持维度大于1024的向量" << std::endl;
			return;
		}
		if (vecs.if_need_copy) {
			cudaMemcpy(vecs.data_dev_ptr, vecs.data_ptr, vecs.size * sizeof(float), cudaMemcpyHostToDevice);
		}
		if (query.if_need_copy) {
			cudaMemcpy(query.data_dev_ptr, query.data_ptr, query.size * sizeof(float), cudaMemcpyHostToDevice);
		}
		if (index.if_need_copy) {
			cudaMemcpy(index.data_dev_ptr, index.data_ptr, index.size * sizeof(int), cudaMemcpyHostToDevice);
		}

		const int query_per_thread = 20;
		dim3 block_size(256);
		dim3 grid_size(ceil(1.0 * index.size / block_size.x));
		auto cuda_status = cudaSuccess;
		float* tmp_result = new float[index.size];

		//32是指分配给答案的内存，以减少同一warp中原子操作的竞争。
		const int shared_memory_size = (dim + 32) * sizeof(float);
		GetOneToNDistanceByIndexKernel <<<grid_size, block_size, shared_memory_size>>> (result_dev, vecs.data_dev_ptr, vecs.size,
			dim, query.data_dev_ptr, index.data_dev_ptr, index.size, query_per_thread);
		cudaDeviceSynchronize();
		cuda_status = cudaGetLastError();
		if (cuda_status != cudaSuccess) {
			fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
			goto Error;
		}
		cuda_status = cudaMemcpy(tmp_result, result_dev, index.size * sizeof(float), cudaMemcpyDeviceToHost);
		if (cuda_status != cudaSuccess) {
			std::cerr << cuda_status << std::endl;
			fprintf(stderr, "Memcpy from device failed: %s\n", cudaGetErrorString(cuda_status));
			goto Error;
		}
		//for (int i = 0; i < index.size; i++) {
		//	tmp_result[i] = GetDistance(query.data_ptr, vecs.data_ptr + index.data_ptr[i] * dim, dim);
		//}
		result.clear();
		for (int i = 0; i < index.size; i++) {
			result.push_back(std::make_pair(sqrt(tmp_result[i]), index.data_ptr[i]));
		}
	Error:
		delete[] tmp_result;
		return;
	}
}

