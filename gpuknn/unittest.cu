#include "nndescent.cu"
#include <random>
using namespace std;
__global__ void TestKNNListInsertKernel(ResultElement* knn_list,
                                        const ResultElement element) {
    __shared__ int local_lock;
    if (threadIdx.x == 0) {
        local_lock = 0;
    }
    __syncthreads();
    InsertToLocalKNNList(knn_list, 16, element, &local_lock);
}

void TestKNNListInsert() {
    dim3 block_size(8);
    dim3 grid_size(1);
    size_t knn_list_size = 16;

    vector<ResultElement> knn_list;
    vector<int> distances;
    // GenerateRandomSequence(distances, knn_list_size, 100);
    distances = {2, 3, 23, 29, 31, 36, 45, 50, 
                 55, 56, 56, 57, 65, 76, 83, 89};
    for (int i = 0; i < knn_list_size; i++) {
        knn_list.push_back(ResultElement(distances[i], i));
    }
    sort(knn_list.begin(), knn_list.end());

    for (auto re : knn_list) {
        cout << re.distance << " ";
    } cout << endl;

    ResultElement re_to_insert(86, 20);

    ResultElement* knn_list_dev;
    cudaMalloc(&knn_list_dev, knn_list_size * sizeof(ResultElement));
    cudaMemcpy(knn_list_dev, knn_list.data(),
               knn_list_size * sizeof(ResultElement),
               cudaMemcpyHostToDevice);
    
    TestKNNListInsertKernel<<<grid_size, block_size>>> (knn_list_dev, 
                                                        re_to_insert);
    cudaDeviceSynchronize();
    
    cudaMemcpy(knn_list.data(), knn_list_dev, 
               knn_list_size * sizeof(ResultElement),
               cudaMemcpyDeviceToHost);
    for (auto re : knn_list) {
        cout << re.distance << " ";
    } cout << endl;

    cudaFree(knn_list_dev);
}