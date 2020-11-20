#include "nndescent.cu"
#include "../tools/filetool.hpp"
#include "../tools/distfunc.hpp"
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

__global__ void TestTiledDistanceCompareKernel(float *distances,
                                               float *vectors, const int size, const int dim,
                                               const int *new_neighbors, const int num_new,
                                               const int *old_neighbors, const int num_old) {
    GetNewOldDistances(distances, vectors, new_neighbors, num_new, old_neighbors, num_old);
}

void TestTiledDistanceCompare() {
    float *vecs;
    int size, dim;
    FileTool::ReadVecs(vecs, size, dim, "/media/data4/huiwang/data/sift10k/sift10k.txt");
    
    float *vecs_dev;
    cudaMalloc(&vecs_dev, size * dim * sizeof(float));
    cudaMemcpy(vecs_dev, vecs, size * dim * sizeof(float), cudaMemcpyHostToDevice);

    vector<int> new_neighbors, old_neighbors;
    for (int i = 0; i < 64; i++) {
        new_neighbors.push_back(i * 15);
    }
    for (int j = 0; j < 75; j++) {
        old_neighbors.push_back(j * 33);
    }

    int *new_neighbors_dev, *old_neighbors_dev;
    cudaMalloc(&new_neighbors_dev, new_neighbors.size() * sizeof(int));
    cudaMalloc(&old_neighbors_dev, old_neighbors.size() * sizeof(int));
    cudaMemcpy(new_neighbors_dev, new_neighbors.data(),
               new_neighbors.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(old_neighbors_dev, old_neighbors.data(),
               old_neighbors.size() * sizeof(int), cudaMemcpyHostToDevice);

    float *distances_dev;
    cudaMalloc(&distances_dev, new_neighbors.size() * old_neighbors.size() * sizeof(float));

    dim3 grid_size(1);
    dim3 block_size(256);
    TestTiledDistanceCompareKernel<<<grid_size, block_size>>>(distances_dev,
                                                              vecs_dev, size, dim,
                                                              new_neighbors_dev, 
                                                              new_neighbors.size(),
                                                              old_neighbors_dev,
                                                              old_neighbors.size());
    cudaDeviceSynchronize();
    float *distances = new float[new_neighbors.size() * old_neighbors.size()];
    cudaMemcpy(distances, distances_dev, 
               new_neighbors.size() * old_neighbors.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    float *grd_distances = new float[new_neighbors.size() * old_neighbors.size()];

    int cnt = 0;
    for (int i = 0; i < new_neighbors.size(); i++) {
        for (int j = 0; j < old_neighbors.size(); j++) {
            grd_distances[cnt++] = 
                GetDistance(&vecs[new_neighbors[i] * dim], &vecs[old_neighbors[j] * dim], dim);
        }
    }

    cnt = 0;
    for (int i = 0; i < new_neighbors.size(); i++) {
        for (int j = 0; j < old_neighbors.size(); j++) {
            if (distances[cnt] != grd_distances[cnt]) {
                printf("%d %d %f %f\n", i, j, distances[cnt], grd_distances[cnt]);
            }
            cnt++;
        }
    }
    delete[] distances;
    delete[] grd_distances;
}