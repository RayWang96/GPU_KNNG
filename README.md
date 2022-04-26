# GPU_KNNG

Source code for CIKM 2021 paper [Fast k-NN Graph Construction by GPU based NN-Descent](https://dl.acm.org/doi/10.1145/3459637.3482344).

TestCUDANNDescent() in main.cu shows a simple demo for constructing k-NN graph.

We are working on a more formal and optimized version, so this library is no longer being updated.

In this version of the code, in order to improve efficiency, we fixed the following parameters in nndescent.cuh.
```cpp
const int VEC_DIM = 128;
const int NEIGHB_NUM_PER_LIST = 64;
const int SAMPLE_NUM = 32; 
const int NND_ITERATION = 6;
```

Above four parameters are main parameters for GNND to construct k-NN graph.
