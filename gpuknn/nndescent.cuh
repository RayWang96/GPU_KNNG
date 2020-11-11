#ifndef XMUKNN_NNDESCENT_CUH
#define XMUKNN_NNDESCENT_CUH
#include "../xmuknn.h"

using namespace std;
using namespace xmuknn;

namespace gpuknn {
    //NNDescent's required item.
    struct NNDItem {
        int id = -1;
        bool visited = false;
        float distance = -1;
        NNDItem() : id(-1), visited(false), distance(-1) {}
        NNDItem(int id, bool visited, float distance) : id(id), visited(visited), distance(distance){}
    };

    Graph NNDescent(const float* vectors, const int vecs_size, const int vecs_dim);
}
#endif
