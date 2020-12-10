#ifndef XMUKNN_NNDESCENT_CUH
#define XMUKNN_NNDESCENT_CUH
#include "../xmuknn.h"
#define LARGE_INT 0x3f3f3f3f
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
        bool operator == (const NNDItem &other) const {
            return (this->id == other.id) && (fabs(this->distance - other.distance) < 1e-9);
        }
        bool operator < (const NNDItem &other) const {
            if (fabs(this->distance - other.distance) < 1e-9) return this->id < other.id;
            return this->distance < other.distance;
        }
    };

    vector<vector<NNDItem>> NNDescent(const float* vectors, const int vecs_size, const int vecs_dim);
}
#endif
