#include "xmuknn.h"
namespace xmuknn {
size_t time_sum = 0;
long long dist_calc_count = 0;
std::uniform_real_distribution<float> unif(0.0, 1.0);
std::default_random_engine rand_engine(time(NULL));
}  // namespace xmuknn