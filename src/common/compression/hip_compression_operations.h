#pragma once
#include "gpu_compression_operations.h"

namespace cgx {
namespace common {
namespace gpu {
template<typename T>
void HIP_quantize_maxmin(unsigned char *input_data, unsigned char *output_data,
                         unsigned char *feedback_data, int num_elems, int bits,
                         int bucket_size, RandState *states,
                         hipStream_t stream);

template<typename T, bool ADD>
void HIP_dequantize_maxmin(unsigned char *input_data,
                         unsigned char *output_data, int num_elems, int bits,
                         int bucket_size, hipStream_t stream);

template<typename T>
void HIP_add(int n, const T *x, T *y, T *sum, hipStream_t stream);

void HIP_init_rand(RandState *states, int num_elems, unsigned int seed,
                   hipStream_t stream);
} // namespace gpu
} // namespace common
} // namespace cgx