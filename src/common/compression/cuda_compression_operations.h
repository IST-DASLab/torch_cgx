#pragma once
#include "gpu_compression_operations.h"

namespace cgx {
namespace common {
namespace gpu {
template<typename T>
void CUDA_quantize_maxmin(unsigned char *input_data, unsigned char *output_data,
                          unsigned char *feedback_data, unsigned char* util_buf,
                          int num_elems, int bits, int bucket_size,
                          RandState *states, cudaStream_t stream);

template<typename T, bool ADD>
void CUDA_dequantize_maxmin(unsigned char *input_data,
                            unsigned char *output_data, unsigned char* util_buf,
                            int num_elems, int bits, int bucket_size,
                            cudaStream_t stream);

template<typename T>
void CUDA_add(int n, const T *x, T *y, T *sum, cudaStream_t stream);

void CUDA_init_rand(RandState *states, int num_elems, unsigned int seed,
                    cudaStream_t stream);

void CUDA_half2float(Half *input,
                     float *output,
                     int numel,
                     cudaStream_t stream);

void CUDA_float2half(float *input,
                     Half *output,
                     int numel,
                     cudaStream_t stream);

} // namespace gpu
} // namespace common
} // namespace cgx