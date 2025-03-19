/*
 * pytorch-cgx
 *
 * Copyright (C) 2022 Institute of Science and Technology Austria (ISTA).
 * All Rights Reserved.
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "gpu_compression_operations.h"

#if HAVE_CUDA
#include "cuda_compression_operations.h"
#elif HAVE_ROCM
#include "hip_compression_operations.h"
#endif

namespace cgx::common::gpu {

size_t get_curand_array_size(int num_elems) {
  return BLOCKS_PER_GRID(num_elems, MAX_THREADS_PER_BLOCK)
      * MAX_THREADS_PER_BLOCK *
      sizeof(RandState);
}

template<typename T>
void quantize_maxmin(const unsigned char *input_data, unsigned char *output_data,
                     unsigned char *feedback_data,
                     int num_elems, int bits,
                     int bucket_size, RandState *states,
                     gpuStream_t stream) {
#if HAVE_CUDA
  CUDA_quantize_maxmin<T>(input_data, output_data, feedback_data,
                          num_elems, bits, bucket_size, states, stream);
#elif HAVE_ROCM
  HIP_quantize_maxmin<T>(input_data, output_data, feedback_data, num_elems, bits,
                          bucket_size, states, stream);
#endif
}

template<typename T, bool ADD>
void dequantize_maxmin(const unsigned char *input_data,
                       unsigned char *output_data,
                       int num_elems, int bits,
                       int bucket_size, gpuStream_t stream) {
#if HAVE_CUDA
  CUDA_dequantize_maxmin<T, ADD>(input_data, output_data, num_elems,
                                 bits, bucket_size, stream);
#elif HAVE_ROCM
  HIP_dequantize_maxmin<T, ADD>(input_data, output_data, num_elems, bits,
                                 bucket_size, stream);
#endif
}

template<typename T>
void add(int n, const T *x, T *y, T *sum, gpuStream_t stream) {
#if HAVE_CUDA
  CUDA_add(n, x, y, sum, stream);
#elif HAVE_ROCM
  HIP_add(n, x, y, sum, stream);
#endif
}

void init_rand_states(RandState *states, size_t num_elems, unsigned int seed,
                      gpuStream_t stream) {
#if HAVE_CUDA
  CUDA_init_rand(states, num_elems, seed, stream);
#elif HAVE_ROCM
  HIP_init_rand(states, num_elems, seed, stream);
#endif
}

void half2float(Half *input, float *output, int numel, gpuStream_t stream) {
#if HAVE_CUDA
  CUDA_half2float(input, output, numel, stream);
#elif HAVE_ROCM
#endif
}

void float2half(float *input, Half *output, int numel, gpuStream_t stream) {
#if HAVE_CUDA
  CUDA_float2half(input, output, numel, stream);
#elif HAVE_ROCM
#endif
}

/* Template declarations */
template void quantize_maxmin<float>(const unsigned char *input_data,
                                     unsigned char *output_data,
                                     unsigned char *feedback_data,
                                     int num_elems,
                                     int bits,
                                     int bucket_size,
                                     RandState *states,
                                     gpuStream_t stream);

template void quantize_maxmin<Half>(const unsigned char *input_data,
                                    unsigned char *output_data,
                                    unsigned char *feedback_data,
                                    int num_elems,
                                    int bits,
                                    int bucket_size,
                                    RandState *states,
                                    gpuStream_t stream);

template
void dequantize_maxmin<float, true>(const unsigned char *input_data,
                                    unsigned char *output_data,
                                    int num_elems,
                                    int bits,
                                    int bucket_size,
                                    gpuStream_t stream);

template
void dequantize_maxmin<float, false>(const unsigned char *input_data,
                                     unsigned char *output_data,
                                     int num_elems,
                                     int bits,
                                     int bucket_size,
                                     gpuStream_t stream);

template
void dequantize_maxmin<Half, true>(const unsigned char *input_data,
                                   unsigned char *output_data,
                                   int num_elems,
                                   int bits,
                                   int bucket_size,
                                   gpuStream_t stream);

template
void dequantize_maxmin<Half, false>(const unsigned char *input_data,
                                    unsigned char *output_data,
                                    int num_elems,
                                    int bits,
                                    int bucket_size,
                                    gpuStream_t stream);

template
void add<float>(int n,
                const float *x,
                float *y,
                float *sum,
                gpuStream_t stream);

template
void add<Half>(int n, const Half *x, Half *y, Half *sum, gpuStream_t stream);

} // namespace cgx::common::gpu
