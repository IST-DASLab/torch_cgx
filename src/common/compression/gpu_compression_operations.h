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

#pragma once
#ifdef HAVE_CUDA
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#elif HAVE_ROCM
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "../gpu_context.h"
#include "gpu_def.h"

namespace cgx::common::gpu {

constexpr int MIN(int a, int b) { return (a > b) ? b : a; }

constexpr int BLOCKS_PER_GRID(int num_elems, int threads_per_block) {
  threads_per_block = (threads_per_block > 0) ? threads_per_block: 1;
  return MIN((num_elems + (threads_per_block - 1)) / threads_per_block,
             MAX_NUMBER_OF_BLOCKS);
}

template<typename T>
void quantize_maxmin(const unsigned char *input_data, unsigned char *output_data,
                     unsigned char *feedback_data,
                     int num_elems, int bits,
                     int bucket_size, RandState *states,
                     gpuStream_t stream);

template<typename T, bool ADD>
void dequantize_maxmin(const unsigned char *input_data,
                       unsigned char *output_data,
                       int num_elems, int bits,
                       int bucket_size, gpuStream_t stream);

void init_rand_states(RandState* states, size_t num_elems, unsigned int seed,
                      gpuStream_t stream);
template<typename T>
void add(int n, const T* x, T* y, T* sum, gpuStream_t stream);

void half2float(Half* input, float* output, int numel, gpuStream_t stream);

void float2half(float* input, Half* output, int numel, gpuStream_t stream);


size_t get_curand_array_size(int num_elems);

} // namespace cgx::common::gpu
