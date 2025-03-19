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
#include "gpu_compression_operations.h"

namespace cgx::common::gpu {

template<typename T>
void CUDA_quantize_maxmin(const unsigned char *input_data, unsigned char *output_data,
                          unsigned char *feedback_data,
                          int num_elems, int bits, int bucket_size,
                          RandState *states, cudaStream_t stream);

template<typename T, bool ADD>
void CUDA_dequantize_maxmin(const unsigned char *input_data,
                            unsigned char *output_data,
                            int num_elems, int bits, int bucket_size,
                            cudaStream_t stream);

template<typename T>
void CUDA_add(int n, const T *x, T *y, T *sum, cudaStream_t stream);

void CUDA_init_rand(RandState *states, size_t num_elems, unsigned int seed,
                    cudaStream_t stream);

void CUDA_half2float(Half *input,
                     float *output,
                     int numel,
                     cudaStream_t stream);

void CUDA_float2half(float *input,
                     Half *output,
                     int numel,
                     cudaStream_t stream);

} // namespace cgx::common::gpu
