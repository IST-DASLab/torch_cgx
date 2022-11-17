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

namespace cgx {
namespace common {
namespace gpu {
template<typename T>
void HIP_quantize_maxmin(const unsigned char *input_data, unsigned char *output_data,
                         unsigned char *feedback_data, int num_elems, int bits,
                         int bucket_size, RandState *states,
                         hipStream_t stream);

template<typename T, bool ADD>
void HIP_dequantize_maxmin(const unsigned char *input_data,
                         unsigned char *output_data, int num_elems, int bits,
                         int bucket_size, hipStream_t stream);

template<typename T>
void HIP_add(int n, const T *x, T *y, T *sum, hipStream_t stream);

void HIP_init_rand(RandState *states, size_t num_elems, unsigned int seed,
                   hipStream_t stream);
} // namespace gpu
} // namespace common
} // namespace cgx