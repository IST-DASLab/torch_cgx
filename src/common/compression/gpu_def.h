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
#include <cstdint>

namespace cgx::common::gpu {

struct xorshift128p_state {
  std::uint64_t a, b;
};

using Half = __half;
using RandState = cgx::common::gpu::xorshift128p_state;

const float EPS = 1e-10;
const int PACK_SIZE = 8;
const int MAX_THREADS_PER_BLOCK = 1024;
const int THREADS_PER_BLOCK_DECOMPRESS = MAX_THREADS_PER_BLOCK;
const int THREADS_PER_BLOCK_COMPRESS = 64;
const int MAX_NUMBER_OF_BLOCKS = 65535;
const int WARP_SIZE = 32;

typedef union {
  uchar2 vec;
  unsigned char a[2];
} U2;

typedef union {
  uchar3 vec;
  unsigned char a[3];
} U3;

typedef union {
  uchar4 vec;
  unsigned char a[4];
} U4;

typedef struct __align__(16) {
half2 x;
half2 y;
half2 z;
half2 w;
} half8;

typedef union {
  half8 vec;
  Half a[8];
} H8;

typedef union {
  float4 vec;
  float a[4];
} F4;

template<typename T>
struct TypeToVectorType;

template<>
struct TypeToVectorType<float> {
  typedef F4 vector_union;
  typedef float4 vector_type;
  static const int num_values = 4;
};

template<>
struct TypeToVectorType<Half> {
  typedef H8 vector_union;
  typedef half8 vector_type;
  static const int num_values = 8;
};

} // namespace cgx::common::gpu
