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
#include <sstream>

#if HAVE_CUDA
#define CUDA_CHECK(condition)                                                  \
do {                                                                           \
  cudaError_t cuda_result = condition;                                         \
  if (cuda_result != cudaSuccess) {                                            \
    printf("%s on line %i in %s returned: %s(code:%i)\n", #condition,          \
           __LINE__, __FILE__, cudaGetErrorString(cuda_result),                \
           cuda_result);                                                       \
    throw std::runtime_error(                                                  \
        std::string(#condition) + " in file " + __FILE__                       \
        + " on line " + std::to_string(__LINE__) +                             \
        " returned: " + cudaGetErrorString(cuda_result));                      \
  }                                                                            \
} while (0)
#elif HAVE_ROCM
#define HIP_CHECK(condition)                                                   \
do {                                                                           \
  hipError_t hip_result = condition;                                           \
  if (hip_result != hipSuccess) {                                              \
    printf("%s on line %i in %s returned: %s(code:%i)\n", #condition,          \
           __LINE__, __FILE__, hipGetErrorString(hip_result),                  \
           hip_result);                                                            \
    throw std::runtime_error(                                                  \
        std::string(#condition) + " in file " + __FILE__                       \
        + " on line " + std::to_string(__LINE__) +                             \
        " returned: " + hipGetErrorString(hip_result));                        \
  }                                                                            \
} while (0)
#endif