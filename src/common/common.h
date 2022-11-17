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
#include <stdio.h>
#include <stdexcept>

#define FUSION_BUFFER_SIZE_MB "CGX_FUSION_BUFFER_SIZE_MB"
#define COMPRESSION_MINIMAL_SIZE "CGX_COMPRESSION_MINIMAL_SIZE"
#define COMPRESSION_QUANTIZATION_BITS "CGX_COMPRESSION_QUANTIZATION_BITS"
#define COMPRESSION_BUCKET_SIZE "CGX_COMPRESSION_BUCKET_SIZE"
#define COMPRESSION_SKIP_INCOMPLETE_BUCKETS "CGX_COMPRESSION_SKIP_INCOMPLETE_BUCKETS"
#define COMPRESSION_FAKE_RATIO "CGX_COMPRESSION_FAKE_RATIO"
#define INNER_COMMUNICATOR_TYPE "CGX_INNER_COMMUNICATOR_TYPE"
#define INNER_REDUCTION_TYPE "CGX_INNER_REDUCTION_TYPE"
#define CROSS_COMMUNICATOR_TYPE "CGX_CROSS_COMMUNICATOR_TYPE"
#define CROSS_REDUCTION_TYPE "CGX_CROSS_REDUCTION_TYPE"
#define REMOTE_BUF_COMPRESSION "CGX_REMOTE_BUF_COMPRESSION"
#define DEBUG_ALL_TO_ALL_REDUCTION "CGX_DEBUG_ALL_TO_ALL_REDUCTION"
#define DEBUG_DUMMY_COMPRESSION "CGX_DEBUG_DUMMY_COMPRESSION"
#define INTRA_BROADCAST "CGX_INTRA_BROADCAST"
#define INTRA_COMPRESS "CGX_INTRA_COMPRESS"

const unsigned int FUSION_SIZE_DEFAULT_MB = 64;
const unsigned int MIN_FUSION_SIZE = 2048;

#define MPI_CHECK(condition)                                                   \
  do {                                                                         \
    int op = condition;                                                        \
    if (op != MPI_SUCCESS) {                                                   \
      int len;                                                                 \
      char estring[MPI_MAX_ERROR_STRING];                                      \
      MPI_Error_string(op, estring, &len);                                     \
      printf("%s on line %i. MPI Error: %s\n", #condition, __LINE__, estring); \
      throw std::runtime_error(std::string(#condition) + " on line " +         \
                               std::to_string(__LINE__) + " failed: ");        \
    }                                                                          \
  } while (0)
