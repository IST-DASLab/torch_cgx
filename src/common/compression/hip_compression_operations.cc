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

#include "hip_compression_operations.h"
#include "gpu_fp16_util.h"
#include "gpu_rand.h"
#include "hip/hip_runtime.h"

#define HIP_CHECK(cmd)                                                         \
  do {                                                                         \
    hipError_t error = cmd;                                                    \
    if (error != hipSuccess) {                                                 \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error),  \
              error, __FILE__, __LINE__);                                      \
      throw std::runtime_error(std::string(#cmd) + " on line " +               \
                               std::to_string(__LINE__) +                      \
                               " returned: " + hipGetErrorString(error));      \
    }                                                                          \
  } while (0)

namespace cgx::common::gpu {
const bool VECTORIZE_COMPRESS = false;
const bool VECTORIZE_DECOMPRESS = false;

__global__ void _init_rand(unsigned int seed, RandState *states) {
  unsigned int index = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  states[index] = xorshift128_init(seed * index);
}

template <typename T>
__global__ void _add(int64_t n, const T *x, const T *y, T *sum_result) {
  int index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  int stride = hipBlockDim_x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    sum_result[i] = sum(x[i], y[i]);
  }
}

// Single value quantization functions
template <typename T, bool EF>
inline __device__ unsigned char
MaxMinEncodeValue(const T input, T *feedback, unsigned char *meta_info, float rand) {
  T *maxmin = ((T *)meta_info);
  float min = type2float(maxmin[1]);
  float unit = type2float(maxmin[0]);
  if (unit < EPS) {
    return 0;
  }
  float input_f = type2float(input);
  float d = ((input_f - min) / unit) + rand;
  unsigned char level = floor(d);
  if (EF)
    *feedback = float2type<T>(input_f - (min + unit * level));
  return level;
}

template <typename T>
inline __device__ T MaxMinDecodeValue(const unsigned char input,
                                      const unsigned char *meta_info,
                                      const unsigned int idx,
                                      const int bucket_size) {
  const unsigned int bucket_no = idx / bucket_size;
  const T *maxmin = ((T *)meta_info) + 2 * bucket_no;
  T min = maxmin[1];
  T unit = maxmin[0];
  return sum(min, mul_int(unit, (int)input));
}

template <typename T, int BITS>
__device__ void find_meta_parallel(const T *input, unsigned char *meta,
                                   int num_elems) {
  unsigned int tid = hipThreadIdx_x;
  unsigned int block_size = hipBlockDim_x;
  T *meta_buf = (T *)meta;
  const unsigned int divisor = (1 << BITS) - 1;
  extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];
  T *sdata = reinterpret_cast<T *>(my_smem);
  meta_buf[0] = input[0];
  meta_buf[1] = input[0];
  unsigned int num_iters_per_bucket = (num_elems + block_size - 1) / block_size;
  for (int i = 0; i < num_iters_per_bucket; i++) {
    unsigned int idx = i * hipBlockDim_x + tid;
    if (idx < num_elems) {
      sdata[tid] = input[idx];
      sdata[block_size + tid] = input[idx];
    }
    __syncthreads();

    for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
      if (tid < s && idx + s < num_elems) {
        sdata[tid] = max(sdata[tid + s], sdata[tid]);
        sdata[block_size + tid] =
            min(sdata[block_size + tid + s], sdata[block_size + tid]);
      }
      __syncthreads();
    }

    if (tid == 0) {
      meta_buf[0] = max(meta_buf[0], sdata[tid]);
      meta_buf[1] = min(meta_buf[1], sdata[block_size + tid]);
    }
  }
  if (tid == 0) {
    float max = type2float(meta_buf[0]);
    float min = type2float(meta_buf[1]);
    meta_buf[0] = float2type<T>((max - min) / divisor);
  }
  __syncthreads();
}

template <int BITS>
inline __device__ void pack_value(const uint64_t value, unsigned char *output,
                                  unsigned int shift = 0) {
#pragma unroll BITS
  for (unsigned int j = 0; j < BITS; j++) {
    output[j] = value >> (PACK_SIZE * j) & 0xFF;
  }
}

template <>
inline __device__ void
pack_value<2>(const uint64_t value, unsigned char *output, unsigned int shift) {
  U2 output2;
#pragma unroll 2
  for (unsigned int j = 0; j < 2; j++) {
    output2.a[j] = value >> (PACK_SIZE * (j + shift)) & 0xFF;
  }
  uchar2 *output_p = reinterpret_cast<uchar2 *>(output);
  output_p[0] = output2.vec;
}

template <>
inline __device__ void
pack_value<3>(const uint64_t value, unsigned char *output, unsigned int shift) {
  U3 output3;
#pragma unroll 3
  for (unsigned int j = 0; j < 3; j++) {
    output3.a[j] = value >> (PACK_SIZE * (j + shift)) & 0xFF;
  }
  uchar3 *output_p = reinterpret_cast<uchar3 *>(output);
  output_p[0] = output3.vec;
}

template <>
inline __device__ void
pack_value<4>(const uint64_t value, unsigned char *output, unsigned int shift) {
  U4 output4;
#pragma unroll 4
  for (unsigned int j = 0; j < 4; j++) {
    output4.a[j] = value >> (PACK_SIZE * (j + shift)) & 0xFF;
  }
  uchar4 *output_p = reinterpret_cast<uchar4 *>(output);
  output_p[0] = output4.vec;
}

template <>
inline __device__ void
pack_value<6>(const uint64_t value, unsigned char *output, unsigned int shift) {
  pack_value<3>(value, output, 0);
  pack_value<3>(value, output + 3, 3);
}

template <>
inline __device__ void
pack_value<8>(const uint64_t value, unsigned char *output, unsigned int shift) {
  pack_value<4>(value, output, 0);
  pack_value<4>(value, output + 4, 4);
}

template <typename T, bool EF, int BITS>
__device__ void CompressBucket(const T *input, unsigned char *output,
                               T *feedback_data, unsigned char *meta_info,
                               int num_elems, RandState *state) {
  unsigned int tid = hipThreadIdx_x;
  unsigned int num_threads = hipBlockDim_x;
  float rand;
  int num_char = (BITS * num_elems + PACK_SIZE - 1) / PACK_SIZE;
  T *feedback_ = nullptr;
  for (unsigned int i = tid; i < (num_elems + PACK_SIZE - 1) / PACK_SIZE;
       i += num_threads) {
    uint64_t value = 0;
    if (VECTORIZE_COMPRESS) {
      typename TypeToVectorType<T>::vector_union input_vector;
      if (num_elems - i * PACK_SIZE >= PACK_SIZE) {
#pragma unroll
        for (unsigned int j = 0; j < PACK_SIZE;
             j += TypeToVectorType<T>::num_values) {
          int idx = i * PACK_SIZE + j;
          input_vector.vec =
              (reinterpret_cast<typename TypeToVectorType<T>::vector_type *>(
                  const_cast<T*>(input + idx)))[0];
#pragma unroll
          for (int k = 0; k < TypeToVectorType<T>::num_values; k++) {
            rand = GetRand(state);
            if (EF)
              feedback_ = feedback_data + idx + k;
            uint64_t encoded = MaxMinEncodeValue<T, EF>(
                input_vector.a[k], feedback_, meta_info, rand);
            value += (encoded << ((j + k) * BITS));
          }
        }
      } else {
        for (unsigned int j = 0; j < num_elems - i * PACK_SIZE; j++) {
          int idx = i * PACK_SIZE + j;
          if (EF)
            feedback_ = feedback_data + idx;
          rand = GetRand(state);
          unsigned encoded =
              MaxMinEncodeValue<T, EF>(input[idx], feedback_, meta_info, rand);
          value += (encoded << (j * BITS));
        }
      }
      if (num_char - i * BITS < BITS) {
        for (unsigned int j = 0; j < num_char - i * BITS; j++) {
          output[i * BITS + j] = value >> (PACK_SIZE * j) & 0xFF;
        }
      } else {
        pack_value<BITS>(value, output + i * BITS);
      }
    } else {
      for (unsigned int j = 0; j < PACK_SIZE && i * PACK_SIZE + j < num_elems;
           j++) {
        int idx = i * PACK_SIZE + j;
        if (EF)
          feedback_ = feedback_data + idx;
        rand = GetRand(state);
        uint64_t encoded =
            MaxMinEncodeValue<T, EF>(input[idx], feedback_, meta_info, rand);
        value += (encoded << (j * BITS));
      }
      for (unsigned int j = 0; j < BITS && i * BITS + j < num_char; j++) {
        output[i * BITS + j] = value >> (PACK_SIZE * j) & 0xFF;
      }
    }
  }
}

template <typename T, bool EF, int BITS>
__global__ void quantize(const unsigned char *input_data,
                         unsigned char *output_data,
                         unsigned char *feedback_data, int num_elems,
                         int bucket_size, RandState *states) {
  unsigned num_blocks = gridDim.x;
  unsigned int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  unsigned int bid = hipBlockIdx_x;
  unsigned int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  unsigned int cur_bucket_size;
  T *meta = (T *)output_data;
  unsigned char *output;
  const int META_MULTIPLIER = 2;
  output = output_data + META_MULTIPLIER * sizeof(T) * num_buckets;

  unsigned int compressed_size =
      (bucket_size * BITS + PACK_SIZE - 1) / PACK_SIZE;

  T *input = (T *)input_data;
  for (int bucket_id = bid; bucket_id < num_buckets; bucket_id += num_blocks) {
    cur_bucket_size = umin(bucket_size, num_elems - bucket_id * bucket_size);
    find_meta_parallel<T, BITS>(
        input + bucket_size * bucket_id,
        (unsigned char *)(meta + META_MULTIPLIER * bucket_id), cur_bucket_size);
  }
  RandState local_state = states[tid];
  for (int bucket_id = bid; bucket_id < num_buckets; bucket_id += num_blocks) {
    cur_bucket_size = umin(bucket_size, num_elems - bucket_id * bucket_size);
    CompressBucket<T, EF, BITS>(
        input + bucket_size * bucket_id, output + compressed_size * bucket_id,
        (T *)feedback_data,
        (unsigned char *)(meta + META_MULTIPLIER * bucket_id), cur_bucket_size,
        &local_state);
  }
  states[tid] = local_state;
}

template <int BITS>
inline __device__ void unpack_value(const unsigned char *input, uint64_t &value,
                                    const unsigned shift = 0) {
  for (unsigned int j = 0; j < BITS; j++) {
    value |= ((uint64_t)input[j]) << (j * PACK_SIZE);
  }
}

template <>
inline __device__ void unpack_value<2>(const unsigned char *input, uint64_t &value,
                                       const unsigned int shift) {
  U2 input2;
  input2.vec = reinterpret_cast<uchar2 *>(const_cast<unsigned char*>(input))[0];
#pragma unroll 3
  for (unsigned int j = 0; j < 2; j++) {
    value |= ((uint64_t)input2.a[j]) << ((j + shift) * PACK_SIZE);
  }
}

template <>
inline __device__ void unpack_value<3>(cosnt unsigned char *input, uint64_t &value,
                                       const unsigned int shift) {
  U3 input3;
  input3.vec = reinterpret_cast<uchar3 *>(const_cast<unsigned char*>(input))[0];
#pragma unroll 3
  for (unsigned int j = 0; j < 3; j++) {
    value |= ((uint64_t)input3.a[j]) << ((j + shift) * PACK_SIZE);
  }
}

template <>
inline __device__ void unpack_value<4>(const unsigned char *input, uint64_t &value,
                                       const unsigned int shift) {
  U4 input4;
  input4.vec = reinterpret_cast<uchar4 *>(const_cast<unsigned char*>(input))[0];
#pragma unroll 4
  for (unsigned int j = 0; j < 4; j++) {
    value |= ((uint64_t)input4.a[j]) << ((j + shift) * PACK_SIZE);
  }
}

template <>
inline __device__ void unpack_value<6>(const unsigned char *input, uint64_t &value,
                                       const unsigned int shift) {
  unpack_value<3>(input, value, 0);
  unpack_value<3>(input + 3, value, 3);
}

template <>
inline __device__ void unpack_value<8>(const unsigned char *input, uint64_t &value,
                                       const unsigned int shift) {
  unpack_value<4>(input, value, 0);
  unpack_value<4>(input + 4, value, 4);
}

template <typename T, bool ADD, int BITS>
__global__ void UnpackArray(const unsigned char *input, const unsigned char *meta_info,
                            T *output, int num_elems, int bucket_size) {
  unsigned int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  unsigned int stride = gridDim.x * hipBlockDim_x;
  int num_char = (BITS * num_elems + PACK_SIZE - 1) / PACK_SIZE;
  const unsigned int divisor = 1 << BITS;
  for (unsigned int i = tid; i < (num_elems + PACK_SIZE - 1) / PACK_SIZE;
       i += stride) {
    uint64_t value = 0;
    if (VECTORIZE_DECOMPRESS) {
      if ((i + 1) * BITS > num_char) {
        for (unsigned int j = 0; j < num_char - i * BITS; j++)
          value |= ((uint64_t)input[i * BITS + j]) << (j * PACK_SIZE);
      } else {
        unpack_value<BITS>(input + i * BITS, value);
      }

      if ((i + 1) * PACK_SIZE > num_elems) {
        for (unsigned int j = 0; j < num_elems - i * PACK_SIZE; j++) {
          unsigned char encoded_value = (value >> (j * BITS)) & (divisor - 1);
          T d = MaxMinDecodeValue<T>(encoded_value, meta_info,
                                     i * PACK_SIZE + j, bucket_size);
          if (ADD) {
            output[i * PACK_SIZE + j] = sum(output[i * PACK_SIZE + j], d);
          } else {
            output[i * PACK_SIZE + j] = d;
          }
        }
      } else {
        typename TypeToVectorType<T>::vector_union output_union;
#pragma unroll
        for (int j = 0; j < PACK_SIZE; j += 4) {
          typename TypeToVectorType<T>::vector_type *output_p =
              reinterpret_cast<typename TypeToVectorType<T>::vector_type *>(
                  &output[i * PACK_SIZE + j]);
          if (ADD)
            output_union.vec = *output_p;
#pragma unroll
          for (int k = 0; k < TypeToVectorType<T>::num_values; k++) {
            unsigned char encoded_value =
                (value >> ((j + k) * BITS)) & (divisor - 1);
            T d = MaxMinDecodeValue<T>(encoded_value, meta_info,
                                       i * PACK_SIZE + j + k, bucket_size);
            if (ADD) {
              output_union.a[k] = sum((T)(output_union.a[k]), d);
            } else {
              output_union.a[k] = d;
            }
            *output_p = output_union.vec;
          }
          typename TypeToVectorType<T>::vector_type *output_p =
              reinterpret_cast<typename TypeToVectorType<T>::vector_type *>(
                  &output[i * PACK_SIZE + j]);
          *output_p = output_union.vec;
        }
      }
    } else {
      for (int j = 0; j < BITS && i * BITS + j < num_char; j++) {
        value |= ((uint64_t)input[i * BITS + j]) << (j * PACK_SIZE);
      }
      for (int j = 0; j < PACK_SIZE && i * PACK_SIZE + j < num_elems; j++) {
        unsigned char encoded_value = (value >> (j * BITS)) & (divisor - 1);
        T d = MaxMinDecodeValue<T>(encoded_value, meta_info, i * PACK_SIZE + j,
                                   bucket_size);
        if (ADD) {
          output[i * PACK_SIZE + j] = sum(output[i * PACK_SIZE + j], d);
        } else {
          output[i * PACK_SIZE + j] = d;
        }
      }
    }
  }
}

/*-------------------Host functions------------------------*/
void HIP_init_rand(RandState *states, size_t num_elems, unsigned int seed,
                   hipStream_t stream) {
  hipLaunchKernelGGL(
      (_init_rand),
      dim3(BLOCKS_PER_GRID(num_elems, THREADS_PER_BLOCK_COMPRESS)),
      dim3(THREADS_PER_BLOCK_COMPRESS), 0, stream, seed, states);
}

template <typename T>
void HIP_add(int n, const T *x, T *y, T *sum, hipStream_t stream) {
  int num_threads = umin(n, MAX_THREADS_PER_BLOCK);
  int blocks = BLOCKS_PER_GRID(n, num_threads);
  hipLaunchKernelGGL((_add<T>), dim3(blocks), dim3(num_threads), 0, stream, n,
                     x, y, sum);
  HIP_CHECK(hipGetLastError());
}

template <typename T, bool EF>
inline void QUANTIZE(const unsigned char *input_data,
                     unsigned char *output_data, unsigned char *feedback_data,
                     int num_elems, int bits, int bucket_size,
                     RandState *states, hipStream_t stream, int num_blocks,
                     int num_threads, int shared_memory_block_size) {
  switch (bits) {
  case 1:
    hipLaunchKernelGGL((quantize<T, EF, 1>), dim3(num_blocks),
                       dim3(num_threads), shared_memory_block_size, stream,
                       input_data, output_data, feedback_data, num_elems,
                       bucket_size, states);
    break;
  case 2:
    hipLaunchKernelGGL((quantize<T, EF, 2>), dim3(num_blocks),
                       dim3(num_threads), shared_memory_block_size, stream,
                       input_data, output_data, feedback_data, num_elems,
                       bucket_size, states);
    break;
  case 3:
    hipLaunchKernelGGL((quantize<T, EF, 3>), dim3(num_blocks),
                       dim3(num_threads), shared_memory_block_size, stream,
                       input_data, output_data, feedback_data, num_elems,
                       bucket_size, states);
    break;
  case 4:
    hipLaunchKernelGGL((quantize<T, EF, 4>), dim3(num_blocks),
                       dim3(num_threads), shared_memory_block_size, stream,
                       input_data, output_data, feedback_data, num_elems,
                       bucket_size, states);
    break;
  case 5:
    hipLaunchKernelGGL((quantize<T, EF, 5>), dim3(num_blocks),
                       dim3(num_threads), shared_memory_block_size, stream,
                       input_data, output_data, feedback_data, num_elems,
                       bucket_size, states);
    break;
  case 6:
    hipLaunchKernelGGL((quantize<T, EF, 6>), dim3(num_blocks),
                       dim3(num_threads), shared_memory_block_size, stream,
                       input_data, output_data, feedback_data, num_elems,
                       bucket_size, states);
    break;
  case 7:
    hipLaunchKernelGGL((quantize<T, EF, 7>), dim3(num_blocks),
                       dim3(num_threads), shared_memory_block_size, stream,
                       input_data, output_data, feedback_data, num_elems,
                       bucket_size, states);
    break;
  case 8:
    hipLaunchKernelGGL((quantize<T, EF, 8>), dim3(num_blocks),
                       dim3(num_threads), shared_memory_block_size, stream,
                       input_data, output_data, feedback_data, num_elems,
                       bucket_size, states);
    break;
  default:
    printf("Wrong number of bits %i!!!\n", bits);
  }
  HIP_CHECK(hipGetLastError());
}

template <typename T>
void HIP_quantize_maxmin(const unsigned char *input_data,
                         unsigned char *output_data,
                         unsigned char *feedback_data, int num_elems, int bits,
                         int bucket_size, RandState *states,
                         hipStream_t stream) {
  int num_blocks =
      umin((num_elems + bucket_size - 1) / bucket_size, MAX_NUMBER_OF_BLOCKS);
  int num_threads = umin(THREADS_PER_BLOCK_COMPRESS, bucket_size);
  int shared_memory_block_size = 2 * MAX_THREADS_PER_BLOCK * sizeof(T);
  QUANTIZE<T, false>(input_data, output_data, feedback_data, num_elems, bits,
                     bucket_size, states, stream, num_blocks, num_threads,
                     shared_memory_block_size);
}

template <typename T, bool ADD>
inline void DEQUANTIZE(const unsigned char *input, const unsigned char *meta_info,
                       T *output, int num_elems, int bucket_size, int bits,
                       hipStream_t stream, int num_blocks, int num_threads) {
  switch (bits) {
  case 1:
    hipLaunchKernelGGL((UnpackArray<T, ADD, 1>), dim3(num_blocks),
                       dim3(num_threads), 0, stream, input, meta_info, output,
                       num_elems, bucket_size);
    break;
  case 2:
    hipLaunchKernelGGL((UnpackArray<T, ADD, 2>), dim3(num_blocks),
                       dim3(num_threads), 0, stream, input, meta_info, output,
                       num_elems, bucket_size);
    break;
  case 3:
    hipLaunchKernelGGL((UnpackArray<T, ADD, 3>), dim3(num_blocks),
                       dim3(num_threads), 0, stream, input, meta_info, output,
                       num_elems, bucket_size);
    break;
  case 4:
    hipLaunchKernelGGL((UnpackArray<T, ADD, 4>), dim3(num_blocks),
                       dim3(num_threads), 0, stream, input, meta_info, output,
                       num_elems, bucket_size);
    break;
  case 5:
    hipLaunchKernelGGL((UnpackArray<T, ADD, 1>), dim3(num_blocks),
                       dim3(num_threads), 0, stream, input, meta_info, output,
                       num_elems, bucket_size);
    break;
  case 6:
    hipLaunchKernelGGL((UnpackArray<T, ADD, 6>), dim3(num_blocks),
                       dim3(num_threads), 0, stream, input, meta_info, output,
                       num_elems, bucket_size);
    break;
  case 7:
    hipLaunchKernelGGL((UnpackArray<T, ADD, 7>), dim3(num_blocks),
                       dim3(num_threads), 0, stream, input, meta_info, output,
                       num_elems, bucket_size);
    break;
  case 8:
    hipLaunchKernelGGL((UnpackArray<T, ADD, 8>), dim3(num_blocks),
                       dim3(num_threads), 0, stream, input, meta_info, output,
                       num_elems, bucket_size);
    break;
  default:
    printf("Wrong number of bits %i!!!\n", bits);
  }
  HIP_CHECK(hipGetLastError());
}

template <typename T, bool ADD>
void HIP_dequantize_maxmin(const unsigned char *input_data,
                           unsigned char *output_data, int num_elems, int bits,
                           int bucket_size, hipStream_t stream) {
  T *output = (T *)output_data;
  const unsigned char *meta_info = input_data;
  int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  unsigned char *input = input_data + 2 * sizeof(T) * num_buckets;
  int num_threads = THREADS_PER_BLOCK_DECOMPRESS;
  int num_blocks = BLOCKS_PER_GRID(num_elems / PACK_SIZE, num_threads);
  DEQUANTIZE<T, ADD>(input, meta_info, output, num_elems, bucket_size, bits,
                     stream, num_blocks, num_threads);
}

/* Functions declarations */
template void HIP_add<float>(int n, const float *x, float *y, float *sum,
                             hipStream_t stream);
template void HIP_add<Half>(int n, const Half *x, Half *y, Half *sum,
                            hipStream_t stream);

template void HIP_quantize_maxmin<float>(const unsigned char *input_data,
                                         unsigned char *output_data,
                                         unsigned char *feedback_data,
                                         int num_elems, int bits,
                                         int bucket_size, RandState *states,
                                         hipStream_t stream);
template void HIP_quantize_maxmin<Half>(const unsigned char *input_data,
                                        unsigned char *output_data,
                                        unsigned char *feedback_data,
                                        int num_elems, int bits,
                                        int bucket_size, RandState *states,
                                        hipStream_t stream);

template void HIP_dequantize_maxmin<float, true>(const unsigned char *input_data,
                                                 unsigned char *output_data,
                                                 int num_elems, int bits,
                                                 int bucket_size,
                                                 hipStream_t stream);
template void HIP_dequantize_maxmin<float, false>(const unsigned char *input_data,
                                                  unsigned char *output_data,
                                                  int num_elems, int bits,
                                                  int bucket_size,
                                                  hipStream_t stream);

template void HIP_dequantize_maxmin<Half, true>(const unsigned char *input_data,
                                                unsigned char *output_data,
                                                int num_elems, int bits,
                                                int bucket_size,
                                                hipStream_t stream);
template void HIP_dequantize_maxmin<Half, false>(const unsigned char *input_data,
                                                 unsigned char *output_data,
                                                 int num_elems, int bits,
                                                 int bucket_size,
                                                 hipStream_t stream);

} // namespace cgx::common::gpu
