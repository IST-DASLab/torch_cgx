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

#include "cuda_compression_operations.h"
#include "gpu_common.h"
#include "gpu_fp16_util.h"
#include "gpu_rand.h"

namespace cgx {
namespace common {
namespace gpu {
#if CUDA_VECTORIZED
const bool VECTORIZE_COMPRESS = true;
const bool VECTORIZE_DECOMPRESS = true;
#else
const bool VECTORIZE_COMPRESS = false;
const bool VECTORIZE_DECOMPRESS = false;
#endif

const int MAXMIN_META_MULTIPLIER = 2;

__global__ void _init_rand(unsigned int seed, RandState *states) {
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  states[index] = xorshift128_init(seed * index);
}

__global__ void _float2half(float *input, Half *output, int numel) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < numel; i += stride) {
    output[i] = __float2half(input[i]);
  }
}

__global__ void _half2float(Half *input, float *output, int numel) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < numel; i += stride) {
    output[i] = __half2float(input[i]);
  }
}

template <typename T>
__global__ void _add(int64_t n, const T *x, const T *y, T *sum_result) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    sum_result[i] = sum(x[i], y[i]);
  }
}

// Single value quantization functions
template <typename T, bool EF, int BITS>
inline __device__ unsigned char MaxMinEncodeValue(T input, T *feedback,
                                                  unsigned char *meta_info,
                                                  const float rand) {
  T *maxmin = ((T *)meta_info);
  const float unit_bucket = type2float(maxmin[0]);
  if (unit_bucket < EPS) {
    return 0;
  }
  const float min_bucket = type2float(maxmin[1]);
  const float input_f = type2float(input);
  const float d = ((input_f - min_bucket) / unit_bucket) + rand;
  const unsigned char level = min(floor(d), (1 << BITS) - 1);
  if (EF)
    *feedback = float2type<T>(input_f - (min_bucket + unit_bucket * level));
  return level;
}

template <typename T>
inline __device__ T MaxMinDecodeValue(const unsigned char input,
                                      const unsigned char *meta_info,
                                      const const unsigned int idx,
                                      const int bucket_size) {
  const unsigned int bucket_no = idx / bucket_size;
  const T *maxmin = ((T *)meta_info) + MAXMIN_META_MULTIPLIER * bucket_no;
  const T min = maxmin[1];
  const T unit = maxmin[0];
  return sum(min, mul_int(unit, (int)input));
}

template <typename T, int BITS>
__device__ void find_meta_parallel(const T *input, T *meta,
                                   const int num_elems) {
  unsigned int tid = threadIdx.x;
  unsigned int block_size = blockDim.x;
  const unsigned int divisor = (1 << BITS) - 1;
  extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];
  T *sdata = reinterpret_cast<T *>(my_smem);
  meta[0] = input[0];
  meta[1] = input[0];
  unsigned int num_iters_per_bucket = (num_elems + block_size - 1) / block_size;
  for (int i = 0; i < num_iters_per_bucket; i++) {
    unsigned int idx = i * blockDim.x + tid;
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
      meta[0] = max(meta[0], sdata[tid]);
      meta[1] = min(meta[1], sdata[block_size + tid]);
    }
  }
  if (tid == 0) {
    float max = type2float(meta[0]);
    float min = type2float(meta[1]);
    meta[0] = float2type<T>((max - min) / divisor);
  }
  __syncthreads();
}

template <typename T, int BITS>
__global__ void find_meta(const T *input, T *meta, const int num_elems,
                          const int bucket_size) {
  unsigned num_blocks = gridDim.x;
  unsigned int bid = blockIdx.x;
  unsigned int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  unsigned int cur_bucket_size;
  for (unsigned int bucket_id = bid; bucket_id < num_buckets;
       bucket_id += num_blocks) {
    cur_bucket_size = umin(bucket_size, num_elems - bucket_id * bucket_size);
    find_meta_parallel<T, BITS>(input + bucket_size * bucket_id,
                                meta + MAXMIN_META_MULTIPLIER * bucket_id,
                                cur_bucket_size);
  }
}

template <int BITS>
inline __device__ void pack_value(const uint64_t value, unsigned char *output,
                                  const unsigned int shift = 0) {
#pragma unroll BITS
  for (unsigned int j = 0; j < BITS; j++) {
    output[j] = value >> (PACK_SIZE * j) & 0xFF;
  }
}

template <>
inline __device__ void pack_value<2>(const uint64_t value,
                                     unsigned char *output,
                                     const unsigned int shift) {
  U2 output2;
#pragma unroll 2
  for (unsigned int j = 0; j < 2; j++) {
    output2.a[j] = value >> (PACK_SIZE * (j + shift)) & 0xFF;
  }
  uchar2 *output_p = reinterpret_cast<uchar2 *>(output);
  output_p[0] = output2.vec;
}

template <>
inline __device__ void pack_value<3>(const uint64_t value,
                                     unsigned char *output,
                                     const unsigned int shift) {
  U3 output3;
#pragma unroll 3
  for (unsigned int j = 0; j < 3; j++) {
    output3.a[j] = value >> (PACK_SIZE * (j + shift)) & 0xFF;
  }
  uchar3 *output_p = reinterpret_cast<uchar3 *>(output);
  output_p[0] = output3.vec;
}

template <>
inline __device__ void pack_value<4>(const uint64_t value,
                                     unsigned char *output,
                                     const unsigned int shift) {
  U4 output4;
#pragma unroll 4
  for (unsigned int j = 0; j < 4; j++) {
    output4.a[j] = value >> (PACK_SIZE * (j + shift)) & 0xFF;
  }
  uchar4 *output_p = reinterpret_cast<uchar4 *>(output);
  output_p[0] = output4.vec;
}

template <>
inline __device__ void pack_value<6>(const uint64_t value,
                                     unsigned char *output,
                                     const unsigned int shift) {
  pack_value<3>(value, output, 0);
  pack_value<3>(value, output + 3, 3);
}

template <>
inline __device__ void pack_value<8>(const uint64_t value,
                                     unsigned char *output,
                                     const unsigned int shift) {
  pack_value<4>(value, output, 0);
  pack_value<4>(value, output + 4, 4);
}

template <typename T, bool EF, int BITS, bool VECTORIZE>
__device__ void CompressBucket(const T *input, unsigned char *output,
                               T *feedback_data, unsigned char *meta_info,
                               const int num_elems, RandState *state) {
  typename TypeToVectorType<T>::vector_union input_vector;
  const unsigned int tid = threadIdx.x;
  const unsigned int num_threads = blockDim.x;
  float rand = 0.5;
  int num_char = (BITS * num_elems + PACK_SIZE - 1) / PACK_SIZE;
  T *feedback_ = nullptr;
  for (unsigned int i = tid; i < (num_elems + PACK_SIZE - 1) / PACK_SIZE;
       i += num_threads) {
    uint64_t value = 0;
    if (VECTORIZE) {
      if (num_elems - i * PACK_SIZE >= PACK_SIZE) {
#pragma unroll
        for (unsigned int j = 0; j < PACK_SIZE;
             j += TypeToVectorType<T>::num_values) {
          int idx = i * PACK_SIZE + j;
          input_vector.vec =
              (reinterpret_cast<typename TypeToVectorType<T>::vector_type *>(
                  const_cast<T *>(input + idx)))[0];
#pragma unroll
          for (int k = 0; k < TypeToVectorType<T>::num_values; k++) {
            rand = GetRand(state);
            if (EF)
              feedback_ = feedback_data + idx + k;
            uint64_t encoded = MaxMinEncodeValue<T, EF, BITS>(
                input_vector.a[k], feedback_, meta_info, rand);
            value |= (encoded << ((j + k) * BITS));
          }
        }
      } else {
        for (unsigned int j = 0; j < num_elems - i * PACK_SIZE; j++) {
          int idx = i * PACK_SIZE + j;
          if (EF)
            feedback_ = feedback_data + idx;
          rand = GetRand(state);
          uint64_t encoded = MaxMinEncodeValue<T, EF, BITS>(
              input[idx], feedback_, meta_info, rand);
          value |= (encoded << (j * BITS));
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
        uint64_t encoded = MaxMinEncodeValue<T, EF, BITS>(input[idx], feedback_,
                                                          meta_info, rand);
        value |= (encoded << (j * BITS));
      }
      for (unsigned int j = 0; j < BITS && i * BITS + j < num_char; j++) {
        output[i * BITS + j] = value >> (PACK_SIZE * j) & 0xFF;
      }
    }
  }
}

template <typename T, bool EF, int BITS, bool VECTORIZE>
__global__ void pack_array(const T *input, unsigned char *output_data,
                           unsigned char *feedback_data, const int num_elems,
                           const unsigned int bucket_size, RandState *states) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  unsigned char *meta_info = output_data;
  unsigned char *output;
  output = output_data + MAXMIN_META_MULTIPLIER * sizeof(T) * num_buckets;

  const unsigned int stride = gridDim.x * blockDim.x;
  float rand = 0.5;
  int bucket_no;
  int num_char = (BITS * num_elems + PACK_SIZE - 1) / PACK_SIZE;
  T *feedback_;
#if !QSGD_DETERMENISTIC
  RandState *state = &states[tid];
#else
  RandState *state = nullptr;
#endif
  for (unsigned int i = tid; i < (num_elems + PACK_SIZE - 1) / PACK_SIZE;
       i += stride) {
    uint64_t value = 0;
    if (VECTORIZE) {
      typename TypeToVectorType<T>::vector_union input_vector;
      if (num_elems - i * PACK_SIZE >= PACK_SIZE) {
#pragma unroll
        for (unsigned int j = 0; j < PACK_SIZE;
             j += TypeToVectorType<T>::num_values) {
          int idx = i * PACK_SIZE + j;
          input_vector.vec =
              (reinterpret_cast<typename TypeToVectorType<T>::vector_type *>(
                  const_cast<T *>(input + idx)))[0];
#pragma unroll
          for (int k = 0; k < TypeToVectorType<T>::num_values; k++) {
            rand = GetRand(state);
            if (EF)
              feedback_ = ((T *)feedback_data) + idx + k;
            bucket_no = (idx + k) / bucket_size;
            uint64_t encoded = MaxMinEncodeValue<T, EF, BITS>(
                input_vector.a[k], feedback_,
                meta_info + MAXMIN_META_MULTIPLIER * sizeof(T) * bucket_no,
                rand);
            value |= (encoded << ((j + k) * BITS));
          }
        }
      } else {
        for (unsigned int j = 0; j < num_elems - i * PACK_SIZE; j++) {
          int idx = i * PACK_SIZE + j;
          if (EF)
            feedback_ = ((T *)feedback_data) + idx;
          rand = GetRand(state);
          bucket_no = idx / bucket_size;
          uint64_t encoded = MaxMinEncodeValue<T, EF, BITS>(
              input[idx], feedback_,
              meta_info + MAXMIN_META_MULTIPLIER * sizeof(T) * bucket_no, rand);
          value |= (encoded << (j * BITS));
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
          feedback_ = ((T *)feedback_data) + idx;
        rand = GetRand(state);
        bucket_no = idx / bucket_size;
        uint64_t encoded = MaxMinEncodeValue<T, EF, BITS>(
            input[idx], feedback_,
            meta_info + MAXMIN_META_MULTIPLIER * sizeof(T) * bucket_no, rand);
        value |= (encoded << (j * BITS));
      }
      for (unsigned int j = 0; j < BITS && i * BITS + j < num_char; j++) {
        output[i * BITS + j] = value >> (PACK_SIZE * j) & 0xFF;
      }
    }
  }
}

template <typename T, bool EF, int BITS, bool VECTORIZE>
__global__ void quantize(const unsigned char *input_data,
                         unsigned char *output_data,
                         unsigned char *feedback_data, const int num_elems,
                         const unsigned int bucket_size, RandState *states) {
  unsigned num_blocks = gridDim.x;
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int bid = blockIdx.x;
  unsigned int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  unsigned int cur_bucket_size;
  T *meta = (T *)output_data;
  unsigned char *output;
  output = output_data + MAXMIN_META_MULTIPLIER * sizeof(T) * num_buckets;

  unsigned int compressed_size =
      (bucket_size * BITS + PACK_SIZE - 1) / PACK_SIZE;

  T *input = (T *)input_data;
  for (unsigned int bucket_id = bid; bucket_id < num_buckets;
       bucket_id += num_blocks) {
    cur_bucket_size = umin(bucket_size, num_elems - bucket_id * bucket_size);
    find_meta_parallel<T, BITS>(input + bucket_size * bucket_id,
                                (meta + MAXMIN_META_MULTIPLIER * bucket_id),
                                cur_bucket_size);
  }
  RandState local_state = states[tid];
  for (unsigned int bucket_id = bid; bucket_id < num_buckets;
       bucket_id += num_blocks) {
    cur_bucket_size = umin(bucket_size, num_elems - bucket_id * bucket_size);
    CompressBucket<T, EF, BITS, VECTORIZE>(
        input + bucket_size * bucket_id, output + compressed_size * bucket_id,
        (T *)feedback_data,
        (unsigned char *)(meta + MAXMIN_META_MULTIPLIER * bucket_id),
        cur_bucket_size, &local_state);
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
inline __device__ void unpack_value<2>(const unsigned char *input,
                                       uint64_t &value,
                                       const unsigned int shift) {
  U2 input2;
  input2.vec =
      reinterpret_cast<uchar2 *>(const_cast<unsigned char *>(input))[0];
#pragma unroll 2
  for (unsigned int j = 0; j < 2; j++) {
    value |= ((uint64_t)input2.a[j]) << ((j + shift) * PACK_SIZE);
  }
}

template <>
inline __device__ void unpack_value<3>(const unsigned char *input,
                                       uint64_t &value,
                                       const unsigned int shift) {
  U3 input3;
  input3.vec =
      reinterpret_cast<uchar3 *>(const_cast<unsigned char *>(input))[0];
#pragma unroll 3
  for (unsigned int j = 0; j < 3; j++) {
    value |= ((uint64_t)input3.a[j]) << ((j + shift) * PACK_SIZE);
  }
}

template <>
inline __device__ void unpack_value<4>(const unsigned char *input,
                                       uint64_t &value,
                                       const unsigned int shift) {
  U4 input4;
  input4.vec =
      reinterpret_cast<uchar4 *>(const_cast<unsigned char *>(input))[0];
#pragma unroll 4
  for (unsigned int j = 0; j < 4; j++) {
    value |= ((uint64_t)input4.a[j]) << ((j + shift) * PACK_SIZE);
  }
}

template <>
inline __device__ void unpack_value<6>(const unsigned char *input,
                                       uint64_t &value,
                                       const unsigned int shift) {
  unpack_value<3>(input, value, 0);
  unpack_value<3>(input + 3, value, 3);
}

template <>
inline __device__ void unpack_value<8>(const unsigned char *input,
                                       uint64_t &value,
                                       const unsigned int shift) {
  unpack_value<4>(input, value, 0);
  unpack_value<4>(input + 4, value, 4);
}

template <typename T, bool ADD, int BITS, bool VECTORIZE>
__global__ void UnpackArray(const unsigned char *input,
                            const unsigned char *meta_info, T *output,
                            const int num_elems, const int bucket_size) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  int num_char = (BITS * num_elems + PACK_SIZE - 1) / PACK_SIZE;
  const unsigned int divisor = 1 << BITS;
  for (unsigned int i = tid; i < (num_elems + PACK_SIZE - 1) / PACK_SIZE;
       i += stride) {
    uint64_t value = 0;
    if (VECTORIZE) {
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
#pragma unroll(PACK_SIZE / TypeToVectorType <T>::num_values)
        for (int j = 0; j < PACK_SIZE; j += TypeToVectorType<T>::num_values) {
          typename TypeToVectorType<T>::vector_type *output_p =
              reinterpret_cast<typename TypeToVectorType<T>::vector_type *>(
                  &output[i * PACK_SIZE + j]);
          if (ADD)
            output_union.vec = *output_p;
#pragma unroll TypeToVectorType < T> ::num_values
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
          }
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
void CUDA_init_rand(RandState *states, size_t num_elems, unsigned int seed,
                    cudaStream_t stream) {
  _init_rand<<<BLOCKS_PER_GRID(num_elems, THREADS_PER_BLOCK_COMPRESS),
               THREADS_PER_BLOCK_COMPRESS, 0, stream>>>(seed, states);
}

template <typename T>
void CUDA_add(int n, const T *x, T *y, T *sum, cudaStream_t stream) {
  int num_threads = umin(n, MAX_THREADS_PER_BLOCK);
  int blocks = BLOCKS_PER_GRID(n, num_threads);
  _add<T><<<blocks, num_threads, 0, stream>>>(n, x, y, sum);
  CUDA_CHECK(cudaGetLastError());
}

void CUDA_half2float(Half *input, float *output, int numel,
                     cudaStream_t stream) {
  _half2float<<<numel, 1, 0, 0>>>(input, output, numel);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

void CUDA_float2half(float *input, half *output, int numel,
                     cudaStream_t stream) {
  _float2half<<<numel, 1, 0, 0>>>(input, output, numel);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

// Difference between two alternatives QUANTIZE and QUANTIZE2 is that
// in QUANTIZE we only call 1 kernel in which we compute meta and pack values
// for each bucket. It means we use less blocks for packing than we could
// in QUANTIZE2 we first find all meta information in one kernel,
// then pack in the separate one.
template <typename T, bool EF, int BITS, bool VECTORIZE>
inline void QUANTIZE2(const unsigned char *input_data,
                      unsigned char *output_data, unsigned char *feedback_data,
                      int num_elems, int bucket_size, RandState *states,
                      cudaStream_t stream) {
  int num_blocks =
      umin((num_elems + bucket_size - 1) / bucket_size, MAX_NUMBER_OF_BLOCKS);
  int num_threads = umin(THREADS_PER_BLOCK_COMPRESS, bucket_size);
  int shared_memory_block_size =
      MAXMIN_META_MULTIPLIER * num_threads * sizeof(T);
  unsigned int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  const T *input = reinterpret_cast<const T *>(input_data);
  T *meta_info = reinterpret_cast<T *>(output_data);
  find_meta<T, BITS>
      <<<num_blocks, num_threads, shared_memory_block_size, stream>>>(
          input, meta_info, num_elems, bucket_size);

  num_threads = THREADS_PER_BLOCK_COMPRESS;
  num_blocks = BLOCKS_PER_GRID(num_elems / PACK_SIZE, num_threads);
  pack_array<T, EF, BITS, VECTORIZE><<<num_blocks, num_threads, 0, stream>>>(
      input, output_data, feedback_data, num_elems, bucket_size, states);
}

template <typename T, bool EF, bool VECTORIZE>
inline void QUANTIZE1(const unsigned char *input_data,
                      unsigned char *output_data, unsigned char *feedback_data,
                      int num_elems, int bits, int bucket_size,
                      RandState *states, cudaStream_t stream) {
  switch (bits) {
  case 1:
    QUANTIZE2<T, EF, 1, VECTORIZE>(input_data, output_data, feedback_data,
                                   num_elems, bucket_size, states, stream);
    break;
  case 2:
    QUANTIZE2<T, EF, 2, VECTORIZE>(input_data, output_data, feedback_data,
                                   num_elems, bucket_size, states, stream);
    break;
  case 3:
    QUANTIZE2<T, EF, 3, VECTORIZE>(input_data, output_data, feedback_data,
                                   num_elems, bucket_size, states, stream);
    break;
  case 4:
    QUANTIZE2<T, EF, 4, VECTORIZE>(input_data, output_data, feedback_data,
                                   num_elems, bucket_size, states, stream);
    break;
  case 5:
    QUANTIZE2<T, EF, 5, VECTORIZE>(input_data, output_data, feedback_data,
                                   num_elems, bucket_size, states, stream);
    break;
  case 6:
    QUANTIZE2<T, EF, 6, VECTORIZE>(input_data, output_data, feedback_data,
                                   num_elems, bucket_size, states, stream);
    break;
  case 7:
    QUANTIZE2<T, EF, 7, VECTORIZE>(input_data, output_data, feedback_data,
                                   num_elems, bucket_size, states, stream);
    break;
  case 8:
    QUANTIZE2<T, EF, 8, VECTORIZE>(input_data, output_data, feedback_data,
                                   num_elems, bucket_size, states, stream);
    break;
  default:
    printf("Wrong number of bits %i!!!\n", bits);
  }
  CUDA_CHECK(cudaGetLastError());
}

template <typename T, bool EF, bool VECTORIZE>
inline void QUANTIZE(const unsigned char *input_data,
                     unsigned char *output_data, unsigned char *feedback_data,
                     int num_elems, int bits, int bucket_size,
                     RandState *states, cudaStream_t stream) {
  int num_blocks =
      umin((num_elems + bucket_size - 1) / bucket_size, MAX_NUMBER_OF_BLOCKS);
  int num_threads = umin(THREADS_PER_BLOCK_COMPRESS, bucket_size);
  int shared_memory_block_size =
      MAXMIN_META_MULTIPLIER * MAX_THREADS_PER_BLOCK * sizeof(T);
  switch (bits) {
  case 1:
    quantize<T, EF, 1, VECTORIZE>
        <<<num_blocks, num_threads, shared_memory_block_size, stream>>>(
            input_data, output_data, feedback_data, num_elems, bucket_size,
            states);
    break;
  case 2:
    quantize<T, EF, 2, VECTORIZE>
        <<<num_blocks, num_threads, shared_memory_block_size, stream>>>(
            input_data, output_data, feedback_data, num_elems, bucket_size,
            states);
    break;
  case 3:
    quantize<T, EF, 3, VECTORIZE>
        <<<num_blocks, num_threads, shared_memory_block_size, stream>>>(
            input_data, output_data, feedback_data, num_elems, bucket_size,
            states);
    break;
  case 4:
    quantize<T, EF, 4, VECTORIZE>
        <<<num_blocks, num_threads, shared_memory_block_size, stream>>>(
            input_data, output_data, feedback_data, num_elems, bucket_size,
            states);
    break;
  case 5:
    quantize<T, EF, 5, VECTORIZE>
        <<<num_blocks, num_threads, shared_memory_block_size, stream>>>(
            input_data, output_data, feedback_data, num_elems, bucket_size,
            states);
    break;
  case 6:
    quantize<T, EF, 6, VECTORIZE>
        <<<num_blocks, num_threads, shared_memory_block_size, stream>>>(
            input_data, output_data, feedback_data, num_elems, bucket_size,
            states);
    break;
  case 7:
    quantize<T, EF, 7, VECTORIZE>
        <<<num_blocks, num_threads, shared_memory_block_size, stream>>>(
            input_data, output_data, feedback_data, num_elems, bucket_size,
            states);
    break;
  case 8:
    quantize<T, EF, 8, VECTORIZE>
        <<<num_blocks, num_threads, shared_memory_block_size, stream>>>(
            input_data, output_data, feedback_data, num_elems, bucket_size,
            states);
    break;
  default:
    printf("Wrong number of bits %i!!!\n", bits);
  }
  CUDA_CHECK(cudaGetLastError());
}

// get_curand_array_size assumes that compressed will be done with maximum
// THREADS_PER_BLOCK_COMPRESS threads.
template <typename T>
void CUDA_quantize_maxmin(const unsigned char *input_data,
                          unsigned char *output_data,
                          unsigned char *feedback_data, int num_elems, int bits,
                          int bucket_size, RandState *states,
                          cudaStream_t stream) {
  // if the buffer is not aligned for vectorized, fallback to non-vectorized
  if (VECTORIZE_COMPRESS and (((unsigned long)input_data & 15) == 0))
    QUANTIZE1<T, false, true>(input_data, output_data, feedback_data, num_elems,
                              bits, bucket_size, states, stream);
  else
    QUANTIZE1<T, false, false>(input_data, output_data, feedback_data,
                               num_elems, bits, bucket_size, states, stream);
}

template <typename T, bool ADD, bool VECTORIZE>
inline void DEQUANTIZE(const unsigned char *input,
                       const unsigned char *meta_info, T *output, int num_elems,
                       int bucket_size, int bits, cudaStream_t stream,
                       int num_blocks, int num_threads) {
  switch (bits) {
  case 1:
    UnpackArray<T, ADD, 1, VECTORIZE><<<num_blocks, num_threads, 0, stream>>>(
        input, meta_info, output, num_elems, bucket_size);
    break;
  case 2:
    UnpackArray<T, ADD, 2, VECTORIZE><<<num_blocks, num_threads, 0, stream>>>(
        input, meta_info, output, num_elems, bucket_size);
    break;
  case 3:
    UnpackArray<T, ADD, 3, VECTORIZE><<<num_blocks, num_threads, 0, stream>>>(
        input, meta_info, output, num_elems, bucket_size);
    break;
  case 4:
    UnpackArray<T, ADD, 4, VECTORIZE><<<num_blocks, num_threads, 0, stream>>>(
        input, meta_info, output, num_elems, bucket_size);
    break;
  case 5:
    UnpackArray<T, ADD, 5, VECTORIZE><<<num_blocks, num_threads, 0, stream>>>(
        input, meta_info, output, num_elems, bucket_size);
    break;
  case 6:
    UnpackArray<T, ADD, 6, VECTORIZE><<<num_blocks, num_threads, 0, stream>>>(
        input, meta_info, output, num_elems, bucket_size);
    break;
  case 7:
    UnpackArray<T, ADD, 7, VECTORIZE><<<num_blocks, num_threads, 0, stream>>>(
        input, meta_info, output, num_elems, bucket_size);
    break;
  case 8:
    UnpackArray<T, ADD, 8, VECTORIZE><<<num_blocks, num_threads, 0, stream>>>(
        input, meta_info, output, num_elems, bucket_size);
    break;
  default:
    printf("Wrong number of bits %i!!!\n", bits);
  }
  CUDA_CHECK(cudaGetLastError());
}

template <typename T, bool ADD>
void DEQUANTIZE1(const unsigned char *input, const unsigned char *meta_info,
                 T *output, int num_elems, int bucket_size, int bits,
                 cudaStream_t stream, int num_blocks, int num_threads) {
  // if the buffer is not aligned for vectorized, fallback to non-vectorized
  if (VECTORIZE_DECOMPRESS and (((unsigned long)output & 15) == 0))
    DEQUANTIZE<T, ADD, true>(input, meta_info, output, num_elems, bucket_size,
                             bits, stream, num_blocks, num_threads);
  else
    DEQUANTIZE<T, ADD, false>(input, meta_info, output, num_elems, bucket_size,
                              bits, stream, num_blocks, num_threads);
}

template <typename T, bool ADD>
void CUDA_dequantize_maxmin(const unsigned char *input_data,
                            unsigned char *output_data, int num_elems, int bits,
                            int bucket_size, cudaStream_t stream) {
  T *output = (T *)output_data;
  const unsigned char *meta_info = input_data;
  int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  const unsigned char *input =
      input_data + MAXMIN_META_MULTIPLIER * sizeof(T) * num_buckets;
  int num_threads = THREADS_PER_BLOCK_DECOMPRESS;
  int num_blocks =
      BLOCKS_PER_GRID((num_elems + PACK_SIZE - 1) / PACK_SIZE, num_threads);
  DEQUANTIZE1<T, ADD>(input, meta_info, output, num_elems, bucket_size, bits,
                      stream, num_blocks, num_threads);
}

/* Functions declarations */
template void CUDA_add<float>(int n, const float *x, float *y, float *sum,
                              cudaStream_t stream);
template void CUDA_add<Half>(int n, const Half *x, Half *y, Half *sum,
                             cudaStream_t stream);

template void CUDA_quantize_maxmin<float>(const unsigned char *input_data,
                                          unsigned char *output_data,
                                          unsigned char *feedback_data,
                                          int num_elems, int bits,
                                          int bucket_size, RandState *states,
                                          cudaStream_t stream);
template void CUDA_quantize_maxmin<Half>(const unsigned char *input_data,
                                         unsigned char *output_data,
                                         unsigned char *feedback_data,
                                         int num_elems, int bits,
                                         int bucket_size, RandState *states,
                                         cudaStream_t stream);

template void CUDA_dequantize_maxmin<float, true>(
    const unsigned char *input_data, unsigned char *output_data, int num_elems,
    int bits, int bucket_size, cudaStream_t stream);
template void CUDA_dequantize_maxmin<float, false>(
    const unsigned char *input_data, unsigned char *output_data, int num_elems,
    int bits, int bucket_size, cudaStream_t stream);

template void CUDA_dequantize_maxmin<Half, true>(
    const unsigned char *input_data, unsigned char *output_data, int num_elems,
    int bits, int bucket_size, cudaStream_t stream);
template void CUDA_dequantize_maxmin<Half, false>(
    const unsigned char *input_data, unsigned char *output_data, int num_elems,
    int bits, int bucket_size, cudaStream_t stream);

} // namespace gpu
} // namespace common
} // namespace cgx
