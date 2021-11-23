#pragma once
namespace cgx {
namespace common {
namespace gpu {
using uint64_t = unsigned long long int;

struct xorshift128p_state {
  uint64_t a, b;
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

} // namespace gpu
} // namespace common
} // namespace cgx