#pragma once

#if HAVE_CUDA
#include <cuda_fp16.h>
#elif HAVE_ROCM
#include <hip/hip_fp16.h>
#endif

namespace cgx {
namespace common {
namespace gpu {


__device__ __half hmax(__half a, __half b) { return __hge(a, b) ? a : b; }

__device__ __half hmin(__half a, __half b) { return __hge(a, b) ? b : a; }

__device__ __half habs(__half a) {
  return hmax(a, __hneg(a));
}

template <typename T>
__device__ inline T max(T a, T b) {
  return fmaxf(a, b);
}

template <>
__device__ inline __half max(__half a, __half b) {
  return hmax(a, b);
}

template <typename T>
__device__ inline T min(T a, T b) {
  return fminf(a, b);
}

template <>
__device__ inline __half min(__half a, __half b) {
  return hmin(a, b);
}

template <typename T>
__device__ inline T sum(T a, T b) {
  return a + b;
}

template <>
__device__ inline __half sum(__half a, __half b) {
  return __hadd(a, b);
}

template <typename T>
__device__ inline T sub(T a, T b) {
  return a - b;
}

template <>
__device__ inline __half sub(__half a, __half b) {
  return __hsub(a, b);
}

template <typename T>
__device__ inline T mul(T a, T b) {
  return a * b;
}

template <>
__device__ inline __half mul(__half a, __half b) {
  return __hmul(a, b);
}

template <typename T>
__device__ inline T div(T a, T b) {
  return a / b;
}

template <>
__device__ inline __half div(__half a, __half b) {
  return __hdiv(a, b);
}

template <typename T>
__device__ inline T mul_int(T a, int b) {
  return a * b;
}

template <>
__device__ inline __half mul_int(__half a, int b) {
  return __hmul(a, __uint2half_rd(b));
}

template <typename T>
__device__ inline T div_int(T a, unsigned int b) {
  return a / b;
}

template <>
__device__ inline __half div_int(__half a, unsigned int b) {
  return __hdiv(a, __uint2half_rd(b));
}

template <typename T>
__device__ inline T add_float(T a, float b) {
  return a + b;
}

template <>
__device__ inline __half add_float(__half a, float b) {
  return __hadd(a, __float2half(b));
}

template <typename T>
__device__ inline T mul_float(T a, float b) {
  return a * b;
}

template <>
__device__ inline __half mul_float(__half a, float b) {
  return __hmul(a, __float2half(b));
}

template <typename T>
__device__ inline T abs(T a) {
  return fabsf(a);
}

template <>
__device__ inline __half abs(__half a) {
  return habs(a);
}

template <typename T>
__device__ inline T sqrt(T a) {
  return ::sqrt(a);
}

template <>
__device__ inline __half sqrt(__half a) {
  return hsqrt(a);
}

template <typename T>
__device__ inline unsigned char floor(T a) {
  return ::floor(a);
}

template <>
__device__ inline unsigned char floor(__half a) {
  return __half2uint_rd(hfloor(a));
}

template <typename T>
__device__ inline bool lt(T a, T b) {
  return a < b;
}

template <>
__device__ inline bool lt(__half a, __half b) {
  return __hlt(a, b);
}

template <typename T>
__device__ inline bool le(T a, T b) {
  return a <= b;
}

template <>
__device__ inline bool le(__half a, __half b) {
  return __hle(a, b);
}

template <typename T>
__device__ inline T float2type(float a) {
  return (T) a;
}

template <>
__device__ inline __half float2type(float a) {
  return __float2half(a);
}

template <typename T>
__device__ inline float type2float(T a) {
  return (float) a;
}

template <>
__device__ inline float type2float(__half a) {
  return __half2float(a);
}

template <typename T>
__device__ inline bool isnan(T a) {
  return ::isnan(a);
}

template <>
__device__ inline bool isnan(__half a) {
  return __hisnan(a);
}

__global__ void float2half(float* input, __half* output, int numel) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < numel; i += stride) {
    output[i] = __float2half(input[i]);
  }
}

} // namespace cuda
} // namespace common
} // namespace cgx
