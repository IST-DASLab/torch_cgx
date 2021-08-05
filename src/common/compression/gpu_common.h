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