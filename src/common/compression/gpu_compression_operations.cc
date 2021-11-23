#include "gpu_compression_operations.h"

#if HAVE_CUDA
#include "cuda_compression_operations.h"
#elif HAVE_ROCM
#include "hip_compression_operations.h"
#endif
namespace cgx {
namespace common {
namespace gpu {

size_t get_curand_array_size(int num_elems) {
  return BLOCKS_PER_GRID(num_elems, THREADS_PER_BLOCK_COMPRESS)
      * THREADS_PER_BLOCK_COMPRESS *
      sizeof(RandState);
}

template<typename T>
void quantize_maxmin(unsigned char *input_data, unsigned char *output_data,
                     unsigned char *feedback_data, unsigned char *util_buf,
                     int num_elems, int bits,
                     int bucket_size, RandState *states,
                     gpuStream_t stream) {
#if HAVE_CUDA
  CUDA_quantize_maxmin<T>(input_data, output_data, feedback_data, util_buf,
                          num_elems, bits, bucket_size, states, stream);
#elif HAVE_ROCM
  HIP_quantize_maxmin<T>(input_data, output_data, feedback_data, num_elems, bits,
                          bucket_size, states, stream);
#endif
}

template<typename T, bool ADD>
void dequantize_maxmin(unsigned char *input_data,
                       unsigned char *output_data, unsigned char* util_buf,
                       int num_elems, int bits,
                       int bucket_size, gpuStream_t stream) {
#if HAVE_CUDA
  CUDA_dequantize_maxmin<T, ADD>(input_data, output_data, util_buf, num_elems,
                                 bits, bucket_size, stream);
#elif HAVE_ROCM
  HIP_dequantize_maxmin<T, ADD>(input_data, output_data, num_elems, bits,
                                 bucket_size, stream);
#endif
}

template<typename T>
void add(int n, const T *x, T *y, T *sum, gpuStream_t stream) {
#if HAVE_CUDA
  CUDA_add(n, x, y, sum, stream);
#elif HAVE_ROCM
  HIP_add(n, x, y, sum, stream);
#endif
}

void init_rand_states(RandState *states, int num_elems, unsigned int seed,
                      gpuStream_t stream) {
#if HAVE_CUDA
  CUDA_init_rand(states, num_elems, seed, stream);
#elif HAVE_ROCM
  HIP_init_rand(states, num_elems, seed, stream);
#endif
}

void half2float(Half *input, float *output, int numel, gpuStream_t stream) {
#if HAVE_CUDA
  CUDA_half2float(input, output, numel, stream);
#elif HAVE_ROCM
#endif
}

void float2half(float *input, Half *output, int numel, gpuStream_t stream) {
#if HAVE_CUDA
  CUDA_float2half(input, output, numel, stream);
#elif HAVE_ROCM
#endif
}

/* Template declarations */
template void quantize_maxmin<float>(unsigned char *input_data,
                                     unsigned char *output_data,
                                     unsigned char *feedback_data,
                                     unsigned char* util_buf,
                                     int num_elems,
                                     int bits,
                                     int bucket_size,
                                     RandState *states,
                                     gpuStream_t stream);

template void quantize_maxmin<Half>(unsigned char *input_data,
                                    unsigned char *output_data,
                                    unsigned char *feedback_data,
                                    unsigned char* util_buf,
                                    int num_elems,
                                    int bits,
                                    int bucket_size,
                                    RandState *states,
                                    gpuStream_t stream);

template
void dequantize_maxmin<float, true>(unsigned char *input_data,
                                    unsigned char *output_data,
                                    unsigned char* util_buf,
                                    int num_elems,
                                    int bits,
                                    int bucket_size,
                                    gpuStream_t stream);

template
void dequantize_maxmin<float, false>(unsigned char *input_data,
                                     unsigned char *output_data,
                                     unsigned char* util_buf,
                                     int num_elems,
                                     int bits,
                                     int bucket_size,
                                     gpuStream_t stream);

template
void dequantize_maxmin<Half, true>(unsigned char *input_data,
                                   unsigned char *output_data,
                                   unsigned char* util_buf,
                                   int num_elems,
                                   int bits,
                                   int bucket_size,
                                   gpuStream_t stream);

template
void dequantize_maxmin<Half, false>(unsigned char *input_data,
                                    unsigned char *output_data,
                                    unsigned char* util_buf,
                                    int num_elems,
                                    int bits,
                                    int bucket_size,
                                    gpuStream_t stream);

template
void add<float>(int n,
                const float *x,
                float *y,
                float *sum,
                gpuStream_t stream);

template
void add<Half>(int n, const Half *x, Half *y, Half *sum, gpuStream_t stream);

} // namespace gpu
} // namespace common
} // namespace cgx
