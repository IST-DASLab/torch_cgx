#pragma once
#include <memory>
#include "gpu_context.h"
#include "compression/gpu_compression_operations.h"
#include "buffer.h"

namespace qmpi {
namespace common {
const int COMPRESSION_DEFAULT_BUCKET_SIZE = 512;
const int MIN_SIZE_TO_COMPRESS = 16;

struct CompressionModuleConfig {
  int quantization_bits;
  int bucket_size;
  bool skip_incomplete_buckets;
  bool operator==(const CompressionModuleConfig &b) {
    return quantization_bits == b.quantization_bits and
        bucket_size == b.bucket_size and
        skip_incomplete_buckets == b.skip_incomplete_buckets;
  }
};

class Compressor {
public:
  Compressor(GPUContext *gpu_context);
  virtual ~Compressor() = default;
  // Returns size of buffer to allocate for usage in compress (in bytes). We
  // assume that no compression will be done in-place.
  virtual size_t BufferSize(int num_elems, size_t element_size) = 0;
  size_t BufferSize(int num_elems, const std::vector<at::Tensor> &tensors,
                    int fusion_offset);

  size_t Compress(
      unsigned char *output,
      const std::vector<at::Tensor> &tensors,
      int fusion_offset, int chunk_num_elems,
      gpuStream_t stream);

  void Decompress(unsigned char *input_data,
                  const std::vector<at::Tensor> &entries,
                  int fusion_offset,
                  int chunk_num_elems,
                  bool add,
                  gpuStream_t stream);

  // Returns size of compressed size (in bytes). And update error_feedback.
  // If error_feedback is nullptr, it's not updated.
  virtual size_t
  CompressBuffer(unsigned char *input_data,
                 unsigned char *output,
                 unsigned char *feedback_data,
                 int num_elems,
                 at::ScalarType dtype,
                 gpuStream_t stream) = 0;
  // Decompress data from input to output.
  // If add is True sum decompressed data with output.
  virtual void DecompressBuffer(unsigned char *input,
                                unsigned char *output,
                                int num_elems,
                                at::ScalarType dtype,
                                bool add,
                                gpuStream_t stream) = 0;
  static void GetSizesAndOffsets(int num_elements, int world_size,
                                  const std::vector<at::Tensor> &entries,
                                  std::vector<int> &offsets,
                                  std::vector<int> &sizes);
  void Add(int num_elements,
           unsigned char *x,
           unsigned char *y,
           unsigned char *sum,
           at::ScalarType dtype,
           gpuStream_t stream);
  void Float2Half(unsigned char* input, unsigned char* output, int num_elements, gpuStream_t stream);
  void Half2Float(unsigned char* input, unsigned char* output, int num_elements, gpuStream_t stream);

  virtual bool isEnabled(const at::Tensor& tensor) = 0;
  virtual void ResetParamsFromEnv();
protected:
  GPUContext *gpu_context_;
  CompressionModuleConfig default_config;
  size_t tensor_fusion_size_;
  int min_elems_to_compress_;
};

class Quantizer : public Compressor {
public:
  Quantizer(GPUContext *gpu_context);
  static void GetSizesAndOffsets(int num_elements, int world_size,
                          const std::vector<at::Tensor> &tensors,
                          std::vector<int> &offsets,
                          std::vector<int> &sizes);
  virtual void ResetParamsFromEnv() override;
protected:
  gpu::RandState *rand_states_;
  std::unique_ptr<PersistentBuffer> cuda_states_buffer_;
};

class DummyCompressor : public Compressor {
public:
  DummyCompressor(GPUContext *gpu_context)
      : Compressor(gpu_context) {}

  size_t CompressBuffer(unsigned char *input,
                        unsigned char *output,
                        unsigned char *feedback,
                        int num_elems,
                        at::ScalarType dtype,
                        gpuStream_t stream) override;
  void DecompressBuffer(unsigned char *input,
                        unsigned char *output,
                        int num_elems,
                        at::ScalarType dtype,
                        bool add,
                        gpuStream_t stream) override;
  size_t BufferSize(int num_elems, size_t element_size) final;
  bool isEnabled(const at::Tensor& tensor) override;
};

class MaxMinQuantizer : public Quantizer {
public:
  MaxMinQuantizer(GPUContext *gpu_context)
      : Quantizer(gpu_context) {}

  size_t CompressBuffer(unsigned char *input,
                        unsigned char *output,
                        unsigned char *feedback,
                        int num_elems,
                        at::ScalarType dtype,
                        gpuStream_t stream) override;
  void DecompressBuffer(unsigned char *input,
                        unsigned char *output,
                        int num_elems,
                        at::ScalarType dtype,
                        bool add,
                        gpuStream_t stream) override;
  size_t BufferSize(int num_elems, size_t element_size) final;
  bool isEnabled(const at::Tensor& tensor) override;
};

} // namespace common
} // namespace qmpi
