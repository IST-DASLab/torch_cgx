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
#include <memory>
#include <set>
#include <unordered_map>

#include "buffer.h"
#include "compression/gpu_compression_operations.h"
#include "gpu_context.h"
#include "layer.h"

namespace cgx::common {
const int COMPRESSION_DEFAULT_BUCKET_SIZE = 512;

struct CompressionLayerConfig {
  int quantization_bits;
  int bucket_size;
  bool skip_incomplete_buckets;
  bool operator==(const CompressionLayerConfig &b) {
    return quantization_bits == b.quantization_bits and
           bucket_size == b.bucket_size and
           skip_incomplete_buckets == b.skip_incomplete_buckets;
  }
};

class Compressor {
public:
  Compressor(std::shared_ptr<GPUContext> gpu_context);
  virtual ~Compressor() = default;
  // Returns size of buffer to allocate for usage in compress (in bytes). We
  // assume that no compression will be done in-place.
  virtual size_t BufferSize(int num_elems, size_t element_size,
                            const CompressionLayerConfig &config) = 0;
  size_t BufferSize(int num_elems, const std::vector<Layer> &layers,
                    int fusion_offset);

  size_t Compress(unsigned char *output, const std::vector<Layer> &tensors,
                  int fusion_offset, int chunk_num_elems, gpuStream_t stream);

  void Decompress(unsigned char *input_data, const std::vector<Layer> &entries,
                  int fusion_offset, int chunk_num_elems, bool add,
                  gpuStream_t stream);

  // Returns size of compressed size (in bytes). And update error_feedback.
  // If error_feedback is nullptr, it's not updated.
  virtual size_t CompressBuffer(unsigned char *input_data,
                                unsigned char *output,
                                unsigned char *feedback_data, int num_elems,
                                at::ScalarType dtype,
                                const CompressionLayerConfig &config,
                                gpuStream_t stream) = 0;
  // Decompress data from input to output.
  // If add is True sum decompressed data with output.
  virtual void DecompressBuffer(unsigned char *input, unsigned char *output,
                                int num_elems, at::ScalarType dtype, bool add,
                                const CompressionLayerConfig &config,
                                gpuStream_t stream) = 0;
  static void GetSizesAndOffsets(int num_elements, int world_size,
                                 int global_offset,
                                 const std::vector<Layer> &entries,
                                 std::vector<int> &offsets,
                                 std::vector<int> &sizes);
  static void Add(int num_elements, unsigned char *x, unsigned char *y,
                  unsigned char *sum, at::ScalarType dtype, gpuStream_t stream);
  void Float2Half(unsigned char *input, unsigned char *output, int num_elements,
                  gpuStream_t stream);
  void Half2Float(unsigned char *input, unsigned char *output, int num_elements,
                  gpuStream_t stream);

  virtual bool isEnabled(const Layer &tensor) = 0;
  virtual void ResetParamsFromEnv();
  virtual void Init(int elem_size, gpuStream_t stream) {}
  CompressionLayerConfig &GetLayerConfig(const LayerId &name);
  static void SetQBits(const LayerId &layer_id, int bits) {
    auto &config = layers_configs[layer_id];
    config.quantization_bits = bits;
    config.bucket_size = (config.bucket_size > 0) ? config.bucket_size
                                                  : default_config.bucket_size;
  }

  static void SetQBucketSize(const LayerId &layer_id, int bucket_size) {
    auto &config = layers_configs[layer_id];
    config.quantization_bits = (config.quantization_bits > 0)
                                   ? config.quantization_bits
                                   : default_config.quantization_bits;
    ;
    config.bucket_size = bucket_size;
  }

protected:
  struct hash_laierid {
    size_t operator()(const LayerId &id) const {
      auto hash1 = std::hash<unsigned>{}(id.first);
      auto hash2 = std::hash<unsigned>{}(id.second);

      if (hash1 != hash2) {
        return hash1 ^ hash2;
      }
      return hash1;
    }
  };

  std::shared_ptr<GPUContext> gpu_context_;
  static CompressionLayerConfig default_config;
  static std::unordered_map<LayerId, CompressionLayerConfig, hash_laierid>
      layers_configs;
  size_t tensor_fusion_size_;
  int min_elems_to_compress_;
};

class Quantizer : public Compressor {
public:
  explicit Quantizer(std::shared_ptr<GPUContext> gpu_context);
  static void GetSizesAndOffsets(int num_elements, int world_size,
                                 int global_offset,
                                 const std::vector<Layer> &tensors,
                                 std::vector<int> &offsets,
                                 std::vector<int> &sizes);
  void ResetParamsFromEnv() override;

protected:
  gpu::RandState *rand_states_;
  std::unique_ptr<PersistentBuffer> aux_buffer_;
};

class DummyCompressor : public Compressor {
public:
  DummyCompressor(std::shared_ptr<GPUContext> gpu_context)
      : Compressor(gpu_context) {}

  size_t CompressBuffer(unsigned char *input, unsigned char *output,
                        unsigned char *feedback, int num_elems,
                        at::ScalarType dtype,
                        const CompressionLayerConfig &config,
                        gpuStream_t stream) override;
  void DecompressBuffer(unsigned char *input, unsigned char *output,
                        int num_elems, at::ScalarType dtype, bool add,
                        const CompressionLayerConfig &config,
                        gpuStream_t stream) override;
  size_t BufferSize(int num_elems, size_t element_size,
                    const CompressionLayerConfig &config) final;
  bool isEnabled(const Layer &tensor) override;
};

class MaxMinQuantizer : public Quantizer {
public:
  MaxMinQuantizer(std::shared_ptr<GPUContext> gpu_context)
      : Quantizer(gpu_context) {}

  size_t CompressBuffer(unsigned char *input, unsigned char *output,
                        unsigned char *feedback, int num_elems,
                        at::ScalarType dtype,
                        const CompressionLayerConfig &config,
                        gpuStream_t stream) override;
  void DecompressBuffer(unsigned char *input, unsigned char *output,
                        int num_elems, at::ScalarType dtype, bool add,
                        const CompressionLayerConfig &config,
                        gpuStream_t stream) override;
  size_t BufferSize(int num_elems, size_t element_size,
                    const CompressionLayerConfig &config) final;
  bool isEnabled(const Layer &tensor) override;
  virtual void Init(int elem_size, gpuStream_t stream);
};

} // namespace cgx::common
