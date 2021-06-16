#include "compressor.h"
#include "common.h"
#include "utils.h"

namespace qmpi {
namespace common {

Compressor::Compressor(GPUContext *gpu_context) : gpu_context_(gpu_context) {
  unsigned int fusion_size_mb =
      utils::GetIntEnvOrDefault(FUSION_BUFFER_SIZE_MB, FUSION_SIZE_DEFAULT_MB);
  tensor_fusion_size_ =
      std::max(fusion_size_mb * 1024 * 1024, MIN_FUSION_SIZE);
}

void Compressor::ResetParamsFromEnv() {
  default_config.bucket_size = utils::GetIntEnvOrDefault(
      COMPRESSION_BUCKET_SIZE, COMPRESSION_DEFAULT_BUCKET_SIZE);
  default_config.skip_incomplete_buckets = false;
  utils::SetBoolFromEnv(COMPRESSION_SKIP_INCOMPLETE_BUCKETS,
                        default_config.skip_incomplete_buckets, true);
  min_elems_to_compress_ =
      std::max(utils::GetIntEnvOrDefault(COMPRESSION_MINIMAL_SIZE, 0),
               MIN_SIZE_TO_COMPRESS);
}

size_t Compressor::BufferSize(
    int chunk_num_elems,
    const std::vector<at::Tensor> &tensors,
    int fusion_offset) {
  int offset_cumm = 0;
  int nelem = 0;
  size_t sum_result = 0;
  for (auto &tensor : tensors) {
    nelem = tensor.numel();
    if (offset_cumm + nelem <= fusion_offset) {
      offset_cumm += nelem;
      continue;
    }

    if (offset_cumm - fusion_offset >= chunk_num_elems) {
      break;
    }

    if (offset_cumm < fusion_offset) {
      // If the first part of param group is placed in previous slice
      // depending on reduction algorithm.
      nelem = offset_cumm + nelem - fusion_offset;
    }

    if (std::max(offset_cumm, fusion_offset) + nelem >
        fusion_offset + chunk_num_elems) {
      // if layer doesn't fit the rest of slice.
      nelem = fusion_offset + chunk_num_elems -
          std::max(offset_cumm, fusion_offset);
    }
    sum_result += BufferSize(nelem, tensor.element_size());
    offset_cumm += tensor.numel();
  }
  return sum_result;
}

size_t Compressor::Compress(
    unsigned char *output,
    const std::vector<at::Tensor> &tensors,
    int fusion_offset, int chunk_num_elems,
    gpuStream_t stream) {
  size_t total_compressed_size = 0;

  int offset_cumm = 0;
  int nelem = 0;
  int buffer_offset = 0;
  size_t compressed_size;
  for (auto &tensor : tensors) {
    nelem = tensor.numel();
    if (offset_cumm + nelem <= fusion_offset) {
      offset_cumm += nelem;
      continue;
    }

    if (offset_cumm - fusion_offset >= chunk_num_elems) {
      break;
    }
    buffer_offset = 0;
    if (offset_cumm < fusion_offset) {
      // If the first part of the entry is placed in the previous slice.
      nelem = offset_cumm + nelem - fusion_offset;
      buffer_offset = tensor.numel() - nelem;
    }

    if (std::max(offset_cumm, fusion_offset) + nelem >
        fusion_offset + chunk_num_elems) {
      // if entry doesn't fit the rest of slice.
      nelem = fusion_offset + chunk_num_elems -
          std::max(offset_cumm, fusion_offset);
    }
    auto offset = buffer_offset * tensor.element_size();
    auto tensor_data = ((unsigned char *) tensor.data_ptr()) + offset;

    compressed_size =
        CompressBuffer(tensor_data, output, nullptr, nelem,
                       tensor.scalar_type(), stream);
    offset_cumm += tensor.numel();
    output += compressed_size;
    total_compressed_size += compressed_size;
  }
  return total_compressed_size;
}

void Compressor::Decompress(
    unsigned char *input_data,
    const std::vector<at::Tensor> &tensors,
    int fusion_offset, int chunk_num_elems, bool add, gpuStream_t stream) {
  int offset_cumm = 0;
  int nelem = 0;
  int buffer_offset = 0;
  size_t cumm_decompressed = 0;

  for (auto &tensor : tensors) {
    nelem = tensor.numel();
    if (offset_cumm + nelem <= fusion_offset) {
      offset_cumm += nelem;
      continue;
    }
    if (offset_cumm - fusion_offset >= chunk_num_elems)
      break;
    buffer_offset = 0;
    if (offset_cumm < fusion_offset) {
      // If the first part of param group is placed in previous slice
      // depending on reduction algorithm.
      nelem = offset_cumm + nelem - fusion_offset;
      buffer_offset = tensor.numel() - nelem;
    }
    if (std::max(offset_cumm, fusion_offset) + nelem >
        fusion_offset + chunk_num_elems) {
      // if layer doesn't fit the rest of slice.
      nelem = fusion_offset + chunk_num_elems -
          std::max(offset_cumm, fusion_offset);
    }
    auto output = ((unsigned char *) tensor.data_ptr()) +
        buffer_offset * tensor.element_size();
    DecompressBuffer(input_data + cumm_decompressed, output, nelem,
                     tensor.scalar_type(), add, stream);
    cumm_decompressed += BufferSize(nelem, tensor.element_size());
    offset_cumm += tensor.numel();
  }
}

void Compressor::GetSizesAndOffsets(
    int num_elements, int world_size,
    const std::vector<at::Tensor> &tensors, std::vector<int> &offsets,
    std::vector<int> &sizes) {
  int residue = num_elements % world_size;
  int num_elems_per_node = num_elements / world_size;
  int offset = 0;
  for (int rank = 0; rank < world_size; rank++) {
    sizes.push_back(num_elems_per_node + ((rank < residue) ? 1 : 0));
    offsets.push_back(offset);
    offset += sizes.back();
  }
}

void Compressor::Add(int num_elements,
                     unsigned char *x,
                     unsigned char *y,
                     unsigned char *sum,
                     at::ScalarType dtype,
                     gpuStream_t stream) {
  if (dtype == at::kHalf) {
    gpu::add<gpu::Half>(num_elements,
                        reinterpret_cast<gpu::Half *>(x),
                        reinterpret_cast<gpu::Half *>(y),
                        reinterpret_cast<gpu::Half *>(sum),
                        stream);
  } else {
    gpu::add<float>(num_elements,
                    reinterpret_cast<float *>(x),
                    reinterpret_cast<float *>(y),
                    reinterpret_cast<float *>(sum),
                    stream);
  }
}

void Compressor::Float2Half(unsigned char *input,
                            unsigned char *output,
                            int num_elements,
                            gpuStream_t stream) {
  gpu::float2half(reinterpret_cast<float *>(input),
                  reinterpret_cast<gpu::Half *>(output),
                  num_elements,
                  stream);
}

void Compressor::Half2Float(unsigned char *input,
                            unsigned char *output,
                            int num_elements,
                            gpuStream_t stream) {
  gpu::half2float(reinterpret_cast<gpu::Half *>(input),
                  reinterpret_cast<float *>(output),
                  num_elements,
                  stream);
}

size_t DummyCompressor::CompressBuffer(unsigned char *input,
                                       unsigned char *output,
                                       unsigned char *feedback,
                                       int num_elems,
                                       at::ScalarType dtype,
                                       gpuStream_t stream) {
  gpu_context_->MemcpyAsyncD2D(output,
                               input,
                               num_elems * utils::get_sizeof(dtype),
                               stream);
  return num_elems * utils::get_sizeof(dtype);
}

void DummyCompressor::DecompressBuffer(unsigned char *input,
                                       unsigned char *output,
                                       int num_elems,
                                       at::ScalarType dtype,
                                       bool add,
                                       gpuStream_t stream) {
  if (add) {
    Compressor::Add(num_elems, output, input, output, dtype, stream);
  } else {
    gpu_context_->MemcpyAsyncD2D(output, input,
                                 num_elems * utils::get_sizeof(dtype), stream);
  }
}

size_t DummyCompressor::BufferSize(int num_elems, size_t element_size) {
  return num_elems * element_size;
}

bool DummyCompressor::isEnabled(const at::Tensor &tensor) {
//  return tensor.numel() > min_elems_to_compress_;
  return false;
}

Quantizer::Quantizer(GPUContext *gpu_context)
    : Compressor(gpu_context) {
  size_t randstates_sizes = gpu::get_curand_array_size(tensor_fusion_size_);
  cuda_states_buffer_ = std::make_unique<PersistentBuffer>(randstates_sizes);
  rand_states_ =
      static_cast<gpu::RandState *>(cuda_states_buffer_->RawPointer());
}

void Quantizer::ResetParamsFromEnv() {
  Compressor::ResetParamsFromEnv();
  auto quantization_bits =
      common::utils::GetIntEnvOrDefault(COMPRESSION_QUANTIZATION_BITS, 32);
  default_config.quantization_bits = quantization_bits;
}

void Quantizer::GetSizesAndOffsets(int num_elements, int world_size,
                                   const std::vector<at::Tensor> &tensors,
                                   std::vector<int> &offsets,
                                   std::vector<int> &sizes) {
  int offset = 0;
  int num_per_node;
  auto it = tensors.begin();
  int entry_offset = 0;
  int n_elem = std::min((int) it->numel(), num_elements);
  int cur_size = 0;
  int align_unit = (tensors[0].scalar_type() == at::kHalf) ? 8 : 4;
  for (int rank = 0; rank < world_size; rank++) {
    num_per_node = num_elements / (world_size - rank);
    cur_size = 0;
    while (cur_size < num_per_node) {
      if (n_elem <= num_per_node - cur_size) {
        cur_size += n_elem;
        it++;
        if (it == tensors.end())
          break;
        n_elem = std::min((int) it->numel(), num_elements);
      } else {
        int aligned =
            std::min((int) utils::round_to(num_per_node - cur_size, align_unit),
                     n_elem);
        cur_size += aligned;
        n_elem -= aligned;
      }
    }
    num_elements -= cur_size;
    sizes.push_back(cur_size);
    offsets.push_back(offset);
    offset += cur_size;
  }
}

size_t MaxMinQuantizer::CompressBuffer(
    unsigned char *input, unsigned char *output, unsigned char *feedback,
    int num_elems, at::ScalarType dtype,
    gpuStream_t stream) {
  if (num_elems == 0)
    return 0;
  const int bits = default_config.quantization_bits;
  const int bucket_size = default_config.bucket_size;
  const bool skip_incomplete = default_config.skip_incomplete_buckets;
  int num_elems_to_compress = num_elems;
  int residual_elems = 0;
  int compressed_size = 0;
  if (skip_incomplete) {
    num_elems_to_compress = (num_elems / bucket_size) * bucket_size;
    residual_elems = num_elems % bucket_size;
  }
  if (num_elems_to_compress > 0) {
    if (dtype != at::kHalf) {
      gpu::quantize_maxmin<float>(
          input, output, feedback, num_elems_to_compress, bits, bucket_size,
          rand_states_, stream);
    } else {
      gpu::quantize_maxmin<gpu::Half>(
          input, output, feedback, num_elems_to_compress, bits, bucket_size,
          rand_states_, stream);
    }
    compressed_size =
        BufferSize(num_elems_to_compress, utils::get_sizeof(dtype));
  }
  if (skip_incomplete and residual_elems > 0) {
    input += num_elems_to_compress * utils::get_sizeof(dtype);
    output += compressed_size;
    gpu_context_->MemcpyAsyncD2D(
        (void *) output, (void *) input,
        residual_elems * utils::get_sizeof(dtype), stream);
    compressed_size += utils::get_sizeof(dtype) * residual_elems;
  }
  return compressed_size;
}

void MaxMinQuantizer::DecompressBuffer(
    unsigned char *input, unsigned char *output, int num_elems,
    at::ScalarType dtype, bool add, gpuStream_t stream) {
  if (num_elems == 0)
    return;
  const int bits = default_config.quantization_bits;
  const int bucket_size = default_config.bucket_size;
  const bool skip_incomplete = default_config.skip_incomplete_buckets;
  int num_elems_to_decompress = num_elems;
  int residual_elems = 0;
  if (skip_incomplete) {
    num_elems_to_decompress = (num_elems / bucket_size) * bucket_size;
    residual_elems = num_elems % bucket_size;
  }
  if (num_elems_to_decompress > 0) {
    if (add) {
      if (dtype != at::kHalf) {
        gpu::dequantize_maxmin<float, true>(
            input, output, num_elems_to_decompress, bits, bucket_size, stream);
      } else {
        gpu::dequantize_maxmin<gpu::Half, true>(
            input, output, num_elems_to_decompress, bits, bucket_size, stream);
      }
    } else {
      if (dtype != at::kHalf) {
        gpu::dequantize_maxmin<float, false>(
            input, output, num_elems_to_decompress, bits, bucket_size, stream);
      } else {
        gpu::dequantize_maxmin<gpu::Half, false>(
            input, output, num_elems_to_decompress, bits, bucket_size, stream);
      }
    }
  }

  if (skip_incomplete and residual_elems > 0) {
    int compressed_size =
        BufferSize(num_elems_to_decompress, utils::get_sizeof(dtype));
    input += compressed_size;
    output += num_elems_to_decompress * utils::get_sizeof(dtype);
    if (add) {
      if (dtype != at::kHalf)
        gpu::add<float>(residual_elems, (float *) input, (float *) output,
                        (float *) output, stream);
      else
        gpu::add<gpu::Half>(residual_elems,
                            (gpu::Half *) input,
                            (gpu::Half *) output,
                            (gpu::Half *) output,
                            stream);
    } else {
      gpu_context_->MemcpyAsyncD2D(
          (void *) output,
          (void *) input,
          residual_elems * utils::get_sizeof(dtype),
          stream);
    }
  }
}

size_t
MaxMinQuantizer::BufferSize(int num_elems, size_t element_size) {
  if (num_elems == 0)
    return 0;
  const int bits = default_config.quantization_bits;
  const int bucket_size = default_config.bucket_size;
  const bool skip_incomplete = default_config.skip_incomplete_buckets;
  int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  int residuals = 0;
  if (skip_incomplete) {
    num_buckets = num_elems / bucket_size;
    residuals = num_elems % bucket_size;
    num_elems = num_buckets * bucket_size;
  }
  size_t meta_buffer_size = 2 * num_buckets * element_size;
  size_t compressed_values_buffer_size = (num_elems * bits + 7) / 8;
  return meta_buffer_size + utils::aligned_size(compressed_values_buffer_size) +
      residuals * element_size;
}

bool MaxMinQuantizer::isEnabled(const at::Tensor &tensor) {
  return tensor.numel() > min_elems_to_compress_
      and default_config.quantization_bits <= 8;
}

} // namespace common
} // namespace qmpi