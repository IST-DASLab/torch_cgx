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
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include <mpi.h>

#include "common/gpu_context.h"
#include "common/layer.h"
#include "common/mpi_context.h"
#include "common/reducer.h"

namespace cgx {
const int MIN_LAYER_SIZE = 16;

struct MPIAllReduce_Operation {
  MPIAllReduce_Operation();
  int PerformOperation(at::Tensor &bucket, at::cuda::CUDAStream &stream);
  static void RegisterLayer(unsigned bucket_idx, unsigned layer_idx,
                            unsigned layer_numel,
                            int quantization_bits, int bucket_size) {
    assert(layers_sizes_.size() >= bucket_idx && "Registering bucket out of order is not supported");
    if (layers_sizes_.size() == bucket_idx) {
      layers_sizes_.emplace_back();
    }
    assert(layers_sizes_[bucket_idx].size() >= layer_idx && "Registering layer out of order is not supported");
    layers_sizes_[bucket_idx].push_back(layer_numel);

    SetQBits(bucket_idx, layer_idx, quantization_bits);
    SetQBucketSize(bucket_idx, layer_idx, bucket_size);
  }

  static void SetQBits(unsigned bucket_idx, unsigned layer_idx,
                       int quantization_bits) {
    common::Compressor::SetQBits(std::make_pair(bucket_idx, layer_idx), quantization_bits);
  }

  static void SetQBucketSize(unsigned bucket_idx, unsigned layer_idx,
                             int bucket_size) {
    common::Compressor::SetQBucketSize(std::make_pair(bucket_idx, layer_idx), bucket_size);
  }

protected:
  std::shared_ptr<common::GPUContext> gpu_context_;
  std::shared_ptr<common::MPIContext> mpi_context_;
  std::shared_ptr<common::Reducer> intra_reducer_;
  std::shared_ptr<common::Reducer> cross_reducer_;
  std::shared_ptr<common::Compressor> compressor_;
  int64_t tensor_fusion_threshold_;
  float fake_compression_ratio_;

private:
  void extractLayers(const at::Tensor &bucket,
                     std::vector<common::Layer> &layers);
  int performOperationSingle(common::Layer &layer, at::cuda::CUDAStream &stream,
                             bool do_compression);
  int allReduce(int num_elements, int offset,
                std::vector<common::Layer> &tensors,
                at::cuda::CUDAStream &stream, bool do_compression);
  int performOperation(std::vector<common::Layer> &tensors,
                       at::cuda::CUDAStream &stream, bool do_compression);
  static std::vector<std::vector<unsigned>> layers_sizes_;
  unsigned bucket_idx_;
  unsigned training_step_;
  bool intra_broadcast_;
  bool intra_compress_;
};
} // namespace cgx