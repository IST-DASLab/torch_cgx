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

#include "mpi_allreduce_operations.h"
#include "common/common.h"
#include "common/compressor.h"
#include "common/mpi_communicator.h"
#include "common/ring.h"
#include "common/scatter_reduce_allgather.h"
#include "common/utils.h"

#if HAVE_CUDA
#include "common/nccl_reduce.h"
#include "common/shm_communicator.h"
#endif

namespace cgx {

std::vector<std::vector<unsigned>>
    MPIAllReduce_Operation::layers_sizes_;

int NumElements(std::vector<common::Layer> &layers) {
  int sum = 0;
  for (auto &layer : layers) {
    sum += layer.numel();
  }
  return sum;
}

std::shared_ptr<common::Compressor>
CreateCompressor(std::shared_ptr<common::GPUContext> gpu_context) {
  bool dummy = false;
  common::utils::SetBoolFromEnv(DEBUG_DUMMY_COMPRESSION, dummy, true);
  if (dummy)
    return std::make_shared<common::DummyCompressor>(gpu_context);
  else
    return std::make_shared<common::MaxMinQuantizer>(gpu_context);
}

std::shared_ptr<common::Reducer>
CreateReducer(std::shared_ptr<common::GPUContext> gpu_context,
              std::shared_ptr<common::Compressor> compressor,
              std::shared_ptr<common::Communicator> communicator,
              common::utils::ReductionType red_type, int world_size) {
  if (red_type == common::utils::ReductionType::SRA) {
    return std::make_shared<common::MPI_Allreduce_ScatterReduceAllgather>(
        gpu_context, compressor, communicator, world_size);
  } else {
    return std::make_shared<common::MPI_Allreduce_Ring>(
        gpu_context, compressor, communicator, world_size);
  }
}

std::shared_ptr<common::Reducer>
CreateInnerReducer(std::shared_ptr<common::GPUContext> gpu_context,
                   std::shared_ptr<common::Compressor> compressor,
                   std::shared_ptr<common::MPIContext> mpi_context) {
  auto comm_type = common::utils::GetCommTypeFromEnv(
      INNER_COMMUNICATOR_TYPE, common::utils::CommunicatorType::SHM);
  unsigned world_size = mpi_context->GetSize(mpi_context->GetLocalComm());
  std::shared_ptr<common::Communicator> communicator;
#if HAVE_CUDA
  if (comm_type == common::utils::CommunicatorType::NCCL) {
    return std::make_shared<common::NCCL_Reduce>(gpu_context, compressor,
                                                 world_size);
  }
  if (comm_type == common::utils::CommunicatorType::SHM)
    communicator.reset(new common::SHMCommunicator(gpu_context));
#endif
  if (comm_type == common::utils::CommunicatorType::MPI)
    communicator.reset(new common::MPICommunicator(gpu_context));
  if (!communicator)
    throw std::runtime_error("Communicator type is not supported");
  auto red_type = common::utils::GetRedTypeFromEnv(
      INNER_REDUCTION_TYPE, common::utils::ReductionType::SRA);
  return CreateReducer(gpu_context, compressor, communicator, red_type,
                       world_size);
}

std::shared_ptr<common::Reducer>
CreateCrossReducer(std::shared_ptr<common::GPUContext> gpu_context,
                   std::shared_ptr<common::Compressor> compressor,
                   std::shared_ptr<common::MPIContext> mpi_context) {
  auto red_type = common::utils::GetRedTypeFromEnv(
      CROSS_REDUCTION_TYPE, common::utils::ReductionType::Ring);
  unsigned world_size = mpi_context->GetSize(mpi_context->GetCrossComm());
#if HAVE_CUDA
  auto comm_type = common::utils::GetCommTypeFromEnv(
      CROSS_COMMUNICATOR_TYPE, common::utils::CommunicatorType::MPI);
  if (comm_type == common::utils::CommunicatorType::NCCL) {
    return std::make_shared<common::NCCL_Reduce>(gpu_context, compressor, world_size);
  }
#endif

  return CreateReducer(gpu_context, compressor,
                       std::make_shared<common::MPICommunicator>(gpu_context),
                       red_type,
                       world_size);
}

MPIAllReduce_Operation::MPIAllReduce_Operation() {
  gpu_context_ = std::make_shared<common::GPUContext>();
  mpi_context_ = std::make_shared<common::MPIContext>();
  gpu_context_->SetDevice(mpi_context_->GetRank(mpi_context_->GetLocalComm()));
  compressor_ = CreateCompressor(gpu_context_);
  if (mpi_context_->GetSize(mpi_context_->GetLocalComm()) > 1)
    intra_reducer_ =
        CreateInnerReducer(gpu_context_, compressor_, mpi_context_);
  if (mpi_context_->GetSize(mpi_context_->GetCrossComm()) > 1)
    cross_reducer_ =
        CreateCrossReducer(gpu_context_, compressor_, mpi_context_);
  unsigned int fusion_size_mb = common::utils::GetIntEnvOrDefault(
      FUSION_BUFFER_SIZE_MB, FUSION_SIZE_DEFAULT_MB);
  fake_compression_ratio_ =
      common::utils::GetFloatEnvOrDefault(COMPRESSION_FAKE_RATIO, 1.0);
  tensor_fusion_threshold_ =
      std::max(fusion_size_mb * 1024 * 1024, MIN_FUSION_SIZE);
  intra_broadcast_ = common::utils::GetIntEnvOrDefault(INTRA_BROADCAST, 1);
  intra_compress_ = common::utils::GetIntEnvOrDefault(INTRA_COMPRESS, 1);
  bucket_idx_ = 0;
}

int MPIAllReduce_Operation::allReduce(int num_elements, int offset,
                                      std::vector<common::Layer> &layers,
                                      at::cuda::CUDAStream& stream,
                                      bool do_compression) {
  if (num_elements > 1000 and fake_compression_ratio_ < 1.0)
    num_elements = (int)(num_elements * fake_compression_ratio_);
  auto local_comm = mpi_context_->GetLocalComm();
  int status;
  if (mpi_context_->GetSize(local_comm) > 1) {
    if (num_elements < 16) {
      status = intra_reducer_->AllReduceAlltoAll(num_elements, offset, layers,
                                                 (void *)&local_comm, stream);
    } else {
      status = intra_reducer_->AllreduceDivision(
          num_elements, offset, layers, (void *)&local_comm, stream,
          do_compression and intra_compress_);
    }
  }
  if (status < 0)
    return status;
  auto cross_comm = mpi_context_->GetCrossComm();
  if (mpi_context_->GetSize(cross_comm) > 1) {
    if (num_elements < 16) {
      status = cross_reducer_->AllReduceAlltoAll(num_elements, offset, layers,
                                                 (void *)&cross_comm, stream);
    } else {
      if (intra_broadcast_) {
        if (mpi_context_->GetRank(local_comm) == 0)
          status = cross_reducer_->AllreduceDivision(
              num_elements, offset, layers, (void *)&cross_comm, stream,
              do_compression);
        if (mpi_context_->GetSize(local_comm) > 1) {
          mpi_context_->Barrier(local_comm);
          status =
              intra_reducer_->Broadcast(num_elements, offset, layers,
                                        (void *)&local_comm, stream,
                                        do_compression);
        }
      } else {
        status = cross_reducer_->AllreduceDivision(
            num_elements, offset, layers, (void *)&cross_comm, stream,
            do_compression);
      }
    }
  }
  return status;
}

int MPIAllReduce_Operation::performOperationSingle(common::Layer &layer,
                                                   at::cuda::CUDAStream& stream,
                                                   bool do_compression) {
  int max_buffer_size = tensor_fusion_threshold_ / layer.element_size();
  int num_elements = layer.numel();
  std::vector<common::Layer> layers = {layer};
  int status;
  for (int offset = 0; offset < num_elements; offset += max_buffer_size) {
    status = allReduce(std::min(max_buffer_size, num_elements - offset), offset,
                       layers, stream, do_compression);
  }
  return status;
}

int MPIAllReduce_Operation::performOperation(std::vector<common::Layer> &layers,
                                             at::cuda::CUDAStream& stream,
                                             bool do_compression) {
  unsigned max_buffer_size = tensor_fusion_threshold_ / layers[0].element_size();
  int num_elements = NumElements(layers);
  int status;

  if (num_elements < max_buffer_size) {
    return allReduce(num_elements, 0, layers, stream, do_compression);
  }
  std::vector<common::Layer> tmp_layers;
  int cur_size = 0;
  for (auto &layer : layers) {
    if (layer.numel() > max_buffer_size) {
      status = performOperationSingle(layer, stream, do_compression);
      break;
    }
    if (cur_size + layer.numel() > max_buffer_size) {
      status = allReduce(cur_size, 0, tmp_layers, stream, do_compression);
      cur_size = 0;
      tmp_layers.clear();
    }
    tmp_layers.push_back(layer);
    cur_size += layer.numel();
  }
  return status;
}

int MPIAllReduce_Operation::PerformOperation(at::Tensor &bucket,
                                             at::cuda::CUDAStream& stream) {
  std::vector<common::Layer> layers;
  int status;
  if (bucket.numel() < MIN_LAYER_SIZE) {
    layers.emplace_back(bucket);
    status = performOperation(layers, stream, false);
    return status;
  }
  compressor_->ResetParamsFromEnv();
  extractLayers(bucket, layers);
  std::vector<common::Layer> layers_compress;
  std::vector<common::Layer> layers_nocompress;
  for (auto &layer : layers) {
    if (compressor_->isEnabled(layer))
      layers_compress.push_back(layer);
    else
      layers_nocompress.push_back(layer);
  }
  if (!layers_compress.empty()) {
    status = performOperation(layers_compress, stream, true);
  }
  if (!layers_nocompress.empty()) {
    status = performOperation(layers_nocompress, stream, false);
  }
  return status;
}

void MPIAllReduce_Operation::extractLayers(const at::Tensor &bucket,
                                           std::vector<common::Layer> &layers) {
  if (layers_sizes_.empty()) {
    layers.emplace_back(bucket);
    return;
  }
  if (bucket_idx_ > 0 and bucket_idx_ % layers_sizes_.size() == 0) {
    training_step_++;
  }
  bucket_idx_ %= layers_sizes_.size();
  const auto& sizes = layers_sizes_.at(bucket_idx_);
  unsigned cur_numel = 0;
  unsigned layer_idx = 0;
  char *ptr = static_cast<char *>(bucket.data_ptr());
  for (auto& layer_size: sizes) {
    layers.emplace_back(bucket, std::make_pair(bucket_idx_, layer_idx), ptr,
                        layer_size);
    cur_numel += layer_size;
    ptr += layer_size * bucket.element_size();
    layer_idx++;
  }
  if (cur_numel != bucket.numel()) {
    throw std::runtime_error("Error at extracting the layers from bucket. "
                             "Number of elements in bucket is not equal to "
                             "number in the layers expected to be in "
                             "the bucket.");
  }
  bucket_idx_++;
}

} // namespace cgx
