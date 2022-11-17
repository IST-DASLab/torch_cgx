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

#include "nccl_reduce.h"
#include <map>

namespace cgx {
namespace common {

std::map<at::ScalarType, ncclDataType_t> ncclDatatype = {
    {at::kByte, ncclInt8},   {at::kChar, ncclChar}, {at::kDouble, ncclFloat64},
    {at::kFloat, ncclFloat}, {at::kInt, ncclInt},   {at::kLong, ncclInt64},
    {at::kShort, ncclUint8}, {at::kHalf, ncclHalf}};

NCCL_Reduce::NCCL_Reduce(std::shared_ptr<GPUContext> gpu_context,
                         std::shared_ptr<Compressor> compressor, int world_size)
    : Reducer(gpu_context, compressor) {
  nccl_comm_ = nullptr;
  int64_t chunk_size = tensor_fusion_size_;
  chunk_size = utils::aligned_size((chunk_size + world_size - 1) / world_size);
  int64_t buffer_size = chunk_size * world_size + chunk_size * (world_size - 1);
  buffer_ = std::make_unique<PersistentBuffer>(buffer_size);
  void *buffer_data = buffer_->RawPointer();
  gradients_send_ = static_cast<unsigned char *>(buffer_data);
  gradients_recv_ = gradients_send_ + chunk_size * world_size;
}

void NCCL_Reduce::ErrorCheck(std::string op_name, ncclResult_t nccl_result) {
  if (nccl_result != ncclSuccess) {
    ncclCommAbort(nccl_comm_);
    throw std::runtime_error(std::string(op_name) +
                             " failed: " + ncclGetErrorString(nccl_result));
  }
}

void NCCL_Reduce::Init(void *comm_p) {
  if (nccl_comm_ != nullptr)
    return;
  MPI_Comm comm = *(static_cast<MPI_Comm *>(comm_p));
  int world_size, rank;
  MPI_CHECK(MPI_Comm_size(comm, &world_size));
  MPI_CHECK(MPI_Comm_rank(comm, &rank));
  ncclUniqueId nccl_id;
  if (rank == 0) {
    ErrorCheck("ncclGetUniqueId", ncclGetUniqueId(&nccl_id));
  }
  MPI_CHECK(MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, comm));
  auto nccl_result = ncclCommInitRank(&nccl_comm_, world_size, nccl_id, rank);
  ErrorCheck("ncclCommInitRank", nccl_result);
  MPI_CHECK(MPI_Barrier(comm));
}

int NCCL_Reduce::AllreduceDivision(int num_elements, int global_offset,
                                   std::vector<Layer> &layers, void *comm_p,
                                   gpuStream_t gpu_stream,
                                   bool do_compression) {
  if (do_compression) {
    return AllreduceCompressed(num_elements, global_offset, layers, comm_p,
                               gpu_stream);
  } else {
    return AllreduceUncompressed(num_elements, global_offset, layers, comm_p,
                                 gpu_stream);
  }
}

int NCCL_Reduce::AllReduceAlltoAll(int num_elements, int global_offset,
                                   std::vector<Layer> &layers, void *comm,
                                   gpuStream_t gpu_stream) {
  return AllreduceUncompressed(num_elements, global_offset, layers, comm,
                               gpu_stream);
}

int NCCL_Reduce::AllreduceUncompressed(int num_elements, int global_offset,
                                       std::vector<Layer> &layers, void *comm_p,
                                       gpuStream_t gpu_stream) {
  Init(comm_p);
  unsigned char *layers_data = gradients_send_;
  FuseLayerData(&layers_data, layers, num_elements, global_offset, gpu_stream);
  auto nccl_result = ncclAllReduce(layers_data, layers_data, num_elements,
                                   ncclDatatype.at(layers[0].scalar_type()),
                                   ncclSum, nccl_comm_, gpu_stream);
  ErrorCheck("ncclAllReduce", nccl_result);
  UnfuseLayerData(layers_data, layers, num_elements, gpu_stream);
  return 0;
}

int NCCL_Reduce::AllreduceCompressed(int num_elements, int global_offset,
                                     std::vector<Layer> &layers, void *comm_p,
                                     gpuStream_t gpu_stream) {
  Init(comm_p);
  MPI_Comm mpi_comm = *(static_cast<MPI_Comm *>(comm_p));
  int world_size, rank;
  MPI_CHECK(MPI_Comm_size(mpi_comm, &world_size));
  MPI_CHECK(MPI_Comm_rank(mpi_comm, &rank));
  std::vector<int> chunk_sizes, offsets;
  Quantizer::GetSizesAndOffsets(num_elements, world_size, global_offset, layers,
                                offsets, chunk_sizes);
  compressor_->Init(layers[0].element_size(), gpu_stream);
  unsigned char *send_buf = gradients_send_;
  unsigned char *recv_buf = gradients_recv_;
  std::vector<int> nodes;
  std::vector<int> compressed_sizes(world_size, 0);
  int send_num_elems = 0;
  int send_compressed_size = 0;
  int start_elem = offsets[rank];
  int recv_num_elems = chunk_sizes[rank];
  int recv_compressed_size = utils::aligned_size(
      compressor_->BufferSize(recv_num_elems, layers, start_elem));
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }
    int start_offset = offsets[node_rank];
    send_num_elems = chunk_sizes[node_rank];

    send_compressed_size = utils::aligned_size(compressor_->Compress(
        send_buf, layers, start_offset, send_num_elems, gpu_stream));
    compressed_sizes[node_rank] = send_compressed_size;
    send_buf += send_compressed_size;
  }
  send_buf = gradients_send_;

  ErrorCheck("ncclGroupStart", ncclGroupStart());
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }
    send_compressed_size = compressed_sizes[node_rank];
    ErrorCheck("ncclRecv", ncclRecv(recv_buf, recv_compressed_size, ncclChar,
                                    node_rank, nccl_comm_, gpu_stream));
    ErrorCheck("ncclSend", ncclSend(send_buf, send_compressed_size, ncclChar,
                                    node_rank, nccl_comm_, gpu_stream));
    recv_buf += recv_compressed_size;
    send_buf += send_compressed_size;
  }
  ErrorCheck("ncclGroupEnd", ncclGroupEnd());

  recv_buf = gradients_recv_;
  for (int i = 0; i < world_size - 1; i++) {
    compressor_->Decompress(recv_buf, layers, start_elem, recv_num_elems, true,
                            gpu_stream);
    recv_buf += recv_compressed_size;
  }
  // End of the first round.

  compressor_->Compress(gradients_send_, layers, start_elem, recv_num_elems,
                        gpu_stream);
  compressor_->Decompress(gradients_send_, layers, start_elem, recv_num_elems,
                          false, gpu_stream);
  recv_buf = gradients_recv_;
  send_compressed_size = recv_compressed_size;
  ErrorCheck("ncclGroupStart", ncclGroupStart());
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }
    int their_start_offset = offsets[node_rank];
    recv_compressed_size = compressed_sizes[node_rank];
    ErrorCheck("ncclRecv", ncclRecv(recv_buf, recv_compressed_size, ncclChar,
                                    node_rank, nccl_comm_, gpu_stream));
    ErrorCheck("ncclSend",
               ncclSend(gradients_send_, send_compressed_size, ncclChar,
                        node_rank, nccl_comm_, gpu_stream));
    recv_buf += recv_compressed_size;
  }
  ErrorCheck("ncclGroupEnd", ncclGroupEnd());

  recv_buf = gradients_recv_;
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }
    // Offset of the received chunk
    int their_start_offset = offsets[node_rank];
    recv_num_elems = chunk_sizes[node_rank];
    recv_compressed_size = compressed_sizes[node_rank];
    compressor_->Decompress(recv_buf, layers, their_start_offset,
                            recv_num_elems, false, gpu_stream);
    recv_buf += recv_compressed_size;
  }
  return 0;
}

int NCCL_Reduce::Broadcast(int num_elements, int global_offset,
                           std::vector<Layer> &layers, void *comm_p,
                           gpuStream_t gpu_stream, bool do_compression) {
  Init(comm_p);
  unsigned char *layers_data = gradients_send_;
  FuseLayerData(&layers_data, layers, num_elements, global_offset, gpu_stream);

  auto nccl_result = ncclBcast(layers_data, num_elements,
                               ncclDatatype.at(layers[0].scalar_type()), 0,
                               nccl_comm_, gpu_stream);
  ErrorCheck("ncclBcast", nccl_result);
  UnfuseLayerData(layers_data, layers, num_elements, gpu_stream);
  return 0;
}

void NCCL_Reduce::FuseLayerData(unsigned char **layers_data,
                                std::vector<Layer> &layers, int num_elements,
                                int global_offset, gpuStream_t stream) {
  auto element_size = utils::get_sizeof(layers[0].scalar_type());

  if (layers.size() > 1) {
    auto data_tmp = *layers_data;
    for (auto &layer : layers) {
      gpu_context_->MemcpyAsyncD2D(data_tmp, layer.data_ptr(),
                                   layer.numel() * element_size, stream);
      data_tmp += layer.numel() * element_size;
    }
  } else {
    *layers_data = static_cast<unsigned char *>(layers[0].data_ptr()) +
                   global_offset * element_size;
  }
}

void NCCL_Reduce::UnfuseLayerData(unsigned char *layers_data,
                                  std::vector<Layer> &layers, int num_elements,
                                  gpuStream_t stream) {
  auto element_size = utils::get_sizeof(layers[0].scalar_type());

  if (layers.size() > 1) {
    for (auto &layer : layers) {
      gpu_context_->MemcpyAsyncD2D(layer.data_ptr(), layers_data,
                                   layer.numel() * element_size, stream);
      layers_data += layer.numel() * element_size;
    }
  }
}

} // namespace common
} // namespace cgx
