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

#include "scatter_reduce_allgather.h"

#include "assert.h"
#include <utility>
#include "compression/gpu_common.h"

namespace cgx::common {

void printDebug(void *buf, int numel, gpuStream_t gpu_stream) {
  float *host_buf = new float[numel];
  cudaStreamSynchronize(gpu_stream);
  cudaMemcpy(host_buf, buf, numel * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < numel; i++) {
    std::cout << host_buf[i] << " ";
  }
  std::cout << std::endl;
  CUDA_CHECK(cudaGetLastError());
  delete[] host_buf;
}

MPI_Allreduce_ScatterReduceAllgather::MPI_Allreduce_ScatterReduceAllgather(
    std::shared_ptr<GPUContext> gpu_context,
    std::shared_ptr<Compressor> compressor,
    std::shared_ptr<Communicator> communicator, int world_size)
    : MPIReducer(std::move(gpu_context), std::move(compressor), std::move(communicator)) {
  int64_t chunk_size = tensor_fusion_size_;
  all_to_all_reduction_ =
      utils::GetIntEnvOrDefault(DEBUG_ALL_TO_ALL_REDUCTION, 0);
  int64_t buffer_size = 0;
  if (!all_to_all_reduction_) {
    chunk_size =
        utils::aligned_size((chunk_size + world_size - 1) / world_size);
    buffer_size = chunk_size * world_size + chunk_size * (world_size - 1);
  } else {
    buffer_size = chunk_size * world_size;
  }

  buffer_ = std::make_unique<PersistentBuffer>(buffer_size);
  void *buffer_data = buffer_->RawPointer();
  if (!all_to_all_reduction_) {
    gradients_send_ = static_cast<unsigned char *>(buffer_data);
    gradients_recv_ = gradients_send_ + chunk_size * world_size;
  } else {
    gradients_send_ = static_cast<unsigned char *>(buffer_data);
    gradients_recv_ = gradients_send_ + chunk_size;
  }
  remote_buf_compression_enabled_ =
      utils::GetIntEnvOrDefault(REMOTE_BUF_COMPRESSION, 0);
  assert(!(remote_buf_compression_enabled_ and all_to_all_reduction_));
}

int MPI_Allreduce_ScatterReduceAllgather::AllreduceDivision(
    int num_elements, int global_offset, std::vector<Layer> &layers,
    void *comm_p, gpuStream_t gpu_stream, bool do_compression) {
  MPI_Comm comm = *(static_cast<MPI_Comm *>(comm_p));
  int status;
  if (do_compression) {
    if (all_to_all_reduction_) {
      status = AllReduceAlltoAllCompressed(num_elements, global_offset, layers,
                                           comm_p, gpu_stream);
    } else if (remote_buf_compression_enabled_ and
               communicator_->GetType() != Communicator::MPI) {
      status = AllreduceCompressedRemoteBuf(num_elements, global_offset, layers,
                                            comm_p, gpu_stream);
    } else {
      status = AllreduceCompressed(num_elements, global_offset, layers, comm_p, gpu_stream);
    }
  } else {
    status = AllreduceUncompressed(num_elements, global_offset, layers, comm_p, gpu_stream);
  }
  return status;
}

// Perform Scatter-Reduce-AllGather (SRA)
int MPI_Allreduce_ScatterReduceAllgather::AllreduceCompressed(
    int num_elements, int global_offset, std::vector<Layer> &layers,
    void *comm_p, gpuStream_t gpu_stream) {
  MPI_Comm comm = *(static_cast<MPI_Comm *>(comm_p));
  int world_size, rank;
  MPI_CHECK(MPI_Comm_size(comm, &world_size));
  MPI_CHECK(MPI_Comm_rank(comm, &rank));
  std::vector<int> chunk_sizes, offsets;
  Quantizer::GetSizesAndOffsets(num_elements, world_size, global_offset, layers,
                                offsets, chunk_sizes);
  communicator_->Init(world_size, comm_p);
  compressor_->Init(layers[0].element_size(), gpu_stream);
  int start_elem = offsets[rank];
  int recv_num_elems = chunk_sizes[rank];
  int recv_compressed_size = utils::aligned_size(
      compressor_->BufferSize(recv_num_elems, layers, start_elem));
  int send_num_elems = 0;
  int send_compressed_size = 0;
  unsigned char *send_buf = gradients_send_;
  unsigned char *recv_buf = gradients_recv_;
  std::queue<int> send_sizes;
  std::vector<int> nodes;
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }
    int start_offset = offsets[node_rank];
    send_num_elems = chunk_sizes[node_rank];

    send_compressed_size = utils::aligned_size(compressor_->Compress(
        send_buf, layers, start_offset, send_num_elems, gpu_stream));
    send_buf += send_compressed_size;
    send_sizes.push(send_compressed_size);
  }
  send_buf = gradients_send_;
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }
    communicator_->IRecv(recv_buf, recv_compressed_size, node_rank,
                         gpu_stream);
    send_compressed_size = send_sizes.front();
    communicator_->ISend(send_buf, send_compressed_size, node_rank,
                         gpu_stream);
    recv_buf += recv_compressed_size;
    send_buf += send_compressed_size;
    send_sizes.pop();
    nodes.push_back(node_rank);
  }
  while (nodes.size() > 0) {
    for (int i = 0; i < nodes.size(); i++) {
      auto &node_rank = nodes.at(i);
      if (communicator_->TestRecv(node_rank) > 0) {
        auto idx = node_rank - ((node_rank > rank) ? 1 : 0);
        compressor_->Decompress(gradients_recv_ + recv_compressed_size * idx,
                                layers, start_elem, recv_num_elems, true,
                                gpu_stream);
        nodes.erase(nodes.begin() + i);
      }
    }
  }
  communicator_->WaitAllSend();
  // End of the first round.
  compressor_->Compress(gradients_send_, layers, start_elem, recv_num_elems,
                        gpu_stream);
  compressor_->Decompress(gradients_send_, layers, start_elem, recv_num_elems,
                          false, gpu_stream);
  recv_buf = gradients_recv_;
  // second round of SRA. receive the sums from other nodes. Perform
  send_compressed_size = recv_compressed_size;
  std::vector<std::tuple<int64_t, int, int>> recv_offsets;
  int64_t recv_acc_size = 0;
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }
    int their_start_offset = offsets[node_rank];
    recv_num_elems = chunk_sizes[node_rank];
    recv_compressed_size = utils::aligned_size(
        compressor_->BufferSize(recv_num_elems, layers, their_start_offset));
    communicator_->IRecv(recv_buf, recv_compressed_size, node_rank,
                         gpu_stream);
    communicator_->ISend(gradients_send_, send_compressed_size, node_rank,
                         gpu_stream);

    recv_buf += recv_compressed_size;
    recv_offsets.emplace_back(recv_acc_size, their_start_offset,
                              recv_num_elems);
    recv_acc_size += recv_compressed_size;
    nodes.push_back(node_rank);
  }
  int their_start_offset;
  while (nodes.size() > 0) {
    for (int i = 0; i < nodes.size(); i++) {
      auto &node_rank = nodes.at(i);
      if (communicator_->TestRecv(node_rank) > 0) {
        auto idx = node_rank - ((node_rank > rank) ? 1 : 0);
        std::tie(recv_acc_size, their_start_offset, recv_num_elems) =
            recv_offsets[idx];
        compressor_->Decompress(gradients_recv_ + recv_acc_size, layers,
                                their_start_offset, recv_num_elems, false,
                                gpu_stream);
        nodes.erase(nodes.begin() + i);
      }
    }
  }
  communicator_->WaitAllSend();
  return 0;
}

int MPI_Allreduce_ScatterReduceAllgather::AllreduceCompressedRemoteBuf(
    int num_elements, int global_offset, std::vector<Layer> &layers,
    void *comm_p, gpuStream_t gpu_stream) {
  MPI_Comm comm = *(static_cast<MPI_Comm *>(comm_p));
  int world_size, rank;
  MPI_CHECK(MPI_Comm_size(comm, &world_size));
  MPI_CHECK(MPI_Comm_rank(comm, &rank));
  std::vector<int> chunk_sizes, offsets;
  Quantizer::GetSizesAndOffsets(num_elements, world_size, global_offset, layers,
                                offsets, chunk_sizes);
  CommunicatorLocal *communicator_local_ =
      reinterpret_cast<CommunicatorLocal *>(communicator_.get());
  communicator_local_->Init(world_size, comm_p);
  compressor_->Init(layers[0].element_size(), gpu_stream);
  int start_elem, num_elems;
  int send_compressed_size = 0;
  unsigned char *send_buf = gradients_send_;

  unsigned char *remote_buf;
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }

    start_elem = offsets[node_rank];
    num_elems = chunk_sizes[node_rank];
    remote_buf = static_cast<unsigned char *>(
        communicator_local_->GetRemoteBuftoSend(node_rank));
    send_compressed_size = utils::aligned_size(compressor_->Compress(
        remote_buf, layers, start_elem, num_elems, gpu_stream));
  }
  start_elem = offsets[rank];
  num_elems = chunk_sizes[rank];
  int recv_compressed_size = utils::aligned_size(
      compressor_->BufferSize(chunk_sizes[rank], layers, offsets[rank]));
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }
    remote_buf = static_cast<unsigned char *>(
        communicator_local_->GetRemoteBuftoRecv(node_rank));

    compressor_->Decompress(remote_buf, layers, start_elem, num_elems, true,
                            gpu_stream);
  }
  // End of the first round.
  remote_buf = static_cast<unsigned char *>(
      communicator_local_->GetRemoteBroadcastBuftoSend());
  send_compressed_size = utils::aligned_size(compressor_->Compress(
      remote_buf, layers, start_elem, num_elems, gpu_stream));
  compressor_->Decompress(remote_buf, layers, start_elem, num_elems, false,
                          gpu_stream);
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank)
      continue;
    remote_buf = static_cast<unsigned char *>(
        communicator_local_->GetRemoteBroadcastBuftoRecv(node_rank));
    start_elem = offsets[node_rank];
    num_elems = chunk_sizes[node_rank];
    compressor_->Decompress(remote_buf, layers, start_elem, num_elems, false,
                            gpu_stream);
  }
  return 0;
}

int MPI_Allreduce_ScatterReduceAllgather::AllReduceAlltoAllCompressed(
    int num_elements, int global_offset, std::vector<Layer> &layers,
    void *comm_p, gpuStream_t gpu_stream) {
  MPI_Comm comm = *(static_cast<MPI_Comm *>(comm_p));
  int world_size, rank;
  MPI_CHECK(MPI_Comm_size(comm, &world_size));
  MPI_CHECK(MPI_Comm_rank(comm, &rank));
  communicator_->Init(world_size, comm_p);
  compressor_->Init(layers[0].element_size(), gpu_stream);
  int compressed_size = compressor_->Compress(
      gradients_send_, layers, global_offset, num_elements, gpu_stream);
  unsigned char *recv_buf = gradients_recv_;
  std::vector<int> nodes;
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank)
      continue;
    communicator_->ISend(gradients_send_, compressed_size, node_rank,
                         gpu_stream);
    communicator_->IRecv(recv_buf, compressed_size, node_rank, gpu_stream);
    nodes.push_back(node_rank);
    recv_buf += compressed_size;
  }
  communicator_->WaitAllSend();
  compressor_->Decompress(gradients_send_, layers, global_offset, num_elements,
                          false, gpu_stream);
  while (nodes.size() > 0) {
    for (int i = 0; i < nodes.size(); i++) {
      auto &node_rank = nodes.at(i);
      if (communicator_->TestRecv(node_rank) > 0) {
        auto idx = node_rank - ((node_rank > rank) ? 1 : 0);
        compressor_->Decompress(gradients_recv_ + idx * compressed_size, layers,
                                global_offset, num_elements, true, gpu_stream);
        nodes.erase(nodes.begin() + i);
      }
    }
  }
  return 0;
}

int MPI_Allreduce_ScatterReduceAllgather::AllreduceUncompressed(
    int num_elements, int global_offset, std::vector<Layer> &layers,
    void *comm_p, gpuStream_t gpu_stream) {
  MPI_Comm comm = *(static_cast<MPI_Comm *>(comm_p));
  int world_size, rank;
  MPI_CHECK(MPI_Comm_size(comm, &world_size));
  MPI_CHECK(MPI_Comm_rank(comm, &rank));
  int element_size = layers[0].element_size();
  std::vector<int> chunk_sizes, offsets;
  Compressor::GetSizesAndOffsets(num_elements, world_size, global_offset,
                                 layers, offsets, chunk_sizes);
  int send_size;
  int recv_num_elems = chunk_sizes[rank];
  int recv_size = recv_num_elems * element_size;
  unsigned char *send_buf = gradients_send_;
  unsigned char *send_buf_base = send_buf;
  unsigned char *recv_buf = gradients_recv_;

  communicator_->Init(world_size, comm_p);
  if (layers.size() > 1) {
    for (auto &layer : layers) {
      gpu_context_->MemcpyAsyncD2D(send_buf, layer.data_ptr(),
                                   layer.numel() * element_size, gpu_stream);
      send_buf += layer.numel() * element_size;
    }
    send_buf = send_buf_base;
  } else {
    send_buf = static_cast<unsigned char *>(layers[0].data_ptr());
    send_buf_base = send_buf;
  }

  if (num_elements > world_size) {
    std::vector<int> nodes;
    nodes.reserve(world_size - 1);
    for (int node_rank = 0; node_rank < world_size; node_rank++) {
      if (node_rank == rank) {
        send_buf += recv_size;
        continue;
      }
      send_size = chunk_sizes[node_rank] * element_size;
      communicator_->IRecv(recv_buf, recv_size, node_rank, gpu_stream);
      communicator_->ISend(send_buf, send_size, node_rank, gpu_stream);
      recv_buf += recv_size;
      send_buf += send_size;
      nodes.push_back(node_rank);
    }
    send_buf = send_buf_base + offsets[rank] * element_size;
    recv_buf = gradients_recv_;
    while (nodes.size() > 0) {
      for (int i = 0; i < nodes.size(); i++) {
        auto &node_rank = nodes[i];
        if (communicator_->TestRecv(node_rank) > 0) {
          auto idx = node_rank - ((node_rank > rank) ? 1 : 0);
          recv_buf = gradients_recv_ + recv_size * idx;

          Compressor::Add(recv_num_elems, send_buf, recv_buf, send_buf,
                          layers[0].scalar_type(), gpu_stream);
          nodes.erase(nodes.begin() + i);
        }
      }
    }
    communicator_->WaitAllSend();
    send_size = recv_size;
    recv_buf = send_buf_base;
    for (int node_rank = 0; node_rank < world_size; node_rank++) {
      if (node_rank == rank) {
        recv_buf += send_size;
        continue;
      }
      recv_size = chunk_sizes[node_rank] * element_size;
      communicator_->IRecv(recv_buf, recv_size, node_rank, gpu_stream);
      communicator_->ISend(send_buf, send_size, node_rank, gpu_stream);
      recv_buf += recv_size;
    }
    communicator_->WaitAllRecv();
    communicator_->WaitAllSend();
    send_buf = send_buf_base;
  } else {
    send_size = num_elements * element_size;
    for (int node_rank = 0; node_rank < world_size; node_rank++) {
      if (node_rank == rank)
        continue;
      communicator_->ISend(send_buf, send_size, node_rank, gpu_stream);
      communicator_->IRecv(recv_buf, send_size, node_rank, gpu_stream);
      recv_buf += send_size;
    }
    communicator_->WaitAllRecv();
    communicator_->WaitAllSend();
    recv_buf = gradients_recv_;
    for (int node_rank = 0; node_rank < world_size; node_rank++) {
      if (node_rank == rank)
        continue;
      compressor_->Add(num_elements, send_buf, recv_buf, send_buf,
                       layers[0].scalar_type(), gpu_stream);
      recv_buf += send_size;
    }
  }
  if (layers.size() > 1) {
    for (auto &layer : layers) {
      gpu_context_->MemcpyAsyncD2D(layer.data_ptr(), send_buf,
                                   layer.numel() * element_size, gpu_stream);
      send_buf += layer.numel() * element_size;
    }
  }
  return 0;
}

} // namespace cgx::common
