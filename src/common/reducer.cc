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

#include "reducer.h"
namespace cgx {
namespace common {

Reducer::Reducer(std::shared_ptr<GPUContext> gpu_context,
                 std::shared_ptr<Compressor> compressor,
                 std::shared_ptr<Communicator> communicator)
    : compressor_(compressor), gpu_context_(gpu_context),
      communicator_(communicator) {
  unsigned int fusion_size_mb =
      utils::GetIntEnvOrDefault(FUSION_BUFFER_SIZE_MB, FUSION_SIZE_DEFAULT_MB);
  tensor_fusion_size_ = std::max(fusion_size_mb * 1024 * 1024, MIN_FUSION_SIZE);
}

int Reducer::AllReduceAlltoAll(int num_elements, int global_offset,
                               std::vector<Layer> &layers, void *comm_p,
                               gpuStream_t gpu_stream) {
  MPI_Comm comm = *(static_cast<MPI_Comm *>(comm_p));
  int world_size, rank;
  MPI_CHECK(MPI_Comm_size(comm, &world_size));
  MPI_CHECK(MPI_Comm_rank(comm, &rank));
  unsigned char *send_buf = gradients_send_;
  unsigned char *recv_buf = gradients_recv_;
  size_t element_size = layers[0].element_size();
  size_t buf_size = num_elements * element_size;

  communicator_->Init(world_size, comm_p);
  if (layers.size() > 1) {
    for (auto &layer: layers) {
      gpu_context_->MemcpyAsyncD2D(send_buf,
                                   layer.data_ptr(),
                                   layer.numel() * element_size,
                                   gpu_stream);
      send_buf += layer.numel() * element_size;
    }
    send_buf = gradients_send_;
  } else {
    send_buf = static_cast<unsigned char *>(layers[0].data_ptr()) + global_offset * element_size;
  }

  std::vector<int> nodes;
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank)
      continue;
    communicator_->ISend(send_buf, buf_size, node_rank,
                         gpu_stream);
    communicator_->IRecv(recv_buf, buf_size, node_rank, gpu_stream);
    nodes.push_back(node_rank);
    recv_buf += buf_size;
  }
  communicator_->WaitAllSend();

  while (nodes.size() > 0) {
    for (int i = 0; i < nodes.size(); i++) {
      auto &node_rank = nodes.at(i);
      if (communicator_->TestRecv(node_rank) > 0) {
        auto idx = node_rank - ((node_rank > rank) ? 1 : 0);
        Compressor::Add(num_elements, gradients_recv_ + idx * buf_size,
                        send_buf, send_buf, layers[0].scalar_type(), gpu_stream);
        nodes.erase(nodes.begin() + i);
      }
    }
  }
  if (layers.size() > 1) {
    for (auto &layer: layers) {
      gpu_context_->MemcpyAsyncD2D(layer.data_ptr(),
                                   send_buf,
                                   layer.numel() * element_size,
                                   gpu_stream);
      send_buf += layer.numel() * element_size;
    }
  }
  return 0;
}

int Reducer::Broadcast(int num_elements, int global_offset,
                       std::vector<Layer> &layers, void *comm_p,
                       gpuStream_t gpu_stream, bool do_compression) {
  MPI_Comm comm = *(static_cast<MPI_Comm *>(comm_p));
  int world_size, rank;
  MPI_CHECK(MPI_Comm_size(comm, &world_size));
  MPI_CHECK(MPI_Comm_rank(comm, &rank));
  int element_size = layers[0].element_size();

  communicator_->Init(world_size, comm_p);
  if (do_compression)
    compressor_->Init(layers[0].element_size(), gpu_stream);
  if (rank == 0) {
    unsigned char *send_buf = gradients_send_;
    size_t send_size;
    if (do_compression) {
      send_size = utils::aligned_size(compressor_->Compress(
          gradients_send_, layers, global_offset, num_elements, gpu_stream));
      compressor_->Decompress(gradients_send_, layers, num_elements,
                              global_offset, false, gpu_stream);
      send_buf = gradients_send_;
    } else {
      if (layers.size() > 1) {
        for (auto &layer : layers) {
          gpu_context_->MemcpyAsyncD2D(send_buf, layer.data_ptr(),
                                       layer.numel() * element_size,
                                       gpu_stream);
          send_buf += layer.numel() * element_size;
        }
        send_buf = gradients_send_;
      } else {
        send_buf = static_cast<unsigned char *>(layers[0].data_ptr());
      }
      send_size = num_elements * element_size;
    }
    for (int node_rank = 1; node_rank < world_size; node_rank++) {
      communicator_->ISend(send_buf, send_size, node_rank, gpu_stream);
    }
    communicator_->WaitAllSend();
  } else {
    size_t recv_size = 0;
    unsigned char *recv_buf = gradients_send_;
    if (do_compression) {
      recv_size = utils::aligned_size(
          compressor_->BufferSize(num_elements, layers, global_offset));
    } else {
      recv_size = num_elements * element_size;
      if (layers.size() == 1)
        recv_buf = static_cast<unsigned char *>(layers[0].data_ptr());
    }
    communicator_->IRecv(recv_buf, recv_size, 0, gpu_stream);
    communicator_->WaitRecv(0);
    if (do_compression) {
      compressor_->Decompress(recv_buf, layers, global_offset, num_elements,
                              false, gpu_stream);
    } else if (layers.size() > 1) {
      for (auto &layer : layers) {
        gpu_context_->MemcpyAsyncD2D(layer.data_ptr(), recv_buf,
                                     layer.numel() * element_size, gpu_stream);
        recv_buf += layer.numel() * element_size;
      }
    }
  }
  return 0;
}

} // namespace common
} // namespace cgx
