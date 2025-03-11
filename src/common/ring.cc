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

#include "ring.h"

namespace cgx::common {

MPI_Allreduce_Ring::MPI_Allreduce_Ring(
    std::shared_ptr<common::GPUContext> gpu_context,
    std::shared_ptr<Compressor> compressor,
    std::shared_ptr<Communicator> communicator, int world_size)
    : MPIReducer(gpu_context, compressor, communicator) {
  int64_t chunk_size = tensor_fusion_size_;
  chunk_size = utils::aligned_size((chunk_size + world_size - 1) / world_size);
  int64_t buffer_size = chunk_size * world_size + chunk_size * (world_size - 1);

  buffer_ = std::make_unique<PersistentBuffer>(buffer_size);
  void *buffer_data = buffer_->RawPointer();
  gradients_send_ = static_cast<unsigned char *>(buffer_data);
  gradients_recv_ = gradients_send_ + chunk_size * world_size;
}

int MPI_Allreduce_Ring::AllreduceDivision(int num_elements, int global_offset,
                                          std::vector<Layer> &tensors,
                                          void *comm_p, gpuStream_t gpu_stream,
                                          bool do_compression) {
  int status;
  if (do_compression) {
    status = AllreduceDivisionCompressed(num_elements, global_offset, tensors,
                                         comm_p, gpu_stream);
  } else {
    status = AllreduceDivisionUncompressed(num_elements, global_offset, tensors,
                                           comm_p, gpu_stream);
  }
  MPI_Comm comm = *(static_cast<MPI_Comm *>(comm_p));
  MPI_CHECK(MPI_Barrier(comm));
  return status;
}

int MPI_Allreduce_Ring::AllreduceDivisionUncompressed(
    int num_elements, int global_offset, std::vector<Layer> &layers,
    void *comm_p, gpuStream_t gpu_stream) {
  MPI_Comm comm = *(static_cast<MPI_Comm *>(comm_p));
  int world_size, rank;
  std::vector<int> chunk_sizes, offsets;
  unsigned char *send_buf = gradients_send_;
  unsigned char *send_buf_base = send_buf;
  unsigned char *recv_buf = gradients_recv_;
  int element_size = layers[0].element_size();

  MPI_CHECK(MPI_Comm_size(comm, &world_size));
  MPI_CHECK(MPI_Comm_rank(comm, &rank));
  Compressor::GetSizesAndOffsets(num_elements, world_size, global_offset, layers,
                                offsets, chunk_sizes);
  communicator_->Init(world_size, comm_p);
  if (layers.size() > 1) {
    for (auto &layer : layers) {
      gpu_context_->MemcpyAsyncD2D(send_buf, layer.data_ptr(),
                                   layer.numel() * element_size, gpu_stream);
      send_buf += layer.numel() * element_size;
    }
    send_buf = send_buf_base;
  } else {
    send_buf = static_cast<unsigned char *>(layers[0].data_ptr()) +
               element_size * global_offset;
    send_buf_base = send_buf;
  }

  // Receive from your left neighbor with wrap-around.
  const size_t recv_from = (rank - 1 + world_size) % world_size;
  // Send to your right neighbor with wrap-around.
  const size_t send_to = (rank + 1) % world_size;

  int recv_segment_idx, send_segment_idx;
  int buf_send_idx, buf_recv_idx;
  int send_size, recv_size;
  for (int i = 0; i < world_size - 1; i++) {
    recv_segment_idx = (rank - i - 1 + world_size) % world_size;
    send_segment_idx = (rank - i + world_size) % world_size;
    buf_send_idx = offsets[send_segment_idx];
    buf_recv_idx = offsets[recv_segment_idx];
    communicator_->ISend(send_buf_base + buf_send_idx * element_size,
                         chunk_sizes[send_segment_idx] * element_size, send_to,
                         gpu_stream);
    communicator_->IRecv(recv_buf, chunk_sizes[recv_segment_idx] * element_size,
                         recv_from, gpu_stream);
    communicator_->WaitSend(send_to);
    communicator_->WaitRecv(recv_from);
    Compressor::Add(chunk_sizes[recv_segment_idx],
                    send_buf_base + buf_recv_idx * element_size, recv_buf,
                    send_buf_base + buf_recv_idx * element_size,
                    layers[0].scalar_type(), gpu_stream);
  }
  for (int i = 0; i < world_size - 1; i++) {
    send_segment_idx = (rank - i + world_size + 1) % world_size;
    buf_send_idx = offsets[send_segment_idx];
    send_buf = send_buf_base + buf_send_idx * element_size;
    send_size = chunk_sizes[send_segment_idx] * element_size;

    recv_segment_idx = (rank - i + world_size) % world_size;
    buf_recv_idx = offsets[recv_segment_idx];
    recv_size = chunk_sizes[recv_segment_idx] * element_size;
    recv_buf = send_buf_base + buf_recv_idx * element_size;

    communicator_->ISend(send_buf, send_size, send_to, gpu_stream);
    communicator_->IRecv(recv_buf, recv_size, recv_from, gpu_stream);
    communicator_->WaitSend(send_to);
    communicator_->WaitRecv(recv_from);
    send_buf += send_size;
  }
  send_buf = send_buf_base;
  if (layers.size() > 1) {
    for (auto &layer : layers) {
      gpu_context_->MemcpyAsyncD2D(layer.data_ptr(), send_buf,
                                   layer.numel() * element_size, gpu_stream);
      send_buf += layer.numel() * element_size;
    }
  }
  return 0;
}

int MPI_Allreduce_Ring::AllreduceDivisionCompressed(int num_elements,
                                                    int global_offset,
                                                    std::vector<Layer> &layers,
                                                    void *comm_p,
                                                    gpuStream_t gpu_stream) {
  MPI_Comm comm = *(static_cast<MPI_Comm *>(comm_p));
  int world_size, rank;
  MPI_CHECK(MPI_Comm_size(comm, &world_size));
  MPI_CHECK(MPI_Comm_rank(comm, &rank));
  std::vector<int> chunk_sizes, offsets;
  Quantizer::GetSizesAndOffsets(num_elements, world_size, global_offset, layers,
                                offsets, chunk_sizes);
  communicator_->Init(world_size, comm_p);
  compressor_->Init(layers[0].element_size(), gpu_stream);
  int start_elem = offsets[rank] + global_offset;
  int recv_num_elems = chunk_sizes[rank];
  int recv_compressed_size = utils::aligned_size(
      compressor_->BufferSize(recv_num_elems, layers, start_elem));
  int send_num_elems = 0;
  int send_compressed_size = 0;
  unsigned char *send_buf = gradients_send_;
  unsigned char *recv_buf = gradients_recv_;
  int element_size = layers[0].element_size();
  // Receive from your left neighbor with wrap-around.
  const size_t recv_from = (rank - 1 + world_size) % world_size;
  // Send to your right neighbor with wrap-around.
  const size_t send_to = (rank + 1) % world_size;
  int recv_segment_idx, send_segment_idx;
  int buf_send_idx, buf_recv_idx;
  int send_size, recv_size;

  for (int i = 0; i < world_size - 1; i++) {
    recv_segment_idx = (rank - i - 1 + world_size) % world_size;
    send_segment_idx = (rank - i + world_size) % world_size;
    buf_send_idx = offsets[send_segment_idx];
    buf_recv_idx = offsets[recv_segment_idx];

    recv_size = utils::aligned_size(compressor_->BufferSize(
        chunk_sizes[recv_segment_idx], layers, buf_recv_idx));
    communicator_->IRecv(gradients_recv_, recv_size, recv_from, gpu_stream);

    send_size = utils::aligned_size(
        compressor_->Compress(gradients_send_, layers, buf_send_idx,
                              chunk_sizes[send_segment_idx], gpu_stream));
    communicator_->ISend(gradients_send_, send_size, send_to, gpu_stream);
    communicator_->WaitRecv(recv_from);
    communicator_->WaitSend(send_to);
    compressor_->Decompress(gradients_recv_, layers, buf_recv_idx,
                            chunk_sizes[recv_segment_idx], true, gpu_stream);
  }

  send_segment_idx = (rank + world_size + 1) % world_size;
  buf_send_idx = offsets[send_segment_idx];
  send_buf = gradients_send_;
  send_size = utils::aligned_size(compressor_->Compress(
      send_buf, layers, buf_send_idx, chunk_sizes[send_segment_idx], gpu_stream));
  compressor_->Decompress(send_buf, layers, buf_send_idx,
                          chunk_sizes[send_segment_idx], false, gpu_stream);
  recv_buf = send_buf + send_size;
  unsigned char *compressed_buf = recv_buf;

  for (int i = 0; i < world_size - 1; i++) {
    recv_segment_idx = (rank - i + world_size) % world_size;
    buf_recv_idx = offsets[recv_segment_idx];
    recv_size = utils::aligned_size(compressor_->BufferSize(
        chunk_sizes[recv_segment_idx], layers, buf_recv_idx));
    communicator_->ISend(send_buf, send_size, send_to, gpu_stream);
    communicator_->IRecv(recv_buf, recv_size, recv_from, gpu_stream);
    communicator_->WaitSend(send_to);
    communicator_->WaitRecv(recv_from);
    send_buf += send_size;
    recv_buf += recv_size;
    send_size = recv_size;
  }

  // Decompress all chunks we received.
  for (int i = 0; i < world_size - 1; i++) {
    recv_segment_idx = (rank - i + world_size) % world_size;
    buf_recv_idx = offsets[recv_segment_idx];

    compressor_->Decompress(compressed_buf, layers, buf_recv_idx,
                            chunk_sizes[recv_segment_idx], false, gpu_stream);
    recv_size = utils::aligned_size(compressor_->BufferSize(
        chunk_sizes[recv_segment_idx], layers, buf_recv_idx));
    compressed_buf += recv_size;
  }
  return 0;
}

} // namespace cgx::common
