#include "scatter_reduce_allgather.h"

namespace qmpi {
namespace common {

MPI_Allreduce_ScatterReduceAllgather::MPI_Allreduce_ScatterReduceAllgather(
    GPUContext *gpu_context,
    std::shared_ptr<Compressor> compressor,
    int world_size)
    : MPIReducer(gpu_context, compressor) {
  int64_t chunk_size = tensor_fusion_size_;
  chunk_size = utils::aligned_size((chunk_size + world_size - 1) / world_size);
  int64_t buffer_size = chunk_size * (world_size - 1) +
      +chunk_size * (world_size - 1) + chunk_size;

  buffer_ = std::make_unique<PersistentBuffer>(buffer_size);
  void *buffer_data = buffer_->RawPointer();
  gradients_send_ = static_cast<unsigned char *>(buffer_data);
  gradients_recv_ = gradients_send_ + chunk_size * (world_size - 1);
  decompress_buffer_ = gradients_recv_ + chunk_size * (world_size - 1);
  streams_ = new gpuStream_t[world_size];
  for (int i = 0; i < world_size; i++) {
    gpu_context->StreamCreate(&streams_[i]);
  }
}

int MPI_Allreduce_ScatterReduceAllgather::AllreduceDivision(int num_elements,
                                                            int global_offset,
                                                            std::vector<at::Tensor> &tensors,
                                                            void *comm,
                                                            bool do_compression) {
  if (do_compression) {
    return AllreduceDivisionCompressed(num_elements, global_offset, tensors, comm);
  } else {
    return AllreduceDivisionUncompressed(num_elements, global_offset, tensors, comm);
  }
}
// Perform Scatter-Reduce-AllGather (SRA)
int MPI_Allreduce_ScatterReduceAllgather::AllreduceDivisionCompressed(int num_elements,
                                                                      int global_offset,
                                                                      std::vector<
                                                                          at::Tensor> &tensors,
                                                                      void *comm_p) {
  MPI_Comm comm = *(static_cast<MPI_Comm *>(comm_p));
  int world_size, rank;
  MPI_CHECK(MPI_Comm_size(comm, &world_size));
  MPI_CHECK(MPI_Comm_rank(comm, &rank));
  std::vector<int> chunk_sizes, offsets;
  compressor_->GetSizesAndOffsets(num_elements, world_size, tensors, offsets,
                                  chunk_sizes);

  int start_elem = offsets[rank] + global_offset;
  int recv_num_elems = chunk_sizes[rank];
  int recv_compressed_size = utils::aligned_size(
      compressor_->BufferSize(recv_num_elems, tensors, start_elem));
  int send_num_elems = 0;
  int send_compressed_size = 0;
  unsigned char *send_buf = gradients_send_;
  unsigned char *recv_buf = gradients_recv_;
  std::vector<MPI_Request> send_requests;
  std::vector<MPI_Request> recv_requests;
  std::queue<int> send_sizes;

  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }
    int start_offset = offsets[node_rank];
    send_num_elems = chunk_sizes[node_rank];
    send_compressed_size = utils::aligned_size(
        compressor_->Compress(send_buf, tensors, start_offset,
                              send_num_elems, streams_[node_rank]));
    send_buf += send_compressed_size;
    send_sizes.push(send_compressed_size);
  }
//  gpu_context_->StreamSynchronize(stream_);

  send_buf = gradients_send_;
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }
    gpu_context_->StreamSynchronize(streams_[node_rank]);
    recv_requests.push_back(MPI_Request());
    MPI_CHECK(MPI_Irecv(recv_buf, recv_compressed_size, MPI_UNSIGNED_CHAR,
                        node_rank, 0, comm, &recv_requests.back()));
    send_compressed_size = send_sizes.front();
    send_requests.push_back(MPI_Request());
    MPI_CHECK(MPI_Isend(send_buf, send_compressed_size, MPI_UNSIGNED_CHAR,
                        node_rank, 0, comm, &send_requests.back()));

    recv_buf += recv_compressed_size;
    send_buf += send_compressed_size;
    send_sizes.pop();
  }
  std::vector<int> idx_map;
  for (int i = 0; i < world_size - 1; i++) {
    idx_map.push_back(i);
  }

  while (recv_requests.size() > 0) {
    int req_idx;
    MPI_CHECK(MPI_Waitany((int) recv_requests.size(), recv_requests.data(),
                          &req_idx, MPI_STATUSES_IGNORE));
    int idx = idx_map[req_idx];
    recv_requests.erase(recv_requests.begin() + req_idx);
    idx_map.erase(idx_map.begin() + req_idx);
    compressor_->Decompress(gradients_recv_ + idx * recv_compressed_size,
                            tensors, start_elem, recv_num_elems,
                            true, streams_[rank]);
  }
  MPI_CHECK(MPI_Waitall((int) send_requests.size(), send_requests.data(),
                        MPI_STATUSES_IGNORE));
  send_requests.clear();
  // End of the first round.

  compressor_->Compress(gradients_send_,
                        tensors,
                        start_elem,
                        recv_num_elems,
                        streams_[rank]);
  gpu_context_->StreamSynchronize(streams_[rank]);
  compressor_->Decompress(gradients_send_, tensors, start_elem,
                          recv_num_elems, false, streams_[rank]);
  recv_buf = gradients_recv_;

  // second round of SRA. receive the sums from other nodes. Perform
  send_compressed_size = recv_compressed_size;
  std::vector<std::tuple<int64_t, int, int>> recv_offsets;
  int64_t recv_acc_size = 0;
  recv_requests.clear();
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      continue;
    }
    int their_start_offset = offsets[node_rank] + global_offset;
    recv_num_elems = chunk_sizes[node_rank];
    recv_compressed_size = utils::aligned_size(
        compressor_->BufferSize(recv_num_elems, tensors, their_start_offset));

    recv_requests.push_back(MPI_Request());
    MPI_CHECK(MPI_Irecv(recv_buf, recv_compressed_size, MPI_UNSIGNED_CHAR,
                        node_rank, 0, comm, &recv_requests.back()));
    send_requests.push_back(MPI_Request());
    MPI_CHECK(MPI_Isend(gradients_send_, send_compressed_size,
                        MPI_UNSIGNED_CHAR, node_rank, 0, comm,
                        &send_requests.back()));
    recv_buf += recv_compressed_size;
    recv_offsets.emplace_back(recv_acc_size, their_start_offset,
                              recv_num_elems);
    recv_acc_size += recv_compressed_size;
  }
  int stream_id = 0;
  while (recv_requests.size() > 0) {
    int req_idx;
    int their_start_offset;
    MPI_CHECK(MPI_Waitany((int) recv_requests.size(), recv_requests.data(),
                          &req_idx, MPI_STATUSES_IGNORE));

    std::tie(recv_acc_size, their_start_offset, recv_num_elems) =
        recv_offsets[req_idx];
    recv_requests.erase(recv_requests.begin() + req_idx);
    recv_offsets.erase(recv_offsets.begin() + req_idx);
    compressor_->Decompress(gradients_recv_ + recv_acc_size,
                            tensors, their_start_offset, recv_num_elems, false,
                            streams_[stream_id]);
    stream_id++;
  }
  MPI_CHECK(MPI_Waitall((int) send_requests.size(), send_requests.data(),
                        MPI_STATUSES_IGNORE));
  for (int i = 0; i < world_size; i++) {
    gpu_context_->StreamSynchronize(streams_[i]);
  }
//  gpu_context_->StreamSynchronize(stream_);
  return 0;
}

int MPI_Allreduce_ScatterReduceAllgather::AllreduceDivisionUncompressed(int num_elements,
                                                                        int global_offset,
                                                                        std::vector<
                                                                            at::Tensor> &tensors,
                                                                        void *comm_p) {
  gpuStream_t gpu_stream = streams_[0];
  MPI_Comm comm = *(static_cast<MPI_Comm *>(comm_p));
  int world_size, rank;
  MPI_CHECK(MPI_Comm_size(comm, &world_size));
  MPI_CHECK(MPI_Comm_rank(comm, &rank));
  int element_size = tensors[0].element_size();
  auto get_numel_chunk = [=](int node_rank) {
    return num_elements / world_size + ((node_rank < world_size) ? 1 : 0);
  };
  int send_size;
  int recv_num_elems = get_numel_chunk(rank);
  int recv_size = recv_num_elems * element_size;
  unsigned char *send_buf = gradients_send_;
  unsigned char *send_buf_base = send_buf;
  unsigned char *recv_buf = gradients_recv_;
  std::vector<MPI_Request> requests;

  if (tensors.size() > 1) {
    for (auto &tensor: tensors) {
      gpu_context_->MemcpyAsyncD2D(send_buf,
                                   tensor.data_ptr(),
                                   tensor.numel(),
                                   gpu_stream);
      send_buf += tensor.numel() * element_size;
    }
    send_buf = send_buf_base;
  } else {
    send_buf = static_cast<unsigned char *>(tensors[0].data_ptr())
        + element_size * global_offset;
    send_buf_base = send_buf;
  }
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      send_buf += recv_size;
      continue;
    }
    send_size = get_numel_chunk(node_rank) * element_size;
    requests.push_back(MPI_Request());
    MPI_CHECK(MPI_Irecv(recv_buf, recv_size, MPI_UNSIGNED_CHAR,
                        node_rank, 0, comm, &requests.back()));
    requests.push_back(MPI_Request());
    MPI_CHECK(MPI_Isend(send_buf, send_size, MPI_UNSIGNED_CHAR,
                        node_rank, 0, comm, &requests.back()));
    recv_buf += recv_size;
    send_buf += send_size;
  }
  MPI_CHECK(MPI_Waitall((int) requests.size(), requests.data(),
                        MPI_STATUSES_IGNORE));
  requests.clear();
  send_buf = send_buf_base
      + ((num_elements / world_size) * rank + rank) * element_size;
  recv_buf = gradients_recv_;
  for (int node_rank; node_rank < world_size; node_rank++) {
    if (node_rank == rank)
      continue;
    compressor_->Add(recv_num_elems, send_buf, recv_buf, send_buf, tensors[0].scalar_type(), gpu_stream);
    recv_buf += recv_size;
  }
  gpu_context_->StreamSynchronize(gpu_stream);
  send_size = recv_size;
  recv_buf = send_buf_base;
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank) {
      recv_buf += send_size;
      continue;
    }
    recv_size = get_numel_chunk(node_rank) * element_size;
    requests.push_back(MPI_Request());
    MPI_CHECK(MPI_Irecv(recv_buf, recv_size, MPI_UNSIGNED_CHAR,
                        node_rank, 0, comm, &requests.back()));
    requests.push_back(MPI_Request());
    MPI_CHECK(MPI_Isend(send_buf, send_size,
                        MPI_UNSIGNED_CHAR, node_rank, 0, comm,
                        &requests.back()));
    recv_buf += recv_size;
  }
  MPI_CHECK(MPI_Waitall((int) requests.size(), requests.data(),
                        MPI_STATUSES_IGNORE));
  send_buf = send_buf_base;
  if (tensors.size() > 1) {
    for (auto &tensor: tensors) {
      gpu_context_->MemcpyAsyncD2D(tensor.data_ptr(),
                                   send_buf,
                                   tensor.numel(),
                                   gpu_stream);
      send_buf += tensor.numel() * element_size;
    }
  }
  gpu_context_->StreamSynchronize(gpu_stream);
  return 0;
}

} // namespace common
} // namespace qmpi

