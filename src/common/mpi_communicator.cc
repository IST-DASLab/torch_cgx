#include "mpi_communicator.h"

namespace cgx {
namespace common {

void MPICommunicator::Init(int world_size, void *ctx) {
  if (recv_requests.size() == 0) {
    for (int i = 0; i < world_size; i++) {
      recv_requests.push_back(MPI_REQUEST_NULL);
      send_requests.push_back(MPI_REQUEST_NULL);
    }
  }
  comm_ = *(static_cast<MPI_Comm *>(ctx));
  MPI_CHECK(MPI_Comm_rank(comm_, &rank_));
  world_size_ = world_size;
  MPI_Barrier(comm_);
}

void MPICommunicator::IRecv(void *buf,
                            size_t buf_size,
                            int peer_rank,
                            gpuStream_t stream) {
  gpu_context_->StreamSynchronize(stream);
  MPI_CHECK(MPI_Irecv(buf, buf_size, MPI_UNSIGNED_CHAR,
                      peer_rank, 0, comm_, &recv_requests.at(peer_rank)));
}

void MPICommunicator::ISend(void *buf,
                            size_t buf_size,
                            int peer_rank,
                            gpuStream_t stream) {
  gpu_context_->StreamSynchronize(stream);
  MPI_CHECK(MPI_Isend(buf, buf_size, MPI_UNSIGNED_CHAR,
                      peer_rank, 0, comm_, &send_requests.at(peer_rank)));
}

void MPICommunicator::WaitAllSend() {
  for (int peer_rank = 0; peer_rank < world_size_; peer_rank++) {
    if (send_requests.at(peer_rank) == MPI_REQUEST_NULL)
      continue;
    MPI_CHECK(MPI_Wait(&send_requests.at(peer_rank), MPI_STATUSES_IGNORE));
  }
}

int MPICommunicator::TestRecv(int peer_rank) {
  int flag = 0;
  MPI_CHECK(MPI_Test(&recv_requests.at(peer_rank), &flag, MPI_STATUSES_IGNORE));
//  if (flag) {
//    gpu_context_->MemcpyAsyncH2D(recv_bufs[peer_rank], recv_host_bufs[peer_rank], recv_size[peer_rank], recv_streams[peer_rank]);
//    gpu_context_->StreamSynchronize(recv_streams[peer_rank]);
//  }
  return flag;
}

void MPICommunicator::WaitRecv(int peer_rank) {
  MPI_CHECK(MPI_Wait(&recv_requests.at(peer_rank), MPI_STATUSES_IGNORE));
}

void MPICommunicator::WaitSend(int peer_rank) {
  MPI_CHECK(MPI_Wait(&send_requests.at(peer_rank), MPI_STATUSES_IGNORE));
}

void MPICommunicator::WaitAllRecv() {
  for (int peer_rank = 0; peer_rank < world_size_; peer_rank++) {
    if (send_requests.at(peer_rank) == MPI_REQUEST_NULL)
      continue;
    MPI_CHECK(MPI_Wait(&recv_requests.at(peer_rank), MPI_STATUSES_IGNORE));
  }
}

void *MPICommunicator::GetRemoteBuftoSend(int peer_rank) {
  throw std::runtime_error(
      "RemoteBuf primitive is not supported by MPICommunicator\n");
}

void *MPICommunicator::GetRemoteBuftoRecv(int peer_rank) {
  throw std::runtime_error(
      "RemoteBuf primitive is not supported by MPICommunicator\n");
}

void *MPICommunicator::GetRemoteBroadcastBuftoSend() {
  throw std::runtime_error(
      "RemoteBuf primitive is not supported by MPICommunicator\n");
}

void *MPICommunicator::GetRemoteBroadcastBuftoRecv(int peer_rank) {
  throw std::runtime_error(
      "RemoteBuf primitive is not supported by MPICommunicator\n");
}

} // namespace common
} // namespace cgx
