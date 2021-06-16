#include "mpi_communicator.h"

namespace qmpi {
namespace common {

void MPICommunicator::Init(int world_size, void* ctx) {
  if (recv_requests.size() == 0) {
    for (int i = 0; i < world_size; i++)
      recv_requests.push_back(MPI_Request());
  }
  comm_ = *(static_cast<MPI_Comm *>(ctx));
  MPI_CHECK(MPI_Comm_rank(comm_, &rank_));
  world_size_ = world_size;
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
  send_requests.push_back(MPI_Request());
  MPI_CHECK(MPI_Isend(buf, buf_size, MPI_UNSIGNED_CHAR,
                      peer_rank, 0, comm_, &send_requests.back()));
}

void MPICommunicator::WaitAllSend() {
  MPI_CHECK(MPI_Waitall((int) send_requests.size(), send_requests.data(),
                        MPI_STATUSES_IGNORE));
  send_requests.clear();
}

int MPICommunicator::TestRecv(int peer_rank) {
  int flag = 0;
  MPI_CHECK(MPI_Test(&recv_requests.at(peer_rank), &flag, MPI_STATUSES_IGNORE));
  return flag;
}

void MPICommunicator::WaitAllRecv() {
  for (int peer_rank = 0; peer_rank < world_size_; peer_rank++){
    if (peer_rank == rank_)
      continue;
    MPI_CHECK(MPI_Wait(&recv_requests.at(peer_rank), MPI_STATUSES_IGNORE));
  }
}

} // namespace common
} // namespace qmpi
