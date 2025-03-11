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

#include "mpi_communicator.h"

namespace cgx::common {

void MPICommunicator::Init(int world_size, void *ctx) {
  if (recv_requests.empty()) {
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
  MPI_CHECK(MPI_Wait(&send_requests.at(peer_rank), MPI_STATUSES_IGNORE));
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

} // namespace cgx::common

