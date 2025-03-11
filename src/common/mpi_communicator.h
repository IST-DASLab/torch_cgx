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
#include "communicator.h"
#include <utility>

namespace cgx::common {

struct MPICommunicator : public Communicator {
  explicit MPICommunicator(std::shared_ptr<GPUContext> gpu_context) : Communicator(std::move(gpu_context)){
    communicator_type_ = CommunicatorType::MPI;
  }
  void Init(int world_size, void *ctx) override;
  void ISend(void *buf, size_t buf_size, int peer_rank,
             gpuStream_t stream) override;
  void IRecv(void *buf, size_t buf_size, int peer_rank,
             gpuStream_t stream) override;
  void WaitSend(int peer_rank) override;
  void WaitRecv(int peer_rank) override;
  void WaitAllSend() override;
  void WaitAllRecv() override;
  int TestRecv(int rank) override;
protected:
  std::vector<MPI_Request> send_requests;
  std::vector<MPI_Request> recv_requests;
};

} // namespace cgx::common
