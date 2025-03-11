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
#include "common.h"
#include "gpu_context.h"
#include <mpi.h>


namespace cgx::common {

struct Communicator {
  enum CommunicatorType { MPI, SHM };
  Communicator(std::shared_ptr<GPUContext> gpu_context)
      : gpu_context_(gpu_context) {}
  virtual void Init(int world_size, void *ctx) = 0;
  virtual void ISend(void *buf, size_t buf_size, int peer_rank,
                     gpuStream_t stream) = 0;
  virtual void IRecv(void *buf, size_t buf_size, int peer_rank,
                     gpuStream_t stream) = 0;
  virtual void WaitSend(int rank) = 0;
  virtual void WaitRecv(int rank) = 0;
  virtual void WaitAllSend() = 0;
  virtual void WaitAllRecv() = 0;

  virtual int TestRecv(int rank) = 0;
  virtual void Barrier() { MPI_CHECK(MPI_Barrier(comm_)); }
  CommunicatorType GetType() { return communicator_type_; }

protected:
  std::shared_ptr<GPUContext> gpu_context_;
  MPI_Comm comm_;
  int rank_;
  int world_size_;
  CommunicatorType communicator_type_;
};

struct CommunicatorLocal : Communicator {
  CommunicatorLocal(std::shared_ptr<GPUContext> gpu_context)
      : Communicator(gpu_context) {}
  virtual void *GetRemoteBuftoSend(int peer_rank) = 0;
  virtual void *GetRemoteBuftoRecv(int peer_rank) = 0;
  virtual void *GetRemoteBroadcastBuftoSend() = 0;
  virtual void *GetRemoteBroadcastBuftoRecv(int peer_rank) = 0;
  virtual void CommitSend(int peer_rank, gpuStream_t stream) = 0;
  virtual int TestRemote(int peer_rank, gpuStream_t stream) = 0;
};

} // namespace cgx::common
