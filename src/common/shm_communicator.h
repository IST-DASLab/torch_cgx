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
#include <semaphore.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cgx::common {
struct SHMCommunicator : public CommunicatorLocal {
  explicit SHMCommunicator(std::shared_ptr<GPUContext> gpu_context)
      : CommunicatorLocal(std::move(gpu_context)) {
    communicator_type_ = CommunicatorType::SHM;
  }

  ~SHMCommunicator();

  void Init(int world_size, void *ctx) override;
  void ISend(void *buf, size_t buf_size, int peer_rank,
             gpuStream_t stream) override;
  void IRecv(void *buf, size_t buf_size, int peer_rank,
             gpuStream_t stream) override;
  void WaitSend(int rank) override;
  void WaitRecv(int rank) override;
  void WaitAllSend() override;
  void WaitAllRecv() override;
  int TestRecv(int rank) override;
  void *GetRemoteBuftoSend(int peer_rank) override;
  void *GetRemoteBuftoRecv(int peer_rank) override;
  void *GetRemoteBroadcastBuftoSend() override;
  void *GetRemoteBroadcastBuftoRecv(int peer_rank) override;
  void CommitSend(int peer_rank, gpuStream_t stream) override;
  int TestRemote(int peer_rank, gpuStream_t stream) override;

private:
  struct gpuEventSync {
    sem_t *sentSem;
    gpuEvent_t sentEvent;
    gpuIpcEventHandle_t sendEventHandle;

    sem_t *recvSem;
    gpuEvent_t recvEvent;
    gpuIpcEventHandle_t recvEventHandle;

    gpuStream_t stream;
  };

  struct shmBuffer {
    int shmSize;
    int shmOffset;
    void *hostMem;
    void *devHostMem;
  };

  struct RecvRequest {
    int recv_size;
    void *dest;
  };

  // Initialize send and receive resources.
  // Calls to sendInit and recvInit
  // must be separated with MPI_Barrier.
  void sendInit(shmBuffer *resource, int peer_rank, size_t shm_size);
  void recvInit(shmBuffer *resource, int peer_rank, size_t shm_size,
                bool broadcast);
  void cleanupBroadcast();
  // Initialize IPC primitives.
  void initEventSend(gpuEventSync &sendEventSync, gpuEventSync &recvEventSync,
                     int peer_rank, std::vector<MPI_Request> &send_requests);
  void initEventRecv(gpuEventSync &sendEventSync, gpuEventSync &recvEventSync,int peer_rank);

  static void freeBuffer(shmBuffer *buffer);
  void freeEventSync(gpuEventSync *eventSend, bool send);
  void unlinkSem(int peer_rank);

  std::unordered_map<int, std::pair<shmBuffer, gpuEventSync>> send_resources;
  std::unordered_map<int, std::pair<shmBuffer, gpuEventSync>> recv_resources;
  std::unordered_map<int, std::pair<shmBuffer, gpuEventSync>>
      broadcast_recv_resources;
  std::unordered_map<int, RecvRequest> recv_requests;
  bool initialized_ = false;
};

} // namespace cgx::common
