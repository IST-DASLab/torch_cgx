/*************************************************************************
* Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
*
* Modifications copyright (C) 2022, IST Austria.
************************************************************************/

#include "shm_communicator.h"
#include "compression/gpu_common.h"
#include "shm_utils.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <fcntl.h>

#define TRIV_CHECK(cmd)                                                        \
  do {                                                                         \
    int e = cmd;                                                               \
    if (e != 0) {                                                              \
      printf("Failed: %s:%d: %s\n", __FILE__, __LINE__, strerror(errno));      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

namespace cgx {
namespace common {

const char *sendSemFmt = "/cgx-sem-send-%d-%d";
const char *recvSemFmt = "/cgx-sem-recv-%d-%d";

SHMCommunicator::~SHMCommunicator() {
  // Can't deallocate buffers here, because at this point destuctor is called
  // CUDA driver is in the middle of deinitialization.
  if (!initialized_)
    return;
  for (auto &resource : send_resources) {
    if (resource.first == rank_)
      continue;
    freeEventSync(&resource.second.second, true);
    //    freeBuffer(&resource.second.first);
  }
  for (auto &resource : recv_resources) {
    if (resource.first == rank_)
      continue;
    freeEventSync(&resource.second.second, false);
    //    freeBuffer(&resource.second.first);
  }
  MPI_Barrier(comm_);
  for (auto &resource : send_resources) {
    if (resource.first == rank_)
      continue;
    unlinkSem(resource.first);
  }
}

void SHMCommunicator::Init(int world_size, void *ctx) {
  if (initialized_) {
    for (auto &resource : send_resources) {
      resource.second.first.shmOffset = 0;
    }
    for (auto &resource : recv_resources) {
      resource.second.first.shmOffset = 0;
    }
    return;
  }
  comm_ = *(static_cast<MPI_Comm *>(ctx));
  MPI_CHECK(MPI_Comm_rank(comm_, &rank_));
  world_size_ = world_size;
  unsigned int fusion_size_mb =
      utils::GetIntEnvOrDefault(FUSION_BUFFER_SIZE_MB, FUSION_SIZE_DEFAULT_MB);
  unsigned int buf_size =
      std::max(fusion_size_mb * 1024 * 1024, MIN_FUSION_SIZE);
  // Initialize shared memory buffers.
  for (int peer_rank = 0; peer_rank < world_size; peer_rank++) {
    auto &send_resource = send_resources[peer_rank];
    sendInit(&send_resource.first, peer_rank, buf_size);
  }
  MPI_Barrier(comm_);
  for (int peer_rank = 0; peer_rank < world_size; peer_rank++) {
    if (peer_rank == rank_)
      continue;
    auto &recv_resource = recv_resources[peer_rank];
    recvInit(&recv_resource.first, peer_rank, buf_size, false);
    auto &broadcast_recv_resource = broadcast_recv_resources[peer_rank];
    recvInit(&broadcast_recv_resource.first, peer_rank, buf_size, true);
  }
  MPI_Barrier(comm_);
  cleanupBroadcast();
  // Initialize IPC primitives.
  std::vector<MPI_Request> send_requests;
  int count = 0;
  for (int peer_rank = 0; peer_rank < world_size; peer_rank++) {
    if (peer_rank == rank_)
      continue;
    auto &send_resource = send_resources[peer_rank];
    auto &recv_resource = recv_resources[peer_rank];
    initEventSend(send_resource.second, recv_resource.second, peer_rank,
                  send_requests);
  }
  for (int peer_rank = 0; peer_rank < world_size; peer_rank++) {
    if (peer_rank == rank_)
      continue;
    auto &send_resource = send_resources[peer_rank];
    auto &recv_resource = recv_resources[peer_rank];
    initEventRecv(send_resource.second, recv_resource.second, peer_rank);
  }
  MPI_Waitall(send_requests.size(), send_requests.data(), MPI_STATUSES_IGNORE);
  MPI_Barrier(comm_);
  initialized_ = true;
}
static int counter = 0;
void SHMCommunicator::ISend(void *buf, size_t buf_size, int peer_rank,
                            gpuStream_t stream) {
  auto &send_resource = send_resources[peer_rank];
  auto &shm_buf = send_resource.first;
  auto &eventSync = send_resource.second;
  eventSync.stream = stream;
  // Performance optimization for SRA.
  // We can postpone waiting for recv ACK of the previous send in allreduce and
  // send new data into a piece of shared memory (shifted by shm_buf.shmOffset).
  if (shm_buf.shmOffset == 0) {
    TRIV_CHECK(sem_wait(eventSync.recvSem));
    gpu_context_->StreamWaitEvent(eventSync.stream, eventSync.recvEvent);
  }

  assert(shm_buf.shmOffset + buf_size < shm_buf.shmSize);
  gpu_context_->MemcpyAsyncD2D(static_cast<char *>(shm_buf.devHostMem) +
                                   shm_buf.shmOffset,
                               buf, buf_size, stream);
  gpu_context_->EventRecord(eventSync.sentEvent, stream);

  if (shm_buf.shmOffset > 0) {
    // Postponed wait for the previous recv ACK.
    TRIV_CHECK(sem_wait(eventSync.recvSem));
  }
  // Notify peer that send event has been recorded.
  sem_post(eventSync.sentSem);
  shm_buf.shmOffset += buf_size;
}

void SHMCommunicator::IRecv(void *buf, size_t buf_size, int peer_rank,
                            gpuStream_t stream) {
  auto &recv_resource = recv_resources[peer_rank];
  auto &shm_buf = recv_resource.first;
  assert(shm_buf.shmOffset + buf_size < shm_buf.shmSize);
  auto &eventSync = recv_resource.second;
  eventSync.stream = stream;
  auto &recv_request = recv_requests[peer_rank];
  recv_request.recv_size = buf_size;
  recv_request.dest = buf;
}

int SHMCommunicator::TestRecv(int peer_rank) {
  auto &recv_resource = recv_resources[peer_rank];
  auto &shm_buf = recv_resource.first;
  auto &eventSync = recv_resource.second;
  // Wait notification of sent data.
  // Need to wait till the send event from peer is recorded.
  int ret = sem_trywait(eventSync.sentSem);
  if (ret < 0) {
    if (errno == EAGAIN) {
      return 0;
    } else {
      TRIV_CHECK(ret);
    }
  }
  // Wait on stream till data transfer is finished.
  gpu_context_->StreamWaitEvent(eventSync.stream, eventSync.sentEvent);
  auto &recv_request = recv_requests[peer_rank];
  gpu_context_->MemcpyAsyncD2D(recv_request.dest,
                               static_cast<char *>(shm_buf.devHostMem) +
                                   shm_buf.shmOffset,
                               recv_request.recv_size, eventSync.stream);
  gpu_context_->EventRecord(eventSync.recvEvent, eventSync.stream);
  TRIV_CHECK(sem_post(eventSync.recvSem));
  shm_buf.shmOffset += recv_request.recv_size;
  counter++;
  return 1;
}

void SHMCommunicator::WaitAllSend() {}

void SHMCommunicator::WaitSend(int rank) {
  send_resources.at(rank).first.shmOffset = 0;
}

void SHMCommunicator::WaitAllRecv() {
  std::vector<int> nodes;
  for (int peer_rank = 0; peer_rank < world_size_; peer_rank++) {
    if (peer_rank == rank_)
      continue;
    nodes.push_back(peer_rank);
  }
  while (nodes.size() > 0) {
    for (int i = 0; i < nodes.size(); i++) {
      if (TestRecv(nodes[i])) {
        nodes.erase(nodes.begin() + i);
      }
    }
  }
}

void SHMCommunicator::WaitRecv(int rank) {
  while (!TestRecv(rank)) {
  }
  recv_resources.at(rank).first.shmOffset = 0;
}

void *SHMCommunicator::GetRemoteBuftoSend(int peer_rank) {
  auto &send_resource = send_resources.at(peer_rank);
  auto &shm_buf = send_resource.first;
  return shm_buf.devHostMem;
}

void *SHMCommunicator::GetRemoteBuftoRecv(int peer_rank) {
  auto &recv_resource = recv_resources.at(peer_rank);
  auto &shm_buf = recv_resource.first;
  return shm_buf.devHostMem;
}

void *SHMCommunicator::GetRemoteBroadcastBuftoSend() {
  auto &send_resource = send_resources.at(rank_);
  auto &shm_buf = send_resource.first;
  return shm_buf.devHostMem;
}

void *SHMCommunicator::GetRemoteBroadcastBuftoRecv(int peer_rank) {
  auto &recv_resource = broadcast_recv_resources.at(peer_rank);
  auto &shm_buf = recv_resource.first;
  return shm_buf.devHostMem;
}

void SHMCommunicator::CommitSend(int peer_rank, gpuStream_t stream) {
  auto &send_resource = send_resources.at(rank_);
  auto &eventSync = send_resource.second;
  gpu_context_->EventRecord(eventSync.sentEvent, stream);
  TRIV_CHECK(sem_post(eventSync.sentSem));
}

int SHMCommunicator::TestRemote(int peer_rank, gpuStream_t stream) { return 0; }

void SHMCommunicator::sendInit(shmBuffer *buffer, int peer_rank,
                               size_t shm_size) {
  char shmName[utils::MAX_SHM_NAME_LEN];
  buffer->shmSize = shm_size;
  buffer->shmOffset = 0;
  sprintf(shmName, "cgx-shm-send-%d-%d", rank_, peer_rank);
  TRIV_CHECK(utils::shmOpen(shmName, buffer->shmSize, (void **)&buffer->hostMem,
                            (void **)&buffer->devHostMem, 1));
}

void SHMCommunicator::recvInit(shmBuffer *buffer, int peer_rank,
                               size_t shm_size, bool broadcast) {
  char shmName[utils::MAX_SHM_NAME_LEN];
  buffer->shmSize = shm_size;
  buffer->shmOffset = 0;
  sprintf(shmName, "cgx-shm-send-%d-%d", peer_rank,
          broadcast ? peer_rank : rank_);
  TRIV_CHECK(utils::shmOpen(shmName, buffer->shmSize, (void **)&buffer->hostMem,
                            (void **)&buffer->devHostMem, 0));
  // in case of broadcast we cleanup later.
  if (!broadcast)
    TRIV_CHECK(utils::shmUnlink(shmName););
}

void SHMCommunicator::cleanupBroadcast() {
  char shmName[utils::MAX_SHM_NAME_LEN];
  sprintf(shmName, "cgx-shm-send-%d-%d", rank_, rank_);
  TRIV_CHECK(utils::shmUnlink(shmName););
}

void SHMCommunicator::initEventSend(gpuEventSync &sendEventSync,
                                    gpuEventSync &recvEventSync, int peer_rank,
                                    std::vector<MPI_Request> &send_requests) {
  char semName[utils::MAX_SHM_NAME_LEN];
  sprintf(semName, sendSemFmt, rank_, peer_rank);
  int ret = sem_unlink(semName);
  if (ret < 0 and errno != ENOENT) {
    TRIV_CHECK(ret);
  }
  sendEventSync.sentSem = sem_open(semName, O_CREAT | O_EXCL, 0644, 0);
  TRIV_CHECK(!sendEventSync.sentSem);

  sprintf(semName, recvSemFmt, peer_rank, rank_);
  ret = sem_unlink(semName);
  if (ret < 0 and errno != ENOENT) {
    TRIV_CHECK(ret);
  }
  recvEventSync.recvSem = sem_open(semName, O_CREAT | O_EXCL, 0644, 1);
  TRIV_CHECK(!recvEventSync.recvSem);

  gpu_context_->EventCreate(&sendEventSync.sentEvent);
  gpu_context_->IpcGetEventHandle(&sendEventSync.sendEventHandle,
                                  sendEventSync.sentEvent);
  send_requests.push_back(MPI_Request());
  MPI_CHECK(MPI_Isend((void *)(&sendEventSync.sendEventHandle),
                      sizeof(sendEventSync.sendEventHandle), MPI_UNSIGNED_CHAR,
                      peer_rank, 0, comm_, &send_requests.back()));

  gpu_context_->EventCreate(&recvEventSync.recvEvent);
  gpu_context_->IpcGetEventHandle(&recvEventSync.recvEventHandle,
                                  recvEventSync.recvEvent);
  send_requests.push_back(MPI_Request());
  MPI_CHECK(MPI_Isend((void *)(&recvEventSync.recvEventHandle),
                      sizeof(recvEventSync.recvEventHandle), MPI_UNSIGNED_CHAR,
                      peer_rank, 0, comm_, &send_requests.back()));
}

void SHMCommunicator::initEventRecv(gpuEventSync &sendEventSync,
                                    gpuEventSync &recvEventSync,
                                    int peer_rank) {
  MPI_CHECK(MPI_Recv((void *)(&recvEventSync.sendEventHandle),
                     sizeof(recvEventSync.sendEventHandle), MPI_UNSIGNED_CHAR,
                     peer_rank, 0, comm_, MPI_STATUSES_IGNORE));
  gpu_context_->IpcOpenEventHandle(&recvEventSync.sentEvent,
                                   recvEventSync.sendEventHandle);

  MPI_CHECK(MPI_Recv((void *)(&sendEventSync.recvEventHandle),
                     sizeof(sendEventSync.recvEventHandle), MPI_UNSIGNED_CHAR,
                     peer_rank, 0, comm_, MPI_STATUSES_IGNORE));
  gpu_context_->IpcOpenEventHandle(&sendEventSync.recvEvent,
                                   sendEventSync.recvEventHandle);

  char semName[utils::MAX_SHM_NAME_LEN];
  sprintf(semName, sendSemFmt, peer_rank, rank_);
  recvEventSync.sentSem = sem_open(semName, O_RDWR);
  TRIV_CHECK(!recvEventSync.sentSem);

  sprintf(semName, recvSemFmt, rank_, peer_rank);
  sendEventSync.recvSem = sem_open(semName, O_RDWR);
  TRIV_CHECK(!sendEventSync.recvSem);
}

void SHMCommunicator::freeBuffer(shmBuffer *buffer) {
  TRIV_CHECK(
      utils::shmClose(buffer->hostMem, buffer->devHostMem, buffer->shmSize));
}

void SHMCommunicator::freeEventSync(gpuEventSync *eventSync, bool send) {
  if (send)
    gpu_context_->EventDestroy(eventSync->sentEvent);
  else
    gpu_context_->EventDestroy(eventSync->recvEvent);

  TRIV_CHECK(sem_close(eventSync->sentSem));
  TRIV_CHECK(sem_close(eventSync->recvSem));
}

void SHMCommunicator::unlinkSem(int peer_rank) {
  char semName[utils::MAX_SHM_NAME_LEN];
  sprintf(semName, sendSemFmt, rank_, peer_rank);
  TRIV_CHECK(sem_unlink(semName));
  sprintf(semName, recvSemFmt, rank_, peer_rank);
  TRIV_CHECK(sem_unlink(semName));
}

} // namespace common
} // namespace cgx
