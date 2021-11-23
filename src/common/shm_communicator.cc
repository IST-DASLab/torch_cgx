#include "shm_communicator.h"
#include "shm_utils.h"
#include "compression/gpu_common.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define TRIV_CHECK(cmd)                                                        \
  do {                                                                         \
    int e = cmd;                                                               \
    if (e != 0) {                                                              \
      printf("Failed: Error %s:%d'\n", __FILE__, __LINE__);                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

namespace cgx {
namespace common {

SHMCommunicator::~SHMCommunicator() {
  // Can't deallocate buffers here, because at this point destuctor is called
  // CUDA driver is in the middle of deinitialization.
  for (auto &resource : send_resources) {
    if (resource.first == rank_)
      continue;
    freeEventSync(&resource.second.second);
    //    freeBuffer(&resource.second.first);
  }
  //  for (auto& resource: recv_resources) {
  //    freeBuffer(&resource.second.first);
  //  }
}

void SHMCommunicator::Init(int world_size, void *ctx) {
  if (initialized_) {
    for (auto &resource: send_resources) {
      resource.second.first.shmOffset = 0;
    }
    for (auto &resource: recv_resources) {
      resource.second.first.shmOffset = 0;
    }
    return;
  }
  comm_ = *(static_cast<MPI_Comm *>(ctx));
  MPI_CHECK(MPI_Comm_rank(comm_, &rank_));
  world_size_ = world_size;
  unsigned int fusion_size_mb =
      utils::GetIntEnvOrDefault(CGX_FUSION_BUFFER_SIZE_MB, FUSION_SIZE_DEFAULT_MB);
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
    send_requests.push_back(MPI_Request());
    initEventSend(&send_resource.second, peer_rank, &send_requests.back());
  }
  for (int peer_rank = 0; peer_rank < world_size; peer_rank++) {
    if (peer_rank == rank_)
      continue;
    auto &recv_resource = recv_resources[peer_rank];
    initEventRecv(&recv_resource.second, peer_rank);
  }
  MPI_Waitall(send_requests.size(), send_requests.data(), MPI_STATUSES_IGNORE);
  initialized_ = true;
}

void printDebug1(unsigned char* buf, int numel) {
  float* host_buf = new float[numel];
#if HAVE_CUDA
  cudaMemcpy(host_buf, buf, numel * sizeof(float), cudaMemcpyDeviceToHost);
  CUDA_CHECK(cudaGetLastError());
#elif HAVE_ROCM
  hipMemcpy(host_buf, buf, numel * sizeof(float), cudaMemcpyDeviceToHost);
#endif
  for (int i = 0; i < numel; i++) {
    std::cout << host_buf[i] << " ";
  }
  std::cout << std::endl;
}


void SHMCommunicator::ISend(void *buf, size_t buf_size, int peer_rank,
                            gpuStream_t stream) {
  auto &send_resource = send_resources[peer_rank];
  auto &shm_buf = send_resource.first;
  auto &eventSync = send_resource.second;
  // TODO: ensure that the size never overflows.
  //  We only allocated fusion_buffer_size per connection.
  //  In case of All-to-All and SRA reductions it works but may fail in case
  //  of other reduction schemes.
  assert(shm_buf.shmOffset + buf_size < shm_buf.shmSize);
  gpu_context_->MemcpyAsyncD2D(static_cast<char *>(shm_buf.devHostMem) + shm_buf.shmOffset, buf,
                               buf_size, stream);
  gpu_context_->EventRecord(eventSync.event, stream);
  // Notify peer that data transfer has started.
  MPI_CHECK(MPI_Isend((void *) &eventSync.dummy, 1, MPI_UNSIGNED_CHAR,
                      peer_rank, 0, comm_, &eventSync.request));
  shm_buf.shmOffset += buf_size;
}

void SHMCommunicator::IRecv(void *buf, size_t buf_size, int peer_rank,
                            gpuStream_t stream) {
  auto &recv_resource = recv_resources[peer_rank];
  auto &shm_buf = recv_resource.first;
  auto &eventSync = recv_resource.second;
  assert(shm_buf.shmOffset + buf_size < shm_buf.shmSize);
  // Non-blocking receive the notification about the start of the data transfer.
  MPI_CHECK(MPI_Irecv(&eventSync.dummy, 1, MPI_UNSIGNED_CHAR,
                      peer_rank, 0, comm_, &eventSync.request));
  auto &recv_request = recv_requests[peer_rank];
  recv_request.recv_size = buf_size;
  recv_request.dest = buf;
  recv_request.stream = stream;
}


int SHMCommunicator::TestRecv(int peer_rank) {
  auto &recv_resource = recv_resources[peer_rank];
  auto &shm_buf = recv_resource.first;
  auto &eventSync = recv_resource.second;
  int flag;
  // Check if notification has arrived.
  MPI_CHECK(MPI_Test(&eventSync.request, &flag, MPI_STATUSES_IGNORE));
  if (!flag)
    return 0;
  // Wait on stream till data transfer is finished.
  auto &recv_request = recv_requests[peer_rank];
  gpu_context_->StreamWaitEvent(recv_request.stream, eventSync.event);
  gpu_context_->MemcpyAsyncD2D(recv_request.dest,
                               static_cast<char *>(shm_buf.devHostMem)
                                   + shm_buf.shmOffset,
                               recv_request.recv_size,
                               recv_request.stream);
  shm_buf.shmOffset += recv_request.recv_size;
  return 1;
}

void SHMCommunicator::WaitAllSend() {
  for (auto &resource : send_resources) {
    auto &eventSync = resource.second.second;
    MPI_Wait(&eventSync.request, MPI_STATUSES_IGNORE);
  }
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
  while (!TestRecv(rank)){}
}

void SHMCommunicator::WaitSend(int rank) {
  auto eventSync = send_resources.at(rank).second;
  MPI_Wait(&eventSync.request, MPI_STATUSES_IGNORE);
}

void * SHMCommunicator::GetRemoteBuftoSend(int peer_rank) {
  auto &send_resource = send_resources.at(peer_rank);
  auto &shm_buf = send_resource.first;
  return shm_buf.devHostMem;
}

void * SHMCommunicator::GetRemoteBuftoRecv(int peer_rank) {
  auto &recv_resource = recv_resources.at(peer_rank);
  auto &shm_buf = recv_resource.first;
  return shm_buf.devHostMem;
}

void* SHMCommunicator::GetRemoteBroadcastBuftoSend() {
  auto &send_resource = send_resources.at(rank_);
  auto &shm_buf = send_resource.first;
  return shm_buf.devHostMem;
}

void* SHMCommunicator::GetRemoteBroadcastBuftoRecv(int peer_rank) {
  auto &recv_resource = broadcast_recv_resources.at(peer_rank);
  auto &shm_buf = recv_resource.first;
  return shm_buf.devHostMem;
}

void SHMCommunicator::sendInit(shmBuffer *buffer,
                               int peer_rank,
                               size_t shm_size) {
  char shmName[utils::MAX_SHM_NAME_LEN];
  buffer->shmSize = shm_size;
  buffer->shmOffset = 0;
  sprintf(shmName, "cgx-shm-send-%d-%d", rank_, peer_rank);
  TRIV_CHECK(utils::shmOpen(shmName,
                            buffer->shmSize,
                            (void **) &buffer->hostMem,
                            (void **) &buffer->devHostMem,
                            1));
}

void SHMCommunicator::recvInit(shmBuffer *buffer,
                               int peer_rank,
                               size_t shm_size, bool broadcast) {
  char shmName[utils::MAX_SHM_NAME_LEN];
  buffer->shmSize = shm_size;
  buffer->shmOffset = 0;
  sprintf(shmName, "cgx-shm-send-%d-%d", peer_rank, broadcast? peer_rank: rank_);
  TRIV_CHECK(utils::shmOpen(shmName,
                            buffer->shmSize,
                            (void **) &buffer->hostMem,
                            (void **) &buffer->devHostMem,
                            0));
  // in case of broadcast we cleanup later.
  if (!broadcast)
    TRIV_CHECK(utils::shmUnlink(shmName););
}
void SHMCommunicator::cleanupBroadcast() {
  char shmName[utils::MAX_SHM_NAME_LEN];
  sprintf(shmName, "cgx-shm-send-%d-%d", rank_, rank_);
  TRIV_CHECK(utils::shmUnlink(shmName););
}

void SHMCommunicator::initEventSend(gpuEventSync *eventSync,
                                    int recv_rank, MPI_Request *request) {
  gpu_context_->EventCreate(&eventSync->event);
  gpu_context_->IpcGetEventHandle(&eventSync->eventHandle, eventSync->event);
  MPI_CHECK(MPI_Isend((void *) (&eventSync->eventHandle),
                      sizeof(eventSync->eventHandle), MPI_UNSIGNED_CHAR,
                      recv_rank, 0, comm_, request));
}

void SHMCommunicator::initEventRecv(gpuEventSync *eventSync,
                                    int send_rank) {
  MPI_CHECK(MPI_Recv((void *) (&eventSync->eventHandle),
                     sizeof(eventSync->eventHandle), MPI_UNSIGNED_CHAR,
                     send_rank, 0, comm_, MPI_STATUSES_IGNORE));
  gpu_context_->IpcOpenEventHandle(&eventSync->event, eventSync->eventHandle);
  eventSync->request = MPI_Request();
}

void SHMCommunicator::freeBuffer(shmBuffer *buffer) {
  TRIV_CHECK(utils::shmClose(buffer->hostMem,
                             buffer->devHostMem,
                             buffer->shmSize));
}

void SHMCommunicator::freeEventSync(gpuEventSync *eventSend) {
  gpu_context_->EventDestroy(eventSend->event);
}

} // namespace common
} // namespace horovod
