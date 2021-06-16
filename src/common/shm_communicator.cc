#include "shm_communicator.h"
#include "shm_utils.h"
#include "compression/cuda_common.h"
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

namespace qmpi {
namespace common {

SHMCommunicator::~SHMCommunicator() {
  // Can't deallocate buffers here, because at this point destuctor is called
  // CUDA driver is in the middle of deinitialization.
  for (auto &resource : send_resources) {
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
      utils::GetIntEnvOrDefault(FUSION_BUFFER_SIZE_MB, FUSION_SIZE_DEFAULT_MB);
  unsigned int buf_size =
      std::max(fusion_size_mb * 1024 * 1024, MIN_FUSION_SIZE);
  // Initialize shared memory buffers.
  for (int peer_rank = 0; peer_rank < world_size; peer_rank++) {
    if (peer_rank == rank_)
      continue;
    auto &send_resource = send_resources[peer_rank];
    sendInit(&send_resource.first, peer_rank, buf_size);
  }
  MPI_Barrier(comm_);
  for (int peer_rank = 0; peer_rank < world_size; peer_rank++) {
    if (peer_rank == rank_)
      continue;
    auto &recv_resource = recv_resources[peer_rank];
    recvInit(&recv_resource.first, peer_rank, buf_size);
  }

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

#include "compression/cuda_common.h"

void printDebug1(unsigned char* buf, int numel) {
  float* host_buf = new float[numel];
  cudaMemcpy(host_buf, buf, numel * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < numel; i++) {
    std::cout << host_buf[i] << " ";
  }
  std::cout << std::endl;
  CUDA_CHECK(cudaGetLastError());
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
//  CUDA_CHECK(cudaMemcpyAsync(
//      static_cast<char *>(shm_buf.devHostMem) + shm_buf.shmOffset, buf,
//      buf_size, cudaMemcpyDeviceToDevice, stream));
//  CUDA_CHECK(cudaMemcpyAsync(
//      static_cast<char *>(shm_buf.hostMem) + shm_buf.shmOffset, buf,
//      buf_size, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpy(
      static_cast<char *>(shm_buf.devHostMem) + shm_buf.shmOffset, buf,
      buf_size, cudaMemcpyDeviceToDevice));
//  CUDA_CHECK(cudaEventRecord(eventSync.event, stream));
//  CUDA_CHECK(cudaStreamSynchronize(stream));
//  if (rank_ == 0 and shm_buf.shmOffset == 0) {
//    std::cout << "Sending ";
//    printDebug1(static_cast<unsigned char*>(buf), 8);
//
//    float* buf_f = static_cast<float*>(shm_buf.hostMem);
//    float* host_buf_f = static_cast<float*>((void*)host_buf);
//    std::cout << "Sent side. Host shm buffer: ";
//    for (int i = 0; i < 8; i++) {
//      std::cout << buf_f[i] << " ";
//    }
//    std::cout << std::endl;
//  }
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
//  CUDA_CHECK(cudaStreamWaitEvent(recv_request.stream, eventSync.event, 0));
  CUDA_CHECK(cudaStreamSynchronize(recv_request.stream));
//  if (rank_ == 2)
//    printDebug((unsigned char*) shm_buf.devHostMem, recv_request.recv_size / sizeof(float));
//  if (rank_ == 1 and shm_buf.shmOffset == 0) {
//    float* buf = static_cast<float*>(shm_buf.hostMem);
//    std::cout << "Recv side. Host shm buffer: ";
//    for (int i = 0; i < 8; i++) {
//      std::cout << buf[i] << " ";
//    }
//    std::cout << std::endl;
//  }
//  CUDA_CHECK(cudaMemcpyAsync(recv_request.dest,
//                             static_cast<char *>(shm_buf.devHostMem)
//                                 + shm_buf.shmOffset,
//                             recv_request.recv_size,
//                             cudaMemcpyDeviceToDevice,
//                             recv_request.stream));
  CUDA_CHECK(cudaMemcpyAsync(recv_request.dest,
                             static_cast<char *>(shm_buf.hostMem)
                                 + shm_buf.shmOffset,
                             recv_request.recv_size,
                             cudaMemcpyHostToDevice,
                             recv_request.stream));
//  MPI_Request request;
//  char a;
//  MPI_CHECK(MPI_Isend((void *) &a, 1, MPI_UNSIGNED_CHAR,
//                      peer_rank, 0, comm_, &request));
//  MPI_Wait(&request, MPI_STATUSES_IGNORE);
  shm_buf.shmOffset += recv_request.recv_size;
  return 1;
}

void SHMCommunicator::WaitAllSend() {
  for (auto &resource : send_resources) {
    auto &eventSync = resource.second.second;
    MPI_Wait(&eventSync.request, MPI_STATUSES_IGNORE);
//    char a;
//    MPI_CHECK(MPI_Recv(&a, 1, MPI_UNSIGNED_CHAR, resource.first, 0, comm_, MPI_STATUSES_IGNORE));
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

void SHMCommunicator::sendInit(shmBuffer *buffer,
                               int peer_rank,
                               size_t shm_size) {
  char shmName[utils::MAX_SHM_NAME_LEN];
  buffer->shmSize = shm_size;
  buffer->shmOffset = 0;
  sprintf(shmName, "qmpi-shm-send-%d-%d", rank_, peer_rank);
  TRIV_CHECK(utils::shmOpen(shmName,
                            buffer->shmSize,
                            (void **) &buffer->hostMem,
                            (void **) &buffer->devHostMem,
                            1));
}

void SHMCommunicator::recvInit(shmBuffer *buffer,
                               int peer_rank,
                               size_t shm_size) {
  char shmName[utils::MAX_SHM_NAME_LEN];
  buffer->shmSize = shm_size;
  buffer->shmOffset = 0;
  sprintf(shmName, "qmpi-shm-send-%d-%d", peer_rank, rank_);
  TRIV_CHECK(utils::shmOpen(shmName,
                            buffer->shmSize,
                            (void **) &buffer->hostMem,
                            (void **) &buffer->devHostMem,
                            0));
  TRIV_CHECK(utils::shmUnlink(shmName));
}

void SHMCommunicator::initEventSend(cudaEventSync *eventSync,
                                    int recv_rank, MPI_Request *request) {
  CUDA_CHECK(cudaEventCreateWithFlags(
      &eventSync->event, cudaEventDisableTiming | cudaEventInterprocess));
  CUDA_CHECK(cudaIpcGetEventHandle(
      (cudaIpcEventHandle_t * ) & eventSync->eventHandle, eventSync->event));
  MPI_CHECK(MPI_Isend((void *) (&eventSync->eventHandle),
                      sizeof(eventSync->eventHandle), MPI_UNSIGNED_CHAR,
                      recv_rank, 0, comm_, request));
}

void SHMCommunicator::initEventRecv(cudaEventSync *eventSync,
                                    int send_rank) {
  MPI_CHECK(MPI_Recv((void *) (&eventSync->eventHandle),
                     sizeof(eventSync->eventHandle), MPI_UNSIGNED_CHAR,
                     send_rank, 0, comm_, MPI_STATUSES_IGNORE));
  CUDA_CHECK(cudaIpcOpenEventHandle(&eventSync->event, eventSync->eventHandle));
  eventSync->request = MPI_Request();
}

void SHMCommunicator::freeBuffer(shmBuffer *buffer) {
  TRIV_CHECK(utils::shmClose(buffer->hostMem,
                             buffer->devHostMem,
                             buffer->shmSize));
}

void SHMCommunicator::freeEventSync(cudaEventSync *eventSend) {
  cudaEventDestroy(eventSend->event);
}

} // namespace common
} // namespace horovod
