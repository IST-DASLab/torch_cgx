#pragma once

#include <mpi.h>
#include "communicator.h"
#include <unordered_map>
#include <vector>

namespace qmpi {
namespace common {

struct SHMCommunicator : public Communicator {
  SHMCommunicator(GPUContext* gpu_context) : Communicator(gpu_context) {}

  ~SHMCommunicator();

  virtual void Init(int world_size, void *ctx) override;
  virtual void ISend(void *buf, size_t buf_size, int peer_rank,
                     gpuStream_t stream) override;
  virtual void IRecv(void *buf, size_t buf_size, int peer_rank,
                     gpuStream_t stream) override;
  virtual void WaitAllSend() override;
  virtual void WaitAllRecv() override;
  virtual int TestRecv(int rank) override;

private:
  struct gpuEventSync {
    gpuEvent_t event;
    gpuIpcEventHandle_t eventHandle;
    MPI_Request request;
    unsigned char dummy;
  };

  struct shmBuffer {
    int shmSize;
    int shmOffset;
    void* hostMem;
    void* devHostMem;
  };

  struct RecvRequest {
    int recv_size;
    void* dest;
    gpuStream_t stream;
  };

  // Initialize send and receive resources.
  // Calls to sendInit and recvInit
  // must be separated with MPI_Barrier.
  void sendInit(shmBuffer* resource, int peer_rank, size_t shm_size);
  void recvInit(shmBuffer* resource, int peer_rank, size_t shm_size);
  // Initialize IPC primitives.
  void initEventSend(gpuEventSync* eventSync, int recv_rank,
                     MPI_Request* request);
  void initEventRecv(gpuEventSync* eventSync, int send_rank);

  static void freeBuffer(shmBuffer* buffer);
  void freeEventSync(gpuEventSync* eventSend);

  std::unordered_map<int, std::pair<shmBuffer, gpuEventSync>> send_resources;
  std::unordered_map<int, std::pair<shmBuffer, gpuEventSync>> recv_resources;
  std::unordered_map<int, RecvRequest> recv_requests;
  bool initialized_ = false;
  MPI_Comm comm_;
  int rank_;
  int world_size_;
};

} // namespace common
} // namespace qmpi

