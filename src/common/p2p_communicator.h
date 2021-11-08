#pragma once
#include "communicator.h"
#include <unordered_map>
#include <vector>
#include "buffer.h"

namespace qmpi {
namespace common {

struct P2PCommunicator : public Communicator {
  P2PCommunicator(GPUContext* gpu_context) : Communicator(gpu_context) {}


  virtual void Init(int world_size, void *ctx) override;
  virtual void ISend(void *buf, size_t buf_size, int peer_rank,
                     gpuStream_t stream) override;
  virtual void IRecv(void *buf, size_t buf_size, int peer_rank,
                     gpuStream_t stream) override;
  virtual void WaitSend(int rank) override;
  virtual void WaitRecv(int rank) override;
  virtual void WaitAllSend() override;
  virtual void WaitAllRecv() override;
  virtual int TestRecv(int rank) override;

private:
  struct CommData {
    CommData()
        : buf(nullptr), request(MPI_REQUEST_NULL) {}
    CommData(void* buf_)
        : buf(buf_), request(MPI_REQUEST_NULL) {}

    cudaEvent_t event;
    cudaIpcMemHandle_t memHandle;
    cudaIpcEventHandle_t eventHandle;
    MPI_Request request;
    void* buf;
    unsigned char dummy;

    void* target_buf;
    gpuStream_t stream;
    size_t buf_size;
  };

  std::unordered_map<int, CommData> send_comms;
  std::unordered_map<int, CommData> recv_comms;
  std::unique_ptr<PersistentBuffer> buffer_;
  bool initialized_ = false;
};

} // namespace common
} // namespace qmpi

