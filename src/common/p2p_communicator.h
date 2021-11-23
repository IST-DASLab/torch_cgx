#pragma once
#include "communicator.h"
#include <unordered_map>
#include <vector>
#include "buffer.h"

namespace cgx {
namespace common {

struct P2PCommunicator : public Communicator {
  P2PCommunicator(GPUContext* gpu_context) : Communicator(gpu_context){
    communicator_type_ = CommunicatorType::P2P;
  }

  // The buffers allocated here are used for the life-time of the app,
  // so they don't need to be freed, they will be initialized with the
  // the gpu Context.
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
  virtual void* GetRemoteBuftoSend(int peer_rank) override;
  virtual void* GetRemoteBuftoRecv(int peer_rank) override;
  virtual void* GetRemoteBroadcastBuftoSend() override;
  virtual void* GetRemoteBroadcastBuftoRecv(int peer_rank) override;

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
  std::unordered_map<int, CommData> p2p_recv_comms;
  std::unordered_map<int, CommData> broadcast_recv_comms;
  std::unique_ptr<PersistentBuffer> buffer_;
  bool initialized_ = false;
};

} // namespace common
} // namespace cgx

