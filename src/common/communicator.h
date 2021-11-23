#pragma once
#include "gpu_context.h"
#include "common.h"
#include <mpi.h>

namespace cgx {
namespace common {

struct Communicator {
  enum CommunicatorType {
    MPI,
    SHM,
    P2P
  };
  Communicator(GPUContext *gpu_context): gpu_context_(gpu_context){}
  virtual void Init(int world_size, void* ctx) = 0;
  virtual void ISend(void* buf, size_t buf_size, int peer_rank,
                     gpuStream_t stream) = 0;
  virtual void IRecv(void* buf, size_t buf_size, int peer_rank,
                     gpuStream_t stream) = 0;
  virtual void WaitSend(int rank) = 0;
  virtual void WaitRecv(int rank) = 0;
  virtual void WaitAllSend() = 0;
  virtual void WaitAllRecv() = 0;
  virtual int TestRecv(int rank) = 0;
  virtual void* GetRemoteBuftoSend(int peer_rank) = 0;
  virtual void* GetRemoteBuftoRecv(int peer_rank) = 0;
  virtual void* GetRemoteBroadcastBuftoSend() = 0;
  virtual void* GetRemoteBroadcastBuftoRecv(int peer_rank) = 0;
  virtual void Barrier() {  MPI_CHECK(MPI_Barrier(comm_)); }
  CommunicatorType GetType() {return communicator_type_;}
protected:
  GPUContext *gpu_context_;
  MPI_Comm comm_;
  int rank_;
  int world_size_;
  CommunicatorType communicator_type_;
};

} // namespace common
} // namespace cgx