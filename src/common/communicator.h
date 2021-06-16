#pragma once
#include "gpu_context.h"
#include "common.h"

namespace qmpi {
namespace common {

struct Communicator {
  Communicator(GPUContext *gpu_context): gpu_context_(gpu_context){}
  virtual void Init(int world_size, void* ctx) = 0;
  virtual void ISend(void* buf, size_t buf_size, int peer_rank,
                     gpuStream_t stream) = 0;
  virtual void IRecv(void* buf, size_t buf_size, int peer_rank,
                     gpuStream_t stream) = 0;
  virtual void WaitAllSend() = 0;
  virtual void WaitAllRecv() = 0;
  virtual int TestRecv(int rank) = 0;
protected:
  GPUContext *gpu_context_;
};

} // namespace common
} // namespace qmpi