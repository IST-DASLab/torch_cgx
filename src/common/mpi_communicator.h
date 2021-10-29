#pragma once
#include <mpi.h>
#include "communicator.h"

namespace qmpi {
namespace common {

struct MPICommunicator : public Communicator {
  MPICommunicator(GPUContext *gpu_context) : Communicator(gpu_context) {}
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
protected:
  std::vector<MPI_Request> send_requests;
  std::vector<MPI_Request> recv_requests;
  MPI_Comm comm_;
  int rank_;
  int world_size_;
};

} // namespace common
} // namespace qmpi