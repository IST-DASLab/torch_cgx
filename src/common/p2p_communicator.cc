#include "p2p_communicator.h"
#include "utils.h"
#include "compression/gpu_common.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace cgx {
namespace common {
void P2PCommunicator::Init(int world_size, void *ctx) {
  if (initialized_)
    return;
  comm_ = *(static_cast<MPI_Comm *>(ctx));
  world_size_ = world_size;
  cudaGetDevice(&rank_);
  unsigned int fusion_size_mb =
      utils::GetIntEnvOrDefault(CGX_FUSION_BUFFER_SIZE_MB, FUSION_SIZE_DEFAULT_MB);
  unsigned int buf_size =
      std::max(fusion_size_mb * 1024 * 1024, MIN_FUSION_SIZE);
  buffer_ = std::make_unique<PersistentBuffer>(buf_size * world_size);
  unsigned char *bufs = static_cast<unsigned char*>(buffer_->RawPointer());

  for (int peer_rank = 0; peer_rank < world_size; peer_rank++) {
    send_comms.emplace(peer_rank, CommData(bufs));
    if (peer_rank != rank_) {
      int canAccessPeer;
      CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessPeer, rank_, peer_rank));
      if (!canAccessPeer)
        throw std::runtime_error("P2P Communicator can not be enabled");
//    send_comms.emplace(peer_rank, CommData());
//    p2p_recv_comms.emplace(peer_rank, CommData(bufs));
      p2p_recv_comms.emplace(peer_rank, CommData());
      broadcast_recv_comms.emplace(peer_rank, CommData());
    }
    bufs += buf_size;
  }
  // Create p2p mem handles
  std::vector<MPI_Request> send_requests;
  for (int peer_rank = 0; peer_rank < world_size; peer_rank++) {
    if (peer_rank == rank_)
      continue;
    CUDA_CHECK(cudaDeviceEnablePeerAccess(peer_rank, 0));
    CommData &commData = send_comms[peer_rank];
    CUDA_CHECK(cudaIpcGetMemHandle(&commData.memHandle, commData.buf));
    CUDA_CHECK(cudaEventCreate(&commData.event,
                               cudaEventDisableTiming | cudaEventInterprocess));
    CUDA_CHECK(cudaIpcGetEventHandle(&commData.eventHandle, commData.event));

    send_requests.push_back(MPI_Request());
    MPI_CHECK(MPI_Isend((void *) (&commData.memHandle),
                       sizeof(commData.memHandle), MPI_UNSIGNED_CHAR,
                       peer_rank, 0, comm_, &send_requests.back()));
    send_requests.push_back(MPI_Request());
    MPI_CHECK(MPI_Isend((void *) &commData.eventHandle,
                        sizeof(commData.eventHandle), MPI_UNSIGNED_CHAR,
                        peer_rank, 0, comm_, &send_requests.back()));
  }

  for (int peer_rank = 0; peer_rank < world_size; peer_rank++) {
    if (peer_rank == rank_)
      continue;
    CommData &commData = p2p_recv_comms[peer_rank];
    MPI_CHECK(MPI_Recv((void *) (&commData.memHandle),
                        sizeof(commData.memHandle),
                        MPI_UNSIGNED_CHAR,
                        peer_rank,
                        0,
                        comm_,
                        MPI_STATUS_IGNORE));
    CUDA_CHECK(cudaIpcOpenMemHandle((void **) &commData.buf, commData.memHandle,
                                    cudaIpcMemLazyEnablePeerAccess));
    MPI_CHECK(MPI_Recv((void *) (&commData.eventHandle),
                   sizeof(commData.eventHandle), MPI_UNSIGNED_CHAR,
                   peer_rank, 0, comm_, MPI_STATUS_IGNORE));
    CUDA_CHECK(cudaIpcOpenEventHandle(&commData.event, commData.eventHandle));
  }

  MPI_CHECK(MPI_Waitall(send_requests.size(), send_requests.data(),
                        MPI_STATUS_IGNORE));

  // Create Broadcast mem handles
  for (int node_rank = 0; node_rank < world_size; node_rank++) {
    if (node_rank == rank_) {
      auto& commData = send_comms[node_rank];
      CUDA_CHECK(cudaIpcGetMemHandle(&commData.memHandle, commData.buf));
      CUDA_CHECK(cudaEventCreate(&commData.event,
                                 cudaEventDisableTiming | cudaEventInterprocess));
      CUDA_CHECK(cudaIpcGetEventHandle(&commData.eventHandle, commData.event));
      MPI_CHECK(MPI_Bcast((void *) (&commData.memHandle),
                          sizeof(commData.memHandle), MPI_UNSIGNED_CHAR, node_rank, comm_));
      MPI_CHECK(MPI_Bcast((void *) &commData.eventHandle,
                          sizeof(commData.eventHandle), MPI_UNSIGNED_CHAR, node_rank, comm_));
    } else {
      auto& commData = broadcast_recv_comms[node_rank];
      MPI_CHECK(MPI_Bcast((void *) (&commData.memHandle),
                          sizeof(commData.memHandle), MPI_UNSIGNED_CHAR, node_rank, comm_));
      MPI_CHECK(MPI_Bcast((void *) &commData.eventHandle,
                          sizeof(commData.eventHandle), MPI_UNSIGNED_CHAR, node_rank, comm_));
      CUDA_CHECK(cudaIpcOpenMemHandle((void **) &commData.buf, commData.memHandle,
                                      cudaIpcMemLazyEnablePeerAccess));
      CUDA_CHECK(cudaIpcOpenEventHandle(&commData.event, commData.eventHandle));
    }
  }
  MPI_CHECK(MPI_Barrier(comm_));
  initialized_ = true;
}

void P2PCommunicator::ISend(void *buf,
                            size_t buf_size,
                            int peer_rank,
                            gpuStream_t stream) {
  CommData &commData = send_comms[peer_rank];
  gpu_context_->MemcpyAsyncD2D(commData.buf,
                               buf,
                               buf_size, stream);
  CUDA_CHECK(cudaEventRecord(commData.event, stream));
  MPI_CHECK(MPI_Isend(&commData.dummy,
                      sizeof(commData.dummy),
                      MPI_UNSIGNED_CHAR,
                      peer_rank,
                      0,
                      comm_,
                      &commData.request));
}

void P2PCommunicator::IRecv(void *buf,
                            size_t buf_size,
                            int peer_rank,
                            gpuStream_t stream) {
  CommData &commData = p2p_recv_comms[peer_rank];
  MPI_CHECK(MPI_Irecv(&commData.dummy, 1, MPI_UNSIGNED_CHAR, peer_rank, 0,
                      comm_, &commData.request));
  commData.target_buf = buf;
  commData.stream = stream;
  commData.buf_size = buf_size;
}

int P2PCommunicator::TestRecv(int rank) {
  CommData &commData = p2p_recv_comms[rank];
  int flag;
  MPI_CHECK(MPI_Test(&commData.request, &flag, MPI_STATUSES_IGNORE));
  if (!flag)
    return 0;
  CUDA_CHECK(cudaStreamWaitEvent(commData.stream, commData.event, 0));
  gpu_context_->MemcpyAsyncD2D(commData.target_buf,
                               commData.buf,
                               commData.buf_size,
                               commData.stream);
  return 1;
}

void P2PCommunicator::WaitRecv(int rank) {
  while (!TestRecv(rank)){}
}

void P2PCommunicator::WaitSend(int rank) {
  CommData &commData = send_comms[rank];
  MPI_CHECK(MPI_Wait(&commData.request, MPI_STATUSES_IGNORE));
}

void P2PCommunicator::WaitAllRecv() {
  for (int peer_rank = 0; peer_rank < world_size_; peer_rank++){
    if (peer_rank == rank_)
      continue;
    WaitRecv(peer_rank);
  }
}

void P2PCommunicator::WaitAllSend() {
  for (int peer_rank = 0; peer_rank < world_size_; peer_rank++){
    if (peer_rank == rank_)
      continue;
    WaitSend(peer_rank);
  }
}

void* P2PCommunicator::GetRemoteBuftoSend(int peer_rank) {
  CommData &commData = send_comms[peer_rank];
  return commData.buf;
}

void* P2PCommunicator::GetRemoteBuftoRecv(int peer_rank) {
  CommData &commData = p2p_recv_comms[peer_rank];
  return commData.buf;
}

void * P2PCommunicator::GetRemoteBroadcastBuftoSend() {
  CommData &comm_data = send_comms[rank_];
  return comm_data.buf;
}

void* P2PCommunicator::GetRemoteBroadcastBuftoRecv(int peer_rank) {
  CommData &comm_data = broadcast_recv_comms[peer_rank];
  return comm_data.buf;
}

} // namespace common
} // namespace cgx
