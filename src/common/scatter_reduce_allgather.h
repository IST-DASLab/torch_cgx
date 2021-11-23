#pragma once
#include "reducer.h"
#include "communicator.h"

namespace cgx {
namespace common {

class MPI_Allreduce_ScatterReduceAllgather : public MPIReducer {
public:
  MPI_Allreduce_ScatterReduceAllgather(GPUContext *gpu_context,
                                       std::shared_ptr<Compressor> compressor,
                                       std::shared_ptr<Communicator> communicator,
                                       int world_size);

  int AllreduceDivision(int num_elements, int global_offset,
                        std::vector<Layer> &tensors,
                        void *comm, bool do_compression) override;
private:
  int AllreduceCompressed(int num_elements, int global_offset,
                        std::vector<Layer> &layers,
                        void *comm);
  int AllreduceUncompressed(int num_elements, int global_offset,
                        std::vector<Layer> &layers,
                        void *comm);
  int AllreduceCompressedRemoteBuf(int num_elements,
                               int global_offset,
                               std::vector<
                                   Layer> &layers,
                               void *comm_p);
  int AllReduceAlltoAll(int num_elements, int global_offset,
                        std::vector<Layer> &layers,
                        void *comm);
  gpuStream_t* streams_;
private:
  bool remote_buf_compression_enabled_;
  bool all_to_all_reduction_;
};

} // namespace common
} // namespace cgx
