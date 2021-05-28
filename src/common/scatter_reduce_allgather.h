#pragma once
#include "reducer.h"

namespace qmpi {
namespace common {

class MPI_Allreduce_ScatterReduceAllgather : public MPIReducer {
public:
  MPI_Allreduce_ScatterReduceAllgather(GPUContext *gpu_context,
                                       std::shared_ptr<Compressor> compressor,
                                       int world_size);

  int AllreduceDivision(int num_elements, int global_offset,
                        std::vector<at::Tensor> &tensors,
                        void *comm, bool do_compression) override;
private:
  int AllreduceDivisionCompressed(int num_elements, int global_offset,
                        std::vector<at::Tensor> &tensors,
                        void *comm);
  int AllreduceDivisionUncompressed(int num_elements, int global_offset,
                        std::vector<at::Tensor> &tensors,
                        void *comm);

  gpuStream_t* streams_;
};

} // namespace common
} // namespace qmpi
