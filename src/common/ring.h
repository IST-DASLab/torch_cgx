#pragma once

#include "reducer.h"

namespace cgx {
namespace common {

class MPI_Allreduce_Ring : public MPIReducer {
public:
MPI_Allreduce_Ring(GPUContext *gpu_context,
std::shared_ptr<Compressor> compressor,
    std::shared_ptr<Communicator> communicator,
int world_size);

int AllreduceDivision(int num_elements, int global_offset,
                      std::vector<Layer> &tensors,
                      void *comm, bool do_compression) override;
private:
int AllreduceDivisionCompressed(int num_elements, int global_offset,
                                std::vector<Layer> &layers,
                                void *comm);
int AllreduceDivisionUncompressed(int num_elements, int global_offset,
                                  std::vector<Layer> &layers,
                                  void *comm);
gpuStream_t stream_;
};

} // namespace common
} // namespace cgx
