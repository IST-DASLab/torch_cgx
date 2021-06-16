#pragma once

#include <mpi.h>
#include "compressor.h"
#include "gpu_context.h"
#include "utils.h"
#include "common.h"

namespace qmpi {
namespace common {

class Reducer {
public:
  Reducer(GPUContext *gpu_context, std::shared_ptr<Compressor> compressor);

  virtual ~Reducer() = default;
  virtual int AllreduceDivision(int num_elements, int global_offset,
                                std::vector<at::Tensor> &tensors,
                                void *comm, bool do_compression) = 0;
protected:
  std::shared_ptr<Compressor> compressor_;
  GPUContext *gpu_context_;

  // We only need some framework agnostic Buffer Manager so we reuse
  // FussionBufferManager. Our usage of it is not related to tensor fusion
  // buffer.
  std::unique_ptr<PersistentBuffer> buffer_;
  unsigned char *gradients_send_ = nullptr;
  unsigned char *gradients_recv_ = nullptr;
  unsigned char *decompress_buffer_ = nullptr;
  size_t tensor_fusion_size_;
};

class MPIReducer : public Reducer {
public:
  MPIReducer(GPUContext *gpu_context, std::shared_ptr<Compressor> compressor)
      : Reducer(gpu_context, compressor) {}
};

} // namespace common
} // namespace qmpi
