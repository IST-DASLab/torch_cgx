#pragma once
#include <torch/torch.h>

#include <mpi.h>

#include "common/gpu_context.h"
#include "common/mpi_context.h"
#include "common/reducer.h"

namespace qmpi {
struct MPIAllReduce_Operation {
  MPIAllReduce_Operation();
  int PerformOperation(std::vector<at::Tensor> &tensors);

protected:
  common::GPUContext gpu_context_;
  common::MPIContext mpi_context_;
  std::shared_ptr<common::Reducer> reducer_;
  std::shared_ptr<common::Compressor> compressor_;
  int64_t tensor_fusion_threshold_;
private:
  int performOperationSingle(at::Tensor &tensor, bool do_compression);
  int allReduce(int num_elements,
                int offset,
                std::vector<at::Tensor> &tensors,
                bool do_compression);
  int performOperation(std::vector<at::Tensor> &tensors,
                       bool do_compression);
};
} // namespace qmpi