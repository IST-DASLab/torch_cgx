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
  static void RegisterModel(std::vector<std::pair<std::string, int>>& model_parameters) {
    for (auto rit = model_parameters.rbegin(); rit != model_parameters.rend(); rit++) {
      model_parameters_.push_back(std::move(*rit));
    }
  }
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
  static std::vector<std::pair<std::string, int>> model_parameters_;
  int counter;
};
} // namespace qmpi