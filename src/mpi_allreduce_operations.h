#pragma once
#include <torch/torch.h>

#include <mpi.h>

#include "common/gpu_context.h"
#include "common/mpi_context.h"
#include "common/reducer.h"
#include "common/layer.h"

namespace cgx {
const int MIN_LAYER_SIZE = 16;

struct MPIAllReduce_Operation {
  MPIAllReduce_Operation();
  int PerformOperation(at::Tensor &bucket);
  static void RegisterModel(std::vector<std::pair<std::string, int>>& model_parameters) {
    for (auto rit = model_parameters.rbegin(); rit != model_parameters.rend(); rit++) {
      model_parameters_.push_back(std::move(*rit));
    }
  }
  static void ExcludeLayer(const std::string& layer);
protected:
  common::GPUContext gpu_context_;
  common::MPIContext mpi_context_;
  std::shared_ptr<common::Reducer> intra_reducer_;
  std::shared_ptr<common::Reducer> cross_reducer_;
  std::shared_ptr<common::Compressor> compressor_;
  int64_t tensor_fusion_threshold_;
  float fake_compression_ratio_;
private:
  void extractLayers(const at::Tensor& bucket, std::vector<common::Layer>& layers);
  int performOperationSingle(common::Layer &layer, bool do_compression);
  int allReduce(int num_elements,
                int offset,
                std::vector<common::Layer> &tensors,
                bool do_compression);
  int performOperation(std::vector<common::Layer> &tensors,
                       bool do_compression);
  static std::vector<std::pair<std::string, int>> model_parameters_;
  int counter_;
};
} // namespace cgx