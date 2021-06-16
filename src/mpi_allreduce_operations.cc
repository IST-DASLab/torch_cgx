#include "mpi_allreduce_operations.h"
#include "common/utils.h"
#include "common/compressor.h"
#include "common/common.h"
#include "common/scatter_reduce_allgather.h"

namespace qmpi {

int NumElements(std::vector<at::Tensor> &tensors) {
  int sum = 0;
  for (auto &tensor: tensors) {
    sum += tensor.numel();
  }
  return sum;
}

std::shared_ptr<common::Compressor> CreateCompressor(common::GPUContext *gpu_context) {
  return std::make_shared<common::MaxMinQuantizer>(gpu_context);
//  return std::make_shared<common::DummyCompressor>(gpu_context);
}

std::shared_ptr<common::Reducer>
CreateReducer(common::GPUContext *gpu_context,
              std::shared_ptr<common::Compressor> compressor, int world_size) {
  return std::make_shared<common::MPI_Allreduce_ScatterReduceAllgather>(
      gpu_context, compressor, world_size);
}

MPIAllReduce_Operation::MPIAllReduce_Operation() {
  gpu_context_.SetDevice(mpi_context_.GetRank(mpi_context_.GetLocalComm()));
  compressor_ = CreateCompressor(&gpu_context_);
  reducer_ = CreateReducer(&gpu_context_,
                           compressor_,
                           mpi_context_.GetSize(mpi_context_.GetGlobalComm()));
  unsigned int fusion_size_mb =
      common::utils::GetIntEnvOrDefault(FUSION_BUFFER_SIZE_MB,
                                        FUSION_SIZE_DEFAULT_MB);
  tensor_fusion_threshold_ =
      std::max(fusion_size_mb * 1024 * 1024, MIN_FUSION_SIZE);
}

int MPIAllReduce_Operation::allReduce(int num_elements,
                                      int offset,
                                      std::vector<at::Tensor> &tensors,
                                      bool do_compression) {
  auto comm = mpi_context_.GetGlobalComm();
  return reducer_->AllreduceDivision(num_elements, offset, tensors,
                                     (void *) &comm, do_compression);
}

int MPIAllReduce_Operation::performOperationSingle(at::Tensor &tensor,
                                                   bool do_compression) {
  int max_buffer_size = tensor_fusion_threshold_ / tensor.element_size();
  int num_elements = tensor.numel();
  std::vector<at::Tensor> tensors = {tensor};
  int status;
  for (int offset = 0; offset < num_elements;
       offset += max_buffer_size) {
    status =
        allReduce(std::min(max_buffer_size, num_elements - offset),
                  offset, tensors, do_compression);
  }
  return status;
}

int MPIAllReduce_Operation::performOperation(std::vector<at::Tensor> &tensors,
                                             bool do_compression) {
  int max_buffer_size = tensor_fusion_threshold_ / tensors[0].element_size();
  int num_elements = NumElements(tensors);
  if (num_elements < max_buffer_size) {
    return allReduce(num_elements, 0, tensors, do_compression);
  }
  int status;
  std::vector<at::Tensor> tmp_tensors;
  int cur_size = 0;
  for (auto &tensor: tensors) {
    if (tensor.numel() > max_buffer_size) {
      status = performOperationSingle(tensor, do_compression);
      break;
    }
    if (cur_size + tensor.numel() > max_buffer_size) {
      status = allReduce(cur_size, 0, tmp_tensors, do_compression);
      cur_size = 0;
      tmp_tensors.clear();
    }
    tmp_tensors.push_back(tensor);
    cur_size += tensor.numel();
  }
  return status;
}

int MPIAllReduce_Operation::PerformOperation(std::vector<at::Tensor> &tensors) {
  compressor_->ResetParamsFromEnv();
  std::vector<at::Tensor> tensors_compress;
  std::vector<at::Tensor> tensors_nocompress;
  for (auto &tensor: tensors) {
    if (compressor_->isEnabled(tensor))
      tensors_compress.push_back(tensor);
    else
      tensors_nocompress.push_back(tensor);
  }
  int status;
  if (!tensors_compress.empty()) {
    status = performOperation(tensors_compress, true);
  }
  if (!tensors_nocompress.empty()) {
    status = performOperation(tensors_nocompress, false);
  }
  return status;
}

} // namespace qmpi
