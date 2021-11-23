#include "mpi_allreduce_operations.h"
#include "common/utils.h"
#include "common/compressor.h"
#include "common/common.h"
#include "common/scatter_reduce_allgather.h"
#include "common/ring.h"
#include "common/mpi_communicator.h"

#if HAVE_CUDA
#include "common/shm_communicator.h"
#include "common/p2p_communicator.h"
#endif

namespace cgx {

std::vector<std::pair<std::string, int>>
    MPIAllReduce_Operation::model_parameters_;

int NumElements(std::vector<common::Layer> &layers) {
  int sum = 0;
  for (auto &layer: layers) {
    sum += layer.numel();
  }
  return sum;
}

std::shared_ptr<common::Compressor> CreateCompressor(common::GPUContext *gpu_context) {
  bool dummy = common::utils::GetIntEnvOrDefault(CGX_DUMMY_COMPRESSION, 0);
  if (dummy)
    return std::make_shared<common::DummyCompressor>(gpu_context);
  else
    return std::make_shared<common::MaxMinQuantizer>(gpu_context);
}

std::shared_ptr<common::Reducer>
CreateReducer(common::GPUContext *gpu_context,
              std::shared_ptr<common::Compressor> compressor,
              std::shared_ptr<common::Communicator> communicator,
              common::utils::ReductionType red_type, int world_size) {
  if (red_type == common::utils::ReductionType::SRA) {
    return std::make_shared<common::MPI_Allreduce_ScatterReduceAllgather>(
        gpu_context, compressor, communicator, world_size);
  } else {
    return std::make_shared<common::MPI_Allreduce_Ring>(
        gpu_context, compressor, communicator, world_size);
  }
}

std::shared_ptr<common::Reducer>
CreateInnerReducer(common::GPUContext *gpu_context,
                   std::shared_ptr<common::Compressor> compressor,
                   int world_size) {
  auto comm_type = common::utils::GetCommTypeFromEnv(CGX_INNER_COMMUNICATOR_TYPE,
                                                     common::utils::CommunicatorType::SHM);
  std::shared_ptr<common::Communicator> communicator;
#if HAVE_CUDA
  if (comm_type == common::utils::CommunicatorType::SHM)
    communicator.reset(new common::SHMCommunicator(gpu_context));
  else if (comm_type == common::utils::CommunicatorType::P2P)
    communicator.reset(new common::P2PCommunicator(gpu_context));
#endif
  if (comm_type == common::utils::CommunicatorType::MPI)
    communicator.reset(new common::MPICommunicator(gpu_context));
  if (!communicator)
    throw std::runtime_error("Communicator type is not supported");
  auto red_type = common::utils::GetRedTypeFromEnv(CGX_INNER_REDUCTION_TYPE,
                                                   common::utils::ReductionType::SRA);
  return CreateReducer(gpu_context,
                       compressor,
                       communicator,
                       red_type,
                       world_size);
}

std::shared_ptr<common::Reducer>
CreateCrossReducer(common::GPUContext *gpu_context,
                   std::shared_ptr<common::Compressor> compressor,
                   int world_size) {
  auto red_type = common::utils::GetRedTypeFromEnv(CGX_CROSS_REDUCTION_TYPE,
                                                   common::utils::ReductionType::Ring);
  return CreateReducer(gpu_context,
                       compressor,
                       std::make_shared<common::MPICommunicator>(gpu_context),
                       red_type,
                       world_size);
}

MPIAllReduce_Operation::MPIAllReduce_Operation() {
  gpu_context_.SetDevice(mpi_context_.GetRank(mpi_context_.GetLocalComm()));
  compressor_ = CreateCompressor(&gpu_context_);
  if (mpi_context_.GetSize(mpi_context_.GetLocalComm()) > 1)
    intra_reducer_ = CreateInnerReducer(&gpu_context_,
                                      compressor_,
                                      mpi_context_.GetSize(mpi_context_.GetLocalComm()));
  if (mpi_context_.GetSize(mpi_context_.GetCrossComm()) > 1)
    cross_reducer_ = CreateCrossReducer(&gpu_context_, compressor_,
                                        mpi_context_.GetSize(mpi_context_.GetCrossComm()));
  unsigned int fusion_size_mb =
      common::utils::GetIntEnvOrDefault(CGX_FUSION_BUFFER_SIZE_MB,
                                        FUSION_SIZE_DEFAULT_MB);
  fake_compression_ratio_ =
      common::utils::GetFloatEnvOrDefault(CGX_COMPRESSION_FAKE_RATIO, 1.0);
  tensor_fusion_threshold_ =
      std::max(fusion_size_mb * 1024 * 1024, MIN_FUSION_SIZE);
  counter_ = 0;
}

int MPIAllReduce_Operation::allReduce(int num_elements,
                                      int offset,
                                      std::vector<common::Layer> &layers,
                                      bool do_compression) {
  if (num_elements > 1000 and fake_compression_ratio_ < 1.0)
    num_elements = (int) (num_elements * fake_compression_ratio_);
  auto comm = mpi_context_.GetLocalComm();
  int status;
  if (mpi_context_.GetSize(comm) > 1) {
    status = intra_reducer_->AllreduceDivision(num_elements, offset, layers,
                                               (void *) &comm, do_compression);
  }
  if (status < 0)
    return status;
  comm = mpi_context_.GetCrossComm();
  if (mpi_context_.GetSize(comm) > 1) {
    status = cross_reducer_->AllreduceDivision(num_elements, offset, layers,
                                               (void *) &comm, do_compression);
  }
  return status;
}

int MPIAllReduce_Operation::performOperationSingle(common::Layer &layer,
                                                   bool do_compression) {
  int max_buffer_size = tensor_fusion_threshold_ / layer.element_size();
  int num_elements = layer.numel();
  std::vector<common::Layer> layers = {layer};
  int status;
  for (int offset = 0; offset < num_elements; offset += max_buffer_size) {
    status =
        allReduce(std::min(max_buffer_size, num_elements - offset),
                  offset, layers, do_compression);
  }
  return status;
}

int MPIAllReduce_Operation::performOperation(std::vector<common::Layer> &layers,
                                             bool do_compression) {
  int max_buffer_size = tensor_fusion_threshold_ / layers[0].element_size();
  int num_elements = NumElements(layers);
  int status;

  if (num_elements < max_buffer_size) {
    return allReduce(num_elements, 0, layers, do_compression);
  }
  std::vector<common::Layer> tmp_layers;
  int cur_size = 0;
  for (auto &layer: layers) {
    if (layer.numel() > max_buffer_size) {
      status = performOperationSingle(layer, do_compression);
      break;
    }
    if (cur_size + layer.numel() > max_buffer_size) {
      status = allReduce(cur_size, 0, tmp_layers, do_compression);
      cur_size = 0;
      tmp_layers.clear();
    }
    tmp_layers.push_back(layer);
    cur_size += layer.numel();
  }
  return status;
}

int MPIAllReduce_Operation::PerformOperation(at::Tensor &bucket) {
  std::vector<common::Layer> layers;
  int status;
  if (bucket.numel() < MIN_LAYER_SIZE) {
    layers.emplace_back(bucket);
    status = performOperation(layers, false);
    return status;
  }
  compressor_->ResetParamsFromEnv();
  extractLayers(bucket, layers);
  std::vector<common::Layer> layers_compress;
  std::vector<common::Layer> layers_nocompress;
  for (auto &tensor: layers) {
    if (compressor_->isEnabled(tensor))
      layers_compress.push_back(tensor);
    else
      layers_nocompress.push_back(tensor);
  }
  if (!layers_compress.empty()) {
    status = performOperation(layers_compress, true);
  }
  if (!layers_nocompress.empty()) {
    status = performOperation(layers_nocompress, false);
  }
  return status;
}

void MPIAllReduce_Operation::extractLayers(const at::Tensor &bucket,
                                           std::vector<common::Layer> &layers) {
  if (model_parameters_.empty()) {
    layers.emplace_back(bucket);
    return;
  }
  int bucket_numel = bucket.numel();
  int cur_numel = 0;
  char *ptr = static_cast<char *>(bucket.data_ptr());
  while (cur_numel < bucket_numel) {
    auto &param = model_parameters_.at(counter_);
    layers.emplace_back(bucket, param.first, ptr, param.second);
    cur_numel += param.second;
    ptr += param.second * bucket.element_size();
    counter_++;
  }
  if (cur_numel != bucket_numel) {
    throw std::runtime_error("Error at extracting the layers from bucket. "
                             "Number of elements in bucket is not equal to "
                             "number in the layers expected to be in "
                             "the bucket.");
  }
  counter_ %= model_parameters_.size();
}

void MPIAllReduce_Operation::ExcludeLayer(const std::string &layer) {
  common::Compressor::ExcludeLayer(layer);
}

} // namespace cgx
