/*
 * pytorch-cgx
 *
 * Copyright (C) 2022 Institute of Science and Technology Austria (ISTA).
 * All Rights Reserved.
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "ProcessGroupCGX.h"

#include <limits>
#include <map>

#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAGuard.h>
#include <mpi-ext.h> // Needed for CUDA-aware check

namespace cgx {

#define MPI_CHECK(cmd)                                                         \
  do {                                                                         \
    int mpiStatus = cmd;                                                       \
    if (mpiStatus != MPI_SUCCESS) {                                            \
      std::string err = "MPI error in: " + std::string(__FILE__) + ":" +       \
                        std::to_string(__LINE__) +                             \
                        ", with error code: " + std::to_string(mpiStatus);     \
      throw std::runtime_error(err);                                           \
    }                                                                          \
  } while (0)

namespace {

// Op mapping
std::map<c10d::ReduceOp, MPI_Op> mpiOp = {
    {c10d::ReduceOp::MIN, MPI_MIN},
    {c10d::ReduceOp::MAX, MPI_MAX},
    {c10d::ReduceOp::SUM, MPI_SUM},
    {c10d::ReduceOp::PRODUCT, MPI_PROD},
};
// Type mapping
std::map<at::ScalarType, MPI_Datatype> mpiDatatype = {
    {at::kByte, MPI_UNSIGNED_CHAR},
    {at::kChar, MPI_CHAR},
    {at::kDouble, MPI_DOUBLE},
    {at::kFloat, MPI_FLOAT},
    {at::kInt, MPI_INT},
    {at::kLong, MPI_LONG},
    {at::kShort, MPI_SHORT},
};

// Checking CUDA-aware MPI support, currently we only support CUDA aware
// MPI ops through Open MPI
bool cudaAwareMpiCheck() {
// Run time check
#if defined(MPIX_CUDA_AWARE_SUPPORT)
  if (MPIX_Query_cuda_support() == 1) {
    return true;
  } else {
    return false;
  }
#else // !defined(MPIX_CUDA_AWARE_SUPPORT)
  return false;
#endif // MPIX_CUDA_AWARE_SUPPORT
}

// Checking the input tensor's validity
void checkSingleTensorHelper(const at::Tensor &tensor) {
  if (!tensor.is_contiguous()) {
    throw std::runtime_error("input tensor has to be contiguous");
  }
  if (tensor.is_sparse()) {
    throw std::runtime_error("input tensor has to be dense");
  }
  if (tensor.is_cuda() && !cudaAwareMpiCheck()) {
    throw std::runtime_error("CUDA tensor detected and the MPI used doesn't "
                             "have CUDA-aware MPI support");
  }
}

void checkSingleTensor(const std::vector<at::Tensor> &tensors) {
  if (tensors.size() != 1) {
    throw std::runtime_error(
        "MPI process group does not support multi-GPU collectives");
  }
  checkSingleTensorHelper(tensors[0]);
}

void checkSameSizeAndType(const at::Tensor &tensor,
                          const std::vector<at::Tensor> &tensors) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    if ((tensors[i].numel() != tensor.numel()) ||
        (tensors[i].scalar_type() != tensor.scalar_type())) {
      throw std::runtime_error("Tensors are not equal in size or data type");
    }
    checkSingleTensorHelper(tensors[i]);
  }
}

} // namespace

std::vector<at::Tensor> ProcessGroupCGX::WorkMPI::result() {
  return outputTensors_;
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupCGX::WorkMPI::getFuture() {
  return future_;
}

void ProcessGroupCGX::WorkMPI::finishWorkMPIError(std::exception_ptr eptr) {
  future_->setError(eptr);
  finish(eptr);
}

void ProcessGroupCGX::WorkMPI::finishWorkMPI() {
  if (compressed_) {
    c10::cuda::CUDAStreamGuard streamGuard(*cgx_stream);
    future_->markCompleted(at::IValue(outputTensors_));
  } else {
    future_->markCompleted(at::IValue(outputTensors_));
  }
  finish();
}

void ProcessGroupCGX::WorkMPI::synchronize() {
  if (compressed_) {
    auto data = outputTensors_[0];
    c10::DeviceGuard guard(data.device());
    std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
    endEvent_->block(at::cuda::getCurrentCUDAStream());
  }
}

ProcessGroupCGX::AsyncWork::AsyncWork(
    MPI_Request request, std::vector<at::Tensor> outputTensors,
    const char *profilingTitle,
    const c10::optional<std::vector<at::Tensor>> &inputTensors)
    : c10d::Work(-1, c10d::OpType::UNKNOWN, profilingTitle,
                               inputTensors),
      outputTensors_(std::move(outputTensors)), request_(request) {
  memset(&status_, 0, sizeof(status_));
}

ProcessGroupCGX::AsyncWork::~AsyncWork() {
  if (request_ != MPI_REQUEST_NULL) {
    std::cerr
        << "Attempted destruction of AsyncWork before work has completed, "
        << "terminating the program." << std::endl;
    std::terminate();
  }
}

bool ProcessGroupCGX::AsyncWork::isCompleted() {
  if (request_ == MPI_REQUEST_NULL) {
    return true;
  }

  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  int flag = 0;
  MPI_CHECK(MPI_Test(&request_, &flag, &status_));
  if (request_ != MPI_REQUEST_NULL) {
    return false;
  }

  // request_ == MPI_REQUEST_NULL; the work has completed
  // Populate exception if request was not successful
  if (status_.MPI_ERROR != MPI_SUCCESS) {
    populateException();
  }

  return true;
}

bool ProcessGroupCGX::AsyncWork::isSuccess() const {
  if (request_ != MPI_REQUEST_NULL) {
    throw std::runtime_error(
        "Invalid call to AsyncWork::isSuccess before work has completed");
  }

  return status_.MPI_ERROR == MPI_SUCCESS;
}

int ProcessGroupCGX::AsyncWork::sourceRank() const {
  return status_.MPI_SOURCE;
}

bool ProcessGroupCGX::AsyncWork::wait(std::chrono::milliseconds /* unused */) {
  if (request_ == MPI_REQUEST_NULL) {
    return true;
  }

  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  MPI_CHECK(MPI_Wait(&request_, &status_));
  auto ok = (status_.MPI_ERROR == MPI_SUCCESS);
  if (!ok) {
    populateException();
    std::rethrow_exception(exception_);
  }
  // Always return true, because abort API is not implemented.
  return true;
}

void ProcessGroupCGX::AsyncWork::abort(){
    TORCH_CHECK(false, "ProcessGroupCGX::AsyncWork::abort not implemented.")}

std::vector<at::Tensor> ProcessGroupCGX::AsyncWork::result() {
  return outputTensors_;
}

void ProcessGroupCGX::AsyncWork::populateException() {
  std::array<char, MPI_MAX_ERROR_STRING> buf;
  int len = buf.size();
  MPI_CHECK(MPI_Error_string(status_.MPI_ERROR, buf.data(), &len));
  exception_ =
      std::make_exception_ptr(std::runtime_error(std::string(buf.data(), len)));
}

// Static global states
int ProcessGroupCGX::mpiThreadSupport_ = 0;
std::mutex ProcessGroupCGX::pgGlobalMutex_;
// We only want to initialize once
std::once_flag ProcessGroupCGX::onceFlagInitMPI;

void ProcessGroupCGX::mpiExit() {
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  if (mpiDatatype.find(at::kHalf) != mpiDatatype.end()) {
    MPI_Type_free(&mpiDatatype[at::kHalf]);
  }
  MPI_CHECK(MPI_Finalize());
}

void ProcessGroupCGX::initMPIOnce() {
  // Initialize MPI environment
  std::call_once(onceFlagInitMPI, []() {
    MPI_CHECK(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_SERIALIZED,
                              &mpiThreadSupport_));
    if (mpiThreadSupport_ < MPI_THREAD_SERIALIZED) {
      throw std::runtime_error("Used MPI implementation doesn't have the "
                               "minimum level of threading support: "
                               "MPI_THREAD_SERIALIZED. This is required by "
                               "c10d package");
    }
    if (std::atexit(ProcessGroupCGX::mpiExit)) {
      throw std::runtime_error("Fail to register the MPI exit handler");
    }
  });
}

c10::intrusive_ptr<c10d::ProcessGroup> ProcessGroupCGX::createProcessGroupCGX(
    const c10::intrusive_ptr<c10d::Store> &store, int rank, int size,
    const std::chrono::duration<float> &timeout) {
  // Once initialization
  initMPIOnce();

  MPI_Comm groupComm = MPI_COMM_WORLD;
  return c10::make_intrusive<ProcessGroupCGX>(rank, size, groupComm);
}

ProcessGroupCGX::ProcessGroupCGX(int rank, int size, MPI_Comm pgComm)
    : ProcessGroup(rank, size), stop_(false), pgComm_(pgComm) {
  if (pgComm_ == MPI_COMM_NULL) {
    throw std::runtime_error("pgComm_ must not be MPI_COMM_NULL");
  }
  allreduce_operator = std::make_unique<MPIAllReduce_Operation>();
  // Start the worker thread accepting MPI calls
  workerThread_ = std::thread(&ProcessGroupCGX::runLoop, this);
}

ProcessGroupCGX::~ProcessGroupCGX() { destroy(); }

void ProcessGroupCGX::destroy() {
  std::unique_lock<std::mutex> lock(pgMutex_);
  queueConsumeCV_.wait(lock, [&] { return queue_.empty(); });
  // Queue is empty, signal stop
  stop_ = true;

  // Release lock to allow threads to terminate
  lock.unlock();
  queueProduceCV_.notify_all();

  // Join the single worker thread
  workerThread_.join();
}

void ProcessGroupCGX::abort() {
  destroy();
  MPI_Abort(pgComm_, EXIT_FAILURE);
}

void ProcessGroupCGX::runLoop() {
  std::unique_lock<std::mutex> lock(pgMutex_);
  while (!stop_) {
    if (queue_.empty()) {
      queueProduceCV_.wait(lock);
      continue;
    }
    auto workTuple = std::move(queue_.front());
    queue_.pop();
    auto &workEntry = std::get<0>(workTuple);
    auto &work = std::get<1>(workTuple);
    queueConsumeCV_.notify_one();
    try {
      workEntry->run(workEntry);
      work->finishWorkMPI();
    } catch (std::exception &e) {
      work->finishWorkMPIError(std::current_exception());
    }
  }
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCGX::enqueue(
    std::unique_ptr<WorkEntry> entry, const char *profilingTitle,
    const c10::optional<std::vector<at::Tensor>> &inputTensors, bool compressed,
    const std::shared_ptr<at::cuda::CUDAStream>& stream) {
  auto work =
      c10::make_intrusive<WorkMPI>(entry->dst, profilingTitle, inputTensors,
                                   entry->endEvent_, compressed, stream);
  if (compressed) {
    c10::cuda::CUDAStreamGuard streamGuard(*(stream));
    work->future_ = c10::make_intrusive<at::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()),
        std::vector<at::Device>({entry->dst[0].device()}));
  }
  std::unique_lock<std::mutex> lock(pgMutex_);
  queue_.push(std::make_tuple(std::move(entry), work));
  lock.unlock();
  queueProduceCV_.notify_one();
  return work;
}

c10::intrusive_ptr<c10d::Work>
ProcessGroupCGX::broadcast(std::vector<at::Tensor> &tensors,
                            const c10d::BroadcastOptions &opts) {
  checkSingleTensor(tensors);
  std::function<void(std::unique_ptr<WorkEntry> &)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry> &entry) {
        auto data = (entry->src)[0];
        c10::DeviceGuard guard(data.device());
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_Datatype dtype;
        if (mpiDatatype.find(data.scalar_type()) != mpiDatatype.end())
          dtype = mpiDatatype.at(data.scalar_type());
        else {
          assert(data.scalar_type() == at::kHalf);
          MPI_CHECK(MPI_Type_contiguous(2, MPI_BYTE, &dtype));
          MPI_CHECK(MPI_Type_commit(&dtype));
          mpiDatatype[data.scalar_type()] = dtype;
        }
        MPI_CHECK(MPI_Bcast(data.data_ptr(), data.numel(),
                            mpiDatatype.at(data.scalar_type()), opts.rootRank,
                            pgComm_));
      };
  auto entry = std::unique_ptr<WorkEntry>(
      new WorkEntry(&tensors, nullptr, std::move(runFunc)));
  return enqueue(std::move(entry), "mpi:broadcast",
                 c10::optional<std::vector<at::Tensor>>(tensors));
}

c10::intrusive_ptr<c10d::Work>
ProcessGroupCGX::allreduce(std::vector<at::Tensor> &tensors,
                            const c10d::AllreduceOptions &opts) {
  checkSingleTensor(tensors);
  auto &tensor = tensors[0];
  bool do_compress = (tensor.scalar_type() == at::kFloat or
                      tensor.scalar_type() == at::kHalf) and
                     opts.reduceOp == c10d::ReduceOp::SUM and
                     tensor.device().type() == at::kCUDA;
  auto device_idx = tensor.device().index();
  if (streams_.find(device_idx) == streams_.end()) {
    streams_.emplace(device_idx, std::make_shared<at::cuda::CUDAStream>(
                                     std::move(at::cuda::getStreamFromPool())));
  }
  if (cuda_start_events_.find(device_idx) == cuda_start_events_.end()) {
    cuda_start_events_.emplace(device_idx,
                               std::make_shared<at::cuda::CUDAEvent>());
    cuda_end_events_.emplace(device_idx,
                             std::make_shared<at::cuda::CUDAEvent>());
  }

  std::function<void(std::unique_ptr<WorkEntry> &)> runFunc =
      [opts, this, do_compress, device_idx](std::unique_ptr<WorkEntry> &entry) {
        auto &bucket = entry->src[0];
        c10::DeviceGuard guard(bucket.device());
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        if (do_compress) {
          (c10::cuda::getFreeMutex())->lock();
          auto &cgx_event_start = *(cuda_start_events_.at(device_idx));
          auto &cgx_stream = *(streams_.at(device_idx));
          const auto &currentStream =
              at::cuda::getCurrentCUDAStream(device_idx);
          cgx_event_start.record(currentStream);
          cgx_event_start.block(cgx_stream);
          c10::cuda::CUDACachingAllocator::recordStream(
              bucket.storage().data_ptr(), cgx_stream);
          allreduce_operator->PerformOperation(bucket, cgx_stream);
          entry->endEvent_->record(cgx_stream);
          (c10::cuda::getFreeMutex())->unlock();
        } else {
          MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, bucket.data_ptr(),
                                  bucket.numel(),
                                  mpiDatatype.at(bucket.scalar_type()),
                                  mpiOp.at(opts.reduceOp), pgComm_));
        }
      };
  auto entry = std::unique_ptr<WorkEntry>(new WorkEntry(
      &tensors, &tensors, std::move(runFunc), cuda_end_events_.at(device_idx)));
  return enqueue(std::move(entry), "mpi:allreduce",
                 c10::optional<std::vector<at::Tensor>>(tensors), do_compress,
                 streams_.at(device_idx));
}

c10::intrusive_ptr<c10d::Work>
ProcessGroupCGX::allreduce_coalesced(
    std::vector<at::Tensor> &tensors,
    const c10d::AllreduceCoalescedOptions &opts) {
  throw std::runtime_error(
      "allreduce_coalesced is currently not supported with MPI");
}

c10::intrusive_ptr<c10d::Work>
ProcessGroupCGX::reduce(std::vector<at::Tensor> &tensors,
                         const c10d::ReduceOptions &opts) {
  checkSingleTensor(tensors);

  std::function<void(std::unique_ptr<WorkEntry> &)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry> &entry) {
        auto data = (entry->src)[0];
        auto dataPtr = (entry->src)[0].data_ptr();
        void *sendbuf = (rank_ == opts.rootRank) ? MPI_IN_PLACE : dataPtr;
        void *recvbuf = (rank_ == opts.rootRank) ? dataPtr : nullptr;

        c10::DeviceGuard guard(data.device());
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Reduce(sendbuf, recvbuf, data.numel(),
                             mpiDatatype.at(data.scalar_type()),
                             mpiOp.at(opts.reduceOp), opts.rootRank, pgComm_));
      };
  auto entry = std::unique_ptr<WorkEntry>(
      new WorkEntry(&tensors, &tensors, std::move(runFunc)));
  return enqueue(std::move(entry), "mpi:reduce",
                 c10::optional<std::vector<at::Tensor>>(tensors));
}

c10::intrusive_ptr<c10d::Work>
ProcessGroupCGX::allgather(std::vector<std::vector<at::Tensor>> &outputTensors,
                            std::vector<at::Tensor> &inputTensors,
                            const c10d::AllgatherOptions &opts) {
  checkSingleTensor(inputTensors);
  if (outputTensors.size() != 1) {
    throw std::runtime_error("MPI process group only supports a single "
                             "tensor op");
  }
  if (static_cast<size_t>(size_) != outputTensors[0].size()) {
    throw std::runtime_error(
        "All gather: number of output tensors should equal "
        "to the world size");
  }

  checkSameSizeAndType(inputTensors[0], outputTensors[0]);

  std::function<void(std::unique_ptr<WorkEntry> &)> runFunc =
      [this](std::unique_ptr<WorkEntry> &entry) {
        auto data = (entry->src)[0];
        std::vector<at::Tensor> outputDataVec = entry->dst;
        auto flatOutputTensor = c10d::newLikeFlat(outputDataVec);

        c10::DeviceGuard guard(data.device());
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Allgather(data.data_ptr(), data.numel(),
                                mpiDatatype.at(data.scalar_type()),
                                flatOutputTensor.data_ptr(), data.numel(),
                                mpiDatatype.at(data.scalar_type()), pgComm_));

        for (size_t i = 0; i < outputDataVec.size(); ++i) {
          outputDataVec[i].copy_(flatOutputTensor[i]);
        }
      };
  auto entry = std::unique_ptr<WorkEntry>(
      new WorkEntry(&inputTensors, &outputTensors[0], std::move(runFunc)));
  return enqueue(std::move(entry), "mpi:allgather",
                 c10::optional<std::vector<at::Tensor>>(inputTensors));
}

c10::intrusive_ptr<c10d::Work>
ProcessGroupCGX::allgather_coalesced(
    std::vector<std::vector<at::Tensor>> & /* unused */,
    std::vector<at::Tensor> & /* unused */,
    const c10d::AllgatherOptions & /* unused */) {
  throw std::runtime_error(
      "ProcessGroupCGX does not support allgather_coalesced");
}

c10::intrusive_ptr<c10d::Work>
ProcessGroupCGX::gather(std::vector<std::vector<at::Tensor>> &outputTensors,
                         std::vector<at::Tensor> &inputTensors,
                         const c10d::GatherOptions &opts) {
  checkSingleTensor(inputTensors);

  if (rank_ != opts.rootRank) {
    if (outputTensors.size() > 0) {
      throw std::runtime_error("Gather: number of output tensors should be 0 "
                               "for non-root");
    }
  } else {
    if (outputTensors.size() != 1) {
      throw std::runtime_error("Gather: multi-GPU collective is not supported");
    }
    if (static_cast<size_t>(size_) != outputTensors[0].size()) {
      throw std::runtime_error("Gather: number of output tensors should equal "
                               "to the world size");
    }
    checkSameSizeAndType(inputTensors[0], outputTensors[0]);
  }

  std::function<void(std::unique_ptr<WorkEntry> &)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry> &entry) {
        auto data = (entry->src)[0];
        void *recvbuf = nullptr;
        at::Tensor flatOutputTensor;

        std::vector<at::Tensor> dstdata = entry->dst;
        if (rank_ == opts.rootRank) {
          flatOutputTensor = c10d::newLikeFlat(dstdata);
          recvbuf = flatOutputTensor.data_ptr();
        }

        c10::DeviceGuard guard(data.device());
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Gather(data.data_ptr(), data.numel(),
                             mpiDatatype.at(data.scalar_type()), recvbuf,
                             data.numel(), mpiDatatype.at(data.scalar_type()),
                             opts.rootRank, pgComm_));

        if (rank_ == opts.rootRank) {
          const std::vector<at::Tensor> &outputDataVec = entry->dst;
          // copy the flattened output tensors to the outputs
          for (size_t i = 0; i < outputDataVec.size(); ++i) {
            outputDataVec.at(i).copy_(flatOutputTensor[i]);
          }
        }
      };

  if (rank_ == opts.rootRank) {
    auto entry = std::unique_ptr<WorkEntry>(
        new WorkEntry(&inputTensors, &outputTensors[0], std::move(runFunc)));
    return enqueue(std::move(entry), "mpi:gather",
                   c10::optional<std::vector<at::Tensor>>(inputTensors));
  } else {
    auto entry = std::unique_ptr<WorkEntry>(
        new WorkEntry(&inputTensors, nullptr, std::move(runFunc)));
    return enqueue(std::move(entry), "mpi:gather",
                   c10::optional<std::vector<at::Tensor>>(inputTensors));
  }
}

c10::intrusive_ptr<c10d::Work>
ProcessGroupCGX::scatter(std::vector<at::Tensor> &outputTensors,
                          std::vector<std::vector<at::Tensor>> &inputTensors,
                          const c10d::ScatterOptions &opts) {
  checkSingleTensor(outputTensors);

  if (rank_ != opts.rootRank) {
    if (inputTensors.size() > 0) {
      throw std::runtime_error("Scatter: number of input tensors should be 0 "
                               "for non-root");
    }
  } else {
    if (inputTensors.size() != 1) {
      throw std::runtime_error(
          "Scatter: multi-GPU collective is not supported");
    }
    if (static_cast<size_t>(size_) != inputTensors[0].size()) {
      throw std::runtime_error("Scatter: number of input tensors should equal "
                               "to the world size");
    }
    checkSameSizeAndType(outputTensors[0], inputTensors[0]);
  }

  std::function<void(std::unique_ptr<WorkEntry> &)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry> &entry) {
        auto data = (entry->dst)[0];
        void *sendbuf = nullptr;
        at::Tensor flatInputTensor;

        if (rank_ == opts.rootRank) {
          std::vector<at::Tensor> &inputDataVec = entry->src;
          flatInputTensor = c10d::newLikeFlat(inputDataVec);
          sendbuf = flatInputTensor.data_ptr();

          // copy the input tensors to the flatten large send buffer
          for (size_t i = 0; i < inputDataVec.size(); ++i) {
            flatInputTensor[i].copy_(inputDataVec.at(i));
          }
        }

        c10::DeviceGuard guard(data.device());
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Scatter(
            sendbuf, data.numel(), mpiDatatype.at(data.scalar_type()),
            data.data_ptr(), data.numel(), mpiDatatype.at(data.scalar_type()),
            opts.rootRank, pgComm_));
      };

  if (rank_ == opts.rootRank) {
    auto entry = std::unique_ptr<WorkEntry>(
        new WorkEntry(&inputTensors[0], &outputTensors, std::move(runFunc)));
    return enqueue(std::move(entry), "mpi:scatter",
                   inputTensors.size() > 0
                       ? c10::optional<std::vector<at::Tensor>>(inputTensors[0])
                       : c10::nullopt);
  } else {
    auto entry = std::unique_ptr<WorkEntry>(
        new WorkEntry(nullptr, &outputTensors, std::move(runFunc)));
    return enqueue(std::move(entry), "mpi:scatter",
                   inputTensors.size() > 0
                       ? c10::optional<std::vector<at::Tensor>>(inputTensors[0])
                       : c10::nullopt);
  }
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCGX::reduce_scatter(
    std::vector<at::Tensor> &outputTensors,
    std::vector<std::vector<at::Tensor>> &inputTensors,
    const c10d::ReduceScatterOptions &opts) {
  throw std::runtime_error("ProcessGroupCGX does not support reduce_scatter");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCGX::alltoall_base(
    at::Tensor &outputTensor, at::Tensor &inputTensor,
    std::vector<int64_t> &outputSplitSizes,
    std::vector<int64_t> &inputSplitSizes, const c10d::AllToAllOptions &opts) {
  checkSingleTensorHelper(inputTensor);
  checkSingleTensorHelper(outputTensor);

  if (outputSplitSizes.size() == 0 && inputSplitSizes.size() == 0) {
    // We can use alltoall
    TORCH_CHECK(outputTensor.numel() == inputTensor.numel() &&
                    outputTensor.type() == inputTensor.type(),
                "Tensors are not equal in size or data type");
    TORCH_CHECK(outputTensor.size(0) % size_ == 0,
                "Tensor's dim 0 does not divide equally across group size");

    std::function<void(std::unique_ptr<WorkEntry> &)> runFunc =
        [opts, this](std::unique_ptr<WorkEntry> &entry) {
          auto srcdata = (entry->src)[0];
          auto dstdata = (entry->dst)[0];
          c10::DeviceGuard guard(srcdata.device());
          std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
          MPI_CHECK(MPI_Alltoall(srcdata.data_ptr(), srcdata.numel() / size_,
                                 mpiDatatype.at(srcdata.scalar_type()),
                                 dstdata.data_ptr(), dstdata.numel() / size_,
                                 mpiDatatype.at(dstdata.scalar_type()),
                                 pgComm_));
        };
    std::vector<at::Tensor> inputTensors = {inputTensor};
    std::vector<at::Tensor> outputTensors = {outputTensor};
    auto entry = std::unique_ptr<WorkEntry>(
        new WorkEntry(&inputTensors, &outputTensors, std::move(runFunc)));
    return enqueue(std::move(entry), "mpi:all_to_all",
                   c10::optional<std::vector<at::Tensor>>(inputTensors));
  } else {
    // Need alltoallv
    c10d::checkSplitSizes(inputSplitSizes, inputTensor, size_);
    c10d::checkSplitSizes(outputSplitSizes, outputTensor, size_);
    std::function<void(std::unique_ptr<WorkEntry> &)> runFunc =
        [opts, this, inputSplitSizes,
         outputSplitSizes](std::unique_ptr<WorkEntry> &entry) {
          auto srcdata = (entry->src)[0];
          auto dstdata = (entry->dst)[0];
          std::vector<int> send_lengths(size_);
          std::vector<int> recv_lengths(size_);
          std::vector<int> send_offsets(size_);
          std::vector<int> recv_offsets(size_);
          c10d::computeLengthsAndOffsets(inputSplitSizes, srcdata,
                                         &send_lengths, &send_offsets);
          c10d::computeLengthsAndOffsets(outputSplitSizes, dstdata,
                                         &recv_lengths, &recv_offsets);
          c10::DeviceGuard guard(srcdata.device());
          std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
          MPI_CHECK(MPI_Alltoallv(
              srcdata.data_ptr(), send_lengths.data(), send_offsets.data(),
              mpiDatatype.at(srcdata.scalar_type()), dstdata.data_ptr(),
              recv_lengths.data(), recv_offsets.data(),
              mpiDatatype.at(dstdata.scalar_type()), pgComm_));
        };
    std::vector<at::Tensor> inputTensors = {inputTensor};
    std::vector<at::Tensor> outputTensors = {outputTensor};
    auto entry = std::unique_ptr<WorkEntry>(
        new WorkEntry(&inputTensors, &outputTensors, std::move(runFunc)));
    return enqueue(std::move(entry), "mpi:all_to_all",
                   c10::optional<std::vector<at::Tensor>>(inputTensors));
  }
}
c10::intrusive_ptr<c10d::Work>
ProcessGroupCGX::alltoall(std::vector<at::Tensor> &outputTensors,
                           std::vector<at::Tensor> &inputTensors,
                           const c10d::AllToAllOptions &opts) {
  TORCH_CHECK(inputTensors.size() == size_,
              "Number of input tensors are not equal to group size");
  TORCH_CHECK(outputTensors.size() == size_,
              "Number of output tensors are not equal to group size");
  std::function<void(std::unique_ptr<WorkEntry> &)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry> &entry) {
        std::vector<int> send_lengths(size_);
        std::vector<int> recv_lengths(size_);
        std::vector<int> send_offsets(size_);
        std::vector<int> recv_offsets(size_);
        auto srcdata = entry->src;
        auto dstdata = entry->dst;
        int64_t src_len = c10d::computeLengthsAndOffsets(srcdata, &send_lengths,
                                                         &send_offsets);
        int64_t dst_len = c10d::computeLengthsAndOffsets(dstdata, &recv_lengths,
                                                         &recv_offsets);
        std::vector<int64_t> send_lengthsL(send_lengths.begin(),
                                           send_lengths.end());
        std::vector<int64_t> recv_lengthsL(recv_lengths.begin(),
                                           recv_lengths.end());
        at::Tensor srcFlatData = at::empty({src_len}, srcdata[0].options());
        at::Tensor dstFlatData = at::empty({dst_len}, dstdata[0].options());
        auto srcFlatDataSplits =
            srcFlatData.split_with_sizes(c10::IntArrayRef(send_lengthsL), 0);
        for (int i = 0; i < size_; i++) {
          srcFlatDataSplits[i].copy_(srcdata[i].view({-1}));
        }
        c10::DeviceGuard guard1(srcdata[0].device());
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Alltoallv(
            srcFlatData.data_ptr(), send_lengths.data(), send_offsets.data(),
            mpiDatatype.at(srcdata[0].scalar_type()), dstFlatData.data_ptr(),
            recv_lengths.data(), recv_offsets.data(),
            mpiDatatype.at(dstdata[0].scalar_type()), pgComm_));

        auto dstFlatDataSplits =
            dstFlatData.split_with_sizes(c10::IntArrayRef(recv_lengthsL), 0);
        for (int i = 0; i < size_; i++) {
          dstdata[i].view({-1}).copy_(dstFlatDataSplits[i]);
        }
      };
  auto entry = std::unique_ptr<WorkEntry>(
      new WorkEntry(&inputTensors, &outputTensors, std::move(runFunc)));
  return enqueue(std::move(entry), "mpi:all_to_all",
                 c10::optional<std::vector<at::Tensor>>(inputTensors));
}

c10::intrusive_ptr<c10d::Work>
ProcessGroupCGX::send(std::vector<at::Tensor> &tensors, int dstRank, int tag) {
  checkSingleTensor(tensors);

  auto &tensor = tensors[0];
  MPI_Request request = MPI_REQUEST_NULL;

  {
    c10::DeviceGuard guard(tensor.device());
    std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
    MPI_CHECK(MPI_Isend(tensor.data_ptr(), tensor.numel(),
                        mpiDatatype.at(tensor.scalar_type()), dstRank, tag,
                        pgComm_, &request));
  }

  return c10::make_intrusive<AsyncWork>(
      request, std::vector<at::Tensor>(), "mpi:send",
      c10::optional<std::vector<at::Tensor>>(tensors));
}

c10::intrusive_ptr<c10d::Work>
ProcessGroupCGX::recv(std::vector<at::Tensor> &tensors, int srcRank, int tag) {
  checkSingleTensor(tensors);

  auto &tensor = tensors[0];
  MPI_Request request = MPI_REQUEST_NULL;

  {
    c10::DeviceGuard guard(tensor.device());
    std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
    MPI_CHECK(MPI_Irecv(tensor.data_ptr(), tensor.numel(),
                        mpiDatatype.at(tensor.scalar_type()), srcRank, tag,
                        pgComm_, &request));
  }

  return c10::make_intrusive<AsyncWork>(
      request, tensors, "mpi:recv",
      c10::optional<std::vector<at::Tensor>>(tensors));
}

c10::intrusive_ptr<c10d::Work>
ProcessGroupCGX::recvAnysource(std::vector<at::Tensor> &tensors, int tag) {
  checkSingleTensor(tensors);

  auto &tensor = tensors[0];
  MPI_Request request = MPI_REQUEST_NULL;

  {
    c10::DeviceGuard guard(tensor.device());
    std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
    MPI_CHECK(MPI_Irecv(tensor.data_ptr(), tensor.numel(),
                        mpiDatatype.at(tensor.scalar_type()), MPI_ANY_SOURCE,
                        tag, pgComm_, &request));
  }

  return c10::make_intrusive<AsyncWork>(
      request, tensors, "mpi:recvAnySource",
      c10::optional<std::vector<at::Tensor>>(tensors));
}

c10::intrusive_ptr<c10d::Work>
ProcessGroupCGX::barrier(const c10d::BarrierOptions &opts) {
  std::function<void(std::unique_ptr<WorkEntry> &)> runFunc =
      [this](std::unique_ptr<WorkEntry> &entry) {
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Barrier(pgComm_));
      };
  auto entry = std::unique_ptr<WorkEntry>(
      new WorkEntry(nullptr, nullptr, std::move(runFunc)));
  return enqueue(std::move(entry), "mpi:barrier", c10::nullopt);
}

c10::intrusive_ptr<c10d::Work>
ProcessGroupCGX::_allgather_base(at::Tensor & /*unused */,
                                  at::Tensor & /*unused */,
                                  const c10d::AllgatherOptions & /*unused */) {
  throw std::runtime_error(
      "no support for allgather_base in MPI process group");
}

MPI_Datatype ProcessGroupCGX::float16_type;

void RegisterLayer(unsigned bucket_idx, unsigned layer_idx,
                   unsigned layer_numel, int bits, int bucket_size) {
  MPIAllReduce_Operation::RegisterLayer(bucket_idx, layer_idx, layer_numel,
                                        bits, bucket_size);
}


void SetQBits(unsigned bucket_idx, unsigned layer_idx, int bits) {
  MPIAllReduce_Operation::SetQBits(bucket_idx, layer_idx, bits);
}

void SetQBucketSize(unsigned bucket_idx, unsigned layer_idx, int bucket_size) {
  MPIAllReduce_Operation::SetQBits(bucket_idx, layer_idx, bucket_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createProcessGroupCGX", &ProcessGroupCGX::createProcessGroupCGX);
  m.def("register_layer", &RegisterLayer);
  m.def("set_quantization_bits", &SetQBits);
  m.def("set_quantization_bucket_size", &SetQBucketSize);
}

} // namespace cgx
