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

#pragma once

#include <torch/extension.h>

#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <c10d/ProcessGroup.hpp>
#include <c10d/Store.hpp>
#include <c10d/Types.hpp>
#include <c10d/Utils.hpp>

#include "mpi_allreduce_operations.h"
#include <mpi.h>

#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>

#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>

namespace cgx {

constexpr const char *CGX_BACKEND_NAME = "cgx";

// WorkEntry is the state associated with a single MPI run instance.
// It include the source Tensor list and destination Tensor list, as well as
// The actual run function that will operate either on src or dst or both.
struct WorkEntry {
  explicit WorkEntry(std::vector<at::Tensor> *srcPtr,
                     std::vector<at::Tensor> *dstPtr,
                     std::function<void(std::unique_ptr<WorkEntry> &)> run,
                     std::shared_ptr<at::cuda::CUDAEvent> endEvent = nullptr)
      : dst(dstPtr ? *dstPtr : std::vector<at::Tensor>()), run(std::move(run)),
        endEvent_(endEvent) {
    if (srcPtr) {
      src = *srcPtr;
    }
  }

  // Not copyable
  WorkEntry(const WorkEntry &) = delete;
  // Not copy assignable
  WorkEntry &operator=(const WorkEntry &) = delete;

  // For input and output tensors (in-place), we will always use src
  std::vector<at::Tensor> src;
  const std::vector<at::Tensor> dst;
  std::shared_ptr<at::cuda::CUDAEvent> endEvent_ = nullptr;
  // src rank returned, for recv only
  int *srcRank = nullptr;
  std::function<void(std::unique_ptr<WorkEntry> &)> run;
};

// ProcessGroupCGX implements MPI bindings with quantization for c10d.
//
// All functions on this class are expected to be called in the same
// order across processes in the group. This is the only way that we
// can guarantee to match up the same calls across processes.
//
// All MPI functions provided by this class is asynchronously scheduled on a
// Worker thread. Therefore, ProcessGroupCGX requires the MPI implementation
// that is used to have a minimum thread support value of MPI_THREAD_SERIALIZED.
// That is, The process may be multi-threaded, and multiple threads may make
// MPI calls, but only one at a time: MPI calls are not made concurrently from
// two distinct threads (all MPI calls are serialized). However, with
// MPI_THREAD_SERIALIZED, ProcessGroupCGX will only support a singe process
// group. In other words, no more than 1 process group can be created globally.
//
// If you would like to use multiple ProcessGroupCGX, it requres your MPI
// implemenation to have a thread support value of MPI_THREAD_MULTIPLE, that is,
// multiple threads may call MPI, with no restriction.
//
// Also note that ProcessGroupCGX only supports a single Tensor operation. In
// other words, the size of the input Tensor vector should always be 1.
//
// CUDA tensor can be supported if the MPI used is CUDA-aware MPI, and
// ProcessGroupCGX will automatically detect this support.
class ProcessGroupCGX : public c10d::ProcessGroup {
public:
  class WorkMPI : public c10d::ProcessGroup::Work {
  public:
    explicit WorkMPI(
        std::vector<at::Tensor> outputTensors,
        const char *profilingTitle = nullptr,
        const c10::optional<std::vector<at::Tensor>> &inputTensors =
            c10::nullopt,
        std::shared_ptr<at::cuda::CUDAEvent> endEvent = nullptr,
        bool compressed = false,
        const std::shared_ptr<at::cuda::CUDAStream> stream = nullptr)
        : ProcessGroup::Work(-1, c10d::OpType::UNKNOWN, profilingTitle,
                             inputTensors),
          outputTensors_(std::move(outputTensors)),
          future_(c10::make_intrusive<at::ivalue::Future>(
              c10::ListType::create(c10::TensorType::get()))),
          endEvent_(endEvent), compressed_(compressed), cgx_stream(stream) {}

    std::vector<at::Tensor> result() override;
    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

    void synchronize() override;

  protected:
    friend class ProcessGroupCGX;

  private:
    void finishWorkMPI();
    void finishWorkMPIError(std::exception_ptr eptr);
    unsigned counter_;
    std::shared_ptr<at::cuda::CUDAEvent> endEvent_;
    bool compressed_;
    const std::shared_ptr<at::cuda::CUDAStream> cgx_stream;
    std::vector<at::Tensor> outputTensors_;
    c10::intrusive_ptr<at::ivalue::Future> future_;
  };

  class AsyncWork : public c10d::ProcessGroup::Work {
  public:
    AsyncWork(MPI_Request request, std::vector<at::Tensor> outputTensors,
              const char *profilingTitle,
              const c10::optional<std::vector<at::Tensor>> &inputTensors);
    virtual ~AsyncWork();

    bool isCompleted() override;

    bool isSuccess() const override;

    int sourceRank() const override;

    bool wait(std::chrono::milliseconds timeout = c10d::kUnsetTimeout) override;

    void abort() override;
    std::vector<at::Tensor> result() override;
    unsigned counter_;

  protected:
    void populateException();

  private:
    const std::vector<at::Tensor> outputTensors_;
    MPI_Request request_;
    MPI_Status status_;
  };

  // Constructor will spawn up the worker thread loop
  explicit ProcessGroupCGX(int rank, int size, MPI_Comm pgComm);

  virtual ~ProcessGroupCGX();

  // Abort the MPI program, needs to be called when exception is detected
  void abort();

  const std::string getBackendName() const override {
    return std::string(CGX_BACKEND_NAME);
  }

  c10::intrusive_ptr<c10d::ProcessGroup::Work> broadcast(
      std::vector<at::Tensor> &data,
      const c10d::BroadcastOptions &opts = c10d::BroadcastOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> allreduce(
      std::vector<at::Tensor> &tensors,
      const c10d::AllreduceOptions &opts = c10d::AllreduceOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work>
  allreduce_coalesced(std::vector<at::Tensor> &tensors,
                      const c10d::AllreduceCoalescedOptions &opts =
                          c10d::AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work>
  reduce(std::vector<at::Tensor> &tensors,
         const c10d::ReduceOptions &opts = c10d::ReduceOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> allgather(
      std::vector<std::vector<at::Tensor>> &outputTensors,
      std::vector<at::Tensor> &inputTensors,
      const c10d::AllgatherOptions &opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> _allgather_base(
      at::Tensor &outputbuffer, at::Tensor &inputbuffer,
      const c10d::AllgatherOptions &opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>> &outputTensorLists,
      std::vector<at::Tensor> &inputTensors,
      const c10d::AllgatherOptions &opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work>
  gather(std::vector<std::vector<at::Tensor>> &outputTensors,
         std::vector<at::Tensor> &inputTensors,
         const c10d::GatherOptions &opts = c10d::GatherOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work>
  scatter(std::vector<at::Tensor> &outputTensors,
          std::vector<std::vector<at::Tensor>> &inputTensors,
          const c10d::ScatterOptions &opts = c10d::ScatterOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work>
  reduce_scatter(std::vector<at::Tensor> &outputTensors,
                 std::vector<std::vector<at::Tensor>> &inputTensors,
                 const c10d::ReduceScatterOptions &opts =
                     c10d::ReduceScatterOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> alltoall_base(
      at::Tensor &outputTensor, at::Tensor &inputTensor,
      std::vector<int64_t> &outputSplitSizes,
      std::vector<int64_t> &inputSplitSizes,
      const c10d::AllToAllOptions &opts = c10d::AllToAllOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> alltoall(
      std::vector<at::Tensor> &outputTensors,
      std::vector<at::Tensor> &inputTensors,
      const c10d::AllToAllOptions &opts = c10d::AllToAllOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work>
  send(std::vector<at::Tensor> &tensors, int dstRank, int tag) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work>
  recv(std::vector<at::Tensor> &tensors, int srcRank, int tag) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work>
  recvAnysource(std::vector<at::Tensor> &tensor, int tag) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work>
  barrier(const c10d::BarrierOptions &opts = c10d::BarrierOptions()) override;

  // Creating a new ProcessGroupCGX, will initiialize MPI if not initialized
  static c10::intrusive_ptr<c10d::ProcessGroup>
  createProcessGroupCGX(const c10::intrusive_ptr<c10d::Store> &store, int rank,
                         int size, const std::chrono::duration<float> &timeout);

  static void ProcessGroupCGXConstructor() __attribute__((constructor)) {
    py::object module = py::module::import("torch.distributed");
    py::object register_backend =
        module.attr("Backend").attr("register_backend");
    register_backend(CGX_BACKEND_NAME, py::cpp_function(createProcessGroupCGX));
  }

  // Support float16 in MPI
  static MPI_Datatype float16_type;

protected:
  using WorkType =
      std::tuple<std::unique_ptr<WorkEntry>, c10::intrusive_ptr<WorkMPI>>;
  // Worker thread loop
  void runLoop();

  // Helper function that is called by the destructor
  void destroy();

  c10::intrusive_ptr<c10d::ProcessGroup::Work>
  enqueue(std::unique_ptr<WorkEntry> entry, const char *profilingTitle,
          const c10::optional<std::vector<at::Tensor>> &inputTensors,
          bool compressed = false,
          const std::shared_ptr<at::cuda::CUDAStream> stream = nullptr);

  bool stop_;

  std::mutex pgMutex_;
  std::thread workerThread_;

  std::queue<WorkType> queue_;
  std::condition_variable queueProduceCV_;
  std::condition_variable queueConsumeCV_;

  // Global states
  static void initMPIOnce();
  static void mpiExit();
  static std::once_flag onceFlagInitMPI;

  static std::mutex pgGlobalMutex_;
  static int mpiThreadSupport_;

  MPI_Comm pgComm_;
  std::unique_ptr<MPIAllReduce_Operation> allreduce_operator;
  std::unordered_map<unsigned int, std::shared_ptr<at::cuda::CUDAStream>> streams_;
  std::unordered_map<unsigned int, std::shared_ptr<at::cuda::CUDAEvent>> cuda_start_events_;
  std::unordered_map<unsigned int, std::shared_ptr<at::cuda::CUDAEvent>>
      cuda_end_events_;
  unsigned counter_;
};

} // namespace cgx
