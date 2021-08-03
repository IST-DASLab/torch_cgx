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

#include <mpi.h>
#include "mpi_allreduce_operations.h"

#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>

namespace qmpi {

constexpr const char *QMPI_BACKEND_NAME = "qmpi";

// WorkEntry is the state associated with a single MPI run instance.
// It include the source Tensor list and destination Tensor list, as well as
// The actual run function that will operate either on src or dst or both.
struct WorkEntry {
  explicit WorkEntry(
      std::vector<at::Tensor> *srcPtr,
      std::vector<at::Tensor> *dstPtr,
      std::function<void(std::unique_ptr<WorkEntry> &)> run)
      : run(run) {
    if (srcPtr) {
      src = *srcPtr;
    }
    if (dstPtr) {
      dst = *dstPtr;
    }
  }

  // Not copyable
  WorkEntry(const WorkEntry &) = delete;
  // Not copy assignable
  WorkEntry &operator=(const WorkEntry &) = delete;

  // For input and output tensors (in-place), we will always use src
  std::vector<at::Tensor> src;
  std::vector<at::Tensor> dst;
  // src rank returned, for recv only
  int *srcRank = nullptr;
  std::function<void(std::unique_ptr<WorkEntry> &)> run;
};

// ProcessGroupQMPI implements MPI bindings with quantization for c10d.
//
// All functions on this class are expected to be called in the same
// order across processes in the group. This is the only way that we
// can guarantee to match up the same calls across processes.
//
// All MPI functions provided by this class is asynchronously scheduled on a
// Worker thread. Therefore, ProcessGroupQMPI requires the MPI implementation
// that is used to have a minimum thread support value of MPI_THREAD_SERIALIZED.
// That is, The process may be multi-threaded, and multiple threads may make
// MPI calls, but only one at a time: MPI calls are not made concurrently from
// two distinct threads (all MPI calls are serialized). However, with
// MPI_THREAD_SERIALIZED, ProcessGroupQMPI will only support a singe process
// group. In other words, no more than 1 process group can be created globally.
//
// If you would like to use multiple ProcessGroupQMPI, it requres your MPI
// implemenation to have a thread support value of MPI_THREAD_MULTIPLE, that is,
// multiple threads may call MPI, with no restriction.
//
// Also note that ProcessGroupQMPI only supports a single Tensor operation. In
// other words, the size of the input Tensor vector should always be 1.
//
// CUDA tensor can be supported if the MPI used is CUDA-aware MPI, and
// ProcessGroupQMPI will automatically detect this support.
class ProcessGroupQMPI : public c10d::ProcessGroup {
 public:
class WorkMPI : public c10d::ProcessGroup::Work {
   protected:
    friend class ProcessGroupQMPI;
  };

  class AsyncWork : public c10d::ProcessGroup::Work {
   public:
    AsyncWork(at::Tensor tensor, MPI_Request request);
    virtual ~AsyncWork();

    bool isCompleted() override;

    bool isSuccess() const override;

    int sourceRank() const override;

    bool wait(std::chrono::milliseconds timeout = c10d::kUnsetTimeout) override;

    void abort() override;

   protected:
    void populateException();

    at::Tensor tensor_;
    MPI_Request request_;
    MPI_Status status_;
  };

  // Constructor will spawn up the worker thread loop
  explicit ProcessGroupQMPI(int rank, int size, MPI_Comm pgComm);

  virtual ~ProcessGroupQMPI();

  // Abort the MPI program, needs to be called when exception is detected
  void abort();

  const std::string getBackendName() const override {
    return std::string(QMPI_BACKEND_NAME);
  }

  c10::intrusive_ptr<c10d::ProcessGroup::Work> broadcast(
      std::vector<at::Tensor> &data,
      const c10d::BroadcastOptions &opts = c10d::BroadcastOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> allreduce(
      std::vector<at::Tensor> &tensors,
      const c10d::AllreduceOptions &opts = c10d::AllreduceOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> allreduce_coalesced(
      std::vector<at::Tensor> &tensors,
      const c10d::AllreduceCoalescedOptions &opts =
      c10d::AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> reduce(
      std::vector<at::Tensor> &tensors,
      const c10d::ReduceOptions &opts = c10d::ReduceOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> allgather(
      std::vector<std::vector<at::Tensor>> &outputTensors,
      std::vector<at::Tensor> &inputTensors,
      const c10d::AllgatherOptions &opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> allgather_base(
      at::Tensor &outputbuffer,
      at::Tensor &inputbuffer,
      const c10d::AllgatherOptions &opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>> &outputTensorLists,
      std::vector<at::Tensor> &inputTensors,
      const c10d::AllgatherOptions &opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> gather(
      std::vector<std::vector<at::Tensor>> &outputTensors,
      std::vector<at::Tensor> &inputTensors,
      const c10d::GatherOptions &opts = c10d::GatherOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> scatter(
      std::vector<at::Tensor> &outputTensors,
      std::vector<std::vector<at::Tensor>> &inputTensors,
      const c10d::ScatterOptions &opts = c10d::ScatterOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> reduce_scatter(
      std::vector<at::Tensor> &outputTensors,
      std::vector<std::vector<at::Tensor>> &inputTensors,
      const c10d::ReduceScatterOptions &opts = c10d::ReduceScatterOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> alltoall_base(
      at::Tensor &outputTensor,
      at::Tensor &inputTensor,
      std::vector<int64_t> &outputSplitSizes,
      std::vector<int64_t> &inputSplitSizes,
      const c10d::AllToAllOptions &opts = c10d::AllToAllOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> alltoall(
      std::vector<at::Tensor> &outputTensors,
      std::vector<at::Tensor> &inputTensors,
      const c10d::AllToAllOptions &opts = c10d::AllToAllOptions()) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> send(
      std::vector<at::Tensor> &tensors,
      int dstRank,
      int tag) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> recv(
      std::vector<at::Tensor> &tensors,
      int srcRank,
      int tag) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> recvAnysource(
      std::vector<at::Tensor> &tensor,
      int tag) override;

  c10::intrusive_ptr<c10d::ProcessGroup::Work> barrier(
      const c10d::BarrierOptions &opts = c10d::BarrierOptions()) override;

  // Creating a new ProcessGroupQMPI, will initiialize MPI if not initialized
  static c10::intrusive_ptr<c10d::ProcessGroup> createProcessGroupQMPI(
      const c10::intrusive_ptr<c10d::Store> &store,
      int rank,
      int size,
      const std::chrono::duration<float> &timeout);

  static void ProcessGroupQMPIConstructor() __attribute__((constructor)) {
    py::object module = py::module::import("torch.distributed");
    py::object register_backend = module.attr("Backend").attr("register_backend");
    register_backend(QMPI_BACKEND_NAME, py::cpp_function(createProcessGroupQMPI));
  }

  // Support float16 in MPI
  static MPI_Datatype float16_type;

protected:
  using WorkType =
  std::tuple<std::unique_ptr<WorkEntry>, c10::intrusive_ptr < WorkMPI>>;
  // Worker thread loop
  void runLoop();

  // Helper function that is called by the destructor
  void destroy();

  c10::intrusive_ptr<c10d::ProcessGroup::Work> enqueue(std::unique_ptr<WorkEntry> entry);

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

};

} // namespace qmpi
