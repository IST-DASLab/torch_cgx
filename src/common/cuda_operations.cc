// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Uber Technologies, Inc.
// Modifications copyright (C) 2022 IST Austria.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "gpu_context.h"

#include <thread>

namespace cgx {
namespace common {

class GPUContext::impl {
public:
  void ErrorCheck(std::string op_name, cudaError_t cuda_result) {
    if (cuda_result != cudaSuccess) {
      throw std::logic_error(
          std::string(op_name) + " failed: " + cudaGetErrorString(cuda_result));
    }
  }

  void EventCreate(cudaEvent_t *event) {
    ErrorCheck("cudaEventCreateWithFlags",
               cudaEventCreateWithFlags(event,
                                        cudaEventDisableTiming
                                            | cudaEventInterprocess));
  }

  void EventDestroy(cudaEvent_t &event) {
    ErrorCheck("cudaEventDestroy", cudaEventDestroy(event));
  }

  void IpcGetEventHandle(cudaIpcEventHandle_t *eventHandle, cudaEvent_t &event) {
    ErrorCheck("cudaIpcGetEventHandle",
               cudaIpcGetEventHandle(eventHandle, event));
  }

  void IpcOpenEventHandle(cudaEvent_t *event, cudaIpcEventHandle_t &eventHandle) {
    ErrorCheck("cudaIpcOpenEventHandle",
               cudaIpcOpenEventHandle(event, eventHandle));
  }

  void EventRecord(cudaEvent_t &event,
                   cudaStream_t &stream) {
    ErrorCheck("cudaEventRecord", cudaEventRecord(event, stream));
  }

  void StreamCreate(cudaStream_t *stream) {
//    int greatest_priority;
//    ErrorCheck("cudaDeviceGetStreamPriorityRange",
//               cudaDeviceGetStreamPriorityRange(NULL, &greatest_priority));
//    ErrorCheck("cudaStreamCreateWithPriority",
//               cudaStreamCreateWithPriority(stream,
//                                            cudaStreamNonBlocking,
//                                            greatest_priority));
    ErrorCheck("cudaStreamCreateWithFlags",
               cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking));
  }

  void StreamSynchronize(cudaStream_t stream) {
    ErrorCheck("cudaStreamSynchronize", cudaStreamSynchronize(stream));
  }

  void StreamWaitEvent(cudaStream_t stream, cudaEvent_t &event) {
    ErrorCheck("cudaStreamWaitEvent", cudaStreamWaitEvent(stream, event, 0));
  }

  int GetDevice() {
    int device;
    ErrorCheck("cudaGetDevice", cudaGetDevice(&device));
    return device;
  }

  void SetDevice(int device) {
    ErrorCheck("cudaSetDevice", cudaSetDevice(device));
  }

  void MemcpyAsyncD2D(void *dst,
                      const void *src,
                      size_t count,
                      cudaStream_t stream) {
    ErrorCheck("cudaMemcpyAsync",
               cudaMemcpyAsync(dst,
                               src,
                               count,
                               cudaMemcpyDeviceToDevice,
                               stream));
  }

  void MemcpyAsyncH2D(void *dst,
                      const void *src,
                      size_t count,
                      cudaStream_t stream) {
    ErrorCheck("cudaMemcpyAsync",
               cudaMemcpyAsync(dst,
                               src,
                               count,
                               cudaMemcpyHostToDevice,
                               stream));
  }

  void MemcpyAsyncD2H(void *dst,
                      const void *src,
                      size_t count,
                      cudaStream_t stream) {
    ErrorCheck("cudaMemcpyAsync",
               cudaMemcpyAsync(dst,
                               src,
                               count,
                               cudaMemcpyDeviceToHost,
                               stream));
  }

  void DeviceSynchronize() {
    ErrorCheck("cudaDeviceSynchronize", cudaDeviceSynchronize());
  }

  void MemcpyD2D(void *dst, const void *src, size_t count) {
    ErrorCheck("cudaMemcpy",
               cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice));
  }

private:
  // We reuse CUDA events as it appears that their creation carries non-zero cost.
  std::unordered_map<int, std::queue<cudaEvent_t>> cuda_events;
};

#include "gpu_context_impl.cc"

} // namespace common
} // namespace cgx
