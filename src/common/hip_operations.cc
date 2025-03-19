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

#include <mutex>

namespace cgx::common {

class GPUContext::impl {
public:
  void ErrorCheck(std::string op_name, hipError_t hip_result) {
    if (hip_result != hipSuccess) {
      throw std::logic_error(
          std::string(op_name) + " failed: " + hipGetErrorString(hip_result));
    }
  }

  void EventCreate(hipEvent_t *event) {
    ErrorCheck("hipEventCreateWithFlags",
               hipEventCreateWithFlags(event,
                                       hipEventDisableTiming
                                           | hipEventInterprocess));
  }

  void EventRecord(hipEvent_t &event, hipStream_t &stream) {
    ErrorCheck("hipEventRecord", hipEventRecord(event, stream));
  }

  void EventDestroy(hipEvent_t &event) {
    ErrorCheck("hipEventDestroy", hipEventDestroy(event));
  }

  void IpcGetEventHandle(hipIpcEventHandle_t *eventHandle, hipEvent_t &event) {
    ErrorCheck("hipIpcGetEventHandle",
               hipIpcGetEventHandle(eventHandle, event));
  }

  void IpcOpenEventHandle(hipEvent_t *event, hipIpcEventHandle_t &eventHandle) {
    ErrorCheck("hipIpcOpenEventHandle",
               hipIpcOpenEventHandle(event, eventHandle));
  }

  void StreamCreate(hipStream_t *stream) {
    int greatest_priority;
    ErrorCheck("hipDeviceGetStreamPriorityRange",
               hipDeviceGetStreamPriorityRange(nullptr, &greatest_priority));
    ErrorCheck("hipStreamCreateWithPriority",
               hipStreamCreateWithPriority(stream,
                                           hipStreamNonBlocking,
                                           greatest_priority));
  }

  void StreamSynchronize(hipStream_t stream) {
    ErrorCheck("hipStreamSynchronize", hipStreamSynchronize(stream));
  }

  int GetDevice() {
    int device;
    ErrorCheck("hipGetDevice", hipGetDevice(&device));
    return device;
  }

  void SetDevice(int device) {
    ErrorCheck("hipSetDevice", hipSetDevice(device));
  }

  void MemcpyAsyncD2D(void *dst,
                      const void *src,
                      size_t count,
                      hipStream_t stream) {
    ErrorCheck("hipMemcpyAsync",
               hipMemcpyAsync(dst,
                              src,
                              count,
                              hipMemcpyDeviceToDevice,
                              stream));
  }

  void MemcpyAsyncH2D(void *dst,
                      const void *src,
                      size_t count,
                      hipStream_t stream) {
    ErrorCheck("hipMemcpyAsync",
               hipMemcpyAsync(dst, src, count, hipMemcpyHostToDevice, stream));
  }

  void MemcpyAsyncD2H(void *dst,
                      const void *src,
                      size_t count,
                      hipStream_t stream) {
    ErrorCheck("hipMemcpyAsync",
               hipMemcpyAsync(dst, src, count, hipMemcpyDeviceToHost, stream));
  }

  void MemcpyD2D(void *dst, const void *src, size_t count) {
    ErrorCheck("hipMemcpyAsync",
               hipMemcpy(dst, src, count, hipMemcpyDeviceToDevice));
  }

  void DeviceSynchronize() {
    ErrorCheck("hipDeviceSynchronize", hipDeviceSynchronize());
  }

  void StreamWaitEvent(hipStream_t stream, hipEvent_t &event) {
    ErrorCheck("hipStreamWaitEvent", hipStreamWaitEvent(stream, event, 0));
  }

private:
  // We reuse HIP events as it appears that their creation carries non-zero cost.
  std::unordered_map<int, std::queue<hipEvent_t>> hip_events;
  std::mutex hip_events_mutex;
};

#include "gpu_context_impl.cc"

} // namespace cgx::common
