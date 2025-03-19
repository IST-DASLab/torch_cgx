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
// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Uber Technologies, Inc.
// Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
// Modifications copyright (C) 2022, IST Austria.
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
#if HAVE_CUDA
#include <cuda_fp16.h>
#include <cuda_runtime.h>
using gpuError_t = cudaError_t;
using gpuEvent_t = cudaEvent_t;
using gpuStream_t = cudaStream_t;
using gpuIpcEventHandle_t = cudaIpcEventHandle_t;
#elif HAVE_ROCM
#include <hip/hip_runtime_api.h>
using gpuError_t = hipError_t;
using gpuEvent_t = hipEvent_t;
using gpuStream_t = hipStream_t;
using gpuIpcEventHandle_t = hipIpcEventHandle_t;
#endif
#include <queue>
#include <vector>
#include <string>
#include <functional>
#include <unordered_map>
#include <cassert>
#include <memory>
#include <stdexcept>

#ifndef TEST
#include <torch/torch.h>
#endif

namespace cgx::common {

class GPUContext {
public:
  GPUContext();
  ~GPUContext();

  void ErrorCheck(const char* op_name, gpuError_t gpu_result);

  void EventCreate(gpuEvent_t* event);
  void EventDestroy(gpuEvent_t& event);
  void EventRecord(gpuEvent_t& event, gpuStream_t &stream);

  void IpcGetEventHandle(gpuIpcEventHandle_t *eventHandle, gpuEvent_t &event);
  void IpcOpenEventHandle(gpuEvent_t *event, gpuIpcEventHandle_t &eventHandle);

  void StreamCreate(gpuStream_t *stream);
  void StreamSynchronize(gpuStream_t stream);
  void StreamWaitEvent(gpuStream_t& stream, gpuEvent_t& event);

  void DeviceSynchronize();

  int GetDevice();

  void SetDevice(int device);

  void MemcpyAsyncD2D(void *dst, const void *src, size_t count, gpuStream_t stream);
  void MemcpyAsyncH2D(void *dst, const void *src, size_t count, gpuStream_t stream);
  void MemcpyAsyncD2H(void *dst, const void *src, size_t count, gpuStream_t stream);

  void MemcpyD2D(void *dst, const void* srt, size_t count);

private:
  class impl;
  std::unique_ptr<impl> pimpl;
};
} // namespace cgx::common
