
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

GPUContext::GPUContext() : pimpl{new impl} {}
GPUContext::~GPUContext() = default;

void GPUContext::ErrorCheck(const char* op_name, gpuError_t gpu_result) {
  pimpl->ErrorCheck(op_name, gpu_result);
}

void GPUContext::EventCreate(gpuEvent_t *event) {
  pimpl->EventCreate(event);
}

void GPUContext::EventDestroy(gpuEvent_t &event) {
  pimpl->EventDestroy(event);
}

void GPUContext::EventRecord(gpuEvent_t &event,
                             gpuStream_t &stream) {
  pimpl->EventRecord(event, stream);
}

void GPUContext::IpcGetEventHandle(gpuIpcEventHandle_t *eventHandle, gpuEvent_t &event) {
  pimpl->IpcGetEventHandle(eventHandle, event);
}

void GPUContext::IpcOpenEventHandle(gpuEvent_t *event,
                                   gpuIpcEventHandle_t &eventHandle) {
  pimpl->IpcOpenEventHandle(event, eventHandle);
}

void GPUContext::StreamCreate(gpuStream_t *stream) {
  pimpl->StreamCreate(stream);
}

void GPUContext::StreamSynchronize(gpuStream_t stream) {
  pimpl->StreamSynchronize(stream);
}

int GPUContext::GetDevice() {
  return pimpl->GetDevice();
}

void GPUContext::SetDevice(int device) {
  pimpl->SetDevice(device);
}

void GPUContext::MemcpyAsyncD2D(void *dst,
                                const void *src,
                                size_t count,
                                gpuStream_t stream) {
  pimpl->MemcpyAsyncD2D(dst, src, count, stream);
}

void GPUContext::MemcpyAsyncH2D(void *dst,
                                const void *src,
                                size_t count,
                                gpuStream_t stream) {
  pimpl->MemcpyAsyncH2D(dst, src, count, stream);
}

void GPUContext::MemcpyAsyncD2H(void *dst,
                                const void *src,
                                size_t count,
                                gpuStream_t stream) {
  pimpl->MemcpyAsyncD2H(dst, src, count, stream);
}

void GPUContext::MemcpyD2D(void *dst,
                           const void *src,
                           size_t count) {
  pimpl->MemcpyD2D(dst, src, count);
}

void GPUContext::DeviceSynchronize() {
  pimpl->DeviceSynchronize();
}

void GPUContext::StreamWaitEvent(gpuStream_t &stream, gpuEvent_t &event) {
  pimpl->StreamWaitEvent(stream, event);
}
