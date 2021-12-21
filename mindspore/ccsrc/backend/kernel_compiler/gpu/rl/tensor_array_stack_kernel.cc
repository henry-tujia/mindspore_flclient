

/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "backend/kernel_compiler/gpu/rl/tensor_array_stack_kernel.h"
#include <algorithm>
#include "backend/kernel_compiler/common_utils.h"
#include "runtime/device/gpu/gpu_tensor_array.h"
#include "runtime/device/tensor_array_manager.h"

namespace mindspore {
namespace kernel {
using mindspore::device::TensorArrayMgr;
using mindspore::device::TensorArrayPtr;
TensorArrayStackKernel::TensorArrayStackKernel()
    : handle_(0), value_size_(0), ele_size_(0), stream_ptr_(nullptr), type_(nullptr) {
  ResetResource();
}

const std::vector<size_t> &TensorArrayStackKernel::GetInputSizeList() const { return input_size_list_; }

const std::vector<size_t> &TensorArrayStackKernel::GetOutputSizeList() const { return output_size_list_; }

const std::vector<size_t> &TensorArrayStackKernel::GetWorkspaceSizeList() const { return workspace_size_list_; }

bool TensorArrayStackKernel::Init(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_node_ = kernel_node;
  auto shape = GetAttr<std::vector<int64_t>>(kernel_node, "element_shape");
  auto max_element = GetAttr<int64_t>(kernel_node, "max_element");
  for (auto i : shape) {
    shapes_.push_back(LongToSize(i));
  }
  type_ = GetAttr<TypePtr>(kernel_node, "dtype");
  ele_size_ = GetTypeByte(type_);
  for (auto i : shapes_) {
    ele_size_ *= i;
  }
  value_size_ = ele_size_ * LongToSize(max_element);
  InitSizeLists();
  return true;
}

void TensorArrayStackKernel::PostExecute() {
  CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_, cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_ptr_)),
                             "TensorArrayStack cudaStreamSynchronized failed");
  TensorArrayPtr tensors_ = TensorArrayMgr::GetInstance().GetTensorArray(handle_);
  MS_EXCEPTION_IF_NULL(tensors_);
  size_t tensor_size = tensors_->GetValidSize();
  auto shape = shapes_;
  shape.insert(shape.begin(), tensor_size);
  MS_LOG(DEBUG) << "After postexecute, the real shape of TensorArrayStack is " << shape;
  AnfAlgo::SetOutputInferTypeAndShape({type_->type_id()}, {shape}, kernel_node_.lock().get());
}

void TensorArrayStackKernel::ResetResource() noexcept {
  handle_ = 0;
  value_size_ = 0;
  ele_size_ = 0;
  stream_ptr_ = nullptr;
  shapes_.clear();
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

void TensorArrayStackKernel::InitSizeLists() {
  output_size_list_.push_back(value_size_);
  input_size_list_.push_back(sizeof(int64_t));
}

bool TensorArrayStackKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                    const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  stream_ptr_ = stream_ptr;
  auto handle_addr = GetDeviceAddress<int64_t>(inputs, 0);
  auto out_value = GetDeviceAddress<unsigned char>(outputs, 0);
  MS_ERROR_IF_NULL(out_value);
  MS_ERROR_IF_NULL(handle_addr);
  CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                             cudaMemcpyAsync(&handle_, handle_addr, sizeof(int64_t), cudaMemcpyDeviceToHost,
                                             reinterpret_cast<cudaStream_t>(stream_ptr)),
                             "Get handle to host failed");
  TensorArrayPtr tensors_ = TensorArrayMgr::GetInstance().GetTensorArray(handle_);
  MS_ERROR_IF_NULL(tensors_);
  if (tensors_->GetValidSize() > tensors_->GetRealSize()) {
    MS_LOG(EXCEPTION) << "Invalid TensorArray size, maybe should Clear() TensorArray before next usage.";
  }
  for (size_t i = 0; i < tensors_->GetValidSize(); i++) {
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(out_value + ele_size_ * i, tensors_->GetTensorAddr(i), ele_size_,
                                               cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "Stack value failed");
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
