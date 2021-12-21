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

#include "backend/kernel_compiler/cpu/rl/tensor_array_size_kernel.h"
#include "backend/kernel_compiler/common_utils.h"
#include "runtime/device/cpu/cpu_tensor_array.h"
#include "runtime/device/tensor_array_manager.h"

namespace mindspore {
namespace kernel {
using mindspore::device::TensorArrayMgr;
TensorArrayCPUSizeKernel::TensorArrayCPUSizeKernel() {}

const std::vector<size_t> &TensorArrayCPUSizeKernel::GetInputSizeList() const { return input_size_list_; }

const std::vector<size_t> &TensorArrayCPUSizeKernel::GetOutputSizeList() const { return output_size_list_; }

const std::vector<size_t> &TensorArrayCPUSizeKernel::GetWorkspaceSizeList() const { return workspace_size_list_; }

void TensorArrayCPUSizeKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  input_size_list_.push_back(sizeof(int64_t));
  output_size_list_.push_back(sizeof(int64_t));
}

bool TensorArrayCPUSizeKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                      const std::vector<AddressPtr> &outputs) {
  auto handle_addr = GetDeviceAddress<int64_t>(inputs, 0);
  auto out_addr = GetDeviceAddress<int64_t>(outputs, 0);
  MS_EXCEPTION_IF_NULL(handle_addr);
  MS_EXCEPTION_IF_NULL(out_addr);
  auto tensors_ = TensorArrayMgr::GetInstance().GetTensorArray(handle_addr[0]);
  MS_ERROR_IF_NULL(tensors_);
  int64_t valid_size = SizeToLong(tensors_->GetValidSize());
  out_addr[0] = valid_size;
  MS_LOG(DEBUG) << "Launch TensorArraySize, valid size is " << out_addr[0];
  return true;
}
}  // namespace kernel
}  // namespace mindspore
