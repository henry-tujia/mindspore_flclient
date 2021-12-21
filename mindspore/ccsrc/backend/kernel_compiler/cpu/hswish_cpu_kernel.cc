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

#include "backend/kernel_compiler/cpu/hswish_cpu_kernel.h"
#include <algorithm>
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kHSwishInputsNum = 1;
constexpr size_t kHSwishOutputsNum = 1;
}  // namespace

template <typename T>
void HSwishCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  x_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  for (const uint64_t &d : x_shape_) {
    tensor_size_ *= d;
  }
}

template <typename T>
bool HSwishCPUKernel<T>::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                                const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kHSwishInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kHSwishOutputsNum, kernel_name_);
  const auto *x = reinterpret_cast<T *>(inputs[0]->addr);
  auto *y = reinterpret_cast<T *>(outputs[0]->addr);
  auto zero = static_cast<T>(0);
  auto three = static_cast<T>(3);
  auto six = static_cast<T>(6);

  auto task = [&](size_t start, size_t end) {
    for (uint64_t i = start; i < end; ++i) {
      if (x[i] + three <= zero) {
        y[i] = zero;
      } else if (x[i] >= three) {
        y[i] = x[i];
      } else {
        y[i] = x[i] * (x[i] + three) / six;
      }
    }
  };
  ParallelLaunchAutoSearch(task, tensor_size_, this, &parallel_search_info_);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
