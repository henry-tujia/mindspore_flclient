/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/smooth_l1_loss_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSmoothL1LossInputsNum = 2;
constexpr size_t kSmoothL1LossOutputsNum = 1;
}  // namespace

template <typename T>
void SmoothL1LossCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  beta_ = AnfAlgo::GetNodeAttr<float>(kernel_node, "beta");
  if (beta_ == 0.0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", the 'beta' should not be 0.";
  }
  std::vector<size_t> x_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  for (const uint64_t &d : x_shape) {
    tensor_size_ *= d;
  }
}

template <typename T>
bool SmoothL1LossCPUKernel<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> &,
                                      const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSmoothL1LossInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSmoothL1LossOutputsNum, kernel_name_);
  const auto *predict_addr = reinterpret_cast<T *>(inputs[0]->addr);
  const auto *target_addr = reinterpret_cast<T *>(inputs[1]->addr);
  auto *result_addr = reinterpret_cast<T *>(outputs[0]->addr);
  T zero = (T)0.0;
  T half = (T)0.5;
  T beta = (T)beta_;
  auto task = [&](size_t start, size_t end) {
    for (uint64_t i = start; i < end; ++i) {
      T diff = predict_addr[i] - target_addr[i];
      if (diff < zero) {
        diff = -diff;
      }
      if (diff < beta) {
        result_addr[i] = half * diff * diff / beta;
      } else {
        result_addr[i] = diff - (half * beta);
      }
    }
  };
  ParallelLaunchAutoSearch(task, tensor_size_, this, &parallel_search_info_);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
