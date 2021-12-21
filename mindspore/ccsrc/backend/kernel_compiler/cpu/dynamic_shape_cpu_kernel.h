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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_DYNAMIC_SHAPE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_DYNAMIC_SHAPE_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class DynamicShapeCPUKernel : public CPUKernel {
 public:
  DynamicShapeCPUKernel() = default;
  ~DynamicShapeCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;
};

MS_REG_CPU_KERNEL_T(DynamicShape, KernelAttr(), DynamicShapeCPUKernel, float);
MS_REG_CPU_KERNEL_T(DynamicShape, KernelAttr(), DynamicShapeCPUKernel, int8_t)
MS_REG_CPU_KERNEL_T(DynamicShape, KernelAttr(), DynamicShapeCPUKernel, int16_t)
MS_REG_CPU_KERNEL_T(DynamicShape, KernelAttr(), DynamicShapeCPUKernel, int32_t)
MS_REG_CPU_KERNEL_T(DynamicShape, KernelAttr(), DynamicShapeCPUKernel, int64_t)
MS_REG_CPU_KERNEL_T(DynamicShape, KernelAttr(), DynamicShapeCPUKernel, uint8_t)
MS_REG_CPU_KERNEL_T(DynamicShape, KernelAttr(), DynamicShapeCPUKernel, uint16_t)
MS_REG_CPU_KERNEL_T(DynamicShape, KernelAttr(), DynamicShapeCPUKernel, uint32_t)
MS_REG_CPU_KERNEL_T(DynamicShape, KernelAttr(), DynamicShapeCPUKernel, uint64_t)
MS_REG_CPU_KERNEL_T(DynamicShape, KernelAttr(), DynamicShapeCPUKernel, bool)
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_DYNAMIC_SHAPE_CPU_KERNEL_H_
