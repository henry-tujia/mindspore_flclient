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
#include "backend/kernel_compiler/gpu/other/dynamic_broadcastto_gpu_kernel.h"

namespace mindspore {
namespace kernel {
MS_REG_GPU_KERNEL_TWO(
  DynamicBroadcastTo,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
  DynamicBroadcastToGpuKernel, double, int64_t)
MS_REG_GPU_KERNEL_TWO(
  DynamicBroadcastTo,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
  DynamicBroadcastToGpuKernel, float, int64_t)
MS_REG_GPU_KERNEL_TWO(
  DynamicBroadcastTo,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
  DynamicBroadcastToGpuKernel, half, int64_t)
MS_REG_GPU_KERNEL_TWO(
  DynamicBroadcastTo,
  KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt16),
  DynamicBroadcastToGpuKernel, int16_t, int64_t)
MS_REG_GPU_KERNEL_TWO(
  DynamicBroadcastTo,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
  DynamicBroadcastToGpuKernel, int32_t, int64_t)
MS_REG_GPU_KERNEL_TWO(
  DynamicBroadcastTo,
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
  DynamicBroadcastToGpuKernel, int64_t, int64_t)
MS_REG_GPU_KERNEL_TWO(
  DynamicBroadcastTo,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
  DynamicBroadcastToGpuKernel, double, int32_t)
MS_REG_GPU_KERNEL_TWO(
  DynamicBroadcastTo,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
  DynamicBroadcastToGpuKernel, float, int32_t)
MS_REG_GPU_KERNEL_TWO(
  DynamicBroadcastTo,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
  DynamicBroadcastToGpuKernel, half, int32_t)
MS_REG_GPU_KERNEL_TWO(
  DynamicBroadcastTo,
  KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
  DynamicBroadcastToGpuKernel, int16_t, int32_t)
MS_REG_GPU_KERNEL_TWO(
  DynamicBroadcastTo,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  DynamicBroadcastToGpuKernel, int32_t, int32_t)
MS_REG_GPU_KERNEL_TWO(
  DynamicBroadcastTo,
  KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
  DynamicBroadcastToGpuKernel, int64_t, int32_t)
}  // namespace kernel
}  // namespace mindspore
