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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ELTWISE_GRAD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ELTWISE_GRAD_CPU_KERNEL_H_
#include <memory>
#include <vector>
#include <limits>
#include <string>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
constexpr size_t kInputMinNum = 2;
constexpr size_t kOutputNum = 1;
template <typename T>
class EltWiseGradCPUKernel : public CPUKernel {
 public:
  EltWiseGradCPUKernel() = default;
  ~EltWiseGradCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  void InitComputeFunc();
  void ReluGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const;
  void ReLU6Grad(const T *input1, const T *input2, T *out, size_t start, size_t end) const;
  void AbsGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const;
  void SigmoidGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const;
  void SqrtGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const;
  void TanhGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const;
  void GeluGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const;
  void AsinGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const;
  void ACosGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const;
  void AtanGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const;
  void AsinhGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const;
  void AcoshGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const;
  void SoftplusGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const;

  using TypeComputeFunc = std::function<void(EltWiseGradCPUKernel *, const T *, const T *, T *, size_t, size_t)>;
  TypeComputeFunc compute_func_{nullptr};
};

MS_REG_CPU_KERNEL_T(
  ReluGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  EltWiseGradCPUKernel, float);
MS_REG_CPU_KERNEL_T(
  ReLU6Grad,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  EltWiseGradCPUKernel, float);
MS_REG_CPU_KERNEL_T(
  AbsGrad, KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
  EltWiseGradCPUKernel, int);
MS_REG_CPU_KERNEL_T(
  AbsGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  EltWiseGradCPUKernel, float);
MS_REG_CPU_KERNEL_T(
  AbsGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  EltWiseGradCPUKernel, double);
MS_REG_CPU_KERNEL_T(
  SigmoidGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  EltWiseGradCPUKernel, float);
MS_REG_CPU_KERNEL_T(
  SqrtGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  EltWiseGradCPUKernel, float);
MS_REG_CPU_KERNEL_T(
  SqrtGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
  EltWiseGradCPUKernel, double);
MS_REG_CPU_KERNEL_T(
  TanhGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  EltWiseGradCPUKernel, float);
MS_REG_CPU_KERNEL_T(GeLUGrad,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddOutputAttr(kNumberTypeFloat32),
                    EltWiseGradCPUKernel, float);
MS_REG_CPU_KERNEL_T(
  AsinGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  EltWiseGradCPUKernel, float);
MS_REG_CPU_KERNEL_T(
  ACosGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  EltWiseGradCPUKernel, float);
MS_REG_CPU_KERNEL_T(
  AtanGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  EltWiseGradCPUKernel, float);
MS_REG_CPU_KERNEL_T(
  AsinhGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  EltWiseGradCPUKernel, float);
MS_REG_CPU_KERNEL_T(
  AcoshGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  EltWiseGradCPUKernel, float);
MS_REG_CPU_KERNEL_T(
  SoftplusGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
  EltWiseGradCPUKernel, float);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ELTWISE_GRAD_CPU_KERNEL_H_
