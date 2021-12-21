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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RL_TENSOR_ARRAY_WRITE_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RL_TENSOR_ARRAY_WRITE_KERNEL_H_

#include <string>
#include <vector>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class TensorArrayCPUWriteKernel : public CPUKernel {
 public:
  TensorArrayCPUWriteKernel();
  ~TensorArrayCPUWriteKernel() = default;

  const std::vector<size_t> &GetInputSizeList() const override;
  const std::vector<size_t> &GetOutputSizeList() const override;
  const std::vector<size_t> &GetWorkspaceSizeList() const override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;
  void InitKernel(const CNodePtr &kernel_node) override;

 private:
  size_t value_size_;
  std::vector<size_t> shapes_;
  TypeId type_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};

// index int64
MS_REG_CPU_KERNEL(TensorArrayWrite,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddOutputAttr(kNumberTypeInt64),
                  TensorArrayCPUWriteKernel);
MS_REG_CPU_KERNEL(TensorArrayWrite,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddOutputAttr(kNumberTypeInt64),
                  TensorArrayCPUWriteKernel);
MS_REG_CPU_KERNEL(TensorArrayWrite,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt16)
                    .AddOutputAttr(kNumberTypeInt64),
                  TensorArrayCPUWriteKernel);
MS_REG_CPU_KERNEL(TensorArrayWrite,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeUInt64)
                    .AddOutputAttr(kNumberTypeInt64),
                  TensorArrayCPUWriteKernel);
MS_REG_CPU_KERNEL(TensorArrayWrite,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeUInt32)
                    .AddOutputAttr(kNumberTypeInt64),
                  TensorArrayCPUWriteKernel);
MS_REG_CPU_KERNEL(TensorArrayWrite,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeUInt16)
                    .AddOutputAttr(kNumberTypeInt64),
                  TensorArrayCPUWriteKernel);
MS_REG_CPU_KERNEL(TensorArrayWrite,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeUInt8)
                    .AddOutputAttr(kNumberTypeInt64),
                  TensorArrayCPUWriteKernel);
MS_REG_CPU_KERNEL(TensorArrayWrite,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeInt64),
                  TensorArrayCPUWriteKernel);
MS_REG_CPU_KERNEL(TensorArrayWrite,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeFloat16)
                    .AddOutputAttr(kNumberTypeInt64),
                  TensorArrayCPUWriteKernel);
MS_REG_CPU_KERNEL(TensorArrayWrite,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeBool)
                    .AddOutputAttr(kNumberTypeInt64),
                  TensorArrayCPUWriteKernel);
// index int32
MS_REG_CPU_KERNEL(TensorArrayWrite,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddOutputAttr(kNumberTypeInt64),
                  TensorArrayCPUWriteKernel);
MS_REG_CPU_KERNEL(TensorArrayWrite,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddOutputAttr(kNumberTypeInt64),
                  TensorArrayCPUWriteKernel);
MS_REG_CPU_KERNEL(TensorArrayWrite,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeInt16)
                    .AddOutputAttr(kNumberTypeInt64),
                  TensorArrayCPUWriteKernel);
MS_REG_CPU_KERNEL(TensorArrayWrite,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeUInt64)
                    .AddOutputAttr(kNumberTypeInt64),
                  TensorArrayCPUWriteKernel);
MS_REG_CPU_KERNEL(TensorArrayWrite,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeUInt32)
                    .AddOutputAttr(kNumberTypeInt64),
                  TensorArrayCPUWriteKernel);
MS_REG_CPU_KERNEL(TensorArrayWrite,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeUInt16)
                    .AddOutputAttr(kNumberTypeInt64),
                  TensorArrayCPUWriteKernel);
MS_REG_CPU_KERNEL(TensorArrayWrite,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeUInt8)
                    .AddOutputAttr(kNumberTypeInt64),
                  TensorArrayCPUWriteKernel);
MS_REG_CPU_KERNEL(TensorArrayWrite,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeInt64),
                  TensorArrayCPUWriteKernel);
MS_REG_CPU_KERNEL(TensorArrayWrite,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeFloat16)
                    .AddOutputAttr(kNumberTypeInt64),
                  TensorArrayCPUWriteKernel);
MS_REG_CPU_KERNEL(TensorArrayWrite,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeBool)
                    .AddOutputAttr(kNumberTypeInt64),
                  TensorArrayCPUWriteKernel);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RL_TENSOR_ARRAY_WRITE_KERNEL_H_
