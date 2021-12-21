/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_GATHER_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_GATHER_H_

#include <arm_neon.h>
#include <vector>
#include "include/errorcode.h"
#include "src/inner_kernel.h"
#include "nnacl/gather_parameter.h"
#include "nnacl/base/gather_base.h"

namespace mindspore::kernel {
class GatherFp16CPUKernel : public InnerKernel {
 public:
  GatherFp16CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                      const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx) {}
  ~GatherFp16CPUKernel() override;

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int DoGather(int task_id);

 private:
  int *indices_data_ = nullptr;
  int AssignIndicesData(bool isIndicesInt32, int indices_num, const lite::Tensor *indices_tensor);
  void FreeIndicesData();
  float16_t *input_data_ = nullptr;
  bool const_input_ = false;
  bool is_indices_int32_ = false;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_GATHER_H_
