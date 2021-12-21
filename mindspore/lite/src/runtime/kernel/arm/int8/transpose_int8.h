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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_TRANSPOSE_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_TRANSPOSE_INT8_H_

#include <vector>
#include "nnacl/int8/pack_int8.h"
#include "nnacl/int8/transpose_int8.h"
#include "src/kernel_registry.h"
#include "src/inner_kernel.h"
#include "include/errorcode.h"

namespace mindspore::kernel {

typedef void (*TransposeFunc)(const void *src, void *dst, int batch, int plane, int channel);

class TransposeInt8CPUKernel : public InnerKernel {
 public:
  TransposeInt8CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                         const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx) {
    transpose_param_ = reinterpret_cast<TransposeParameter *>(op_parameter_);
  }
  ~TransposeInt8CPUKernel() = default;

  int Prepare() override;
  int ReSize() override;
  int Run() override;

 public:
  int DoTranspose(int task_id);

 private:
  void GetNHNCTransposeFunc(const lite::Tensor *in_tensor, const lite::Tensor *out_tensor,
                            const TransposeParameter *param);
  TransposeParameter *transpose_param_;
  TransposeFunc NHNCTransposeFunc_ = nullptr;
  int8_t *in_ptr_ = nullptr;
  int8_t *out_ptr_ = nullptr;
  int in_shape_[20] = {0};
  int out_shape_[20] = {0};
  int nhnc_param_[3] = {0};
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_TRANSPOSE_INT8_H_
