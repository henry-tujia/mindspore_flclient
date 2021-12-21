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

#include "src/runtime/kernel/arm/fp16/fused_batchnorm_fp16.h"
#include "nnacl/fp16/batchnorm_fp16.h"
#include "nnacl/fp16/cast_fp16.h"
#include "src/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_FusedBatchNorm;

namespace mindspore::kernel {
constexpr static int kInScaleIdx = 1;
constexpr static int kInOffsetIdx = 2;
constexpr static int kInCurrentMeanIdx = 3;
constexpr static int kInCurrentVarIdx = 4;
constexpr static int kMaxInIdx = 5;
constexpr static int kOutScaleIdx = 1;
constexpr static int kOutOffsetIdx = 2;
constexpr static int kOutCurrentMeanIdx = 3;
constexpr static int kOutCurrentVarIdx = 4;

void FusedBatchnormFp16CPUKernel::CalcMeanVar(float16_t *in, float16_t *scale, float16_t *offset, float16_t *save_mean,
                                              float16_t *save_variance) {
  auto param = reinterpret_cast<BatchNormParameter *>(op_parameter_);
  MS_ASSERT(param != nullptr);
  float16_t *current_mean = static_cast<float16_t *>(mean_);
  float16_t *current_var = static_cast<float16_t *>(variance_);

  std::fill(current_mean, current_mean + in_tensors_.at(kInCurrentMeanIdx)->ElementsNum(), 0.f);
  std::fill(current_var, current_var + in_tensors_.at(kInCurrentVarIdx)->ElementsNum(), 0.f);
  FusedBatchNormFp16MeanVar(in, current_mean, current_var, param, save_mean, save_variance);

  MS_ASSERT(out_tensors_.at(kOutScaleIdx)->data() != nullptr);
  MS_ASSERT(out_tensors_.at(kOutOffsetIdx)->data() != nullptr);
  MS_ASSERT(out_tensors_.at(kOutCurrentMeanIdx)->data() != nullptr);
  MS_ASSERT(out_tensors_.at(kOutCurrentVarIdx)->data() != nullptr);
  memcpy(out_tensors_.at(kOutScaleIdx)->data(), scale, out_tensors_.at(kOutScaleIdx)->Size());
  memcpy(out_tensors_.at(kOutOffsetIdx)->data(), offset, out_tensors_.at(kOutOffsetIdx)->Size());
  memcpy(out_tensors_.at(kOutCurrentMeanIdx)->data(), current_mean, out_tensors_.at(kOutCurrentMeanIdx)->Size());
  memcpy(out_tensors_.at(kOutCurrentVarIdx)->data(), current_var, out_tensors_.at(kOutCurrentVarIdx)->Size());

  // Copy to local variables
  memcpy(scale_, scale, in_tensors_.at(kInScaleIdx)->Size());
  memcpy(offset_, offset, in_tensors_.at(kInOffsetIdx)->Size());

  trained_ = true;  // trained at least once
}

int FusedBatchnormFp16CPUKernel::DoExecute(int task_id) {
  auto param = reinterpret_cast<BatchNormParameter *>(op_parameter_);
  CHECK_NULL_RETURN(in_tensors_.at(0)->data());
  CHECK_NULL_RETURN(out_tensors_.at(0)->data());
  if (IsTrain() && IsTrainable() && in_tensors_.size() >= kMaxInIdx) {
    CalcMeanVar(static_cast<float16_t *>(in_tensors_.at(0)->data()),
                static_cast<float16_t *>(in_tensors_.at(kInScaleIdx)->data()),
                static_cast<float16_t *>(in_tensors_.at(kInOffsetIdx)->data()),
                static_cast<float16_t *>(in_tensors_.at(kInCurrentMeanIdx)->data()),
                static_cast<float16_t *>(in_tensors_.at(kInCurrentVarIdx)->data()));
  }
  FusedBatchNormFp16(in_tensors_.at(0)->data(), scale_, offset_, mean_, variance_, param, task_id,
                     out_tensors_.at(0)->data());
  return RET_OK;
}

int FusedBatchnormFp16CPUKernel::Eval() {
  InnerKernel::Eval();
  if (trained_) {
    float16_t *save_mean = static_cast<float16_t *>(in_tensors_.at(kInCurrentMeanIdx)->data());
    float16_t *save_var = static_cast<float16_t *>(in_tensors_.at(kInCurrentVarIdx)->data());
    float16_t *scale = static_cast<float16_t *>(in_tensors_.at(kInScaleIdx)->data());
    float16_t *bias = static_cast<float16_t *>(in_tensors_.at(kInOffsetIdx)->data());
    CHECK_NULL_RETURN(save_mean);
    CHECK_NULL_RETURN(save_var);
    CHECK_NULL_RETURN(scale);
    CHECK_NULL_RETURN(bias);

    // Copy to local variables
    memcpy(scale_, scale, in_tensors_.at(kInScaleIdx)->Size());
    memcpy(offset_, bias, in_tensors_.at(kInOffsetIdx)->Size());
    memcpy(mean_, save_mean, in_tensors_.at(kInCurrentMeanIdx)->Size());
    memcpy(variance_, save_var, in_tensors_.at(kInCurrentVarIdx)->Size());
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_FusedBatchNorm, LiteKernelCreator<FusedBatchnormFp16CPUKernel>)
}  // namespace mindspore::kernel
