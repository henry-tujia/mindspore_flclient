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

#include "src/runtime/kernel/arm/int8/sub_int8.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_SubFusion;

namespace mindspore::kernel {
SubInt8CPUKernel::~SubInt8CPUKernel() {
  if (quant_param_ != nullptr) {
    free(quant_param_);
    quant_param_ = nullptr;
  }
}

int SubInt8CPUKernel::Prepare() {
  lite::Tensor *input0 = in_tensors_.at(0);
  lite::Tensor *input1 = in_tensors_.at(1);
  lite::Tensor *output = out_tensors_.at(0);
  MS_ASSERT(input0);
  MS_ASSERT(input1);
  MS_ASSERT(output);

  broadcast_ = input0->ElementsNum() != input1->ElementsNum();

  quant_param_ = reinterpret_cast<SubQuantArg *>(malloc(sizeof(SubQuantArg)));
  if (quant_param_ == nullptr) {
    MS_LOG(ERROR) << "Malloc SubQuantArg for Sub int8 op failed!";
    return RET_ERROR;
  }
  quant_param_->in0_args_.scale_ = static_cast<float>(input0->quant_params().front().scale);
  quant_param_->in0_args_.zp_ = -input0->quant_params().front().zeroPoint;
  quant_param_->in1_args_.scale_ = static_cast<float>(input1->quant_params().front().scale);
  quant_param_->in1_args_.zp_ = -input1->quant_params().front().zeroPoint;
  quant_param_->out_args_.scale_ = static_cast<float>(output->quant_params().front().scale);
  quant_param_->out_args_.zp_ = output->quant_params().front().zeroPoint;

  const int left_shift = 20;
  const double twice_max_input_scale = 2 * std::max(quant_param_->in0_args_.scale_, quant_param_->in1_args_.scale_);
  const double real_input0_multiplier = quant_param_->in0_args_.scale_ / twice_max_input_scale;
  const double real_input1_multiplier = quant_param_->in1_args_.scale_ / twice_max_input_scale;
  const double real_output_multiplier = twice_max_input_scale / ((1 << left_shift) * quant_param_->out_args_.scale_);

  QuantizeMultiplierSmallerThanOne(real_input0_multiplier, &quant_param_->input0_multiplier_,
                                   &quant_param_->input0_shift_);
  QuantizeMultiplierSmallerThanOne(real_input1_multiplier, &quant_param_->input1_multiplier_,
                                   &quant_param_->input1_shift_);
  QuantizeMultiplierSmallerThanOne(real_output_multiplier, &quant_param_->output_multiplier_,
                                   &quant_param_->output_shift_);

  quant_param_->output_activation_min_ = std::numeric_limits<int8_t>::min();
  quant_param_->output_activation_max_ = std::numeric_limits<int8_t>::max();

  int left_shift0 = -quant_param_->input0_shift_ > 0 ? -quant_param_->input0_shift_ : 0;
  quant_param_->right_shift0_ = -quant_param_->input0_shift_ > 0 ? 0 : quant_param_->input0_shift_;

  int left_shift1 = -quant_param_->input1_shift_ > 0 ? -quant_param_->input1_shift_ : 0;
  quant_param_->right_shift1_ = -quant_param_->input1_shift_ > 0 ? 0 : quant_param_->input1_shift_;

  quant_param_->left_shift_out_ = -quant_param_->output_shift_ > 0 ? -quant_param_->output_shift_ : 0;
  quant_param_->right_shift_out_ = -quant_param_->output_shift_ > 0 ? 0 : quant_param_->output_shift_;

  quant_param_->left_shift_result0_ = (1 << left_shift) * ((1 << left_shift0));
  quant_param_->left_shift_result1_ = (1 << left_shift) * ((1 << left_shift1));

  MS_ASSERT(left_shift + left_shift0 == left_shift);
  MS_ASSERT(left_shift + left_shift1 == left_shift);

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int SubInt8CPUKernel::ReSize() { return RET_OK; }

int SubInt8CPUKernel::DoExecute(int task_id) {
  auto input0_data_ = static_cast<int8_t *>(in_tensors_.at(0)->MutableData());
  auto input1_data_ = static_cast<int8_t *>(in_tensors_.at(1)->MutableData());
  auto output_data_ = static_cast<int8_t *>(out_tensors_.at(0)->MutableData());
  auto element_num = out_tensors_[0]->ElementsNum();

  MS_ASSERT(op_parameter_->thread_num_ != 0);
  int stride = UP_DIV(element_num, op_parameter_->thread_num_);
  int count = MSMIN(stride, element_num - stride * task_id);
  if (count <= 0) {
    return RET_OK;
  }

  auto ret = RET_OK;
  if (broadcast_) {
    ret = SubInt8(tile0_data_ + task_id * stride, tile1_data_ + task_id * stride, output_data_ + task_id * stride,
                  count, quant_param_);
  } else {
    ret = SubInt8(input0_data_ + task_id * stride, input1_data_ + task_id * stride, output_data_ + task_id * stride,
                  count, quant_param_);
  }

  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Subint8 function error error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SubInt8Run(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto sub_kernel = reinterpret_cast<SubInt8CPUKernel *>(cdata);
  auto ret = sub_kernel->DoExecute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SubInt8 DoExecute error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SubInt8CPUKernel::Run() {
  if (broadcast_) {
    ArithmeticParameter tile_para;
    tile_para.ndim_ = out_tensors_.at(0)->shape().size();
    for (size_t i = 0; i < tile_para.ndim_; i++) {
      tile_para.in_shape0_[i] = in_tensors_.at(0)->DimensionSize(i);
      tile_para.in_shape1_[i] = in_tensors_.at(1)->DimensionSize(i);
      tile_para.out_shape_[i] = out_tensors_.at(0)->DimensionSize(i);
    }
    tile0_data_ = static_cast<int8_t *>(ms_context_->allocator->Malloc(out_tensors_.at(0)->Size()));
    if (tile0_data_ == nullptr) {
      MS_LOG(ERROR) << "malloc memory fail!";
      return RET_ERROR;
    }
    tile1_data_ = static_cast<int8_t *>(ms_context_->allocator->Malloc(out_tensors_.at(0)->Size()));
    if (tile1_data_ == nullptr) {
      MS_LOG(ERROR) << "malloc memory fail!";
      ms_context_->allocator->Free(tile0_data_);
      return RET_ERROR;
    }
    TileDimensionsInt8(static_cast<int8_t *>(in_tensors_.at(0)->data()),
                       static_cast<int8_t *>(in_tensors_.at(1)->data()), reinterpret_cast<int8_t *>(tile0_data_),
                       reinterpret_cast<int8_t *>(tile1_data_), &tile_para);
  }
  auto ret = ParallelLaunch(this->ms_context_, SubInt8Run, this, op_parameter_->thread_num_);
  if (broadcast_) {
    ms_context_->allocator->Free(tile0_data_);
    ms_context_->allocator->Free(tile1_data_);
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SubInt8Run function error error_code[" << ret << "]";
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_SubFusion, LiteKernelCreator<SubInt8CPUKernel>)
}  // namespace mindspore::kernel
