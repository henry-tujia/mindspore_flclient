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

#include "src/runtime/kernel/arm/int8/deconvolution_depthwise_int8.h"
#include "include/errorcode.h"
#include "nnacl/int8/conv_depthwise_int8.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
DeconvolutionDepthwiseInt8CPUKernel::~DeconvolutionDepthwiseInt8CPUKernel() {
  if (sliding_ != nullptr) {
    delete sliding_;
    sliding_ = nullptr;
  }
  if (packed_weight_ != nullptr) {
    delete packed_weight_;
    packed_weight_ = nullptr;
  }
  FreeQuantParam();
}

int DeconvolutionDepthwiseInt8CPUKernel::InitWeightBias() {
  // init weight: int8 -> int16
  // o, h, w, i -> o/8, h, w, i, 8; o equals to group, i equals to 1
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  CHECK_NULL_RETURN(weight_tensor);
  auto origin_weight = reinterpret_cast<int8_t *>(weight_tensor->data());
  CHECK_NULL_RETURN(origin_weight);
  if (origin_weight == nullptr) {
    MS_LOG(ERROR) << "origin_weight nullptr";
    return RET_ERROR;
  }
  int OC4 = UP_DIV(weight_tensor->Batch(), C4NUM);
  int pack_weight_size = C4NUM * OC4 * weight_tensor->Height() * weight_tensor->Width();
  packed_weight_ = reinterpret_cast<int16_t *>(malloc(pack_weight_size * sizeof(int16_t)));
  if (packed_weight_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  PackDeconvDepthwiseInt8Weight(origin_weight, packed_weight_, weight_tensor->Height() * weight_tensor->Width(),
                                weight_tensor->Batch(), &(conv_param_->conv_quant_arg_));

  bias_data_ = reinterpret_cast<int32_t *>(malloc(C4NUM * OC4 * sizeof(int32_t)));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  memset(bias_data_, 0, C4NUM * OC4 * sizeof(int32_t));
  if (in_tensors_.size() == kInputSize2) {
    auto bias_tensor = in_tensors_.at(kBiasIndex);
    CHECK_NULL_RETURN(bias_tensor);
    auto ori_bias = reinterpret_cast<int32_t *>(bias_tensor->data());
    CHECK_NULL_RETURN(ori_bias);
    MS_CHECK_GT(bias_tensor->ElementsNum(), 0, RET_ERROR);
    memcpy(bias_data_, ori_bias, bias_tensor->ElementsNum() * sizeof(int32_t));
  }
  conv_param_->thread_num_ = MSMIN(thread_count_, OC4);
  return RET_OK;
}

int DeconvolutionDepthwiseInt8CPUKernel::InitSlideParam() {
  MS_CHECK_TRUE_RET(in_tensors_.front()->shape().size() == DIMENSION_4D, RET_ERROR);
  MS_CHECK_TRUE_RET(out_tensors_.front()->shape().size() == DIMENSION_4D, RET_ERROR);

  conv_param_->input_batch_ = out_tensors_.front()->shape().at(kNHWC_N);
  conv_param_->input_h_ = out_tensors_.front()->shape().at(kNHWC_H);
  conv_param_->input_w_ = out_tensors_.front()->shape().at(kNHWC_W);
  conv_param_->input_channel_ = C4NUM;
  conv_param_->output_batch_ = in_tensors_.front()->shape().at(kNHWC_N);
  conv_param_->output_h_ = in_tensors_.front()->shape().at(kNHWC_H);
  conv_param_->output_w_ = in_tensors_.front()->shape().at(kNHWC_W);
  conv_param_->output_channel_ = in_tensors_.front()->shape().at(kNHWC_C);

  InitSlidingParamConvDw(sliding_, conv_param_, C4NUM);

  sliding_->in_h_step_ = conv_param_->input_w_ * C4NUM;
  sliding_->in_sh_step_ = conv_param_->input_w_ * C4NUM * conv_param_->stride_h_;    // stride H
  sliding_->in_sw_step_ = C4NUM * conv_param_->stride_h_;                            // stride W
  sliding_->in_kh_step_ = conv_param_->input_w_ * C4NUM * conv_param_->dilation_h_;  // kernel H
  sliding_->in_kw_step_ = C4NUM * conv_param_->dilation_w_;                          // kernel W
  return RET_OK;
}

int DeconvolutionDepthwiseInt8CPUKernel::InitBuffer() {
  int pack_input_size = conv_param_->input_batch_ * conv_param_->input_h_ * conv_param_->input_w_ * C4NUM *
                        UP_DIV(conv_param_->input_channel_, C4NUM);
  packed_input_ = reinterpret_cast<int16_t *>(ms_context_->allocator->Malloc(pack_input_size * sizeof(int16_t)));
  if (packed_input_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }

  if (conv_param_->input_channel_ % C4NUM != 0) {
    need_align_ = true;
    int pack_output_size = conv_param_->output_batch_ * conv_param_->output_h_ * conv_param_->output_w_ * C4NUM *
                           UP_DIV(conv_param_->output_channel_, C4NUM);
    packed_output_ = reinterpret_cast<int8_t *>(ms_context_->allocator->Malloc(pack_output_size * sizeof(int8_t)));
    if (packed_output_ == nullptr) {
      MS_LOG(ERROR) << "Malloc buffer failed.";
      return RET_ERROR;
    }
    memset(packed_output_, 0, pack_output_size * sizeof(int8_t));
  }

  output_buffer_ = reinterpret_cast<int32_t *>(ms_context_->allocator->Malloc(
    conv_param_->output_h_ * conv_param_->output_w_ * C4NUM * conv_param_->thread_num_ * sizeof(int32_t)));
  if (output_buffer_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int DeconvolutionDepthwiseInt8CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_NULL_RETURN(in_tensors_.at(kInputIndex));
  CHECK_NULL_RETURN(in_tensors_.at(kWeightIndex));
  CHECK_NULL_RETURN(conv_param_);

  sliding_ = new (std::nothrow) SlidingWindowParam;
  if (sliding_ == nullptr) {
    MS_LOG(ERROR) << "new SlidingWindowParam fail!";
    return RET_ERROR;
  }
  auto ret = ConvolutionBaseCPUKernel::SetQuantParam();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set quant param failed.";
    return ret;
  }
  ret = InitWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Deconv Depthwise int8 InitWeightBias error!";
    return ret;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int DeconvolutionDepthwiseInt8CPUKernel::ReSize() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_.front());
  CHECK_NULL_RETURN(out_tensors_.front());
  CHECK_NULL_RETURN(conv_param_);
  CHECK_NULL_RETURN(sliding_);

  InitSlideParam();
  ConvolutionBaseCPUKernel::Prepare();
  return RET_OK;
}

int DeconvolutionDepthwiseInt8CPUKernel::Execute(int task_id) {
  auto buffer = output_buffer_ + conv_param_->output_h_ * conv_param_->output_w_ * C4NUM * task_id;
  DeconvDwInt8(packed_output_, buffer, packed_input_, packed_weight_, reinterpret_cast<int32_t *>(bias_data_),
               conv_param_, sliding_, task_id);
  return RET_OK;
}

int DeconvDwInt8Run(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto deconv_dw_int8 = reinterpret_cast<DeconvolutionDepthwiseInt8CPUKernel *>(cdata);
  auto ret = deconv_dw_int8->Execute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DeconvolutionDepthwiseInt8Run error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int DeconvolutionDepthwiseInt8CPUKernel::Run() {
  if (conv_param_->input_channel_ != conv_param_->output_channel_) {
    MS_LOG(ERROR) << "Only support input channel equals output channel.";
    return RET_ERROR;
  }
  auto ret = InitBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Deconv Depthwise int8 InitBuffer error!";
    ms_context_->allocator->Free(packed_input_);
    packed_input_ = nullptr;
    ms_context_->allocator->Free(output_buffer_);
    output_buffer_ = nullptr;
    if (need_align_) {
      ms_context_->allocator->Free(packed_output_);
    }
    return ret;
  }

  CHECK_NULL_RETURN(packed_weight_);
  CHECK_NULL_RETURN(bias_data_);

  auto input_tensor = in_tensors_.at(kInputIndex);
  auto output_tensor = out_tensors_.at(kOutputIndex);
  auto input_addr = reinterpret_cast<int8_t *>(input_tensor->data());
  auto output_addr = reinterpret_cast<int8_t *>(output_tensor->data());
  CHECK_NULL_RETURN(input_addr);
  CHECK_NULL_RETURN(output_addr);

  // pack input, assume input format: NHWC -> NHWC4
  PackDepthwiseInt8Input(input_addr, packed_input_, conv_param_);

  if (!need_align_) {
    MS_CHECK_GT(out_tensors_.at(kOutputIndex)->ElementsNum(), 0, RET_ERROR);
    memset(output_addr, 0, out_tensors_.at(kOutputIndex)->ElementsNum() * sizeof(int8_t));
    packed_output_ = output_addr;
  }

  ret = ParallelLaunch(this->ms_context_, DeconvDwInt8Run, this, conv_param_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DeconvDwInt8Run error: error_code[" << ret << "]";
  }

  if (need_align_) {
    PackNHWC4ToNHWCInt8(packed_output_, output_addr, conv_param_->output_batch_,
                        conv_param_->output_h_ * conv_param_->output_w_, conv_param_->output_channel_);
    ms_context_->allocator->Free(packed_output_);
    packed_output_ = nullptr;
  }
  ms_context_->allocator->Free(packed_input_);
  packed_input_ = nullptr;
  ms_context_->allocator->Free(output_buffer_);
  output_buffer_ = nullptr;
  return ret;
}
}  // namespace mindspore::kernel
