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

#include "ops/fusion/conv2d_transpose_fusion.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
void Conv2dTransposeFusion::Init(int64_t in_channel, int64_t out_channel, const std::vector<int64_t> &kernel_size,
                                 int64_t mode, const PadMode &pad_mode, const std::vector<int64_t> &pad,
                                 const std::vector<int64_t> &stride, const std::vector<int64_t> &dilation,
                                 int64_t group, const Format &format, const std::vector<int64_t> &pad_list,
                                 const std::vector<int64_t> &output_paddings, ActivationType activation_type) {
  set_in_channel(in_channel);
  set_out_channel(out_channel);
  set_kernel_size(kernel_size);
  set_mode(mode);
  set_pad_mode(pad_mode);
  set_pad(pad);
  set_stride(stride);
  set_dilation(dilation);
  set_group(group);
  set_format(format);
  set_pad_list(pad_list);
  set_output_paddings(output_paddings);
  set_activation_type(activation_type);
}

void Conv2dTransposeFusion::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  (void)CheckAndConvertUtils::CheckInteger(kKernelSize, SizeToLong(kernel_size.size()), kEqual, 2, name());
  for (int64_t item : kernel_size) {
    (void)CheckAndConvertUtils::CheckInteger(kKernelSize, item, kGreaterEqual, 1, name());
  }
  (void)AddAttr(kKernelSize, MakeValue(kernel_size));
}

void Conv2dTransposeFusion::set_dilation(const std::vector<int64_t> &dilation) {
  (void)CheckAndConvertUtils::CheckInteger(kDilation, SizeToLong(dilation.size()), kEqual, 2, name());
  for (int64_t item : dilation) {
    (void)CheckAndConvertUtils::CheckInteger(kDilation, item, kGreaterEqual, 1, name());
  }
  (void)AddAttr(kDilation, MakeValue(dilation));
}

void Conv2dTransposeFusion::set_output_paddings(const std::vector<int64_t> &output_paddings) {
  (void)CheckAndConvertUtils::CheckInteger(kOutputPaddings, SizeToLong(output_paddings.size()), kGreaterEqual, 1,
                                           name());
  for (int64_t item : output_paddings) {
    (void)CheckAndConvertUtils::CheckInteger(kOutputPaddings, item, kGreaterEqual, 0, name());
  }
  (void)AddAttr(kOutputPaddings, MakeValue(output_paddings));
}

void Conv2dTransposeFusion::set_activation_type(ActivationType activation_type) {
  int64_t swi = activation_type;
  this->AddAttr(kActivationType, MakeValue(swi));
}

std::vector<int64_t> Conv2dTransposeFusion::get_output_paddings() const {
  auto value_ptr = GetAttr(kOutputPaddings);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

ActivationType Conv2dTransposeFusion::get_activation_type() const {
  auto value_ptr = GetAttr(kActivationType);
  return ActivationType(GetValue<int64_t>(value_ptr));
}

REGISTER_PRIMITIVE_C(kNameConv2dTransposeFusion, Conv2dTransposeFusion);
}  // namespace ops
}  // namespace mindspore
