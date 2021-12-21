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

#include "ops/fusion/max_pool_fusion.h"

namespace mindspore {
namespace ops {
void MaxPoolFusion::Init(const std::vector<int64_t> &kernel_size, const std::vector<int64_t> &stride,
                         const PadMode &pad_mode, const Format &format, const std::vector<int64_t> &pad,
                         const RoundMode &round_mode, const bool global, const ActivationType activation_type) {
  this->set_pad_mode(pad_mode);
  this->set_kernel_size(kernel_size);
  this->set_strides(stride);
  this->set_format(format);
  this->set_pad(pad);
  this->set_round_mode(round_mode);
  this->set_global(global);
  this->set_activation_type(activation_type);
}

void MaxPoolFusion::set_global(const bool global) { (void)AddAttr(kGlobal, MakeValue(global)); }

void MaxPoolFusion::set_activation_type(ActivationType activation_type) {
  int64_t swi = activation_type;
  (void)this->AddAttr(kActivationType, MakeValue(swi));
}

bool MaxPoolFusion::get_global() const {
  auto value_ptr = GetAttr(kGlobal);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<bool>(value_ptr);
}

ActivationType MaxPoolFusion::get_activation_type() const {
  auto value_ptr = GetAttr(kActivationType);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return ActivationType(GetValue<int64_t>(value_ptr));
}

namespace {
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto op_name = primitive->name();
  auto in_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShapeTrack())[kShape];
  auto format = Format(GetValue<int64_t>(primitive->GetAttr(kFormat)));
  if (format == NHWC) {
    in_shape = {in_shape[0], in_shape[3], in_shape[1], in_shape[2]};
  }
  const int64_t in_shape_size = 4;
  (void)CheckAndConvertUtils::CheckInteger("x_rank", SizeToLong(in_shape.size()), kEqual, in_shape_size, op_name);
  auto kernel_size = GetValue<std::vector<int64_t>>(primitive->GetAttr(kKernelSize));
  auto pad_mode = PadMode(GetValue<int64_t>(primitive->GetAttr(kPadMode)));
  auto batch = in_shape[0];
  auto channel = in_shape[1];
  auto in_h = in_shape[2];
  auto in_w = in_shape[3];

  auto strides = GetValue<std::vector<int64_t>>(primitive->GetAttr(kStrides));
  auto kernel_h = kernel_size[2];
  auto kernel_w = kernel_size[3];
  auto stride_h = strides[2];
  auto stride_w = strides[3];
  int64_t out_h = abstract::Shape::SHP_ANY;
  int64_t out_w = abstract::Shape::SHP_ANY;
  if (pad_mode == VALID) {
    out_h = static_cast<int64_t>(ceil((in_h - (kernel_h - 1)) / static_cast<float>(stride_h)));
    out_w = static_cast<int64_t>(ceil((in_w - (kernel_w - 1)) / static_cast<float>(stride_w)));
  } else if (pad_mode == SAME) {
    out_h = static_cast<int64_t>(ceil(in_h / static_cast<float>(stride_h)));
    out_w = static_cast<int64_t>(ceil(in_w / static_cast<float>(stride_w)));
  }
  std::vector<int64_t> out_shape = {batch, channel, out_h, out_w};
  if (format == NHWC) {
    out_shape = {batch, out_h, out_w, channel};
  }
  if (std::any_of(out_shape.begin(), out_shape.end(), [](int64_t a) { return a <= 0; })) {
    MS_LOG(EXCEPTION) << "Kernel size is invalid.";
  }
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr InferType(const std::vector<AbstractBasePtr> &input_args) { return input_args[0]->BuildType(); }
}  // namespace

AbstractBasePtr MaxPoolFusionInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (auto item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  return std::make_shared<abstract::AbstractTensor>(InferType(input_args), InferShape(primitive, input_args));
}
REGISTER_PRIMITIVE_C(kNameMaxPoolFusion, MaxPoolFusion);
}  // namespace ops
}  // namespace mindspore
