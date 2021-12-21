/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "nnacl/conv_parameter.h"
#include "src/ops/populate/populate_register.h"
using mindspore::schema::PrimitiveType_Conv2DFusion;

namespace mindspore {
namespace lite {
OpParameter *PopulateConvParameter(const void *prim) {
  auto primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto value = primitive->value_as_Conv2DFusion();
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return nullptr;
  }

  auto *param = reinterpret_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc ConvParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(ConvParameter));

  param->op_parameter_.type_ = primitive->value_type();
  auto kernel_size = value->kernel_size();
  auto stride = value->stride();
  auto pad_list = value->pad_list();
  auto dilation = value->dilation();
  if (kernel_size != nullptr) {
    if (kernel_size->size() < kMinShapeSizeTwo) {
      MS_LOG(ERROR) << "kernel size is invalid.";
      free(param);
      return nullptr;
    }
    param->kernel_h_ = static_cast<int>(*(kernel_size->begin()));
    param->kernel_w_ = static_cast<int>(*(kernel_size->begin() + 1));
  } else {
    param->kernel_h_ = -1;
    param->kernel_w_ = -1;
  }
  if (stride == nullptr || dilation == nullptr) {
    MS_LOG(ERROR) << "kernel_size/stride/dilation is nullptr";
    free(param);
    return nullptr;
  }
  if (stride->size() < kMinShapeSizeTwo || dilation->size() < kMinShapeSizeTwo) {
    MS_LOG(ERROR) << "stride size: " << stride->size() << ", dilation size: " << dilation->size();
    free(param);
    return nullptr;
  }

  param->group_ = static_cast<int>(value->group());
  param->stride_h_ = static_cast<int>(*(stride->begin()));
  param->stride_w_ = static_cast<int>(*(stride->begin() + 1));
  switch (value->pad_mode()) {
    case schema::PadMode_SAME:
      param->pad_mode_ = Pad_same;
      break;
    case schema::PadMode_VALID:
      param->pad_mode_ = Pad_valid;
      break;
    default:
      param->pad_mode_ = Pad_pad;
  }
  if (pad_list == nullptr || pad_list->size() < kMinShapeSizeFour) {
    param->pad_u_ = 0;
    param->pad_d_ = 0;
    param->pad_l_ = 0;
    param->pad_r_ = 0;
  } else {
    param->pad_u_ = static_cast<int>(*(pad_list->begin()));
    param->pad_d_ = static_cast<int>(*(pad_list->begin() + 1));
    param->pad_l_ = static_cast<int>(*(pad_list->begin() + kOffsetTwo));
    param->pad_r_ = static_cast<int>(*(pad_list->begin() + kOffsetThree));
  }
  param->dilation_h_ = static_cast<int>(*(dilation->begin()));
  param->dilation_w_ = static_cast<int>(*(dilation->begin() + 1));
  param->input_channel_ = static_cast<int>(value->in_channel());
  param->output_channel_ = static_cast<int>(value->out_channel());
  auto act_type = value->activation_type();
  switch (act_type) {
    case schema::ActivationType_RELU:
      param->act_type_ = ActType_Relu;
      break;
    case schema::ActivationType_RELU6:
      param->act_type_ = ActType_Relu6;
      break;
    default:
      param->act_type_ = ActType_No;
  }
  return reinterpret_cast<OpParameter *>(param);
}

REG_POPULATE(PrimitiveType_Conv2DFusion, PopulateConvParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
