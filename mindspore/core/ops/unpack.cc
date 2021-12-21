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

#include "ops/unpack.h"

namespace mindspore {
namespace ops {
void Unpack::Init(const int64_t axis) { this->set_axis(axis); }
void Unpack::set_axis(const int64_t axis) { (void)AddAttr(kAxis, MakeValue(axis)); }
int64_t Unpack::get_axis() const { return GetValue<int64_t>(GetAttr(kAxis)); }

AbstractBasePtr UnpackInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  MS_EXCEPTION_IF_NULL(input_args[0]);
  (void)CheckAndConvertUtils::CheckSubClass("x", input_args[0]->BuildType(), {TypeIdToType(kObjectTypeTensorType)},
                                            prim_name);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  int64_t dim = SizeToLong(x_shape.size());
  int64_t axis = GetValue<int64_t>(primitive->GetAttr(kAxis));
  if (axis < 0) {
    axis = axis + dim;
  }
  auto output_num = x_shape[LongToSize(axis)];
  (void)CheckAndConvertUtils::CheckInteger("output_num", output_num, kGreaterThan, 0, prim_name);
  auto output_valid_check = x_shape[LongToSize(axis)] - output_num;
  (void)CheckAndConvertUtils::CheckInteger("The dimension which to unpack divides output_num", output_valid_check,
                                           kEqual, 0, prim_name);
  std::vector<int64_t> infer_shape(x_shape.begin(), x_shape.begin() + axis);
  (void)infer_shape.insert(infer_shape.end(), x_shape.begin() + axis + 1, x_shape.end());
  AbstractBasePtrList output;
  auto tensor_type = input_args[0]->BuildType()->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  auto element = tensor_type->element();
  for (int64_t i = 0; i != output_num; i++) {
    output.push_back(std::make_shared<abstract::AbstractTensor>(element, infer_shape));
  }
  return std::make_shared<abstract::AbstractTuple>(output);
}
REGISTER_PRIMITIVE_C(kNameUnpack, Unpack);
}  // namespace ops
}  // namespace mindspore
