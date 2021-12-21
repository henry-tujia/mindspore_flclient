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

#include <map>
#include <string>
#include "ops/concat.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
namespace mindspore {
namespace ops {
void Concat::Init(const int64_t axis) { this->set_axis(axis); }
int64_t Concat::get_axis() const {
  auto value_ptr = this->GetAttr(kAxis);
  return GetValue<int64_t>(value_ptr);
}

void Concat::set_axis(const int64_t axis) { (void)this->AddAttr(kAxis, MakeValue(axis)); }

AbstractBasePtr ConcatInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, 1, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto input_tuple = input_args[0]->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(input_tuple);
  auto elements = input_tuple->elements();
  const int64_t kOneNum = 1;
  (void)CheckAndConvertUtils::CheckInteger("concat element num", SizeToLong(elements.size()), kGreaterEqual, kOneNum,
                                           prim_name);
  auto element0 = elements[0]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(element0);
  auto element0_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(element0->BuildShape())[kShape];
  auto element0_rank = element0_shape.size();
  auto axis_temp = GetValue<int64_t>(primitive->GetAttr(kAxis));
  CheckAndConvertUtils::CheckInRange<int64_t>("Concat axis", axis_temp, kIncludeBoth,
                                              {-SizeToLong(element0_rank) - kOneNum, SizeToLong(element0_rank)},
                                              prim_name);
  auto axis = axis_temp < 0 ? LongToSize(axis_temp) + element0_rank : LongToSize(axis_temp);

  std::map<std::string, TypePtr> types;
  (void)types.emplace("element0", element0->BuildType());
  int64_t all_shp = element0_shape[axis];
  for (size_t i = 1; i < elements.size(); ++i) {
    std::string elementi = "element" + std::to_string(i);
    auto elementi_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(elements[i]->BuildShape())[kShape];
    (void)CheckAndConvertUtils::CheckInteger(elementi + " shape rank", SizeToLong(elementi_shape.size()), kEqual,
                                             SizeToLong(element0_shape.size()), prim_name);
    for (size_t j = 0; j < element0_rank; ++j) {
      if (j != axis && elementi_shape[j] != element0_shape[j]) {
        MS_LOG(EXCEPTION) << "element " << i << " shape in input can not concat with first element.";
      }
    }
    all_shp = all_shp == -1 || elementi_shape[axis] == -1 ? -1 : all_shp + elementi_shape[axis];
    (void)types.emplace(elementi, elements[i]->BuildType());
  }
  auto infer_type = CheckAndConvertUtils::CheckTensorTypeSame(types, all_types, prim_name);
  auto ret_shape = element0_shape;
  ret_shape[axis] = all_shp;
  return std::make_shared<abstract::AbstractTensor>(infer_type, std::make_shared<abstract::Shape>(ret_shape));
}
REGISTER_PRIMITIVE_C(kNameConcat, Concat);
}  // namespace ops
}  // namespace mindspore
