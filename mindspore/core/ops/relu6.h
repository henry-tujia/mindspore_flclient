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
#ifndef MINDSPORE_CORE_OPS_RELU6_H_
#define MINDSPORE_CORE_OPS_RELU6_H_
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameReLU6 = prim::kReLU6;
/// \brief Computes ReLU (Rectified Linear Unit) upper bounded by 6 of input tensors element-wise.
/// Refer to Python API @ref mindspore.ops.ReLU6 for more details.
class MS_CORE_API ReLU6 : public PrimitiveC {
 public:
  /// \brief Constructor.
  ReLU6() : PrimitiveC(kNameReLU6) { InitIOName({"x"}, {"output"}); }
  /// \brief Destructor.
  ~ReLU6() = default;
  MS_DECLARE_PARENT(ReLU6, PrimitiveC);
  /// \brief Init.
  void Init() {}
};

AbstractBasePtr ReLU6Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_RELU6_H_
