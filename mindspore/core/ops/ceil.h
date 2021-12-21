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

#ifndef MINDSPORE_CORE_OPS_CEIL_H_
#define MINDSPORE_CORE_OPS_CEIL_H_
#include <vector>
#include <memory>

#include "ops/primitive_c.h"
#include "ops/op_utils.h"
#include "abstract/abstract_value.h"

namespace mindspore {
namespace ops {
constexpr auto kNameCeil = "Ceil";
/// \brief Rounds a tensor up to the closest integer element-wise.
/// Refer to Python API @ref mindspore.ops.Ceil for more details.
class MS_CORE_API Ceil : public PrimitiveC {
 public:
  /// \brief Constructor.
  Ceil() : PrimitiveC(kNameCeil) { InitIOName({"x"}, {"y"}); }
  /// \brief Destructor.
  ~Ceil() = default;
  MS_DECLARE_PARENT(Ceil, PrimitiveC);
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.Ceil for the inputs.
  void Init() {}
};
AbstractBasePtr CeilInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_CEIL_H_
