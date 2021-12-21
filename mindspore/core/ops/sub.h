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

#ifndef MINDSPORE_CORE_OPS_SUB_H_
#define MINDSPORE_CORE_OPS_SUB_H_
#include <string>
#include <vector>
#include <memory>

#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSub = prim::kSub;
/// \brief Subtracts the second input tensor from the first input tensor element-wise.
/// Refer to Python API @ref mindspore.ops.Sub for more details.
class MS_CORE_API Sub : public PrimitiveC {
 public:
  /// \brief Constructor.
  Sub() : PrimitiveC(kNameSub) { InitIOName({"x", "y"}, {"output"}); }
  explicit Sub(const std::string k_name) : PrimitiveC(k_name) { InitIOName({"x", "y"}, {"output"}); }
  /// \brief Destructor.
  ~Sub() = default;
  MS_DECLARE_PARENT(Sub, PrimitiveC);
  /// \brief Init.
  void Init() {}
};

AbstractBasePtr SubInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                         const std::vector<AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SUB_H_
