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

#ifndef MINDSPORE_CORE_OPS_TANH_H_
#define MINDSPORE_CORE_OPS_TANH_H_
#include <map>
#include <memory>
#include <vector>
#include <string>

#include "ops/primitive_c.h"
#include "ops/op_utils.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameTanh = "Tanh";
/// \brief Tanh activation function. Refer to Python API @ref mindspore.ops.Tanh for more details.
class MS_CORE_API Tanh : public PrimitiveC {
 public:
  /// \brief Constructor.
  Tanh() : PrimitiveC(kNameTanh) { InitIOName({"x"}, {"y"}); }
  /// \brief Destructor.
  ~Tanh() = default;
  MS_DECLARE_PARENT(Tanh, PrimitiveC);
  /// \brief Init.
  void Init() {}
};
AbstractBasePtr TanhInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args);
using PrimTanhPtr = std::shared_ptr<Tanh>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_TANH_H_
