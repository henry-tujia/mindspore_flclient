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

#ifndef MINDSPORE_CORE_OPS_SELECT_H_
#define MINDSPORE_CORE_OPS_SELECT_H_

#include <map>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "abstract/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSelect = "Select";
/// \brief Returns the selected elements, either from input x or input y, depending on the condition.
/// Refer to Python API @ref mindspore.ops.Select for more details.
class MS_CORE_API Select : public PrimitiveC {
 public:
  /// \brief Constructor.
  Select() : PrimitiveC(kNameSelect) { InitIOName({"condition", "x", "y"}, {"output"}); }
  /// \brief Destructor.
  ~Select() = default;
  MS_DECLARE_PARENT(Select, PrimitiveC);
  /// \brief Init.
  void Init() {}
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SELECT_H_
