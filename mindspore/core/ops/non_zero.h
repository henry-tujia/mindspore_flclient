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

#ifndef MINDSPORE_CORE_OPS_NON_ZERO_H_
#define MINDSPORE_CORE_OPS_NON_ZERO_H_
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameNonZero = "NonZero";
/// \brief Calculate tensor not zero index, by default.
/// Refer to Python API @ref mindspore.ops.NonZero for more details.
class MS_CORE_API NonZero : public PrimitiveC {
 public:
  /// \brief Constructor.
  NonZero() : PrimitiveC(kNameNonZero) {}
  /// \brief Destructor.
  ~NonZero() = default;
  MS_DECLARE_PARENT(NonZero, PrimitiveC);
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_NON_ZERO_H_
