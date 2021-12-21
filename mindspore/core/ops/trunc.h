/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_TRUNC_H_
#define MINDSPORE_CORE_OPS_TRUNC_H_
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameTrunc = "Trunc";
/// \brief Returns a new tensor with the truncated integer values of the elements of input.
class Trunc : public PrimitiveC {
 public:
  /// \brief Constructor.
  Trunc() : PrimitiveC(kNameTrunc) { InitIOName({"input_x"}, {"output_y"}); }
  /// \brief Destructor.
  ~Trunc() = default;
  MS_DECLARE_PARENT(Trunc, PrimitiveC);
};

AbstractBasePtr TruncInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args);

using PrimTruncPtr = std::shared_ptr<Trunc>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_TRUNC_H_
