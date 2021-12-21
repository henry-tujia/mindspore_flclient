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

#ifndef MINDSPORE_CORE_OPS_ARG_MAX_H_
#define MINDSPORE_CORE_OPS_ARG_MAX_H_
#include <string>
#include <vector>
#include <memory>

#include "ops/primitive_c.h"
#include "ops/op_utils.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameArgMax = "Argmax";
/// \brief Returns the indices of the maximum value of a tensor across the axis.
/// Refer to Python API @ref mindspore.ops.Argmax for more details.
class MS_CORE_API ArgMax : public PrimitiveC {
 public:
  /// \brief Constructor.
  ArgMax() : PrimitiveC(kNameArgMax) { InitIOName({"x"}, {"output"}); }
  explicit ArgMax(const std::string k_name) : PrimitiveC(k_name) { InitIOName({"x"}, {"output"}); }
  /// \brief Destructor.
  ~ArgMax() = default;
  MS_DECLARE_PARENT(ArgMax, PrimitiveC);
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.Argmax for the inputs.
  void Init(const int64_t axis = -1, const TypeId output_type = kNumberTypeInt32);
  /// \brief Set axis.
  void set_axis(const int64_t axis);
  /// \brief Set output_type.
  void set_output_type(const TypeId output_type);

  /// \brief Get axis.
  ///
  /// \return axis.
  int64_t get_axis() const;
  /// \brief Get output_type.
  ///
  /// \return output_type.
  TypeId get_output_type() const;
};
AbstractBasePtr ArgMaxInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_ARG_MAX_H_
