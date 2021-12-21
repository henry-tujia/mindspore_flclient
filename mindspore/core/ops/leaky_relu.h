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

#ifndef MINDSPORE_CORE_OPS_LEAKY_RELU_H_
#define MINDSPORE_CORE_OPS_LEAKY_RELU_H_
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/primitive_c.h"
#include "ops/op_utils.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameLeakyRelu = "LeakyRelu";
/// \brief Leaky ReLU activation function. Refer to Python API @ref mindspore.nn.LeakyReLU for more details.
class MS_CORE_API LeakyRelu : public PrimitiveC {
 public:
  /// \brief Constructor.
  LeakyRelu() : PrimitiveC(kNameLeakyRelu) {}
  /// \brief Destructor.
  ~LeakyRelu() = default;
  MS_DECLARE_PARENT(LeakyRelu, PrimitiveC);
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.nn.LeakyReLU for the inputs.
  void Init(const float negative_slope);
  /// \brief Set negative_slope.
  void set_negative_slope(const float negative_slope);
  /// \brief Get negative_slope.
  ///
  /// \return negative_slope.
  float get_negative_slope() const;
};

AbstractBasePtr LeakyReluInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_LEAKY_RELU_H_
