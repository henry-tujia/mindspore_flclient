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

#ifndef MINDSPORE_CORE_OPS_SCATTER_ND_H_
#define MINDSPORE_CORE_OPS_SCATTER_ND_H_
#include <vector>
#include <memory>

#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameScatterNd = "ScatterNd";
/// \brief Scatters a tensor into a new tensor depending on the specified indices.
/// Refer to Python API @ref mindspore.ops.ScatterNd for more details.
class MS_CORE_API ScatterNd : public PrimitiveC {
 public:
  /// \brief Constructor.
  ScatterNd() : PrimitiveC(kNameScatterNd) { InitIOName({"indices", "update", "shape"}, {"output"}); }
  /// \brief Destructor.
  ~ScatterNd() = default;
  MS_DECLARE_PARENT(ScatterNd, PrimitiveC);
  /// \brief Init.
  void Init() {}
};
AbstractBasePtr ScatterNdInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SCATTER_ND_H_
