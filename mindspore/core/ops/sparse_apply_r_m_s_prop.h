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

#ifndef MINDSPORE_CORE_OPS_SPARSE_APPLY_R_M_S_PROP_H_
#define MINDSPORE_CORE_OPS_SPARSE_APPLY_R_M_S_PROP_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSparseApplyRMSProp = "SparseApplyRMSProp";
/// \brief Update relevant entries according to the rmsprop algorithm.
class SparseApplyRMSProp : public PrimitiveC {
 public:
  /// \brief Constructor.
  SparseApplyRMSProp() : PrimitiveC(kNameSparseApplyRMSProp) {
    InitIOName({"var", "ms", "mom", "lr", "grad", "indices"}, {"var", "ms", "mom"});
  }
  /// \brief Destructor.
  ~SparseApplyRMSProp() = default;
  MS_DECLARE_PARENT(SparseApplyRMSProp, PrimitiveC);
};

AbstractBasePtr SparseApplyRMSPropInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SPARSE_APPLY_R_M_S_PROP_H_
