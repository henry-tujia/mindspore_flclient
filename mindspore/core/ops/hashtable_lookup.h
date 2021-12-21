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
#ifndef MINDSPORE_CORE_OPS_HASHTABLE_LOOKUP_H_
#define MINDSPORE_CORE_OPS_HASHTABLE_LOOKUP_H_
#include <vector>
#include <memory>

#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameHashtableLookup = "HashtableLookup";
/// \brief HashtableLookup defined HashtableLookup operator prototype.
class MS_CORE_API HashtableLookup : public PrimitiveC {
 public:
  /// \brief Constructor.
  HashtableLookup() : PrimitiveC(kNameHashtableLookup) {}

  /// \brief Destructor.
  ~HashtableLookup() = default;

  MS_DECLARE_PARENT(HashtableLookup, PrimitiveC);

  /// \brief Method to init the op's attributes.
  void Init() {}
};

AbstractBasePtr HashtableLookupInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_HASHTABLE_LOOKUP_H_
