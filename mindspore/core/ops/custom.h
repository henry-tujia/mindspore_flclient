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

#ifndef MINDSPORE_CORE_OPS_CUSTOM_H_
#define MINDSPORE_CORE_OPS_CUSTOM_H_
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <memory>
#include "ops/primitive_c.h"
#include "ops/op_utils.h"
#include "ir/anf.h"

namespace mindspore {
namespace ops {
constexpr auto kNameCustom = "Custom";
/// \brief Custom defined user-defined operator prototype.
class MS_CORE_API Custom : public PrimitiveC {
 public:
  /// \brief Constructor.
  Custom() : PrimitiveC(kNameCustom) {}

  /// \brief Destructor.
  ~Custom() override = default;

  MS_DECLARE_PARENT(Custom, PrimitiveC);

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] type Define the concrete type of the custom op, which is used to distinguish different custom op.
  /// \param[in] attrs Define the attributes of the custom op.
  void Init(const std::string &type, const std::map<std::string, std::vector<uint8_t>> &attrs);

  /// \brief Method to set type attribute.
  ///
  /// \param[in] type Define the concrete type of the custom op, which is used to distinguish different custom op.
  void set_type(const std::string &type);

  /// \brief Method to get type attribute.
  ///
  /// \return a string
  std::string get_type() const;

  /// \brief Method to set attr attribute.
  ///
  /// \param[in] attrs Define the attributes of the custom op.
  void set_attr(const std::map<std::string, std::vector<uint8_t>> &attrs);

  /// \brief Method to get attr attribute.
  ///
  /// \return a map which contains all attributes of the custom op.
  std::map<std::string, std::vector<uint8_t>> get_attr() const;
};

}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_CUSTOM_H_
