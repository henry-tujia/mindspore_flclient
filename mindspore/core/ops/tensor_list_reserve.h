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

#ifndef MINDSPORE_CORE_OPS_TENSOR_LIST_RESERVE_H_
#define MINDSPORE_CORE_OPS_TENSOR_LIST_RESERVE_H_
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameTensorListReserve = "TensorListReserve";
/// \brief TensorListReserve defined TensorListReserve operator prototype of lite.
class MS_CORE_API TensorListReserve : public PrimitiveC {
 public:
  /// \brief Constructor.
  TensorListReserve() : PrimitiveC(kNameTensorListReserve) {}

  /// \brief Destructor.
  ~TensorListReserve() = default;

  MS_DECLARE_PARENT(TensorListReserve, PrimitiveC);

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] element_dtype Define the element data type in op;
  /// \param[in] shape_type Define the shape type in op;
  void Init(const int64_t element_dtype, const int64_t shape_type);

  /// \brief Method to set the op's element_dtype attributes.
  ///
  /// \param[in] element_dtype Define the element data type in op;
  void set_element_dtype(const int64_t element_dtype);

  /// \brief Method to set the op's shape_type attributes.
  ///
  /// \param[in] shape_type Define the shape type in op;
  void set_shape_type(const int64_t shape_type);

  /// \brief Method to get the op's element_dtype attributes.
  int64_t get_element_dtype() const;

  /// \brief Method to get the op's shape_type attributes.
  int64_t get_shape_type() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_TENSOR_LIST_RESERVE_H_
