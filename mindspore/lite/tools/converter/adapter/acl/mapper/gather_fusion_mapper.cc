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

#include "tools/converter/adapter/acl/mapper/gather_fusion_mapper.h"
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/common/utils.h"

namespace mindspore {
namespace lite {
namespace {
constexpr size_t kNameGatherInputNum = 4;
}

STATUS GatherMapper::Mapper(const CNodePtr &cnode) {
  if (cnode == nullptr) {
    MS_LOG(ERROR) << "Cnode is nullptr.";
    return lite::RET_ERROR;
  }
  if (cnode->size() != kNameGatherInputNum) {
    MS_LOG(ERROR) << "Input size of gather must be four.";
    return lite::RET_ERROR;
  }
  // convert last parameter to const value node
  auto axis_input = cnode->input(kNameGatherInputNum - 1);
  if (!utils::isa<ParameterPtr>(axis_input)) {
    MS_LOG(ERROR) << "The axis node is not parameter.";
    return lite::RET_ERROR;
  }
  ParameterPtr axis_param = axis_input->cast<ParameterPtr>();
  auto data = acl::GetIntParameterData(axis_param);
  int64_t axis = data.empty() ? 0 : static_cast<int64_t>(data.front());
  ValueNodePtr value_node = NewValueNode<int64_t>(axis);
  if (value_node == nullptr) {
    MS_LOG(ERROR) << "New value node failed.";
    return lite::RET_ERROR;
  }
  cnode->set_input(kNameGatherInputNum - 1, value_node);
  return lite::RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameGather, GatherMapper)
}  // namespace lite
}  // namespace mindspore
