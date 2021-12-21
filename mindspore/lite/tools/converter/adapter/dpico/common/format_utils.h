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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_DPICO_COMMON_FORMAT_UTILS_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_DPICO_COMMON_FORMAT_UTILS_H

#include <set>
#include <vector>
#include <string>
#include "include/api/format.h"
#include "common/anf_util.h"

namespace mindspore {
namespace dpico {
const std::set<std::string> &GetAssignedFormatOpSet();
bool IsSpecialType(const mindspore::CNodePtr &cnode);
std::string FormatEnumToString(mindspore::Format format);
}  // namespace dpico
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_DPICO_COMMON_FORMAT_UTILS_H
