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
#include "backend/optimizer/ascend/mindir/update_input_names_strided_slice_grad.h"
#include <memory>
#include <vector>
#include <string>
#include "backend/session/anf_runtime_algorithm.h"
#include "utils/utils.h"
#include "backend/optimizer/common/helper.h"

namespace mindspore {
namespace opt {
const BaseRef StridedSliceGradUpdateInputNames::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto strided_slice_grad_prim = std::make_shared<Primitive>(kStridedSliceGradOpName);
  return VectorRef({strided_slice_grad_prim, Xs});
}

const AnfNodePtr StridedSliceGradUpdateInputNames::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                           const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto strided_slice_grad = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(strided_slice_grad);

  const size_t shapex_index = 1;
  if (AnfAlgo::IsNodeDynamicShape(strided_slice_grad)) {
    auto primitive = AnfAlgo::GetCNodePrimitive(strided_slice_grad);
    MS_EXCEPTION_IF_NULL(primitive);
    auto input_names_ptr = primitive->GetAttr(kAttrInputNames);
    MS_EXCEPTION_IF_NULL(input_names_ptr);
    auto input_names_vec = GetValue<std::vector<std::string>>(input_names_ptr);
    input_names_vec[shapex_index] = "shape";
    AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue(input_names_vec), strided_slice_grad);
  }
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
