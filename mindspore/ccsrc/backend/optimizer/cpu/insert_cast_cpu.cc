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

#include "backend/optimizer/cpu/insert_cast_cpu.h"

#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "backend/optimizer/common/helper.h"
#include "backend/kernel_compiler/kernel_build_info.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/session/kernel_graph.h"
#include "utils/utils.h"
#include "utils/ms_context.h"
#include "backend/kernel_compiler/common_utils.h"
#include "base/core_ops.h"

namespace mindspore {
namespace opt {
namespace {
constexpr unsigned int kLstmReserveIndex = 3;
AnfNodePtr AddCastOpNodeToGraph(const FuncGraphPtr &func_graph, const AnfNodePtr &input, const std::string &format,
                                const TypeId &input_type, const TypeId &output_type,
                                const abstract::BaseShapePtr &origin_shape, const TypeId &origin_type) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::string input_format = format;
  std::string output_format = format;
  CNodePtr cast = func_graph->NewCNode({NewValueNode(std::make_shared<Primitive>(prim::kPrimCast->name())), input});
  MS_EXCEPTION_IF_NULL(cast);
  // set kernel build info
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  builder.SetInputsFormat({input_format});
  builder.SetOutputsFormat({output_format});
  builder.SetInputsDeviceType({input_type});
  builder.SetOutputsDeviceType({output_type});
  if (cast->kernel_info() == nullptr) {
    auto kernel_info = std::make_shared<device::KernelInfo>();
    cast->set_kernel_info(kernel_info);
  }
  if (origin_shape->IsDynamic()) {
    AnfAlgo::SetNodeAttr(kAttrIsDynamicShape, MakeValue(true), cast);
    AnfAlgo::SetNodeAttr(kAttrInputIsDynamicShape, MakeValue(true), cast);
    AnfAlgo::SetNodeAttr(kAttrOutputIsDynamicShape, MakeValue(true), cast);
  }
  AnfAlgo::SetNodeAttr("dst_type", TypeIdToType(output_type), cast);
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), cast.get());
  AnfAlgo::SetOutputTypeAndDetailShape({origin_type}, {origin_shape}, cast.get());
  AnfAlgo::SetNodeAttr(kIsBackendCast, MakeValue(true), cast);
  std::shared_ptr<kernel::CPUKernel> cpu_kernel = kernel::CPUKernelFactory::GetInstance().Create(kCastOpName, cast);
  if (cpu_kernel == nullptr) {
    MS_LOG(EXCEPTION) << "Operator[Cast] " << cast->kernel_info() << " is not support.";
  }
  try {
    cpu_kernel->Init(cast);
    cpu_kernel->InitDynamicKernel(cast);
    auto cpu_dynamic_kernel = cpu_kernel->DynamicKernel();
    MS_EXCEPTION_IF_NULL(cpu_dynamic_kernel);
    cpu_dynamic_kernel->Initialize();
  } catch (std::exception &e) {
    MS_LOG(EXCEPTION) << e.what() << trace::DumpSourceLines(cast);
  }
  AnfAlgo::SetKernelMod(cpu_kernel, cast.get());
  return cast;
}

std::shared_ptr<std::vector<std::pair<AnfNodePtr, int>>> GetNodeUserList(const FuncGraphPtr &graph,
                                                                         const AnfNodePtr &node) {
  auto output_node_list = std::make_shared<std::vector<std::pair<AnfNodePtr, int>>>();
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto iter = manager->node_users().find(node);
  if (iter == manager->node_users().end()) {
    return output_node_list;
  }
  auto output_info_list = iter->second;
  std::copy(output_info_list.begin(), output_info_list.end(), std::back_inserter(*output_node_list));
  return output_node_list;
}

void SyncWeightNodeWithCast(const FuncGraphPtr &func_graph, const CNodePtr &cnode, const AnfNodePtr &cur_input,
                            const AnfNodePtr &cast, const std::string &format, const TypeId &device_type,
                            const TypeId &origin_type, const abstract::BaseShapePtr &origin_shape,
                            std::vector<AnfNodePtr> *make_tuple_inputs) {
  auto first_depend_node =
    func_graph->NewCNode({NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())), cast, cnode});
  first_depend_node->set_abstract(cast->abstract());
  auto post_cast =
    AddCastOpNodeToGraph(func_graph, first_depend_node, format, device_type, origin_type, origin_shape, origin_type);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  kernel_graph->AddRefCorrespondPairs(std::make_pair(post_cast, 0), AnfAlgo::VisitKernel(cur_input, 0));
  make_tuple_inputs->push_back(post_cast);
}

void InsertCast(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  size_t in_num = AnfAlgo::GetInputTensorNum(cnode);
  std::vector<AnfNodePtr> make_tuple_inputs{NewValueNode(std::make_shared<Primitive>(prim::kPrimMakeTuple->name()))};
  for (size_t input_index = 0; input_index < in_num; ++input_index) {
    auto prev_node = AnfAlgo::GetPrevNodeOutput(cnode, input_index);
    auto origin_type = AnfAlgo::GetOutputDeviceDataType(prev_node.first, prev_node.second);
    if (origin_type == kTypeUnknown) {
      origin_type = AnfAlgo::GetOutputInferDataType(prev_node.first, prev_node.second);
    }
    auto cur_input = AnfAlgo::GetInputNode(cnode, input_index);
    MS_EXCEPTION_IF_NULL(cur_input);
    const std::string dev_fmt = AnfAlgo::GetInputFormat(cnode, input_index);
    const abstract::BaseShapePtr origin_shape = AnfAlgo::GetOutputDetailShape(prev_node.first, prev_node.second);
    if (TypeId device_type = AnfAlgo::GetInputDeviceDataType(cnode, input_index); origin_type != device_type) {
      auto cast =
        AddCastOpNodeToGraph(func_graph, cur_input, dev_fmt, origin_type, device_type, origin_shape, device_type);
      MS_EXCEPTION_IF_NULL(cast);
      cast->set_scope(cnode->scope());
      cnode->set_input(input_index + 1, cast);
      auto real_input = AnfAlgo::VisitKernel(cur_input, 0).first;
      if (AnfAlgo::IsUpdateParameterKernel(cnode) && real_input->isa<Parameter>() &&
          AnfAlgo::IsParameterWeight(real_input->cast<ParameterPtr>())) {
        SyncWeightNodeWithCast(func_graph, cnode, cur_input, cast, dev_fmt, device_type, origin_type, origin_shape,
                               &make_tuple_inputs);
      }
    }
    if (make_tuple_inputs.size() > 1) {
      auto make_tuple = func_graph->NewCNode(make_tuple_inputs);
      auto second_depend_node =
        func_graph->NewCNode({NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())), cnode, make_tuple});
      second_depend_node->set_abstract(cnode->abstract());
      auto used_node_list = GetRealNodeUsedList(func_graph, cnode);
      if (used_node_list != nullptr && used_node_list->empty()) {
        used_node_list = GetNodeUserList(func_graph, cnode);
      }
      for (size_t j = 0; j < used_node_list->size(); j++) {
        auto used_node = used_node_list->at(j).first;
        if (!used_node->isa<CNode>()) {
          continue;
        }
        utils::cast<CNodePtr>(used_node)->set_input(used_node_list->at(j).second, second_depend_node);
      }
    }
  }
}

void InsertCastForGraphOutput(const FuncGraphPtr &func_graph, const CNodePtr &cnode, const AnfNodePtr &func_output) {
  MS_EXCEPTION_IF_NULL(cnode);
  size_t output_num = AnfAlgo::GetOutputTensorNum(cnode);
  for (size_t i = 0; i < output_num; i++) {
    auto infer_type = AnfAlgo::GetOutputInferDataType(cnode, i);
    auto device_type = AnfAlgo::GetOutputDeviceDataType(cnode, i);
    const std::string dev_fmt = AnfAlgo::GetOutputFormat(cnode, i);
    // The shape of LSTM's reserved output will be changed in InitKernel, and this output is only used
    // by its gradient operator, so we don't handle it in this pass.
    if (IsPrimitiveCNode(cnode, prim::kPrimLstm) && i == kLstmReserveIndex) {
      continue;
    }
    if (infer_type != device_type) {
      auto used_node_list = GetRealNodeUsedListByOutputIdx(func_graph, cnode, i);
      for (size_t j = 0; j < used_node_list->size(); j++) {
        auto used_node = used_node_list->at(j).first;
        if (used_node != func_output) {
          continue;
        }
        auto used_node_index = static_cast<size_t>(used_node_list->at(j).second - 1);
        auto cur_input = AnfAlgo::GetInputNode(utils::cast<CNodePtr>(used_node), used_node_index);
        const abstract::BaseShapePtr origin_shape =
          AnfAlgo::GetPrevNodeOutputDetailShape(utils::cast<CNodePtr>(used_node), used_node_index);
        auto cast =
          AddCastOpNodeToGraph(func_graph, cur_input, dev_fmt, device_type, infer_type, origin_shape, infer_type);
        MS_EXCEPTION_IF_NULL(cast);
        cast->set_scope(used_node->scope());
        utils::cast<CNodePtr>(used_node)->set_input(used_node_index + 1, cast);
      }
    }
  }
}
}  // namespace

bool InsertCastCPU::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> node_list = TopoSort(func_graph->get_return());
  for (auto node : node_list) {
    if (node != nullptr && node->isa<CNode>() && AnfUtils::IsRealKernel(node)) {
      CNodePtr cnode = node->cast<CNodePtr>();
      InsertCast(func_graph, cnode);
    }
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode) {
    AnfNodePtrList outputs;
    kernel::GetFuncGraphOutputNodes(func_graph, &outputs);
    auto func_output = func_graph->output();
    for (auto node : outputs) {
      if (node != nullptr && node->isa<CNode>() && AnfUtils::IsRealKernel(node)) {
        auto cnode = node->cast<CNodePtr>();
        InsertCastForGraphOutput(func_graph, cnode, func_output);
      }
    }
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
