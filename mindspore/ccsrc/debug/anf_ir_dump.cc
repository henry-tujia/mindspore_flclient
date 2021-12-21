/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "debug/anf_ir_dump.h"
#if defined(_WIN32) || defined(_WIN64)
#include <stdlib.h>
#endif
#include <fstream>
#include <iomanip>
#include <memory>
#include "utils/hash_map.h"
#include "ir/primitive.h"
#include "ir/func_graph.h"
#include "runtime/device/kernel_info.h"
#include "ir/graph_utils.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "pipeline/jit/base.h"
#include "debug/trace.h"
#include "utils/trace_base.h"

namespace mindspore {
const std::string ToShortString(const TypeId &typeId) {
  std::string label = TypeIdLabel(typeId);
  std::string prefix = "kNumberType";
  if (prefix.length() > label.length()) {
    return label;
  }
  auto position = label.find(prefix);
  // Position is 0 when label begins with prefix
  if (position != 0) {
    return label;
  }
  auto sub_position = position + prefix.length();
  if (sub_position >= label.length()) {
    return label;
  }
  return label.substr(sub_position);
}

void PrintKernelFormatAndType(std::ostringstream &buffer, const std::string &fmt, const TypeId &type,
                              const std::vector<size_t> &shape) {
  buffer << "<" << ToShortString(type);
  if (!fmt.empty()) {
    buffer << "x" << fmt << shape;
  }
  buffer << ">";
}

void PrintNodeOutputType(std::ostringstream &buffer, const AnfNodePtr &node) {
  if (node == nullptr) {
    return;
  }

  ValuePtr tensor_value = nullptr;
  auto abstract = node->abstract();
  if (abstract != nullptr && abstract->isa<abstract::AbstractTensor>()) {
    tensor_value = abstract->BuildValue();
  }
  abstract::ShapePtr shape = dyn_cast<abstract::Shape>(node->Shape());
  TypePtr type = dyn_cast<Type>(node->Type());
  if ((shape != nullptr) && (type != nullptr)) {
    buffer << "<" << type << ", " << shape->ToString();
    if (tensor_value != nullptr && tensor_value != kAnyValue) {
      buffer << ", value=...";
    }
    buffer << ">";
  } else if (type != nullptr) {
    buffer << "<" << type;
    if (tensor_value != nullptr && tensor_value != kAnyValue) {
      buffer << ", value=...";
    }
    buffer << ">";
  } else {
    buffer << "<null>";
  }
}

void PrintNodeInputType(std::ostringstream &buffer, const AnfNodePtr &node) {
  if (node == nullptr) {
    return;
  }

  const auto &inputs = GetInputs(node);
  size_t len = inputs.size();
  if (len > 1) {
    // Skip inputs[0] which is Primitive value node
    for (size_t i = 1; i < len; ++i) {
      AnfNodePtr in = inputs[i];
      if (i != 1) {
        buffer << ", ";
      }
      PrintNodeOutputType(buffer, in);
    }
  }
}

void GatherInputAndOutputInferType(std::ostringstream &buffer, const AnfNodePtr &node) {
  buffer << "      : (";
  PrintNodeInputType(buffer, node);
  buffer << ") -> (";
  PrintNodeOutputType(buffer, node);
  buffer << ")";
}

struct SubGraphIRInfo {
  int32_t local_var;
  std::ostringstream buffer;
  OrderedMap<AnfNodePtr, int32_t> local_var_map;
};

void DumpGlobalInfoEntry(const FuncGraphPtr &graph, std::ostringstream &buffer) {
  if (graph == nullptr) {
    return;
  }

  buffer << "#IR entry      : @" << graph->ToString() << std::endl;
  buffer << "#attrs         :" << std::endl;
  for (const auto &attr : graph->attrs()) {
    buffer << attr.first << " : ";
    if (attr.second->isa<BoolImm>()) {
      buffer << GetValue<bool>(attr.second);
    } else if (attr.second->isa<StringImm>()) {
      buffer << GetValue<std::string>(attr.second);
    }
    buffer << std::endl;
  }
}

void DumpKernelInfo(const CNodePtr &node, const std::shared_ptr<SubGraphIRInfo> &gsub) {
  if (node == nullptr || gsub == nullptr) {
    return;
  }
  auto kernel_info = node->kernel_info();
  if (kernel_info == nullptr || !kernel_info->has_build_info()) {
    return;
  }

  gsub->buffer << "      : (";
  size_t input_num = AnfAlgo::GetInputTensorNum(node);
  for (size_t i = 0; i < input_num; ++i) {
    if (i != 0) {
      gsub->buffer << ", ";
    }
    auto format = AnfAlgo::GetInputFormat(node, i);
    auto type = AnfAlgo::GetInputDeviceDataType(node, i);
    auto shape = AnfAlgo::GetInputDeviceShape(node, i);
    PrintKernelFormatAndType(gsub->buffer, format, type, shape);
  }
  gsub->buffer << ") -> (";
  size_t output_num = AnfAlgo::GetOutputTensorNum(node);
  for (size_t i = 0; i < output_num; ++i) {
    if (i != 0) {
      gsub->buffer << ", ";
    }
    auto format = AnfAlgo::GetOutputFormat(node, i);
    auto type = AnfAlgo::GetOutputDeviceDataType(node, i);
    auto shape = AnfAlgo::GetOutputDeviceShape(node, i);
    PrintKernelFormatAndType(gsub->buffer, format, type, shape);
  }
  gsub->buffer << ")";
  gsub->buffer << std::endl;
}

int32_t DumpParams(const FuncGraphPtr &graph, std::ostringstream &buffer, OrderedMap<AnfNodePtr, int32_t> *para_map) {
  if (graph == nullptr) {
    MS_LOG(INFO) << "Parameter \'graph\' should not be null.";
    return 0;
  }
  std::vector<AnfNodePtr> parameters = graph->parameters();
  buffer << "#Total params  : " << parameters.size() << std::endl;
  buffer << std::endl;

  // Dump parameters
  int32_t para = 1;
  for (const auto &p : parameters) {
    if (p == nullptr) {
      continue;
    }
    auto parameter_ptr = p->cast<ParameterPtr>();
    if (parameter_ptr == nullptr) {
      MS_LOG(EXCEPTION) << "p cannot cast to ParameterPtr";
    }
    buffer << "%para" << para << "_" << parameter_ptr->name() << " : ";
    // Print parameters' type and shape
    PrintNodeOutputType(buffer, p);
    auto kernel_info = p->kernel_info();
    if (kernel_info != nullptr && kernel_info->has_build_info()) {
      buffer << "  :  ";
      auto type = AnfAlgo::GetOutputDeviceDataType(p, 0);
      auto format = AnfAlgo::GetOutputFormat(p, 0);
      auto shape = AnfAlgo::GetOutputDeviceShape(p, 0);
      PrintKernelFormatAndType(buffer, format, type, shape);
      buffer << "  :  IsWeight:" << std::boolalpha << AnfAlgo::IsParameterWeight(parameter_ptr);
    }
    buffer << std::endl;

    if (para_map != nullptr) {
      (*para_map)[p] = para++;
    }
    MS_LOG(DEBUG) << "Record param: " << p->ToString() << " graph belong : " << p->func_graph()->ToString();
  }
  return para;
}

void DumpOperator(const AnfNodePtr &node, const std::shared_ptr<SubGraphIRInfo> &gsub) {
  if (gsub == nullptr) {
    MS_LOG(INFO) << "Parameter \'gsub\' should not be null.";
    return;
  }
  auto cnode = dyn_cast<CNode>(node);
  if (cnode == nullptr) {
    MS_LOG(EXCEPTION) << "Parameter \'node\' should be a CNode";
    return;
  }
  AnfNodePtr op = cnode->input(0);
  MS_EXCEPTION_IF_NULL(op);
  if (IsValueNode<FuncGraph>(op)) {
    FuncGraphPtr fg = GetValueNode<FuncGraphPtr>(op);
    if (fg != nullptr) {
      gsub->buffer << "call @" << fg->ToString();
    }
  } else if (op->isa<CNode>()) {
    if (gsub->local_var_map.find(op) != gsub->local_var_map.end()) {
      gsub->buffer << "%" << gsub->local_var_map[op];
    } else {
      auto input = op->cast<CNodePtr>();
      auto fg = input->func_graph();
      gsub->buffer << "$(@" << fg->ToString() << ":" << input->ToString() << ")";
    }
  } else if (op->isa<ValueNode>()) {
    auto value = GetValueNode(op);
    if (value != nullptr) {
      gsub->buffer << value->ToString();
    }
  } else {
    // It's Parameter.
    if (op->func_graph() != nullptr && op->func_graph() != node->func_graph()) {
      gsub->buffer << "$(@" << op->func_graph()->ToString() << ":";
    }
    gsub->buffer << op->ToString();
    if (op->func_graph() != nullptr && op->func_graph() != node->func_graph()) {
      gsub->buffer << ")";
    }
  }
}

void DumpOperands(const AnfNodePtr &node, OrderedMap<AnfNodePtr, int32_t> *para_map,
                  const std::shared_ptr<SubGraphIRInfo> &gsub) {
  if (node == nullptr || para_map == nullptr || gsub == nullptr) {
    return;
  }

  gsub->buffer << "(";
  const auto &inputs = GetInputs(node);
  size_t len = inputs.size();
  if (len > 1) {
    // Skip inputs[0] which is Primitive valuenode
    for (size_t i = 1; i < len; ++i) {
      AnfNodePtr in = inputs[i];
      MS_EXCEPTION_IF_NULL(in);
      if (i != 1) {
        gsub->buffer << ", ";
      }
      if (in->isa<Parameter>()) {
        MS_EXCEPTION_IF_NULL(node->func_graph());
        if (in->func_graph() == nullptr) {
          MS_LOG(ERROR) << "Parameter should belong to a func graph. Check func graph: " << node->func_graph();
        }
        if (in->func_graph() != nullptr && in->func_graph() != node->func_graph()) {
          gsub->buffer << "$(@" << in->func_graph()->ToString() << ":";
        } else {
          gsub->buffer << "%";
        }
        auto iter = para_map->find(in);
        if (iter == para_map->end()) {
          gsub->buffer << "para_" << in->ToString();
        } else {
          gsub->buffer << "para" << iter->second << "_" << in->ToString();
        }
        if (in->func_graph() != nullptr && in->func_graph() != node->func_graph()) {
          gsub->buffer << ")";
        }
      } else if (in->isa<CNode>()) {
        auto iter = gsub->local_var_map.find(in);
        if (iter != gsub->local_var_map.end()) {
          gsub->buffer << "%" << iter->second;
        } else {
          auto input = in->cast<CNodePtr>();
          auto fg = input->func_graph();
          gsub->buffer << "$(@" << fg->ToString() << ":" << input->ToString() << ")";
        }
      } else if (in->isa<ValueNode>() && !IsValueNode<FuncGraph>(in)) {
        // ValueNode except FuncGraph.
        gsub->buffer << GetValueNode(in)->ToString();
      } else if (IsValueNode<FuncGraph>(in)) {
        FuncGraphPtr fg = GetValueNode<FuncGraphPtr>(in);
        gsub->buffer << "@" << fg->ToString();
      } else {
        gsub->buffer << in->ToString();
      }
    }
  }
  gsub->buffer << ")";
}

void DumpParallelInfo(const CNodePtr &node, const std::shared_ptr<SubGraphIRInfo> &gsub) {
  if ((node == nullptr) || (gsub == nullptr)) {
    return;
  }

  auto operator_info = node->user_data<parallel::OperatorInfo>();
  if (operator_info == nullptr) {
    return;
  }

  auto in_strategy = operator_info->strategy();
  if (in_strategy == nullptr) {
    return;
  }

  ValuePtr in_tmp = MakeValue(in_strategy->GetInputDim());
  gsub->buffer << " {in_strategy: ";
  gsub->buffer << in_tmp->ToString();

  auto out_strategy = operator_info->out_strategy();
  if (out_strategy) {
    ValuePtr out_tmp = MakeValue(out_strategy->GetInputDim());
    gsub->buffer << ", out_strategy: ";
    gsub->buffer << out_tmp->ToString();
  }

  gsub->buffer << "}";
}

void DumpAttrs(const mindspore::HashMap<std::string, ValuePtr> &attrs, const std::shared_ptr<SubGraphIRInfo> &gsub,
               bool check_strategy = false) {
  int i = 0;
  for (const auto &attr : attrs) {
    if (check_strategy && attr.first == PARALLEL_STRATEGY) {
      continue;  // Skip the strategy
    }
    if (i++ != 0) {
      gsub->buffer << ", ";
    }
    gsub->buffer << attr.first << ": ";
    if (attr.second == nullptr) {
      gsub->buffer << "null";
    } else {
      gsub->buffer << attr.second->ToString();
    }
  }
}

void DumpOperateAttrs(const AnfNodePtr &op, const std::shared_ptr<SubGraphIRInfo> &gsub) {
  if (op == nullptr || gsub == nullptr) {
    return;
  }

  if (IsValueNode<Primitive>(op)) {
    PrimitivePtr primitive = GetValueNode<PrimitivePtr>(op);
    if (!primitive->instance_name().empty()) {
      gsub->buffer << " {";
      gsub->buffer << "instance name"
                   << ": ";
      gsub->buffer << primitive->instance_name();
      gsub->buffer << "}";
    }
    auto attrs = primitive->attrs();
    if (!attrs.empty()) {
      gsub->buffer << " primitive_attrs: {";
      DumpAttrs(attrs, gsub, true);
      gsub->buffer << "}";
    }
  }
}

void DumpCNodeAttrs(const CNodePtr &op, const std::shared_ptr<SubGraphIRInfo> &gsub) {
  if (op == nullptr || gsub == nullptr) {
    return;
  }
  if (op->attrs().empty()) {
    return;
  }

  auto attrs = op->attrs();
  gsub->buffer << " cnode_attrs: {";
  DumpAttrs(attrs, gsub);
  gsub->buffer << "}";
}

void DumpCNodePrimalAttrs(const CNodePtr &op, const std::shared_ptr<SubGraphIRInfo> &gsub) {
  if (op == nullptr || gsub == nullptr) {
    return;
  }
  if (op->primal_attrs().empty()) {
    gsub->buffer << std::endl;
    return;
  }
  auto primal_attrs = op->primal_attrs();
  gsub->buffer << " cnode_primal_attrs: {";
  DumpAttrs(primal_attrs, gsub);
  gsub->buffer << "}";
  gsub->buffer << std::endl;
}

void DumpShape(const AnfNodePtr &node, const FuncGraphPtr &sub_graph, const std::shared_ptr<SubGraphIRInfo> &gsub) {
  if (node == nullptr || sub_graph == nullptr || gsub == nullptr) {
    return;
  }

  if (node != sub_graph->get_return()) {
    gsub->buffer << "      : (";
    PrintNodeInputType(gsub->buffer, node);
    gsub->buffer << ") -> (";
    PrintNodeOutputType(gsub->buffer, node);
    gsub->buffer << ")";
  } else {
    gsub->buffer << "      : (";
    PrintNodeInputType(gsub->buffer, node);
    gsub->buffer << ")";
  }

  gsub->buffer << std::endl;
}

void DumpPrimalDebugInfos(const CNodePtr &node, const std::shared_ptr<SubGraphIRInfo> &gsub) {
  MS_EXCEPTION_IF_NULL(node);
  auto primal_debug_infos = node->primal_debug_infos();
  if (!primal_debug_infos.empty()) {
    std::string lines;
    for (auto &primal_debug_info : primal_debug_infos) {
      auto debug_info_str = trace::GetDebugInfo(primal_debug_info, "      # ", kSourceLineTipDiscard);
      if (!debug_info_str.empty()) {
        lines += debug_info_str + "\n";
      }
    }
    if (!lines.empty()) {
      gsub->buffer << "      # Corresponding forward node candidate:\n";
      gsub->buffer << lines;
    }
  }
}

void DumpDebugInfo(const CNodePtr &node, const std::shared_ptr<SubGraphIRInfo> &gsub,
                   const LocDumpMode &dump_location) {
  MS_EXCEPTION_IF_NULL(node);
  if (dump_location == kTopStack) {
    auto fused_debug_infos = node->fused_debug_infos();
    if (!fused_debug_infos.empty()) {
      std::string lines;
      for (const auto &debug_info : fused_debug_infos) {
        auto debug_info_str = trace::GetDebugInfo(debug_info, "      # ", kSourceLineTipDiscard);
        if (!debug_info_str.empty()) {
          lines += debug_info_str + "\n";
        }
      }
      if (!lines.empty()) {
        gsub->buffer << "      # Corresponding code candidate:\n";
        gsub->buffer << lines;
      }
    } else {
      auto debug_info_str = trace::GetDebugInfo(node->debug_info(), "      # ", kSourceLineTipDiscard);
      if (!debug_info_str.empty()) {
        gsub->buffer << debug_info_str << "\n";
      }
    }

    DumpPrimalDebugInfos(node, gsub);
  } else if (dump_location == kWholeStack) {
    auto traces = mindspore::trace::GetSourceLineList(node);
    for (auto &trace : traces) {
      gsub->buffer << "      # " << trace;
    }
  }
}

void DumpCNode(const CNodePtr &node, const FuncGraphPtr &sub_graph, OrderedMap<AnfNodePtr, int32_t> *const para_map,
               const std::shared_ptr<SubGraphIRInfo> &gsub, bool dump_full_name = false,
               LocDumpMode dump_location = kOff) {
  if (node == nullptr || sub_graph == nullptr || para_map == nullptr || gsub == nullptr) {
    return;
  }

  if (node != sub_graph->get_return()) {
    gsub->buffer << "  %" << gsub->local_var << "(" << node->ToString() << ")"
                 << " = ";
    gsub->local_var_map[node] = gsub->local_var++;
  } else {
    gsub->buffer << "  ";
  }

  if (node->inputs().empty()) {
    MS_LOG(EXCEPTION) << "Input of apply node is empty";
  }

  // Print operator
  DumpOperator(node, gsub);

  // Print operands
  DumpOperands(node, para_map, gsub);

  // Print operator attrs
  AnfNodePtr op = node->input(0);
  DumpOperateAttrs(op, gsub);

  // Print cnode attrs
  DumpCNodeAttrs(node, gsub);

  // Print cnode primal attrs
  DumpCNodePrimalAttrs(node, gsub);

  // Print parallel info
  DumpParallelInfo(node, gsub);

  // Print shape info
  DumpShape(node, sub_graph, gsub);

  // Print kernel info
  DumpKernelInfo(node, gsub);

  if (dump_full_name) {
    gsub->buffer << "      : (" << node->fullname_with_scope() << ")" << std::endl;
  }

  // Print debug info
  DumpDebugInfo(node, gsub, dump_location);
}

void DumpIRInSubgraph(const std::vector<AnfNodePtr> &nodes, OrderedMap<AnfNodePtr, int32_t> *para_map,
                      OrderedMap<FuncGraphPtr, std::shared_ptr<SubGraphIRInfo>> *const sub_graphs, int32_t total_para,
                      bool dump_full_name = false, LocDumpMode dump_location = kOff) {
  if (para_map == nullptr || sub_graphs == nullptr) {
    return;
  }

  for (const auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    FuncGraphPtr sub_graph = node->func_graph();
    if (sub_graph == nullptr) {
      MS_LOG(DEBUG) << "Node[" << node->ToString() << "] belongs to no graph!";
      continue;
    }
    std::shared_ptr<SubGraphIRInfo> gsub = (*sub_graphs)[sub_graph];
    if (gsub == nullptr) {
      gsub = std::make_shared<SubGraphIRInfo>();
      gsub->local_var = 0;
      (*sub_graphs)[sub_graph] = gsub;
    }
    std::vector<AnfNodePtr> parameters = sub_graph->parameters();
    for (size_t idx = 0; idx < parameters.size(); idx++) {
      MS_EXCEPTION_IF_NULL(parameters[idx]);
      if ((*para_map).count(parameters[idx]) == 0) {
        (*para_map)[parameters[idx]] = total_para++;
      }
    }
    if (!node->isa<Parameter>()) {
      if (node->isa<CNode>()) {
        // Print and record output of operator if it is not 'Return'
        DumpCNode(node->cast<CNodePtr>(), sub_graph, para_map, gsub, dump_full_name, dump_location);
      } else {
        gsub->buffer << "  " << node->ToString() << std::endl;
      }
    }
  }
}

void DumpSubgraph(const OrderedMap<FuncGraphPtr, std::shared_ptr<SubGraphIRInfo>> *sub_graphs,
                  const FuncGraphPtr &graph, OrderedMap<AnfNodePtr, int32_t> *para_map, std::ofstream &fout) {
  if (sub_graphs == nullptr || graph == nullptr) {
    return;
  }

  fout << "#Total subgraph : " << sub_graphs->size() << std::endl;
  fout << std::endl;

  for (const auto &sg : *sub_graphs) {
    fout << "subgraph attr:" << std::endl;
    MS_EXCEPTION_IF_NULL(sg.first);
    for (const auto &attr : sg.first->attrs()) {
      fout << attr.first << " : ";
      if (attr.second->isa<BoolImm>()) {
        fout << GetValue<bool>(attr.second);
      } else if (attr.second->isa<StringImm>()) {
        fout << (GetValue<std::string>(attr.second));
      }
      fout << std::endl;
    }
    fout << "subgraph @" << sg.first->ToString() << "(";
    if (sg.first != graph) {
      std::vector<AnfNodePtr> parameters = sg.first->parameters();
      if (parameters.size() == 1) {
        MS_EXCEPTION_IF_NULL(parameters[0]);
        fout << "%para" << (*para_map)[parameters[0]] << "_" << parameters[0]->ToString();
      } else if (parameters.size() > 1) {
        for (size_t idx = 0; idx < parameters.size() - 1; idx++) {
          MS_EXCEPTION_IF_NULL(parameters[idx]);
          fout << "%para" << (*para_map)[parameters[idx]] << "_" << parameters[idx]->ToString();
          fout << ", ";
        }
        MS_EXCEPTION_IF_NULL(parameters[parameters.size() - 1]);
        fout << "%para" << (*para_map)[parameters[parameters.size() - 1]] << "_"
             << parameters[parameters.size() - 1]->ToString();
      }
    }
    fout << ") {" << std::endl;
    MS_EXCEPTION_IF_NULL(sg.second);
    fout << sg.second->buffer.str();
    fout << "}" << std::endl;
    fout << std::endl;
  }
}

void SetDumpConfigByString(const std::string &str, DumpConfig *dump_config) {
  static mindspore::HashMap<std::string, enum LocDumpMode> dump_level_map = {
    {kDumpConfigLineLevel0, kOff}, {kDumpConfigLineLevel1, kTopStack}, {kDumpConfigLineLevel2, kWholeStack}};
  auto it = dump_level_map.find(str);
  if (it != dump_level_map.end()) {
    dump_config->dump_line_level = it->second;
    return;
  }
  if (str == kDumpConfigDisableBackend) {
    dump_config->disable_backend_dump = true;
    return;
  }
  if (str == kDumpConfigEnablePassIR) {
    dump_config->enable_dump_pass_ir = true;
    return;
  }
}

DumpConfig GetDumpConfig() {
  static std::vector<HashSet<std::string>> config_white_list = {
    {kDumpConfigLineLevel0, kDumpConfigLineLevel1, kDumpConfigLineLevel2},
    {kDumpConfigDisableBackend},
    {kDumpConfigEnablePassIR}};
  static DumpConfig dump_config;
  static bool parsed = false;
  if (parsed) {
    return dump_config;
  }
  parsed = true;
  // Start parse config.
  std::string str(common::GetEnv("MS_DEV_DUMP_IR_CONFIG"));
  std::vector<std::shared_ptr<HashSet<std::string>>> configs = {std::make_shared<HashSet<std::string>>(),
                                                                std::make_shared<HashSet<std::string>>(),
                                                                std::make_shared<HashSet<std::string>>()};
  auto constexpr max_string_len = 100;
  if (str.size() > max_string_len) {
    MS_LOG(WARNING) << "Dump ir config length exceed max length: " << max_string_len;
    return dump_config;
  }
  if (str.empty()) {
    return dump_config;
  }
  size_t start_pos = 0;
  // if '#' is the last char of str, the str is illegal, so we use '<=' but not '<'.
  while (start_pos <= str.size()) {
    auto pos = str.find('#', start_pos);
    if (pos == std::string::npos) {
      pos = str.size();
    }
    auto substr = str.substr(start_pos, pos - start_pos);
    start_pos = pos + 1;
    bool is_illegal_config = true;
    for (size_t i = 0; i < config_white_list.size(); i++) {
      if (config_white_list[i].find(substr) != config_white_list[i].end()) {
        is_illegal_config = false;
        (void)configs[i]->insert(substr);
        if (configs[i]->size() > 1) {
          std::ostringstream buffer;
          std::for_each(configs[i]->begin(), configs[i]->end(), [&buffer](const std::string &config) {
            buffer << "\n" << config;
          });
          MS_LOG(WARNING) << "Dump configs are conflict. Conflict configs: " << buffer.str() << "\n"
                          << "Please keep only one of them.";
          return dump_config;
        }
      }
    }
    if (is_illegal_config) {
      std::ostringstream buffer;
      buffer << "Support configs:\n"
             << "[0]: " << kDumpConfigLineLevel0 << "\n"
             << "[1]: " << kDumpConfigLineLevel1 << "\n"
             << "[2]: " << kDumpConfigLineLevel2 << "\n"
             << "[3]: " << kDumpConfigDisableBackend << "\n"
             << "[4]: " << kDumpConfigEnablePassIR;
      MS_LOG(WARNING) << "Illegal dump config:\n" << substr << "\n" << buffer.str();
      return {};
    }
  }
  for (auto &config : configs) {
    SetDumpConfigByString(*config->begin(), &dump_config);
  }
  return dump_config;
}

void GetEnvDumpIrLineLevel(LocDumpMode *dump_location) {
  const auto &config = GetDumpConfig();
  if (config.dump_line_level != kInValid) {
    *dump_location = config.dump_line_level;
  }
}

#ifdef ENABLE_DUMP_IR
void DumpIR(const std::string &filename, const FuncGraphPtr &graph, bool dump_full_name, LocDumpMode dump_location,
            const std::string &target_file) {
  GetEnvDumpIrLineLevel(&dump_location);
  if (graph == nullptr) {
    return;
  }
  auto path = GetSaveGraphsPathName(Common::AddId(filename, ".ir"));
  if (!target_file.empty()) {
    path = target_file;
  }
  auto realpath = Common::CreatePrefixPath(path);
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get real path failed, path=" << path;
    return;
  }

  ChangeFileMode(realpath.value(), S_IWUSR);
  std::ofstream fout(realpath.value());
  std::ostringstream buffer;
  if (!fout.is_open()) {
    MS_LOG(ERROR) << "Open dump file '" << realpath.value() << "' failed!" << ErrnoToString(errno);
    return;
  }

  auto nodes = TopoSort(graph->get_return(), SuccDeeperSimple, AlwaysInclude);
  OrderedMap<AnfNodePtr, int32_t> para_map;
  // Dump global info
  DumpGlobalInfoEntry(graph, buffer);
  int32_t total_para = DumpParams(graph, buffer, &para_map);

  OrderedMap<FuncGraphPtr, std::shared_ptr<SubGraphIRInfo>> sub_graphs;
  // Dump ir in each sub graph
  DumpIRInSubgraph(nodes, &para_map, &sub_graphs, total_para, dump_full_name, dump_location);

  // Output global info
  fout << buffer.str() << std::endl;

  // Output each sub graph
  DumpSubgraph(&sub_graphs, graph, &para_map, fout);

  fout.close();
  // Set file mode to read only by user
  ChangeFileMode(realpath.value(), S_IRUSR);
}

void DumpIRForRDR(const std::string &filename, const FuncGraphPtr &graph, bool dump_full_name,
                  LocDumpMode dump_location) {
  GetEnvDumpIrLineLevel(&dump_location);
  if (graph == nullptr) {
    return;
  }
  auto path = Common::AddId(filename, ".ir");
  auto realpath = Common::CreatePrefixPath(path);
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get real path failed. path=" << path;
    return;
  }

  ChangeFileMode(realpath.value(), S_IWUSR);
  std::ofstream fout(realpath.value());
  std::ostringstream buffer;
  if (!fout.is_open()) {
    MS_LOG(ERROR) << "Open dump file '" << realpath.value() << "' failed!" << ErrnoToString(errno);
    return;
  }

  auto nodes = TopoSort(graph->get_return(), SuccDeeperSimple, AlwaysInclude);
  OrderedMap<AnfNodePtr, int32_t> para_map;
  // Dump global info
  DumpGlobalInfoEntry(graph, buffer);
  int32_t total_para = DumpParams(graph, buffer, &para_map);

  OrderedMap<FuncGraphPtr, std::shared_ptr<SubGraphIRInfo>> sub_graphs;
  // Dump ir in each sub graph
  DumpIRInSubgraph(nodes, &para_map, &sub_graphs, total_para, dump_full_name, dump_location);

  // Output global info
  fout << buffer.str() << std::endl;

  // Output each sub graph
  DumpSubgraph(&sub_graphs, graph, &para_map, fout);

  fout.close();
  // Set file mode to read only by user
  ChangeFileMode(realpath.value(), S_IRUSR);
}

#else
void DumpIR(const std::string &, const FuncGraphPtr &, bool, LocDumpMode, const std::string &) {
  static bool already_printed = false;
  if (already_printed) {
    return;
  }
  already_printed = true;
  MS_LOG(WARNING) << "The functionality of dumping function graph IR is disabled, "
                  << "please recompile source to enable it. See help of building script.";
}
void DumpIRForRDR(const std::string &, const FuncGraphPtr &, bool, LocDumpMode) {
  static bool already_printed = false;
  if (already_printed) {
    return;
  }
  already_printed = true;
  MS_LOG(WARNING) << "The functionality of dumping function graph IR is disabled, "
                  << "please recompile source to enable it. See help of building script.";
}
#endif
}  // namespace mindspore
