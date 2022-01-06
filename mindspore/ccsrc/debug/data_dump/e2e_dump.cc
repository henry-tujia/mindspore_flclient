/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "debug/data_dump/e2e_dump.h"

#include <unistd.h>
#include <sstream>
#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>
#include "debug/data_dump/dump_json_parser.h"
#include "common/trans.h"
#include "debug/anf_ir_utils.h"
#include "debug/common.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "utils/ms_context.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "utils/config_manager.h"
#include "utils/file_utils.h"
#include "debug/data_dump/tensor_stat_dump.h"
#include "abstract/utils.h"
#ifdef ENABLE_DEBUGGER
#include "debug/debug_services.h"
#include "debug/tensor_load.h"
#include "debug/debugger/debugger.h"
#endif

namespace mindspore {
#ifdef ENABLE_D
using ProtoFormat = debugger::dump::OutputFormat;
using ProtoDataType = debugger::dump::OutputDataType;

constexpr int kDhaAtomicAddInfoSize = 128;
constexpr int kL2AtomicAddInfoSize = 128;
constexpr int kAiCoreInfoSize = 256;
constexpr int kDhaAtomicAddStatusSize = 256;
constexpr int kL2AtomicAddStatusSize = 256;
constexpr int kUint64Size = sizeof(uint64_t);
const std::set<std::pair<std::string, std::string>> kSuppTransFormatPair = {
  // {device format, host format}
  {kOpFormat_FRAC_Z, kOpFormat_NCHW},      {kOpFormat_FRAC_NZ, kOpFormat_NCHW},
  {kOpFormat_NC1HWC0, kOpFormat_NCHW},     {kOpFormat_C1HWNCoC0, kOpFormat_NCHW},
  {kOpFormat_NC1HWC0_C04, kOpFormat_NCHW}, {kOpFormat_NDC1HWC0, kOpFormat_NCHW},
  {kOpFormat_FRACTAL_Z_3D, kOpFormat_NCHW}};

const std::map<ProtoFormat, std::string> kFormatToStringMap = {
  {ProtoFormat::FORMAT_NCHW, kOpFormat_NCHW},
  {ProtoFormat::FORMAT_NHWC, kOpFormat_NHWC},
  {ProtoFormat::FORMAT_ND, kOpFormat_ND},
  {ProtoFormat::FORMAT_NC1HWC0, kOpFormat_NC1HWC0},
  {ProtoFormat::FORMAT_FRACTAL_Z, kOpFormat_FRAC_Z},
  {ProtoFormat::FORMAT_NC1HWC0_C04, kOpFormat_NC1HWC0_C04},
  {ProtoFormat::FORMAT_FRACTAL_Z_C04, kOpFormat_FRACTAL_Z_C04},
  {ProtoFormat::FORMAT_NC1KHKWHWC0, kOpFormat_NC1KHKWHWC0},
  {ProtoFormat::FORMAT_HWCN, kOpFormat_HWCN},
  {ProtoFormat::FORMAT_NDHWC, kOpFormat_NDHWC},
  {ProtoFormat::FORMAT_NCDHW, kOpFormat_NCDHW},
  {ProtoFormat::FORMAT_DHWCN, kOpFormat_DHWCN},
  {ProtoFormat::FORMAT_DHWNC, kOpFormat_DHWNC},
  {ProtoFormat::FORMAT_NDC1HWC0, kOpFormat_NDC1HWC0},
  {ProtoFormat::FORMAT_FRACTAL_Z_3D, kOpFormat_FRACTAL_Z_3D},
  {ProtoFormat::FORMAT_C1HWNCoC0, kOpFormat_C1HWNCoC0},
  {ProtoFormat::FORMAT_FRACTAL_NZ, kOpFormat_FRAC_NZ},
  {ProtoFormat::FORMAT_FRACTAL_ZN_LSTM, kOpFormat_FRACTAL_ZN_LSTM}};

const std::map<ProtoDataType, mindspore::TypeId> kDataTypetoMSTypeMap = {
  {ProtoDataType::DT_UNDEFINED, mindspore::TypeId::kTypeUnknown},
  {ProtoDataType::DT_FLOAT, mindspore::TypeId::kNumberTypeFloat32},
  {ProtoDataType::DT_FLOAT16, mindspore::TypeId::kNumberTypeFloat16},
  {ProtoDataType::DT_INT8, mindspore::TypeId::kNumberTypeInt8},
  {ProtoDataType::DT_UINT8, mindspore::TypeId::kNumberTypeUInt8},
  {ProtoDataType::DT_INT16, mindspore::TypeId::kNumberTypeInt16},
  {ProtoDataType::DT_UINT16, mindspore::TypeId::kNumberTypeUInt16},
  {ProtoDataType::DT_INT32, mindspore::TypeId::kNumberTypeInt32},
  {ProtoDataType::DT_INT64, mindspore::TypeId::kNumberTypeInt64},
  {ProtoDataType::DT_UINT32, mindspore::TypeId::kNumberTypeUInt32},
  {ProtoDataType::DT_UINT64, mindspore::TypeId::kNumberTypeUInt64},
  {ProtoDataType::DT_BOOL, mindspore::TypeId::kNumberTypeBool},
  {ProtoDataType::DT_DOUBLE, mindspore::TypeId::kNumberTypeFloat64},
  {ProtoDataType::DT_STRING, mindspore::TypeId::kObjectTypeString}};
#endif

bool E2eDump::IsDeviceTargetGPU() {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  return context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kGPUDevice;
}

void E2eDump::DumpGPUMemToFile(const std::string &file_path, const std::string &original_kernel_name,
                               const device::DeviceAddress &addr, const ShapeVector &int_shapes,
                               const TypeId &host_type, const TypeId &device_type, bool trans_flag, size_t slot,
                               const Debugger *debugger) {
#ifdef ENABLE_DEBUGGER
  auto format = kOpFormat_DEFAULT;
  MS_EXCEPTION_IF_NULL(debugger);
  auto ret = debugger->DumpTensorToFile(original_kernel_name, trans_flag, file_path, format, int_shapes, host_type,
                                        device_type, addr.format(), slot);
  if (!ret) {
    MS_LOG(INFO) << "DumpTensorToFile Failed: flag:" << trans_flag << ", path:" << file_path
                 << ", host_format:" << format;
  }
#endif
}

void E2eDump::DumpOutput(const session::KernelGraph *graph, const std::string &dump_path, const Debugger *debugger) {
  MS_EXCEPTION_IF_NULL(graph);
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if (!dump_json_parser.OutputNeedDump()) {
    return;
  }
  MS_LOG(INFO) << "Start e2e dump output";
  bool trans_flag = dump_json_parser.trans_flag();
  const auto &apply_kernels = graph->execution_order();
  for (const auto &node : apply_kernels) {
    MS_EXCEPTION_IF_NULL(node);
    std::string kernel_name = GetKernelNodeName(node);
    if (!dump_json_parser.NeedDump(kernel_name)) {
      continue;
    }
    DumpJsonParser::GetInstance().MatchKernel(kernel_name);
    DumpOutputImpl(node, trans_flag, dump_path, &kernel_name, debugger);
  }
}

void E2eDump::DumpOutputSingleNode(const CNodePtr &node, const std::string &dump_path, const Debugger *debugger) {
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if (!dump_json_parser.OutputNeedDump()) {
    return;
  }
  bool trans_flag = dump_json_parser.trans_flag();
  MS_EXCEPTION_IF_NULL(node);
  std::string kernel_name = GetKernelNodeName(node);
  if (!dump_json_parser.NeedDump(kernel_name)) {
    return;
  }
  DumpJsonParser::GetInstance().MatchKernel(kernel_name);
  DumpOutputImpl(node, trans_flag, dump_path, &kernel_name, debugger);
}

void E2eDump::DumpOutputImpl(const CNodePtr &node, bool trans_flag, const std::string &dump_path,
                             std::string *kernel_name, const Debugger *debugger) {
  MS_EXCEPTION_IF_NULL(node);
  GetFileKernelName(NOT_NULL(kernel_name));
  auto output_size = AnfAlgo::GetOutputTensorNum(node);
  for (size_t j = 0; j < output_size; ++j) {
    if (!AnfAlgo::OutputAddrExist(node, j)) {
      continue;
    }
    auto addr = AnfAlgo::GetOutputAddr(node, j);
    MS_EXCEPTION_IF_NULL(addr);
    ShapeVector int_shapes;
    GetDumpIntShape(node, j, NOT_NULL(&int_shapes), trans_flag);
    auto type = AnfAlgo::GetOutputInferDataType(node, j);
    auto device_type = AnfAlgo::GetOutputDeviceDataType(node, j);
    std::string op_type = AnfAlgo::GetCNodeName(node);
    std::string op_name = GetOpNameWithoutScope(*kernel_name);
    uint32_t task_id = 0;
    uint32_t stream_id = 0;
    uint64_t timestamp = GetTimeStamp();
    std::string file_path = dump_path + '/' + op_type + '.' + op_name + '.' + std::to_string(task_id) + '.' +
                            std::to_string(stream_id) + '.' + std::to_string(timestamp) + ".output." +
                            std::to_string(j);
    if (IsDeviceTargetGPU()) {
      if (DumpJsonParser::GetInstance().IsStatisticDump()) {
        TensorStatDump stat_dump(op_type, op_name, task_id, stream_id, timestamp, false, j, j);
        stat_dump.DumpTensorStatsToFile(GetKernelNodeName(node), dump_path, debugger);
      }
      if (DumpJsonParser::GetInstance().IsTensorDump()) {
        DumpGPUMemToFile(file_path, GetKernelNodeName(node), *addr, int_shapes, type, device_type, trans_flag, j,
                         debugger);
      }
    } else {
      DumpMemToFile(file_path, *addr, int_shapes, type, trans_flag);
    }
  }
}

void E2eDump::DumpInput(const session::KernelGraph *graph, const std::string &dump_path, const Debugger *debugger) {
  MS_EXCEPTION_IF_NULL(graph);
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if (!dump_json_parser.InputNeedDump()) {
    return;
  }
  MS_LOG(INFO) << "Start e2e dump input";
  bool trans_flag = dump_json_parser.trans_flag();
  const auto &apply_kernels = graph->execution_order();
  for (const auto &node : apply_kernels) {
    MS_EXCEPTION_IF_NULL(node);
    std::string kernel_name = GetKernelNodeName(node);
    if (!dump_json_parser.NeedDump(kernel_name)) {
      continue;
    }
    DumpJsonParser::GetInstance().MatchKernel(kernel_name);
    DumpInputImpl(node, trans_flag, dump_path, &kernel_name, debugger);
  }
}

void E2eDump::DumpInputSingleNode(const CNodePtr &node, const std::string &dump_path, const Debugger *debugger) {
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if (!dump_json_parser.InputNeedDump()) {
    return;
  }
  bool trans_flag = dump_json_parser.trans_flag();
  MS_EXCEPTION_IF_NULL(node);
  std::string kernel_name = GetKernelNodeName(node);
  if (!dump_json_parser.NeedDump(kernel_name)) {
    return;
  }
  DumpJsonParser::GetInstance().MatchKernel(kernel_name);
  DumpInputImpl(node, trans_flag, dump_path, &kernel_name, debugger);
}

void E2eDump::DumpInputImpl(const CNodePtr &node, bool trans_flag, const std::string &dump_path,
                            std::string *kernel_name, const Debugger *debugger) {
  MS_EXCEPTION_IF_NULL(node);
  GetFileKernelName(NOT_NULL(kernel_name));
  auto input_size = AnfAlgo::GetInputTensorNum(node);
  for (size_t j = 0; j < input_size; ++j) {
    auto kernel_with_index = AnfAlgo::GetPrevNodeOutput(node, j);
    auto input = kernel_with_index.first;
    auto index = kernel_with_index.second;
    if (!AnfAlgo::OutputAddrExist(input, index)) {
      continue;
    }
    auto addr = AnfAlgo::GetOutputAddr(input, index);
    MS_EXCEPTION_IF_NULL(addr);

    std::string tensor_name = GetKernelNodeName(node);
    size_t slot = j;
    if (IsDeviceTargetGPU()) {
      auto input_kernel = node->input(j + 1);
      std::string input_kernel_name = GetKernelNodeName(input_kernel);
      tensor_name = input_kernel_name;
      slot = 0;
    }
    ShapeVector int_shapes;
    GetDumpIntShape(input, index, NOT_NULL(&int_shapes), trans_flag);
    auto type = AnfAlgo::GetOutputInferDataType(input, index);
    auto device_type = AnfAlgo::GetOutputDeviceDataType(input, index);
    std::string op_type = AnfAlgo::GetCNodeName(node);
    std::string op_name = GetOpNameWithoutScope(*kernel_name);
    uint64_t timestamp = GetTimeStamp();
    uint32_t task_id = 0;
    uint32_t stream_id = 0;
    std::string file_path = dump_path + '/' + op_type + '.' + op_name + '.' + std::to_string(task_id) + '.' +
                            std::to_string(stream_id) + '.' + std::to_string(timestamp) + ".input." + std::to_string(j);
    MS_EXCEPTION_IF_NULL(addr);
    if (IsDeviceTargetGPU()) {
      if (DumpJsonParser::GetInstance().IsStatisticDump()) {
        TensorStatDump stat_dump(op_type, op_name, task_id, stream_id, timestamp, true, j, slot);
        stat_dump.DumpTensorStatsToFile(tensor_name, dump_path, debugger);
      }
      if (DumpJsonParser::GetInstance().IsTensorDump()) {
        DumpGPUMemToFile(file_path, tensor_name, *addr, int_shapes, type, device_type, trans_flag, slot, debugger);
      }
    } else {
      DumpMemToFile(file_path, *addr, int_shapes, type, trans_flag);
    }
  }
}

void E2eDump::DumpSingleAnfNode(const AnfNodePtr &anf_node, const size_t output_index, const std::string &dump_path,
                                bool trans_flag, const Debugger *debugger) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if ((!anf_node->isa<Parameter>() && !anf_node->isa<ValueNode>()) || IsValueNode<StringImm>(anf_node)) {
    return;
  }
  std::string node_name = GetKernelNodeName(anf_node);
  if (!dump_json_parser.NeedDump(node_name)) {
    return;
  }
  DumpJsonParser::GetInstance().MatchKernel(node_name);
  GetFileKernelName(NOT_NULL(&node_name));

  std::string dump_name = node_name;
  const std::string cst_prefix = "Default--";
  if (anf_node->isa<ValueNode>()) {
    if (dump_name.find(cst_prefix) == std::string::npos) {
      MS_LOG(INFO) << "Incorrect constant format: " << dump_name;
      return;
    }
    dump_name = node_name.substr(cst_prefix.length());
    trans_flag = false;
  }

  // check if output address exists, if not, return;
  if (!AnfAlgo::OutputAddrExist(anf_node, output_index)) {
    return;
  }
  auto addr = AnfAlgo::GetOutputAddr(anf_node, output_index);
  MS_EXCEPTION_IF_NULL(addr);
  ShapeVector int_shapes;
  GetDumpIntShape(anf_node, output_index, NOT_NULL(&int_shapes), trans_flag);
  auto type = AnfAlgo::GetOutputInferDataType(anf_node, output_index);
  auto device_type = AnfAlgo::GetOutputDeviceDataType(anf_node, output_index);
  uint64_t timestamp = GetTimeStamp();
  uint32_t task_id = 0;
  uint32_t stream_id = 0;
  std::string file_path = dump_path + "/Parameter." + dump_name + '.' + std::to_string(task_id) + '.' +
                          std::to_string(stream_id) + '.' + std::to_string(timestamp) + ".output.0";
  if (IsDeviceTargetGPU()) {
    if (dump_json_parser.IsStatisticDump()) {
      TensorStatDump stat_dump("Parameter", dump_name, task_id, stream_id, timestamp, false, 0, 0);
      stat_dump.DumpTensorStatsToFile(node_name, dump_path, debugger);
    }
    if (dump_json_parser.IsTensorDump()) {
      DumpGPUMemToFile(file_path, node_name, *addr, int_shapes, type, device_type, trans_flag, 0, debugger);
    }
  } else {
    DumpMemToFile(file_path, *addr, int_shapes, type, trans_flag);
  }
}

void E2eDump::DumpParameters(const session::KernelGraph *graph, const std::string &dump_path,
                             const Debugger *debugger) {
  MS_EXCEPTION_IF_NULL(graph);
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if (!dump_json_parser.OutputNeedDump()) {
    return;
  }
  MS_LOG(INFO) << "Start e2e dump parameters";
  bool trans_flag = dump_json_parser.trans_flag();

  // dump parameters
  const auto &parameters = graph->inputs();
  for (auto &item : parameters) {
    DumpSingleAnfNode(item, PARAMETER_OUTPUT_INDEX, dump_path, trans_flag, debugger);
  }
}

void E2eDump::DumpConstantData(const session::KernelGraph *graph, uint32_t rank_id, const Debugger *debugger) {
  MS_EXCEPTION_IF_NULL(graph);
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if (!IsDeviceTargetGPU() || !dump_json_parser.e2e_dump_enabled()) {
    return;
  }
  uint32_t graph_id = graph->graph_id();
  std::string cst_path = GenerateDumpPath(graph_id, rank_id, true);
  if (!Common::FileExists(cst_path)) {
    DumpConstantData(graph, cst_path, debugger);
  }
}

void E2eDump::DumpConstantData(const session::KernelGraph *graph, const std::string &cst_dump_path,
                               const Debugger *debugger) {
  // Dump constant to npy file
  MS_EXCEPTION_IF_NULL(graph);
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  MS_LOG(INFO) << "DumpConstants. Current iteration is " << dump_json_parser.cur_dump_iter();
  MS_LOG(INFO) << "Current graph id is " << graph->graph_id();
  if (!dump_json_parser.OutputNeedDump()) {
    return;
  }
  const auto value_nodes = graph->graph_value_nodes();
  for (auto &item : value_nodes) {
    DumpSingleAnfNode(item, VALUE_NODE_OUTPUT_INDEX, cst_dump_path, false, debugger);
  }
}

void E2eDump::UpdateIterDumpSetup(const session::KernelGraph *graph, bool sink_mode) {
  uint32_t graph_id = graph->graph_id();
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if (IsDeviceTargetGPU()) {
    if (starting_graph_id == INT32_MAX) {
      starting_graph_id = graph_id;
    } else if (starting_graph_id == graph_id && !MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
      // Update dump iter for mindrt runtime is done using UpdateIterGPUDump().
      // Update dump iter for GPU old runtime.
      dump_json_parser.UpdateDumpIter();
    }
    return;
  }
  // If device target is Ascend
  if (sink_mode && graph->IsDatasetGraph()) {
    MS_LOG(INFO) << "No need to update iteration for dataset graph.";
    return;
  }

  // In multi network scripts, dump iter is equal to the number of networks that have been executed so far.
  dump_json_parser.UpdateDumpIter();
}

void E2eDump::DumpSetup(const session::KernelGraph *graph) {
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  bool sink_mode = (ConfigManager::GetInstance().dataset_mode() || E2eDump::isDatasetGraph(graph));

  if (dump_json_parser.async_dump_enabled() || dump_json_parser.e2e_dump_enabled()) {
    UpdateIterDumpSetup(graph, sink_mode);
  }
}

void E2eDump::UpdateIterGPUDump() {
  if (!IsDeviceTargetGPU()) {
    return;
  }
  DumpJsonParser::GetInstance().UpdateDumpIter();
}

void E2eDump::DumpRunIter(const KernelGraphPtr &graph, uint32_t rank_id) {
  auto &json_parser = DumpJsonParser::GetInstance();
  if (!(json_parser.async_dump_enabled() || json_parser.e2e_dump_enabled())) {
    return;
  }
  bool sink_mode = (ConfigManager::GetInstance().dataset_mode() || graph->IsDatasetGraph());
  auto iter_num = SizeToInt(LongToSize(ConfigManager::GetInstance().iter_num()));
  if (graph->IsDatasetGraph()) {
    MS_LOG(INFO) << "graph: " << graph->graph_id() << " is dataset graph, not creating graph history file.";
    return;
  }
  std::string execution_order_path = json_parser.path() + "/rank_" + std::to_string(rank_id) + "/execution_order/";
  std::string file_name_to_check =
    execution_order_path + "/ms_global_execution_order_graph_" + std::to_string(graph->graph_id()) + ".csv";
  auto real_path = Common::CreatePrefixPath(file_name_to_check);
  if (!real_path.has_value()) {
    MS_LOG(WARNING) << "Check file path: " << file_name_to_check << " failed.";
    return;
  }
  std::string file_name = real_path.value();
  ChangeFileMode(file_name, S_IWUSR);
  std::ofstream fout(file_name, std::ofstream::app);
  if (!fout.is_open()) {
    MS_LOG(WARNING) << "Open file for saving graph global execution order failed.";
    return;
  }
  if (sink_mode && json_parser.async_dump_enabled()) {
    // for async dump when sink_mode = true, cur_dump_iter() = current_epoch
    // dump history for all iterations in the epoch
    Debugger::GetInstance()->UpdateGraphIterMap(graph->graph_id(), iter_num);
    auto graph_iter_map = Debugger::GetInstance()->GetGraphIterMap();
    auto step_per_epoch = graph_iter_map[graph->graph_id()];
    for (int i = 0; i < step_per_epoch; i++) {
      auto step = (json_parser.cur_dump_iter() * step_per_epoch) + i;
      fout << (std::to_string(step) + "\n");
    }
  } else {
    fout << std::to_string(json_parser.cur_dump_iter()) + "\n";
  }
  fout.close();
  ChangeFileMode(file_name, S_IRUSR);
}

void E2eDump::DumpData(const session::KernelGraph *graph, uint32_t rank_id, const Debugger *debugger) {
  MS_EXCEPTION_IF_NULL(graph);
  bool success = false;
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  uint32_t graph_id = graph->graph_id();

  if (dump_json_parser.GetIterDumpFlag()) {
    MS_LOG(INFO) << "Start e2e dump. Current iteration is " << dump_json_parser.cur_dump_iter();
    MS_LOG(INFO) << "Current graph id is " << graph_id;
    std::string dump_path = GenerateDumpPath(graph_id, rank_id);
    std::string cst_path = GenerateDumpPath(graph_id, rank_id, true);

    if (dump_json_parser.IsStatisticDump()) {
      TensorStatDump::OpenStatisticsFile(dump_path);
    }
    DumpInput(graph, dump_path, debugger);
    DumpOutput(graph, dump_path, debugger);
    DumpParameters(graph, dump_path, debugger);
    if (IsDeviceTargetGPU() && dump_json_parser.e2e_dump_enabled()) {
      DumpConstantData(graph, cst_path, debugger);
    }
    if (dump_json_parser.IsStatisticDump()) {
      CsvWriter::GetInstance().CloseFile();
    }
    success = true;
  }

  if (success) {
    MS_LOG(DEBUG) << "E2eDump Dump Data completed!";
  } else {
    MS_LOG(DEBUG) << "E2eDump Dump has not occurred!";
  }
}

bool E2eDump::DumpSingleNodeData(const CNodePtr &node, uint32_t graph_id, uint32_t rank_id, const Debugger *debugger) {
  bool success = false;
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if (dump_json_parser.GetIterDumpFlag()) {
    std::string dump_path = GenerateDumpPath(graph_id, rank_id);
    DumpInputSingleNode(node, dump_path, debugger);
    DumpOutputSingleNode(node, dump_path, debugger);
    success = true;
  }
  return success;
}

bool E2eDump::DumpParametersData(const session::KernelGraph *graph, uint32_t rank_id, const Debugger *debugger) {
  bool success = false;
  uint32_t graph_id = graph->graph_id();
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if (dump_json_parser.GetIterDumpFlag()) {
    MS_LOG(INFO) << "DumpParameters. Current iteration is " << dump_json_parser.cur_dump_iter();
    MS_LOG(INFO) << "Current graph id is " << graph_id;
    std::string dump_path = GenerateDumpPath(graph_id, rank_id);
    DumpParameters(graph, dump_path, debugger);
    success = true;
  }
  return success;
}
bool E2eDump::isDatasetGraph(const session::KernelGraph *graph) {
  // check if there is GetNext or InitDataSetQueue node
  const auto &nodes = graph->execution_order();
  for (const auto &node : nodes) {
    auto node_name = AnfAlgo::GetCNodeName(node);
    if (node_name == prim::kPrimGetNext->name() || node_name == prim::kPrimInitDataSetQueue->name()) {
      return true;
    }
  }
  return false;
}

bool E2eDump::DumpDirExists(const std::string &dump_path) {
  DIR *dir = opendir(dump_path.c_str());
  if (dir != nullptr) {
    MS_LOG(INFO) << "Dump dir " << dump_path << " exists";
    if (closedir(dir) == -1) {
      MS_LOG(WARNING) << "Dump dir " << dump_path << " close failed!";
    }
    return true;
  }
  return false;
}

#ifdef ENABLE_D
void E2eDump::DumpTensorToFile(const std::string &dump_path, const debugger::dump::DumpData &dump_data,
                               char *data_ptr) {
  // dump input tensors
  std::vector<debugger::dump::OpInput> input_tensors(dump_data.input().begin(), dump_data.input().end());
  uint64_t offset = 0;
  for (uint32_t slot = 0; slot < input_tensors.size(); slot++) {
    auto in_tensor = input_tensors[slot];
    auto succ = ConvertFormatForTensorAndDump(dump_path, in_tensor, data_ptr + offset, "input", slot);
    if (!succ) {
      MS_LOG(INFO) << "Failed to convert format for tensor " << dump_path << ".input." << slot;
    }
    offset += in_tensor.size();
  }

  // dump output tensors
  std::vector<debugger::dump::OpOutput> output_tensors(dump_data.output().begin(), dump_data.output().end());
  for (uint32_t slot = 0; slot < output_tensors.size(); slot++) {
    auto out_tensor = output_tensors[slot];
    auto succ = ConvertFormatForTensorAndDump(dump_path, out_tensor, data_ptr + offset, "output", slot);
    if (!succ) {
      MS_LOG(INFO) << "Failed to convert format for tensor " << dump_path << ".output." << slot;
    }
    offset += out_tensor.size();
  }
}

template <typename T>
bool DumpTensorStatsIfNeeded(const std::string &dump_path, const T &tensor, char *data_ptr, const std::string &io,
                             uint32_t slot, const ShapeVector &shape, TypeId type) {
  // dump_path: dump_dir/op_type.op_name.task_id.stream_id.timestamp
  if (!DumpJsonParser::GetInstance().IsStatisticDump()) {
    return true;
  }
  size_t pos = dump_path.rfind("/");
  std::string file_name = dump_path.substr(pos + 1);
  size_t first_dot = file_name.find(".");
  size_t fourth_dot = file_name.rfind(".");
  size_t third_dot = file_name.rfind(".", fourth_dot - 1);
  size_t second_dot = file_name.rfind(".", third_dot - 1);
  if (first_dot == std::string::npos || second_dot == std::string::npos || third_dot == std::string::npos ||
      first_dot == second_dot) {
    MS_LOG(ERROR) << "Dump path " << dump_path << " received is not well formed";
    return false;
  }
  std::string op_type = file_name.substr(0, first_dot);
  std::string op_name = file_name.substr(first_dot + 1, second_dot - first_dot - 1);
  std::string task_id = file_name.substr(second_dot + 1, third_dot - second_dot - 1);
  std::string stream_id = file_name.substr(third_dot + 1, fourth_dot - third_dot - 1);
  std::string timestamp = file_name.substr(fourth_dot + 1);
  TensorStatDump stat_dump(op_type, op_name, task_id, stream_id, timestamp, io, slot, slot);
  std::shared_ptr<TensorData> data = std::make_shared<TensorData>();
  if (type <= TypeId::kNumberTypeBegin || type >= TypeId::kNumberTypeComplex64) {
    MS_LOG(ERROR) << "Data type of operator " << file_name << " is not supported by statistic dump";
    return false;
  }
  data->SetType(type);
  data->SetByteSize((size_t)tensor.size());
  data->SetShape(shape);
  data->SetDataPtr(data_ptr);
  return stat_dump.DumpTensorStatsToFile(dump_path.substr(0, pos), data);
}

template <typename T>
bool E2eDump::ConvertFormatForTensorAndDump(std::string dump_path, const T &tensor, char *data_ptr,
                                            const std::string &io, uint32_t slot) {
  // dump_path: dump_dir/op_type.op_name.task_id.stream_id.timestamp
  std::ostringstream dump_path_ss;
  dump_path_ss << dump_path << "." << io << "." << slot << ".";
  std::string dump_path_slot = dump_path_ss.str();
  // get format
  auto iter_fmt = kFormatToStringMap.find(tensor.format());
  if (iter_fmt == kFormatToStringMap.end()) {
    MS_LOG(INFO) << "Unsupported tensor format for tensor " << dump_path << ": unknown(" << tensor.format() << ")";
    return false;
  }
  std::string device_format = iter_fmt->second;
  // get data type
  auto iter_dtype = kDataTypetoMSTypeMap.find(tensor.data_type());
  if (iter_dtype == kDataTypetoMSTypeMap.end()) {
    MS_LOG(INFO) << "Unsupported data type for tensor " << dump_path << ": unknown(" << tensor.data_type() << ")";
    return false;
  }
  auto src_type = iter_dtype->second;
  // get host shape
  std::vector<size_t> device_shape;
  (void)std::copy(tensor.shape().dim().begin(), tensor.shape().dim().end(), std::back_inserter(device_shape));
  std::vector<size_t> host_shape;
  (void)std::copy(tensor.original_shape().dim().begin(), tensor.original_shape().dim().end(),
                  std::back_inserter(host_shape));
  ShapeVector shape_to;
  (void)std::transform(host_shape.begin(), host_shape.end(), std::back_inserter(shape_to), SizeToLong);
  size_t data_size = (size_t)tensor.size();

  bool trans_success = false;
  auto trans_buf = std::vector<uint8_t>(data_size);
  // convert format to host format. It can be either NCHW or ND (non 4-dimemsions).
  const uint8_t kNumFourDim = 4;
  std::string host_format;
  if (host_shape.size() == kNumFourDim) {
    host_format = kOpFormat_NCHW;
  } else {
    host_format = kOpFormat_ND;
  }
  if (device_format != host_format) {
    auto iter = kSuppTransFormatPair.find(std::make_pair(device_format, host_format));
    if (iter == kSuppTransFormatPair.end()) {
      MS_LOG(INFO) << "Do not support convert from format " << device_format << " to " << host_format << " for tensor "
                   << dump_path_slot;
    } else {
      const trans::FormatArgs format_args{data_ptr,   data_size,    host_format, device_format,
                                          host_shape, device_shape, src_type};
      auto group = tensor.sub_format() > 1 ? tensor.sub_format() : 1;
      trans_success = trans::TransFormatFromDeviceToHost(format_args, trans_buf.data(), group);
      if (!trans_success) {
        MS_LOG(ERROR) << "Trans format failed.";
      }
    }
  }
  // dump tensor data into npy file
  bool dump_success = true;
  if (trans_success) {
    dump_success = DumpTensorStatsIfNeeded(dump_path, tensor, reinterpret_cast<char *>(trans_buf.data()), io, slot,
                                           shape_to, src_type);
    if (DumpJsonParser::GetInstance().IsTensorDump()) {
      dump_path_slot += host_format;
      dump_success =
        DumpJsonParser::DumpToFile(dump_path_slot, trans_buf.data(), data_size, shape_to, src_type) && dump_success;
    }
  } else {
    dump_success = DumpTensorStatsIfNeeded(dump_path, tensor, data_ptr, io, slot, shape_to, src_type);
    if (DumpJsonParser::GetInstance().IsTensorDump()) {
      dump_path_slot += device_format;
      dump_success =
        DumpJsonParser::DumpToFile(dump_path_slot, data_ptr, data_size, shape_to, src_type) && dump_success;
    }
  }
  return dump_success;
}

uint64_t UnpackUint64Value(char *ptr) {
#if defined(__APPLE__)
  return *reinterpret_cast<const uint64_t *>(ptr);
#else
  return le16toh(*reinterpret_cast<const uint64_t *>(ptr));
#endif
}

std::string IntToHexString(const uint64_t value) {
  std::stringstream ss;
  ss << "0x" << std::hex << value;
  return ss.str();
}

nlohmann::json E2eDump::ParseOverflowInfo(char *data_ptr) {
  uint32_t index = 0;
  uint64_t model_id = UnpackUint64Value(data_ptr + index);
  index += kUint64Size;
  uint64_t stream_id = UnpackUint64Value(data_ptr + index);
  index += kUint64Size;
  uint64_t task_id = UnpackUint64Value(data_ptr + index);
  index += kUint64Size;
  uint64_t task_type = UnpackUint64Value(data_ptr + index);
  index += kUint64Size;
  uint64_t pc_start = UnpackUint64Value(data_ptr + index);
  index += kUint64Size;
  uint64_t para_base = UnpackUint64Value(data_ptr + index);

  nlohmann::json overflow_info;
  overflow_info["model_id"] = model_id;
  overflow_info["stream_id"] = stream_id;
  overflow_info["task_id"] = task_id;
  overflow_info["task_type"] = task_type;
  overflow_info["pc_start"] = IntToHexString(pc_start);
  overflow_info["para_base"] = IntToHexString(para_base);
  return overflow_info;
}

void E2eDump::DumpOpDebugToFile(const std::string &dump_path, const debugger::dump::DumpData &dump_data,
                                char *data_ptr) {
  std::string out_path = dump_path + ".output.";
  std::vector<debugger::dump::OpOutput> op_debug(dump_data.output().begin(), dump_data.output().end());
  for (uint32_t slot = 0; slot < op_debug.size(); slot++) {
    uint32_t index = 0;
    // parse DHA Atomic Add info
    nlohmann::json dha_atomic_add_info = ParseOverflowInfo(data_ptr + index);
    index += kDhaAtomicAddInfoSize;
    // parse L2 Atomic Add info
    nlohmann::json l2_atomic_add_info = ParseOverflowInfo(data_ptr + index);
    index += kL2AtomicAddInfoSize;
    // parse AICore info
    nlohmann::json ai_core_info = ParseOverflowInfo(data_ptr + index);
    index += kAiCoreInfoSize;
    // parse DHA Atomic Add status
    dha_atomic_add_info["status"] = UnpackUint64Value(data_ptr + index);
    index += kDhaAtomicAddStatusSize;
    // parse L2 Atomic Add status
    l2_atomic_add_info["status"] = UnpackUint64Value(data_ptr + index);
    index += kL2AtomicAddStatusSize;
    // parse AICore status
    uint64_t kernel_code = UnpackUint64Value(data_ptr + index);
    index += kUint64Size;
    uint64_t block_idx = UnpackUint64Value(data_ptr + index);
    index += kUint64Size;
    uint64_t status = UnpackUint64Value(data_ptr + index);
    ai_core_info["kernel_code"] = IntToHexString(kernel_code);
    ai_core_info["block_idx"] = block_idx;
    ai_core_info["status"] = status;

    nlohmann::json opdebug_data;
    opdebug_data["DHA Atomic Add"] = dha_atomic_add_info;
    opdebug_data["L2 Atomic Add"] = l2_atomic_add_info;
    opdebug_data["AI Core"] = ai_core_info;

    // save json to file
    DumpToFile(out_path + std::to_string(slot) + ".json", opdebug_data.dump());
  }
}
#endif  // ENABLE_D
}  // namespace mindspore
