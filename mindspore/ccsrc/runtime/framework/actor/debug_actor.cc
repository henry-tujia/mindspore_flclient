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

#include "runtime/framework/actor/debug_actor.h"
#include <vector>
#include <memory>
#include <string>
#include "runtime/framework/actor/debug_aware_actor.h"
#include "mindrt/include/async/async.h"
#include "utils/log_adapter.h"
#ifndef ENABLE_SECURITY
#include "debug/data_dump/cpu_e2e_dump.h"
#include "debug/data_dump/e2e_dump.h"
#endif
#ifdef ENABLE_DEBUGGER
#include "debug/debugger/debugger.h"
#include "debug/debugger/debugger_utils.h"
#endif

namespace mindspore {
namespace runtime {
void DebugActor::Debug(const AnfNodePtr &node, const KernelLaunchInfo *launch_info_,
                       const DeviceContext *device_context, OpContext<DeviceTensor> *const op_context,
                       const AID *from_aid) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(op_context);
  MS_EXCEPTION_IF_NULL(from_aid);

  if (!node->isa<CNode>()) {
    // Call back to the from actor to process after debug finished.
    ActorDispatcher::Send(*from_aid, &DebugAwareActor::OnDebugFinish, op_context);
    return;
  }

  const auto &cnode = node->cast<CNodePtr>();
  if (device_context->GetDeviceAddressType() == device::DeviceAddressType::kCPU) {
#ifndef ENABLE_SECURITY
    if (DumpJsonParser::GetInstance().GetIterDumpFlag()) {
      auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(cnode->func_graph());
      MS_EXCEPTION_IF_NULL(kernel_graph);
      CPUE2eDump::DumpCNodeData(cnode, kernel_graph->graph_id());
      CPUE2eDump::DumpRunIter(kernel_graph);
    }
#endif
  } else if (device_context->GetDeviceAddressType() == device::DeviceAddressType::kGPU) {
#ifdef ENABLE_DEBUGGER
    auto debugger = Debugger::GetInstance();
    if (debugger != nullptr) {
      std::string kernel_name = cnode->fullname_with_scope();
      debugger->SetCurNode(kernel_name);
      bool read_data = CheckReadData(cnode);
      if (read_data) {
        ReadDataAndDump(cnode, launch_info_, exec_order_);
      }
    }
    exec_order_ += 1;
#endif
  }

  // Call back to the from actor to process after debug finished.
  ActorDispatcher::Send(*from_aid, &DebugAwareActor::OnDebugFinish, op_context);
}

void DebugActor::DebugForGraph(const KernelGraphPtr &graph, const DeviceContext *device_context,
                               OpContext<DeviceTensor> *const op_context, const AID *from_aid) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(op_context);
  MS_EXCEPTION_IF_NULL(from_aid);
#ifdef ENABLE_DEBUGGER
  LoadDataForDump(graph);
#endif
  // Call back to the from actor to process after debug finished.
  ActorDispatcher::Send(*from_aid, &DebugAwareActor::OnDebugFinish, op_context);
}

void DebugActor::DebugOnStepBegin(std::vector<KernelGraphPtr> graphs, std::vector<DeviceContext *> device_contexts,
                                  OpContext<DeviceTensor> *const op_context, const AID *from_aid) {
  MS_EXCEPTION_IF_NULL(op_context);
  MS_EXCEPTION_IF_NULL(from_aid);
#ifdef ENABLE_DEBUGGER
  if (!graphs.empty()) {
    // First graph is the dataset graph when dataset_sink_mode = True
    auto graph = graphs[0];
    std::string error_info = CheckDatasetSinkMode(graph);
    if (!error_info.empty()) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*op_context), error_info);
    }
  }
  auto debugger = Debugger::GetInstance();
  if (debugger != nullptr && debugger->DebuggerBackendEnabled()) {
    debugger->PreExecuteGraphDebugger(graphs);
  }
#endif

#ifndef ENABLE_SECURITY
  if (DumpJsonParser::GetInstance().e2e_dump_enabled()) {
    DumpJsonParser::GetInstance().ClearGraph();
    for (size_t i = 0; i < graphs.size(); ++i) {
      MS_EXCEPTION_IF_NULL(device_contexts[i]);
      if (device_contexts[i]->GetDeviceAddressType() == device::DeviceAddressType::kCPU) {
        DumpJsonParser::GetInstance().SaveGraph(graphs[i].get());
      }
    }
  }
#endif
  // Call back to the from actor to process after debug finished.
  ActorDispatcher::Send(*from_aid, &DebugAwareActor::OnDebugFinish, op_context);
}

void DebugActor::DebugOnStepEnd(OpContext<DeviceTensor> *const op_context, const AID *from_aid) {
  MS_EXCEPTION_IF_NULL(op_context);
  MS_EXCEPTION_IF_NULL(from_aid);

#ifndef ENABLE_SECURITY
  if (DumpJsonParser::GetInstance().GetIterDumpFlag()) {
    CPUE2eDump::DumpParametersData();
    CPUE2eDump::DumpConstantsData();
  }
#endif

#ifdef ENABLE_DEBUGGER
  auto debugger = Debugger::GetInstance();
  if (debugger != nullptr) {
    // Reset exec_order for the next step
    exec_order_ = 0;
    debugger->Debugger::PostExecuteGraphDebugger();
    debugger->Debugger::UpdateStepNumGPU();
  }
#else
#ifndef ENABLE_SECURITY
  DumpJsonParser::GetInstance().UpdateDumpIter();
#endif
#endif

  // Call back to the from actor to process after debug finished.
  ActorDispatcher::Send(*from_aid, &DebugAwareActor::OnDebugFinish, op_context);
}
}  // namespace runtime
}  // namespace mindspore
