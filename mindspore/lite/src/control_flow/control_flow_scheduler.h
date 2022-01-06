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

#ifndef MINDSPORE_LITE_SRC_CONTROL_FLOW_SCHEDULER_H_
#define MINDSPORE_LITE_SRC_CONTROL_FLOW_SCHEDULER_H_
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <algorithm>
#include <queue>
#include <set>
#include <unordered_map>
#include "src/common/utils.h"
#include "src/common/log_util.h"
#include "nnacl/op_base.h"
#include "src/inner_context.h"
#include "src/tensor.h"
#include "src/sub_graph_kernel.h"
#include "include/model.h"

namespace mindspore::lite {
class ControlFlowScheduler {
 public:
  ControlFlowScheduler(InnerContext *ctx, const mindspore::Context *ms_ctx, std::vector<Tensor *> *src_tensors)
      : context_(ctx), ms_context_(ms_ctx), src_tensors_(src_tensors) {}
  ~ControlFlowScheduler() = default;
  int SplitNonTailCallSubGraphs(std::vector<kernel::LiteKernel *> *dst_kernels);
  void RecordSubgraphCaller(const size_t &subgraph_index, kernel::LiteKernel *partial_node);
  // we insert entrance subgraph kernel and exit subgraph kernel define the boundary of the subgraph.
  int BuildBoundaryForMultipleCalledGraph(std::vector<kernel::LiteKernel *> *dst_kernels);
  std::vector<kernel::LiteKernel *> GetNonTailCalls() const { return non_tail_calls_; }

 private:
  int SplitSingleNonTailCallSubGraph(kernel::SubGraphKernel *subgraph_kernel,
                                     std::vector<kernel::LiteKernel *> *subgraph_kernels);
  std::set<kernel::LiteKernel *> GetNonTailCallSubGraphs(std::vector<kernel::LiteKernel *> *dst_kernels);
  void RemoveUselessKernels(std::vector<kernel::LiteKernel *> *dst_kernels,
                            std::set<kernel::LiteKernel *> *useless_kernels);
  void AppendToProcessQ(std::vector<kernel::LiteKernel *> *new_subgraphs,
                        std::set<kernel::LiteKernel *> *all_non_tail_subgraphs);
  // link partial output to call output.
  int RecordNonTailCallLinkInfo();
  kernel::SubGraphKernel *CreateEntranceSubGraph(kernel::SubGraphKernel *subgraph, lite::Tensor *link_tensor);
  kernel::SubGraphKernel *CreateExitSubGraph(kernel::SubGraphKernel *subgraph, lite::Tensor *link_tensor);

 private:
  InnerContext *context_ = nullptr;
  const mindspore::Context *ms_context_ = nullptr;
  int schema_version_ = SCHEMA_VERSION::SCHEMA_CUR;
  std::vector<Tensor *> *src_tensors_ = nullptr;
  std::queue<kernel::LiteKernel *> to_process_q_{};
  std::vector<kernel::LiteKernel *> non_tail_calls_{};
  // key is subgraph index, value is the corresponding partial nodes.
  std::unordered_map<size_t, std::set<kernel::LiteKernel *>> more_than_once_called_partial_nodes_{};
};

using ControlFlowSchedulerPtr = std::shared_ptr<ControlFlowScheduler>;
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_CONTROL_FLOW_SCHEDULER_H_
