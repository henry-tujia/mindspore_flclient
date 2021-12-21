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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_CONTROL_NODE_SCHEDULER_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_CONTROL_NODE_SCHEDULER_H_

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <unordered_map>
#include <map>
#include <set>
#include <algorithm>
#include "runtime/framework/actor/actor_set.h"
#include "runtime/framework/graph_compiler.h"

namespace mindspore {
namespace runtime {
class ControlNodeScheduler {
 public:
  ControlNodeScheduler() = default;
  ~ControlNodeScheduler() = default;
  DISABLE_COPY_AND_ASSIGN(ControlNodeScheduler);

  // Transform the control nodes to control actors.
  ControlActorSetPtr Build(const GraphCompilerInfo &graph_compiler_info, const AID &memory_manager_aid);
  // Link control actors.
  void Link(ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info);

  bool CheckActorValid(const ControlActorSetPtr &control_actor_set);

  // The control flow actor will generate some data in the loop body execution, so need clear on the end of execution.
  void ClearActorData(const ControlActorSet *control_actor_set);

 private:
  // Interface to create control actors.
  std::vector<SwitchActorPtr> BuildSwitchActor(const GraphCompilerInfo &graph_compiler_info);
  std::vector<GatherActorPtr> BuildGatherActor(const GraphCompilerInfo &graph_compiler_info);
  std::vector<EntranceActorPtr> BuildEntranceActor(const GraphCompilerInfo &graph_compiler_info);
  std::vector<ExitActorPtr> BuildExitActor(const GraphCompilerInfo &graph_compiler_info);
  std::vector<StackActorPtr> BuildStackActor(const GraphCompilerInfo &graph_compiler_info);
  void BuildStackActorForControlNode(const GraphCompilerInfo &graph_compiler_info,
                                     std::vector<StackActorPtr> *stack_actors);
  // Interface to link control actors.
  void LinkControlArrowForControlActor(ActorSet *const actor_set, const GraphCompilerInfo &graph_compiler_info);
  void LinkBranchIDArrowForControlActor(ControlActorSet *const control_actor_set);
  // Link all arrows between control actors.
  void LinkArrowForControlActor(ControlActorSet *const control_actor_set, const GraphCompilerInfo &graph_compiler_info);
  void LinkArrowbyFormalParameter(ControlActor *const to_actor, const KernelWithIndex &from_node_with_index,
                                  const KernelWithIndex &to_node_with_index, const ControlNodeParserPtr &parser);
  void LinkArrowByCallNode(const AnfNodePtr &call_node, ControlActor *const to_actor,
                           const KernelWithIndex &from_node_with_index, const KernelWithIndex &to_node_with_index,
                           const ControlNodeParserPtr &parser);
  void LinkArrowByKernel(const AnfNodePtr &kernel, ControlActor *const to_actor,
                         const KernelWithIndex &from_node_with_index, const KernelWithIndex &to_node_with_index,
                         const ControlNodeParserPtr &parser);
  void LinkArrowByParameter(const AnfNodePtr &parameter, ControlActor *const to_actor,
                            const KernelWithIndex &from_node_with_index, const KernelWithIndex &to_node_with_index,
                            const ControlNodeParserPtr &parser);
  void LinkArrowByValueNode(const AnfNodePtr &value_node, ControlActor *const to_actor, size_t from_index,
                            size_t to_index);
  // Link arrow from stack actor to control actor.
  void LinkArrowFromStackActor(StackActor *stack_actor, ControlActor *to_actor);

  // Link data arrow between control actor and actor in frame, including kernel actor, output actor, data source actor.
  void LinkDataArrowForKernelActor(const GraphCompilerInfo &graph_compiler_info);
  void LinkDataArrowByKernelGraph(const KernelGraphPtr &graph, ControlActor *const entrance_actor,
                                  const ControlNodeParserPtr &parser);
  void LinkArrowForRootGraphEntranceActor(const GraphCompilerInfo &graph_compiler_info);
  void LinkControlArrowForLoopCountActor(const ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info);
  void LinkDataArrowForOutputActor(ActorSet *const actor_set, const GraphCompilerInfo &graph_compiler_info);
  void LinkControlArrowForKernelActor(ActorSet *const actor_set, const GraphCompilerInfo &graph_compiler_info);
  void LinkControlArrowByAutoMonad(ControlActor *to_actor, const AnfNodePtr &from_node,
                                   const ControlNodeParserPtr &parser);

  // Interface tool to link arrows between actors.
  void LinkControlArrow(AbstractActor *from_actor, AbstractActor *to_actor);
  void LinkLoopBodyControlArrow(AbstractActor *from_actor, EntranceActor *to_actor);
  // Data arrow with branch id is only exists from gather actor to entrance actor.
  void LinkDataWithBranchIDArrow(GatherActor *const gather_actor, EntranceActor *const entrance_actor,
                                 const FuncGraphPtr &func_graph);
  void LinkPartialArrow(ControlActor *const from_actor, ControlActor *const to_actor, size_t from_index,
                        size_t to_index);
  void LinkDataArrow(AbstractActor *const from_actor, AbstractActor *const to_actor, size_t from_index, size_t to_index,
                     const AnfNodePtr &from_kernel = nullptr);
  void LinkBranchIDArrow(ControlActor *const from_actor, ControlActor *const to_actor);

  // Since the output of exit actor has branches, it needs to be based on a dedicated interface.
  void LinkControlArrowForExitActor(ExitActor *from_actor, AbstractActor *to_actor, int branch_id);
  void LinkDataArrowForExitActor(ExitActor *const exit_actor, AbstractActor *const to_actor, size_t from_index,
                                 size_t to_index, int branch_id);
  void LinkPartialArrowForExitActor(ExitActor *const exit_actor, ControlActor *const to_actor, size_t from_index,
                                    size_t to_index, int branch_id);
  bool IsNoInputActor(const ControlActor *control_actor);

  // The id of memory manager actor.
  AID memory_manager_aid_;
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_CONTROL_NODE_SCHEDULER_H_
