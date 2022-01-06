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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_CLUSTER_CONTEXT_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_CLUSTER_CONTEXT_H_

#include <map>
#include <set>
#include <mutex>
#include <string>
#include <memory>
#include <atomic>
#include <vector>
#include "distributed/constants.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"

#include "ps/core/cluster_config.h"
#include "ps/core/node.h"
#include "ps/core/worker_node.h"
#include "ps/core/server_node.h"
#include "ps/core/scheduler_node.h"

namespace mindspore {
namespace distributed {
namespace cluster {
// The detailed reason of failing to run 'mindspore.communication.init()' with ClusterContext.
constexpr char kDetailedFailureReason[] =
  "Maybe you are trying to call 'mindspore.communication.init()' without using 'mpirun', which will make MindSpore "
  "load several environment variables and check their validation. Please use 'mpirun' to launch this process to fix "
  "this issue, or refer to this link if you want to run distributed training without using 'mpirun': "
  "https://www.mindspore.cn/docs/programming_guide/zh-CN/master/distributed_training_gpu.html#openmpi.";

// Node role based cluster built by MindSpore communication framework.
class ClusterContext {
 public:
  ~ClusterContext();
  DISABLE_COPY_AND_ASSIGN(ClusterContext)
  static std::shared_ptr<ClusterContext> instance();

  // Initialize the cluster configuration and build network.
  bool Initialize();

  // Finalize the cluster and process exits. If timeout is set to UINT32_MAX, this method will block without timeout.
  bool Finalize(uint32_t timeout = kDefaultFinishTimeout);

  // Return node object of this process.
  const std::shared_ptr<ps::core::Node> &node() const;

  // Return node role in this cluster.
  const std::string &node_role() const;

  // Returns total number of the specified node role. This is used as the group size of this node role.
  uint32_t node_num(const std::string &node_role);

  // Return cluster is initialized.
  bool initialized() const;

 private:
  ClusterContext();

  // This initializing cluster configurations. They can be exported by environment variables, set by python API or
  // configuration file.
  void InitClusterConfig();

  // Build the cluster with other processes. This method will not return until the networking is done.
  bool BuildCluster();

  // Load the cluster configuration like worker number, server number and etc.
  void InitNodeRole();
  void InitSchedulerIp();
  void InitSchedulerPort();

  // Register event callbacks for NODE_TIMEOUT, SCHEDULER_TIMEOUT, etc.
  void RegisterEventCallback();

  // The flag that whether this cluster context instance is already initialized.
  std::atomic_bool inited_;

  // The flag that whether this cluster context instance is already finalized.
  std::atomic_bool finalized_;

  // The mutex about exiting status of this node.
  std::mutex finish_mutex_;

  // Node role to role number map.
  std::map<std::string, uint32_t> node_num_each_role_;

  // Scheduler information.
  std::string scheduler_host_;
  uint16_t scheduler_port_;

  // The node could be Worker, Server or Scheduler, etc.
  std::shared_ptr<ps::core::Node> node_;

  // The role of this process in the cluster.
  std::string node_role_;

  // The configuration of this cluster.
  std::unique_ptr<ps::core::ClusterConfig> cluster_config_;
};
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_CLUSTER_CONTEXT_H_
