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

#include "runtime/framework/actor/control_flow/stack_actor.h"
#include "runtime/framework/actor/memory_manager_actor.h"
#include "runtime/framework/control_node_parser.h"

namespace mindspore {
namespace runtime {
StackActor::StackActor(const std::string &name, const AID &memory_manager_aid,
                       const std::vector<KernelWithIndex> &parameters)
    : ControlActor(name, KernelTransformType::kStackActor, memory_manager_aid, parameters, nullptr) {
  input_device_tensors_.resize(parameters.size());
}

void StackActor::Init() {
  ControlActor::Init();
  // The stack actor has 6 parts of input :
  // 1. Directly input data.
  // 2. Direct input partial.
  // 3. Weight.
  // 4. Local tensor.
  // 5. Call input data.
  // 6. Call input partial.
  input_datas_num_ = formal_parameters_.size() - input_stack_data_num_ - input_stack_partials_num_;
  if (input_stack_data_num_ < device_tensor_store_keys_.size() + local_device_tensors_.size()) {
    MS_LOG(EXCEPTION) << "Invalid input parameter data num:" << input_stack_data_num_
                      << " device store num:" << device_tensor_store_keys_.size() << " local device tensor num"
                      << local_device_tensors_.size() << " for actor:" << GetAID();
  }

  // Fetch the total number of input partial.
  int total_partials_num = 0;
  for (const auto &formal_parameter : formal_parameters_) {
    const auto &abstract = formal_parameter.first->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    const auto &real_abstract = FetchAbstractByIndex(abstract, formal_parameter.second);
    MS_EXCEPTION_IF_NULL(real_abstract);
    if (real_abstract->isa<abstract::AbstractFunction>()) {
      total_partials_num++;
    }
  }

  // Fetch call input data num.
  input_datas_num_ = formal_parameters_.size() - total_partials_num - input_stack_data_num_;
  input_partials_num_ = total_partials_num - input_stack_partials_num_;
  // Fetch call input partial num.
  input_stack_data_num_ -= (device_tensor_store_keys_.size() + local_device_tensors_.size());
  // Check if the input num is valid.
  if (input_stack_data_num_ + input_stack_partials_num_ + input_datas_num_ + input_partials_num_ +
        device_tensor_store_keys_.size() + local_device_tensors_.size() !=
      formal_parameters_.size()) {
    MS_LOG(EXCEPTION) << "Invalid input num, input parameter data num:" << input_stack_data_num_
                      << " input parameter partial num:" << input_stack_partials_num_
                      << " input data num:" << input_datas_num_ << " input partial num:" << input_partials_num_
                      << " device tensor store size:" << device_tensor_store_keys_.size()
                      << " need total size:" << formal_parameters_.size() << " for actor:" << GetAID();
  }
}

void StackActor::RunOpData(OpData<DeviceTensor> *const input_data, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(input_data);
  MS_EXCEPTION_IF_NULL(input_data->data_);
  // The parameters from the inside of the subgraph need to be put into the stack.
  if (IntToSize(input_data->index_) < input_stack_data_num_ + device_tensor_store_keys_.size() +
                                        input_stack_partials_num_ + local_device_tensors_.size()) {
    input_stack_data_[context->sequential_num_][input_data->index_].push(input_data->data_);
  } else {
    // The outputs of call nodes are placed directly in the input data.
    input_op_datas_[context->sequential_num_].emplace_back(input_data);
  }

  auto is_run = CheckRunningCondition(context);
  MS_LOG(DEBUG) << "Actor(" << GetAID().Name() << ") receive the input op data and check running condition:" << is_run;
  if (is_run) {
    Run(context);
  }
}

void StackActor::RunOpControl(AID *const input_control, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  auto &sequential_num = context->sequential_num_;
  if (stack_control_aids_.find(*input_control) != stack_control_aids_.end()) {
    if ((input_stack_controls_.find(sequential_num) == input_stack_controls_.end()) ||
        (input_stack_controls_[sequential_num].find(*input_control) == input_stack_controls_[sequential_num].end())) {
      input_stack_controls_[sequential_num][*input_control] = 1;
    } else {
      input_stack_controls_[sequential_num][*input_control]++;
    }
  } else {
    (void)input_op_controls_[sequential_num].emplace_back(input_control);
  }

  if (CheckRunningCondition(context)) {
    Run(context);
  }
}

void StackActor::RunOpPartial(OpPartialPtr partial, size_t position, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  auto self_partial = std::make_shared<OpPartial>();
  *self_partial = *partial;
  // The parameters from the inside of the subgraph need to be put into the stack.
  if (IntToSize(position) < input_stack_data_num_ + device_tensor_store_keys_.size() + input_stack_partials_num_ +
                              local_device_tensors_.size()) {
    input_stack_partials_[context->sequential_num_][position].push(self_partial);
  } else {
    input_op_partials_[context->sequential_num_].emplace_back(position, self_partial);
  }

  auto is_run = CheckRunningCondition(context);
  MS_LOG(DEBUG) << "Actor(" << GetAID().Name()
                << ") receive the input op partial and check running condition:" << is_run;
  if (is_run) {
    Run(context);
  }
}

bool StackActor::CheckRunningCondition(const OpContext<DeviceTensor> *context) const {
  MS_EXCEPTION_IF_NULL(context);
  if (!ControlActor::CheckRunningCondition(context)) {
    return false;
  }

  auto iter = input_branch_ids_.find(context->sequential_num_);
  bool is_branch_id_available = (iter == input_branch_ids_.end() || iter->second.empty());

  if (input_stack_data_num_ != 0) {
    const auto &data_iter = input_stack_data_.find(context->sequential_num_);
    if (data_iter == input_stack_data_.end()) {
      return false;
    }
    if (data_iter->second.size() != input_stack_data_num_) {
      return false;
    }

    if (is_branch_id_available) {
      MS_LOG(ERROR) << "There is no branch id for actor:" << GetAID().Name();
      return false;
    }
    size_t branch_id_size = iter->second.size();
    if (std::any_of(data_iter->second.begin(), data_iter->second.end(),
                    [branch_id_size](const auto &one_stack) { return one_stack.second.size() != branch_id_size; })) {
      return false;
    }
  }

  if (input_stack_partials_num_ != 0) {
    const auto &partial_iter = input_stack_partials_.find(context->sequential_num_);
    if (partial_iter == input_stack_partials_.end()) {
      return false;
    }
    if (partial_iter->second.size() != input_stack_partials_num_) {
      return false;
    }

    if (is_branch_id_available) {
      MS_LOG(ERROR) << "There is no branch id for actor:" << GetAID().Name();
      return false;
    }
    size_t branch_id_size = iter->second.size();
    if (std::any_of(partial_iter->second.begin(), partial_iter->second.end(),
                    [branch_id_size](const auto &one_stack) { return one_stack.second.size() != branch_id_size; })) {
      return false;
    }
  }

  if (input_stack_controls_num_ != 0) {
    const auto &control_iter = input_stack_controls_.find(context->sequential_num_);
    if (control_iter == input_stack_controls_.end()) {
      return false;
    }
    if (control_iter->second.size() != input_stack_controls_num_) {
      return false;
    }

    if (is_branch_id_available) {
      MS_LOG(ERROR) << "There is no branch id for actor:" << GetAID().Name();
      return false;
    }
    size_t branch_id_size = iter->second.size();
    if (std::any_of(control_iter->second.begin(), control_iter->second.end(),
                    [branch_id_size](const auto &one_stack) { return one_stack.second != branch_id_size; })) {
      return false;
    }
  }
  return true;
}

void StackActor::FetchInput(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  if (input_stack_data_num_ != 0) {
    const auto &data_iter = input_stack_data_.find(context->sequential_num_);
    if (data_iter == input_stack_data_.end()) {
      std::string error_info = "Invalid input for actor:" + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    for (const auto &one_stack : data_iter->second) {
      if (one_stack.first >= input_stack_data_num_ + device_tensor_store_keys_.size() + local_device_tensors_.size() +
                               input_stack_partials_num_) {
        std::string error_info = "Invalid input index:" + std::to_string(one_stack.first) +
                                 " need:" + std::to_string(input_stack_data_num_) + " for actor:" + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
      }
      input_device_tensors_[one_stack.first] = one_stack.second.top();
    }
  }

  if (input_stack_partials_num_ != 0) {
    const auto &partial_iter = input_stack_partials_.find(context->sequential_num_);
    if (partial_iter == input_stack_partials_.end()) {
      std::string error_info = "Invalid input for actor:" + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    for (const auto &one_stack : partial_iter->second) {
      if (one_stack.first >= input_stack_data_num_ + device_tensor_store_keys_.size() + local_device_tensors_.size() +
                               input_stack_partials_num_) {
        std::string error_info = "Invalid input index:" + std::to_string(one_stack.first) +
                                 " need:" + std::to_string(input_stack_partials_num_) + " for actor:" + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
      }
      input_partials_[one_stack.first] = one_stack.second.top();
    }
  }
  ControlActor::FetchInput(context);
}

void StackActor::EraseInput(const OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  ControlActor::EraseInput(context);

  if (input_stack_data_num_ != 0) {
    const auto &data_iter = input_stack_data_.find(context->sequential_num_);
    if (data_iter == input_stack_data_.end()) {
      MS_LOG(ERROR) << "Invalid input for actor:" << GetAID();
      return;
    }

    for (auto &one_stack : data_iter->second) {
      if (one_stack.second.empty()) {
        MS_LOG(ERROR) << "Input index:" << one_stack.first << " is null in actor:" << GetAID();
        return;
      }
      one_stack.second.pop();
    }
  }

  if (input_stack_partials_num_ != 0) {
    const auto &partial_iter = input_stack_partials_.find(context->sequential_num_);
    if (partial_iter == input_stack_partials_.end()) {
      MS_LOG(ERROR) << "Invalid input for actor:" << GetAID();
      return;
    }

    for (auto &one_stack : partial_iter->second) {
      if (one_stack.second.empty()) {
        MS_LOG(ERROR) << "Input index:" << one_stack.first << " is null in actor:" << GetAID();
        return;
      }
      one_stack.second.pop();
    }
  }

  if (input_stack_controls_num_ != 0) {
    const auto &control_iter = input_stack_controls_.find(context->sequential_num_);
    if (control_iter == input_stack_controls_.end()) {
      MS_LOG(ERROR) << "Invalid input for actor:" << GetAID();
      return;
    }

    for (auto &one_stack : control_iter->second) {
      if (one_stack.second == 0) {
        MS_LOG(ERROR) << "Input stack control aid:" << one_stack.first << " is null in actor:" << GetAID();
        return;
      }
      one_stack.second--;
    }
  }
}

void StackActor::SendMemoryFreeReq(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  const auto &sequential_num = context->sequential_num_;

  // Collect the input device tensors.
  std::vector<DeviceTensor *> memory_free_list;
  if (input_op_datas_.count(sequential_num) > 0) {
    for (auto &input_data : input_op_datas_[sequential_num]) {
      MS_EXCEPTION_IF_NULL(input_data);
      MS_EXCEPTION_IF_NULL(input_data->data_);
      memory_free_list.emplace_back(input_data->data_);
    }
  }

  if (input_op_partials_.count(sequential_num) > 0) {
    for (auto &input_partial_pair : input_op_partials_[sequential_num]) {
      auto partial_device_tensors = GetAllDeviceTensors(input_partial_pair.second);
      (void)std::copy(partial_device_tensors.begin(), partial_device_tensors.end(),
                      std::back_inserter(memory_free_list));
    }
  }

  if ((input_stack_data_num_ != 0) && (input_stack_data_.count(sequential_num) > 0)) {
    for (auto &stack_data_pair : input_stack_data_[sequential_num]) {
      if (!stack_data_pair.second.empty()) {
        memory_free_list.emplace_back(stack_data_pair.second.top());
      }
    }
  }

  if ((input_stack_partials_num_ != 0) && (input_stack_partials_.count(sequential_num) > 0)) {
    for (auto &stack_partial_pair : input_stack_partials_[sequential_num]) {
      if (!stack_partial_pair.second.empty()) {
        auto partial_device_tensors = GetAllDeviceTensors(stack_partial_pair.second.top());
        (void)std::copy(partial_device_tensors.begin(), partial_device_tensors.end(),
                        std::back_inserter(memory_free_list));
      }
    }
  }

  if (memory_free_list.size() > 0) {
    memory_free_lists_.emplace_back(memory_free_list);
    ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &(memory_free_lists_.back()),
                          device_contexts_[0], context);
  }
}
}  // namespace runtime
}  // namespace mindspore
