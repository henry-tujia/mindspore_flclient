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

#include "backend/kernel_compiler/aicpu/dynamic_aicpu_kernel_mod.h"

#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include "runtime/mem.h"
#include "acl/acl_rt.h"
#include "utils/convert_utils.h"
#include "backend/kernel_compiler/aicpu/aicpu_util.h"
#include "utils/ms_context.h"
#include "runtime/device/kernel_runtime.h"
#include "runtime/kernel.h"
#include "utils/utils.h"
#include "backend/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace kernel {
DynamicAicpuOpKernelMod::DynamicAicpuOpKernelMod(const AnfNodePtr &anf_node_ptr) : AicpuOpKernelMod(anf_node_ptr) {
  unknow_type_ = device::ascend::UnknowShapeOpType::DEPEND_IN_SHAPE;
  auto cnode = anf_node_ptr->cast<CNodePtr>();
  if (cnode != nullptr) {
    auto op_name = AnfAlgo::GetCNodeName(cnode);
    if (kComputeDepend.find(op_name) != kComputeDepend.end()) {
      unknow_type_ = device::ascend::UnknowShapeOpType::DEPEND_COMPUTE;
    }
  }
}

DynamicAicpuOpKernelMod::~DynamicAicpuOpKernelMod() {
  // free dev ptr
  if (ext_info_addr_dev_ == nullptr) {
    return;
  }
  auto ret = rtFree(ext_info_addr_dev_);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "rtFree failed";
  }
}

void DynamicAicpuOpKernelMod::InferOp() {
  auto node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  if (!AnfAlgo::IsDynamicShape(node)) {
    MS_LOG(EXCEPTION) << "The node is not dynamic shape.";
  }
  KernelMod::InferShape();
}

void DynamicAicpuOpKernelMod::InitOp() {
  auto node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (!AnfAlgo::IsDynamicShape(cnode)) {
    MS_LOG(EXCEPTION) << "The node is not dynamic shape: " << cnode->fullname_with_scope();
  }

  MS_LOG(INFO) << "UpdateExtInfo of " << cnode->fullname_with_scope() << " start";
  auto input_num = AnfAlgo::GetInputTensorNum(cnode);
  auto output_num = AnfAlgo::GetOutputTensorNum(cnode);
  if (input_num == 0 && output_num == 0) {
    MS_LOG(INFO) << "Node:" << cnode->fullname_with_scope() << " no need to update output shape";
    return;
  }

  // Parse aicpu ext info
  ext_info_handler_ = std::make_shared<device::ascend::AicpuExtInfoHandler>(
    cnode->fullname_with_scope(), static_cast<uint32_t>(input_num), static_cast<uint32_t>(output_num), unknow_type_);
  MS_EXCEPTION_IF_NULL(ext_info_handler_);
  if (!ext_info_handler_->Parse(ext_info_)) {
    MS_LOG(EXCEPTION) << "Parse AiCpu ext_info_handler failed";
  }

  if (ext_info_.empty()) {
    MS_LOG(INFO) << "No need to copy to device, ext_info_ is empty. ";
    return;
  }

  for (size_t i = 0; i < input_num; ++i) {
    if (!ext_info_handler_->UpdateInputShapeAndType(i, NOT_NULL(cnode))) {
      MS_LOG(EXCEPTION) << "Update input shape failed, cnode:" << cnode->fullname_with_scope() << " input:" << i;
    }
  }

  if (unknow_type_ != device::ascend::UnknowShapeOpType::DEPEND_COMPUTE) {
    for (size_t i = 0; i < output_num; ++i) {
      if (!ext_info_handler_->UpdateOutputShapeAndType(i, NOT_NULL(cnode))) {
        MS_LOG(EXCEPTION) << "Update output shape failed, cnode:" << cnode->fullname_with_scope() << " output:" << i;
      }
    }
  }
}

void DynamicAicpuOpKernelMod::AllocateExtInfoDeviceAddr(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (ext_info_addr_dev_ != nullptr) {
    return;
  }
  // Allocate ext info addr in device
  if (ext_info_.size() != 0) {
    auto ret = rtMalloc(&ext_info_addr_dev_, ext_info_.size(), RT_MEMORY_HBM);
    if (ret != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "Call rtMalloc ext_info_addr_dev_ failed. Op name: " << cnode->fullname_with_scope();
    }
  }
  ext_info_size_ = ext_info_.size();
}

bool DynamicAicpuOpKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                     const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (stream_ptr == nullptr) {
    MS_LOG(ERROR) << "stream_ptr should not be nullptr.";
    return false;
  }
  if (stream_ == nullptr) {
    stream_ = stream_ptr;
  }
  auto node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(INFO) << "Start launch of node: " << cnode->fullname_with_scope();

  // is dynamic shape
  if (!AnfAlgo::IsDynamicShape(cnode)) {
    MS_LOG(EXCEPTION) << "The cnode is not dynamic shape:" << cnode->fullname_with_scope();
  }

  // copy extinfo to device
  AllocateExtInfoDeviceAddr(cnode);
  MS_EXCEPTION_IF_NULL(ext_info_handler_);
  auto ret = aclrtMemcpy(ext_info_addr_dev_, ext_info_size_, ext_info_handler_->GetExtInfo(),
                         ext_info_handler_->GetExtInfoLen(), ACL_MEMCPY_HOST_TO_DEVICE);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "UpdateExtInfo aclrtMemcpy failed. Node info: " << cnode->fullname_with_scope();
    return false;
  }

  AicpuOpKernelMod::CreateCpuKernelInfo(inputs, outputs);
  MS_LOG(INFO) << "Aicpu launch, node_so_:" << node_so_ << ", node name:" << node_name_
               << ", args_size:" << args_.length();
  // cppcheck-suppress unreadVariable
  auto lock = AscendKernelMod::LockRuntime();
  ret = rtCpuKernelLaunchWithFlag(reinterpret_cast<const void *>(node_so_.c_str()),
                                  reinterpret_cast<const void *>(node_name_.c_str()), 1,
                                  reinterpret_cast<const void *>(args_.data()), static_cast<uint32_t>(args_.length()),
                                  nullptr, stream_, RT_KERNEL_DEFAULT);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Aicpu op launch failed!";
    return false;
  }

  if (unknow_type_ == device::ascend::UnknowShapeOpType::DEPEND_COMPUTE) {
    ret = aclrtMemcpyAsync(ext_info_handler_->GetExtInfo(), ext_info_handler_->GetExtInfoLen(), ext_info_addr_dev_,
                           ext_info_size_, ACL_MEMCPY_DEVICE_TO_HOST, stream_);
    if (ret != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "aclrtMemcpyAsync output shape failed. Op name: " << cnode->fullname_with_scope();
      return false;
    }
  }

  return true;
}

void DynamicAicpuOpKernelMod::UpdateOp() {
  auto node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(INFO) << "Aicpu " << cnode->fullname_with_scope() << " PostExecute";
  // is dynamic shape
  if (!AnfAlgo::IsDynamicShape(cnode)) {
    MS_LOG(EXCEPTION) << "The cnode is not dynamic shape:" << cnode->fullname_with_scope();
  }

  if (unknow_type_ != device::ascend::UnknowShapeOpType::DEPEND_COMPUTE) {
    MS_LOG(INFO) << "Node " << node->fullname_with_scope() << " update op skip.";
    return;
  }
  // cppcheck-suppress unreadVariable
  auto lock = AscendKernelMod::LockRuntime();
  auto ret = rtStreamSynchronize(stream_);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call runtime rtStreamSynchronize failed. Op name: " << cnode->fullname_with_scope();
  }

  MS_LOG(INFO) << "Update aicpu kernel output shape from ext_info. Op name: " << cnode->fullname_with_scope();
  UpdateOutputShapeFromExtInfo(cnode);
}

bool DynamicAicpuOpKernelMod::UpdateOutputShapeFromExtInfo(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(INFO) << "UpdateOutputShapeFromExtInfo start. Op name " << cnode->fullname_with_scope();
  MS_EXCEPTION_IF_NULL(ext_info_handler_);

  std::vector<TypeId> type_ids;
  std::vector<std::vector<size_t>> shapes;
  auto output_num = AnfAlgo::GetOutputTensorNum(cnode);
  for (size_t i = 0; i < output_num; ++i) {
    MS_LOG(INFO) << "Get output:" << output_num << " Shape";
    std::vector<int64_t> shape;
    TypeId type_id;
    (void)ext_info_handler_->GetOutputShapeAndType(SizeToUint(i), NOT_NULL(&shape), NOT_NULL(&type_id));
    type_ids.emplace_back(type_id);
    std::vector<size_t> size_t_shape;
    std::transform(shape.begin(), shape.end(), std::back_inserter(size_t_shape), LongToSize);
    shapes.emplace_back(size_t_shape);
  }

  AnfAlgo::SetOutputInferTypeAndShape(type_ids, shapes, cnode.get());
  return true;
}
}  // namespace kernel
}  // namespace mindspore
