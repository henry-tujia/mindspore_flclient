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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NCCL_RECV_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NCCL_RECV_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <functional>
#include "backend/kernel_compiler/gpu/nccl/nccl_gpu_kernel.h"

namespace mindspore {
namespace kernel {
template <typename T>
class NcclRecvGpuKernel : public NcclGpuKernel {
 public:
  NcclRecvGpuKernel() : src_rank_(-1) {}
  ~NcclRecvGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &, const std::vector<AddressPtr> &outputs,
              void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    (void)Recv(output_addr, output_size_list_[0] / sizeof(T), nccl_data_type_, src_rank_,
               reinterpret_cast<cudaStream_t>(stream_ptr), group_name_);
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    auto kernel_name = AnfAlgo::GetCNodeName(kernel_node);
    MS_EXCEPTION_IF_NULL(kernel_node);
    kernel_node_ = kernel_node;
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of outputs should be 1, but got " << output_num;
    }
    src_rank_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "src_rank"));
    group_name_ = GetAttr<std::string>(kernel_node, kAttrGroup);
    nccl_data_type_ = nccl_dtype(AnfAlgo::GetOutputDeviceDataType(kernel_node, 0));

    auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(output_shape, kernel_name, "output");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    size_t output_size =
      std::accumulate(output_shape.begin(), output_shape.end(), sizeof(T), std::multiplies<size_t>());
    output_size_list_.push_back(output_size);
    MS_LOG(INFO) << "NcclRecv source rank is " << src_rank_ << ", group name is " << group_name_;

    use_mpi_ = common::CheckUseMPI();
    if (use_mpi_) {
      collective_handle_ = device::gpu::CollectiveInitializer::instance().collective_handle();
      MS_EXCEPTION_IF_NULL(collective_handle_);
    } else {
      if (!LoadNvidiaCommLib()) {
        return false;
      }
    }
    return true;
  }

 protected:
  void InitSizeLists() override {}

 private:
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  int src_rank_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NCCL_RECV_GPU_KERNEL_H_
