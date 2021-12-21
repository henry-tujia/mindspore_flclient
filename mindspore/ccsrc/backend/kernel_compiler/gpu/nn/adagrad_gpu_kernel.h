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

#ifndef MINDSPORE_ADAGRAD_GPU_KERNEL_H
#define MINDSPORE_ADAGRAD_GPU_KERNEL_H

#include <vector>
#include <string>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/adagrad_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T, typename S, typename G>
class AdagradGpuKernel : public GpuKernel {
 public:
  AdagradGpuKernel()
      : variable_size_(0),
        accumulation_size_(0),
        learning_rate_size_(0),
        gradient_size_(0),
        update_slots(true),
        is_null_input_(false),
        kernel_name_("ApplyAdagrad") {}

  ~AdagradGpuKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *variable = GetDeviceAddress<T>(inputs, 0);
    T *accumulation = GetDeviceAddress<T>(inputs, 1);
    S *learning_rate = GetDeviceAddress<S>(inputs, 2);
    G *gradient = GetDeviceAddress<G>(inputs, 3);
    T *variable_out = GetDeviceAddress<T>(outputs, 0);
    T *accumulation_out = GetDeviceAddress<T>(outputs, 1);
    ApplyAdagrad(inputs[0]->size / sizeof(T), update_slots, learning_rate, gradient, variable, accumulation,
                 reinterpret_cast<cudaStream_t>(stream_ptr));

    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(variable_out, variable, variable_size_, cudaMemcpyDeviceToDevice,
                                               reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync output failed");
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(accumulation_out, accumulation, accumulation_size_,
                                               cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync output failed");

    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    update_slots = AnfAlgo::GetNodeAttr<bool>(kernel_node, "update_slots");
    if (input_num != 4) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 4, but got " << input_num;
    }
    variable_size_ = sizeof(T);
    accumulation_size_ = sizeof(T);
    learning_rate_size_ = sizeof(S);
    gradient_size_ = sizeof(G);

    auto variable_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto accumulation_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto gradient_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 3);
    is_null_input_ = CHECK_SHAPE_NULL(variable_shape, kernel_name_, "var") ||
                     CHECK_SHAPE_NULL(accumulation_shape, kernel_name_, "accum") ||
                     CHECK_SHAPE_NULL(gradient_shape, kernel_name_, "grad");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    for (size_t i = 0; i < variable_shape.size(); i++) {
      variable_size_ *= variable_shape[i];
    }

    for (size_t i = 0; i < accumulation_shape.size(); i++) {
      accumulation_size_ *= accumulation_shape[i];
    }

    for (size_t i = 0; i < gradient_shape.size(); i++) {
      gradient_size_ *= gradient_shape[i];
    }
    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(variable_size_);
    input_size_list_.push_back(accumulation_size_);
    input_size_list_.push_back(learning_rate_size_);
    input_size_list_.push_back(gradient_size_);
    output_size_list_.push_back(variable_size_);
    output_size_list_.push_back(accumulation_size_);
  }

 private:
  size_t variable_size_;
  size_t accumulation_size_;
  size_t learning_rate_size_;
  size_t gradient_size_;
  bool update_slots;
  bool is_null_input_;
  std::string kernel_name_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_ADAGRAD_GPU_KERNEL_H
