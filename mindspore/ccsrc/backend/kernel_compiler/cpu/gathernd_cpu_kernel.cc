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

#include "backend/kernel_compiler/cpu/gathernd_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
#define MAX_INT (((unsigned int)(-1)) >> 1)

constexpr size_t kGatherNdInputsNum = 2;
constexpr size_t kGatherNdOutputsNum = 1;
}  // namespace

template <typename T>
void GatherNdCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  input_shapes_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  indices_shapes_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  output_shapes_ = AnfAlgo::GetOutputInferShape(kernel_node, 0);

  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);

  // ReShape()
  size_t dim_of_indices = 1;
  for (size_t i = 0; i < indices_shapes_.size() - IntToSize(1); ++i) {
    dim_of_indices *= indices_shapes_[i];
  }

  size_t dim_after_indices = 1;
  size_t dim_indices_last = indices_shapes_[indices_shapes_.size() - IntToSize(1)];
  if (dim_indices_last == 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of indices_shapes_[" << indices_shapes_.size()
                      << " - 1] should not be 0.";
  }
  for (size_t i = dim_indices_last; i < input_shapes_.size(); i++) {
    dim_after_indices *= input_shapes_[i];
  }

  (void)dims_.emplace_back(dim_of_indices);
  (void)dims_.emplace_back(dim_after_indices);
  (void)dims_.emplace_back(dim_indices_last);

  batch_strides_.resize(dim_indices_last, 0);
  batch_indices_.resize(dim_indices_last, 0);

  if (dim_indices_last > 0) {
    batch_strides_[dim_indices_last - 1] = input_shapes_[dim_indices_last - 1];
    batch_indices_[dim_indices_last - 1] = dims_[1];
  }

  for (size_t i = dim_indices_last - 1; i > 0; --i) {
    batch_strides_[i - 1] = input_shapes_[i - 1];
    batch_indices_[i - 1] = batch_indices_[i] * SizeToInt(input_shapes_[i]);
  }
}

template <typename T>
bool GatherNdCPUKernel<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                  const std::vector<kernel::AddressPtr> &,
                                  const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kGatherNdInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kGatherNdOutputsNum, kernel_name_);
  const auto *input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  const auto *indices_addr = reinterpret_cast<int *>(inputs[1]->addr);
  auto output_addr = reinterpret_cast<T *>(outputs[0]->addr);

  size_t output_dim0 = dims_[0];
  size_t output_dim1 = dims_[1];
  size_t indices_dim1 = dims_[2];

  size_t num = output_dim0 * output_dim1;
  if (num > MAX_INT) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', exceed MAX_INT: " << MAX_INT << ", dim0: " << output_dim0
                      << ", dim1: " << output_dim1;
  }

  for (size_t write_index = 0; write_index < num; write_index++) {
    size_t i = write_index / output_dim1 % output_dim0;
    size_t j = write_index % output_dim1;

    int read_index = 0;
    for (size_t k = 0; k < indices_dim1; k++) {
      size_t ind = indices_dim1 * i + k;
      int indices_i = indices_addr[ind];
      read_index += indices_i * batch_indices_[k];
    }
    read_index += j;
    output_addr[write_index] = input_addr[read_index];
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
