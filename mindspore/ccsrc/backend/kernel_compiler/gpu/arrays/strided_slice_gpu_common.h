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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_STRIDED_SLICE_GPU_COMMON_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_STRIDED_SLICE_GPU_COMMON_H_

#include <vector>
#include <bitset>
#include <algorithm>
#include "backend/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace kernel {
constexpr size_t MAX_DIMS = 8;
class StridedSliceGpuCommon {
 public:
  StridedSliceGpuCommon() : null_output_(false) {}
  ~StridedSliceGpuCommon() = default;

  void CollectInfo(const CNodePtr &kernel_node) {
    FillEmptyDims(kernel_node);
    ParseMasks(kernel_node);
    FillOutputDim();
    null_output_ = IsNullOutput();
  }

 protected:
  void FillEmptyDims(const CNodePtr &kernel_node) {
    begin_ = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "begin");
    end_ = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "end");
    strides_ = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "strides");

    for (size_t i = 0; i < MAX_DIMS; i++) {
      if (i >= input_shape_.size()) {
        input_shape_.push_back(1);
      }

      if (i < begin_.size()) {
        int64_t dim = input_shape_[i];
        begin_[i] = std::min(begin_[i] < 0 ? std::max(begin_[i] + dim, static_cast<int64_t>(0)) : begin_[i], dim - 1);
      } else {
        begin_.push_back(0);
      }

      if (i < end_.size()) {
        int64_t dim = input_shape_[i];
        end_[i] = std::max(end_[i] < 0 ? end_[i] + dim : std::min(end_[i], dim), static_cast<int64_t>(-1));
      } else {
        end_.push_back(i < input_shape_.size() ? input_shape_[i] : 1);
      }

      if (i >= strides_.size()) {
        strides_.push_back(1);
      }
    }
  }

  void ParseMasks(const CNodePtr &kernel_node) {
    auto begin_mask_int = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "begin_mask");
    auto begin_mask = Dec2Bin(begin_mask_int);
    for (size_t i = 0; i < begin_mask.size(); i++) {
      if (begin_mask[i] && i < MAX_DIMS) {
        begin_[i] = 0;
      }
    }

    auto end_mask_int = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "end_mask");
    auto end_mask = Dec2Bin(end_mask_int);
    for (size_t j = 0; j < end_mask.size(); j++) {
      if (end_mask[j] && j < MAX_DIMS) {
        end_[j] = input_shape_[j];
      }
    }

    auto ellipsis_mask_int = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "ellipsis_mask");
    auto ellipsis_mask = Dec2Bin(ellipsis_mask_int);
    for (size_t k = 0; k < ellipsis_mask.size(); k++) {
      if (ellipsis_mask[k] && k < MAX_DIMS) {
        begin_[k] = 0;
        end_[k] = input_shape_[k];
        strides_[k] = 1;
      }
    }

    auto new_axis_mask_int = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "new_axis_mask");
    auto new_axis_mask = Dec2Bin(new_axis_mask_int);
    for (size_t l = 0; l < new_axis_mask.size(); l++) {
      if (new_axis_mask[l] && l < MAX_DIMS) {
        begin_[l] = 0;
        end_[l] = input_shape_[l];
        strides_[l] = 1;
      }
    }

    auto shrink_axis_mask_int = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "shrink_axis_mask");
    auto shrink_axis_mask = Dec2Bin(shrink_axis_mask_int);
    for (size_t m = 0; m < shrink_axis_mask.size(); m++) {
      if (shrink_axis_mask[m] && m < MAX_DIMS) {
        end_[m] = end_[m] > begin_[m] ? begin_[m] + 1 : begin_[m] - 1;
        strides_[m] = end_[m] > begin_[m] ? 1 : -1;
      }
    }
  }

  std::vector<bool> Dec2Bin(const int64_t &mask) {
    auto mask_str = std::bitset<MAX_DIMS>(mask).to_string();
    int64_t dim_idx = 0;
    std::vector<bool> result(MAX_DIMS, false);
    for (int64_t i = mask_str.size() - 1; i >= 0; i--) {
      if (mask_str[i] == '1') {
        result[dim_idx] = true;
      }
      dim_idx++;
    }
    return result;
  }

  void FillOutputDim() {
    for (size_t i = 0; i < MAX_DIMS; i++) {
      if (begin_[i] <= end_[i] && strides_[i] > 0) {
        output_shape_.push_back((end_[i] - 1 - begin_[i]) / strides_[i] + 1);
      } else if (begin_[i] > end_[i] && strides_[i] < 0) {
        output_shape_.push_back((end_[i] - begin_[i] + 1) / strides_[i] + 1);
      } else {
        output_shape_.push_back(0);
      }
    }
  }

  bool IsNullOutput() {
    for (size_t i = 0; i < MAX_DIMS; i++) {
      if (begin_[i] >= end_[i] && strides_[i] > 0) {
        return true;
      }
      if (begin_[i] < end_[i] && strides_[i] < 0) {
        return true;
      }
    }
    return false;
  }

  std::vector<int64_t> begin_;
  std::vector<int64_t> end_;
  std::vector<int64_t> strides_;
  std::vector<size_t> input_shape_;
  std::vector<size_t> output_shape_;
  bool null_output_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_STRIDED_SLICE_GPU_COMMON_H_
