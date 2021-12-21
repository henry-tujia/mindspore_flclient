/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/arm/fp32/gather_fp32.h"
#include <limits>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Gather;

namespace mindspore::kernel {
namespace {
constexpr int kSecondInput = 2;
}
int GatherCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), kInputSize2);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_.at(kSecondInput)->data());
  axis_ = *(reinterpret_cast<int *>(in_tensors_.at(kSecondInput)->data()));
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int GatherCPUKernel::ReSize() { return RET_OK; }

int GatherCPUKernel::DoGather(int task_id) const {
  auto input_tensor = in_tensors_.at(0);
  auto indices_tensor = in_tensors_.at(1);
  auto out_tensor = out_tensors_.at(0);

  auto in_shape = input_tensor->shape();
  int in_rank = in_shape.size();
  int indices_element_size = indices_tensor->ElementsNum();
  MS_CHECK_LT(axis_, in_rank, RET_ERROR);
  const int limit = in_shape.at(axis_);

  int outer_size = 1, inner_size = 1;
  for (int i = 0; i < axis_; ++i) {
    outer_size *= in_shape.at(i);
  }
  for (int i = axis_ + 1; i < in_rank; ++i) {
    inner_size *= in_shape.at(i);
  }
  int stride = UP_DIV(outer_size, op_parameter_->thread_num_);
  int count = MSMIN(stride, outer_size - stride * task_id);
  if (count <= 0) {
    return RET_OK;
  }
  auto thread_stride = stride * task_id;

  int8_t *int8_in = reinterpret_cast<int8_t *>(input_tensor->data());
  CHECK_NULL_RETURN(int8_in);
  int8_t *int8_out = reinterpret_cast<int8_t *>(out_tensor->data());
  CHECK_NULL_RETURN(int8_out);

  int data_size = static_cast<int>(lite::DataTypeSize(input_tensor->data_type()));
  int8_in += thread_stride * limit * inner_size * data_size;
  int8_out += thread_stride * indices_element_size * inner_size * data_size;

  int error_code = Gather(int8_in, count, inner_size, limit, indices_data_, indices_element_size, int8_out, data_size);

  return error_code;
}

int GatherRun(const void *cdata, int task_id, float, float) {
  auto gather_kernel = reinterpret_cast<const GatherCPUKernel *>(cdata);
  auto error_code = gather_kernel->DoGather(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "GatherRun error task_id[" << task_id << "] error_code[" << error_code << "]";
  }
  return error_code;
}

int GatherCPUKernel::Run() {
  auto indices_tensor = in_tensors_.at(1);
  int indices_num = indices_tensor->ElementsNum();
  bool isIndicesInt32 = indices_tensor->data_type() == kNumberTypeInt32;
  int ret = AssignIndicesData(isIndicesInt32, indices_num, indices_tensor);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "AssignIndicesData failed, error_code[" << ret << "]";
    return ret;
  }

  ret = ParallelLaunch(this->ms_context_, GatherRun, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Gather function error error_code[" << ret << "]";
  }
  if (!isIndicesInt32) {
    ms_context_->allocator->Free(indices_data_);
    indices_data_ = nullptr;
  }
  return ret;
}

int GatherCPUKernel::AssignIndicesData(bool isIndicesInt32, int indices_num, lite::Tensor *indices_tensor) {
  if (!isIndicesInt32) {
    if (indices_num >= std::numeric_limits<int>::max() / static_cast<int>(sizeof(int))) {
      MS_LOG(ERROR) << "Input indices_num is invalid, indices_num: " << indices_num;
      return RET_ERROR;
    }
    indices_data_ = reinterpret_cast<int32_t *>(ms_context_->allocator->Malloc(sizeof(int32_t) * indices_num));
    if (indices_data_ == nullptr) {
      MS_LOG(ERROR) << "Memory allocation failed";
      return RET_ERROR;
    }
    switch (indices_tensor->data_type()) {
      case kNumberTypeInt64:
        for (int i = 0; i < indices_num; i++) {
          indices_data_[i] = static_cast<int>(reinterpret_cast<int64_t *>(indices_tensor->MutableData())[i]);
        }
        break;
      case kNumberTypeFloat:
      case kNumberTypeFloat32:
        for (int i = 0; i < indices_num; i++) {
          indices_data_[i] = static_cast<int>(reinterpret_cast<float *>(indices_tensor->MutableData())[i]);
        }
        break;
      default:
        MS_LOG(ERROR) << "Does not support data type: " << indices_tensor->data_type();
        return RET_ERROR;
    }
  } else {
    indices_data_ = reinterpret_cast<int32_t *>(indices_tensor->MutableData());
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Gather, LiteKernelCreator<GatherCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Gather, LiteKernelCreator<GatherCPUKernel>)
}  // namespace mindspore::kernel
