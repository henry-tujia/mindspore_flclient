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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_BCE_WITH_LOGITS_LOSS_IMPL_CUH_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_BCE_WITH_LOGITS_LOSS_IMPL_CUH_

#define MAX_LOGITS_DIMENSION 8
#include "runtime/device/gpu/cuda_common.h"

template <typename T>
void CalBCEWithLogitsLoss(const size_t input_size, const T *predict, const T *target, const size_t *input_shape,
                          const size_t shape_size, const T *weight, const size_t *weight_shape,
                          const bool weight_need_broadcast, const T *pos_weight, const size_t *pos_weight_shape,
                          const bool pos_weight_need_broadcast, T *shape_broadcasted, T *output,
                          cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_BCE_WITH_LOGITS_LOSS_IMPL_CUH_
