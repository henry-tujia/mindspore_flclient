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
#include "src/runtime/kernel/arm/fp32_grad/lstm_grad_fp32.h"
#include <string>
#include <memory>
#include "utils/ms_utils.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "nnacl/fp32/lstm_fp32.h"

namespace mindspore {
namespace kernel {
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_LSTMGrad;

int LSTMGradCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), DIMENSION_11D);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int LSTMGradCPUKernel::ReSize() { return InitParam(); }

int LSTMGradCPUKernel::Run() {
  auto ret = MallocRunBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "LstmGradCPUKernel MallocRunBuffer error.";
    FreeRunBuffer();
    return RET_ERROR;
  }

  PackWeights();
  auto output = out_tensors_.at(0);
  auto output_ptr = reinterpret_cast<float *>(output->data());
  CHECK_NULL_RETURN(output_ptr);

  LstmBackpropUnidirectional(output_ptr, false);
  FreeRunBuffer();
  return RET_OK;
}

int LSTMGradCPUKernel::LstmBackpropUnidirectional(float *output, bool is_backward) {
  // auto dy_tensor = in_tensors_.at(dy_index);  / * [SeqLen, Batch, insize ] * / AA Temporary
  auto dC_tensor = in_tensors_.at(dC_index); /* [1, Batch, hidden_size ] */
  auto dH_tensor = in_tensors_.at(dH_index); /* [1, Batch, hidden_size ] */
  auto dy_tensor = in_tensors_.at(dy_index);
  auto cell_tensor = in_tensors_.at(cell_state_index); /* [1, Batch, hidden_size ] */
  auto weights_tensor = in_tensors_.at(weights_index); /* [all weights + biases, 1, 1] */
  auto intermediate_tensor = in_tensors_.at(intermediate_data_index);
  auto input_tensor = in_tensors_.at(input_index);
  MS_ASSERT(dy_tensor != nullptr);
  MS_ASSERT(dC_tensor != nullptr);
  MS_ASSERT(dH_tensor != nullptr);
  MS_ASSERT(cell_tensor != nullptr);
  MS_ASSERT(weights_tensor != nullptr);
  MS_ASSERT(intermediate_tensor != nullptr);
  MS_ASSERT(input_tensor != nullptr);
  auto intermediate_data = reinterpret_cast<float *>(intermediate_tensor->data());
  auto dC = reinterpret_cast<float *>(dC_tensor->data());
  auto dH = reinterpret_cast<float *>(dH_tensor->data());
  auto dY = reinterpret_cast<float *>(dy_tensor->data());
  auto weights = reinterpret_cast<float *>(weights_tensor->data());
  auto last_cell = reinterpret_cast<float *>(cell_tensor->data());
  auto input = reinterpret_cast<float *>(input_tensor->data());

  auto state_size = lstm_param_->batch_ * lstm_param_->hidden_size_;
  auto seq_stride = lstm_param_->seq_len_ * state_size;
  float *hidden_state = intermediate_data;
  float *cell_state = intermediate_data + seq_stride * 1;
  float *input_gate = intermediate_data + seq_stride * 2;
  float *output_gate = intermediate_data + seq_stride * 3;
  float *forget_gate = intermediate_data + seq_stride * 4;
  float *cell_gate = intermediate_data + seq_stride * 5;
  float *cell_state_minus1 = intermediate_data + seq_stride * 6;
  for (int t = lstm_param_->seq_len_ - 1; t >= 0; t--) {
    int real_t = is_backward ? lstm_param_->seq_len_ - t - 1 : t;
    auto stride = real_t * state_size;
    float *hidden_state_t = hidden_state + stride;
    float *cell_state_t = cell_state + stride;
    float *input_gate_t = input_gate + stride;
    float *forget_gate_t = forget_gate + stride;
    float *cell_gate_t = cell_gate + stride;
    float *output_gate_t = output_gate + stride;
    float *cell_state_minus1_t = cell_state_minus1 + stride;
    float *output_ptr = output + real_t * lstm_param_->output_step_;
    float *input_ptr = input + real_t * lstm_param_->batch_ * lstm_param_->input_size_;

    std::cout << "seq_len: " << lstm_param_->seq_len_ << " t: " << t << " real_t: " << real_t << " cell_state "
              << cell_state[0] << " last_cell " << last_cell[0] << std::endl;
    LstmGradStepUnit(input_ptr, output_ptr, input_gate_t, forget_gate_t, cell_gate_t, output_gate_t, hidden_state_t,
                     cell_state_t, dC, dH, dY, cell_state_minus1_t, weights, workspace_, lstm_param_);
  }
  return RET_OK;
}

int LSTMGradCPUKernel::DoGrad(int thread_id) { return RET_OK; }

float *LSTMGradCPUKernel::InputWeightPtr() { return reinterpret_cast<float *>(in_tensors_.at(weights_index)->data()); }

float *LSTMGradCPUKernel::StateWeightPtr() {
  int weight_i_size = num_of_gates * lstm_param_->hidden_size_ * lstm_param_->input_size_;
  return (reinterpret_cast<float *>(in_tensors_.at(weights_index)->data()) + weight_i_size);
}

float *LSTMGradCPUKernel::InputBiasPtr() {
  int bias_offset = num_of_gates * lstm_param_->hidden_size_ * (lstm_param_->input_size_ + lstm_param_->hidden_size_);
  return (reinterpret_cast<float *>(in_tensors_.at(weights_index)->data()) + bias_offset);
}

float *LSTMGradCPUKernel::StateBiasPtr() {
  int bias_offset = num_of_gates * lstm_param_->hidden_size_ * (lstm_param_->input_size_ + lstm_param_->hidden_size_);
  bias_offset += (num_of_gates * (num_of_gates + lstm_param_->hidden_size_));
  return (reinterpret_cast<float *>(in_tensors_.at(weights_index)->data()) + bias_offset);
}

void LSTMGradCPUKernel::FreeTmpBuffer() {
  if (weight_i_ptr_ != nullptr) {
    free(weight_i_ptr_);
    weight_i_ptr_ = nullptr;
  }
  if (input_bias_ != nullptr) {
    free(input_bias_);
    input_bias_ = nullptr;
  }
#ifdef ENABLE_AVX
  if (weight_h_ptr_ != nullptr) {
    free(weight_h_ptr_);
    weight_h_ptr_ = nullptr;
  }
#else
  if (!state_is_vec_) {
    if (weight_h_ptr_ != nullptr) {
      free(weight_h_ptr_);
      weight_h_ptr_ = nullptr;
    }
  }
#endif
  if (state_bias_ != nullptr) {
    free(state_bias_);
    state_bias_ = nullptr;
  }
}

int LSTMGradCPUKernel::InitParam() {
  auto input = in_tensors_.front();
  MS_ASSERT(input != nullptr);
  std::vector<int> in_shape = input->shape();
  lstm_param_->seq_len_ = in_shape.at(FIRST_INPUT);
  lstm_param_->batch_ = in_shape.at(SECOND_INPUT);
  lstm_param_->input_size_ = in_shape.at(THIRD_INPUT);

  auto dy = in_tensors_.at(dy_index);
  MS_ASSERT(dy != nullptr);
  std::vector<int> dy_shape = dy->shape();
  lstm_param_->hidden_size_ = dy_shape.at(THIRD_INPUT);

  int dir_multiplier = lstm_param_->bidirectional_ ? 2 : 1;
  lstm_param_->output_step_ = dir_multiplier * lstm_param_->batch_ * lstm_param_->hidden_size_;
  weight_batch_ = dir_multiplier * num_of_gates;
  state_is_vec_ = lstm_param_->batch_ == 1;

#ifdef ENABLE_AVX
  row_tile_ = C6NUM;
  col_tile_ = C16NUM;
#elif defined(ENABLE_ARM32)
  row_tile_ = C12NUM;
  col_tile_ = C4NUM;
#elif defined(ENABLE_SSE)
  row_tile_ = C4NUM;
  col_tile_ = C8NUM;
#else
  row_tile_ = C12NUM;
  col_tile_ = C8NUM;
#endif
  lstm_param_->input_row_align_ = UP_ROUND(lstm_param_->seq_len_ * lstm_param_->batch_, row_tile_);
  lstm_param_->input_col_align_ = UP_ROUND(lstm_param_->hidden_size_, col_tile_);
  input_size_align_ = UP_ROUND(lstm_param_->input_size_, row_tile_);
  input_thread_count_ = MSMIN(op_parameter_->thread_num_, UP_DIV(lstm_param_->input_col_align_, col_tile_));
  input_thread_stride_ = UP_DIV(UP_DIV(lstm_param_->input_col_align_, col_tile_), input_thread_count_);

  state_row_tile_ = row_tile_;
  state_col_tile_ = col_tile_;

  lstm_param_->state_row_align_ = state_is_vec_ ? 1 : UP_ROUND(lstm_param_->batch_, state_row_tile_);
  lstm_param_->state_col_align_ =
    state_is_vec_ ? lstm_param_->hidden_size_ : UP_ROUND(lstm_param_->hidden_size_, state_col_tile_);

  return AllocateWeights();
}

int LSTMGradCPUKernel::AllocateWeights() {
  if (weight_i_ptr_ == nullptr) {
    weight_i_ptr_ = reinterpret_cast<float *>(
      malloc(weight_batch_ * lstm_param_->input_size_ * lstm_param_->input_col_align_ * sizeof(float)));
    if (weight_i_ptr_ == nullptr) {
      MS_LOG(ERROR) << "LstmCPUKernel malloc weight_i_ptr_ error.";
      return RET_ERROR;
    }
  }
  if (input_bias_ == nullptr) {
    input_bias_ = reinterpret_cast<float *>(malloc(weight_batch_ * lstm_param_->input_col_align_ * sizeof(float)));
    if (input_bias_ == nullptr) {
      MS_LOG(ERROR) << "LstmCPUKernel malloc input_bias_ error.";
      return RET_ERROR;
    }
  }
  if (weight_h_ptr_ == nullptr) {
    weight_h_ptr_ = reinterpret_cast<float *>(
      malloc(weight_batch_ * lstm_param_->state_col_align_ * lstm_param_->hidden_size_ * sizeof(float)));
    if (weight_h_ptr_ == nullptr) {
      MS_LOG(ERROR) << "LstmCPUKernel malloc weight_h_ptr_ error.";
      return RET_ERROR;
    }
  }
  if (state_bias_ == nullptr) {
    state_bias_ = reinterpret_cast<float *>(malloc(weight_batch_ * lstm_param_->state_col_align_ * sizeof(float)));
    if (state_bias_ == nullptr) {
      MS_LOG(ERROR) << "LstmCPUKernel malloc state_bias_ error.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int LSTMGradCPUKernel::MallocRunBuffer() {
  int workspace_size = GetRunWorkspaceSize(lstm_param_);
  if ((workspace_size == 0) || (workspace_size > LSTMGRAD_MAX_WORKSPACE_SIZE)) {
    MS_LOG(ERROR) << "LstmCPUKernel malloc run workspace 0 error.";
    return RET_ERROR;
  }
  workspace_ = reinterpret_cast<float *>(malloc(workspace_size * sizeof(float)));
  if (workspace_ == nullptr) {
    MS_LOG(ERROR) << "LstmCPUKernel malloc run workspace error.";
    return RET_ERROR;
  }
  return RET_OK;
}

void LSTMGradCPUKernel::FreeRunBuffer() {
  if (workspace_ != nullptr) {
    free(workspace_);
    workspace_ = nullptr;
  }
}

int LSTMGradCPUKernel::PackWeights() {
  auto weight_i_data = InputWeightPtr();
  CHECK_NULL_RETURN(weight_i_data);
  PackLstmWeightTranspose(weight_i_ptr_, weight_i_data, weight_batch_, lstm_param_->hidden_size_,
                          lstm_param_->input_size_, input_size_align_);

  auto weight_h_data = StateWeightPtr();
  CHECK_NULL_RETURN(weight_h_data);
  PackLstmWeightTranspose(weight_h_ptr_, weight_h_data, weight_batch_, lstm_param_->hidden_size_,
                          lstm_param_->hidden_size_, lstm_param_->state_col_align_);
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LSTMGrad, LiteKernelCreator<LSTMGradCPUKernel>)
}  // namespace kernel
}  // namespace mindspore
