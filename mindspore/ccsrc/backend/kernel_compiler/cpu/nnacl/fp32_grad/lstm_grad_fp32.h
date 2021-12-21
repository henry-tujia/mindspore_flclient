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

#ifndef MINDSPORE_NNACL_FP32_GRAD_LSTM_GRAD_H_
#define MINDSPORE_NNACL_FP32_GRAD_LSTM_GRAD_H_

#include "nnacl/lstm_parameter.h"
#ifdef __cplusplus
extern "C" {
#endif

int GetRunWorkspaceSize(const LstmParameter *lstm_param);

void PackLstmWeightTranspose(float *dst, const float *src, int batch, int col, int row, int row_align);

void LstmGradStepUnit(float *packed_input, float *output, float *input_gate, float *forget_gate, float *cell_gate,
                      float *output_gate, float *hidden_state, float *cell_state, float *dC, float *dH, float *dY,
                      float *last_cell, float *weights, float *workspace, const LstmParameter *lstm_param);

#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_NNACL_FP32_GRAD_LSTM_GRAD_H_
