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
#include "nnacl/fp32/sub_fp32.h"

int ElementOptSub(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param) {
#ifdef ENABLE_AVX
  MS_FLOAT32X8 vin0_opt_8 = MS_MOV256_F32(in0[0]);
  MS_FLOAT32X8 vin1_opt_8 = MS_MOV256_F32(in1[0]);
#endif
#ifdef ENABLE_NEON
  float32x4_t vin0_opt = vdupq_n_f32(in0[0]);
  float32x4_t vin1_opt = vdupq_n_f32(in1[0]);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_AVX
    for (; index <= size - C8NUM; index += C8NUM) {
      MS_FLOAT32X8 vin1 = MS_LD256_F32(in1 + index);
      MS_FLOAT32X8 vout = MS_SUB256_F32(vin0_opt_8, vin1);
      MS_ST256_F32(out + index, vout);
    }
#endif
#ifdef ENABLE_NEON
    for (; index <= size - C4NUM; index += C4NUM) {
      float32x4_t vin1 = vld1q_f32(in1 + index);
      float32x4_t vout = vsubq_f32(vin0_opt, vin1);
      vst1q_f32(out + index, vout);
    }
#endif
    for (; index < size; index++) {
      out[index] = in0[0] - in1[index];
    }
  } else {
#ifdef ENABLE_AVX
    for (; index <= size - C8NUM; index += C8NUM) {
      MS_FLOAT32X8 vin0 = MS_LD256_F32(in0 + index);
      MS_FLOAT32X8 vout = MS_SUB256_F32(vin0, vin1_opt_8);
      MS_ST256_F32(out + index, vout);
    }
#endif
#ifdef ENABLE_NEON
    for (; index <= size - C4NUM; index += C4NUM) {
      float32x4_t vin0 = vld1q_f32(in0 + index);
      float32x4_t vout = vsubq_f32(vin0, vin1_opt);
      vst1q_f32(out + index, vout);
    }
#endif
    for (; index < size; index++) {
      out[index] = in0[index] - in1[0];
    }
  }
  return NNACL_OK;
}

int ElementOptSubInt(const int *in0, const int *in1, int *out, int size, const ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  int32x4_t vin0_opt = vdupq_n_s32(in0[0]);
  int32x4_t vin1_opt = vdupq_n_s32(in1[0]);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= size - C4NUM; index += C4NUM) {
      int32x4_t vin1 = vld1q_s32(in1 + index);
      int32x4_t vout = vsubq_s32(vin0_opt, vin1);
      vst1q_s32(out + index, vout);
    }
#endif
    for (; index < size; index++) {
      out[index] = in0[0] - in1[index];
    }
  } else {
#ifdef ENABLE_NEON
    for (; index <= size - C4NUM; index += C4NUM) {
      int32x4_t vin0 = vld1q_s32(in0 + index);
      int32x4_t vout = vsubq_s32(vin0, vin1_opt);
      vst1q_s32(out + index, vout);
    }
#endif
    for (; index < size; index++) {
      out[index] = in0[index] - in1[0];
    }
  }
  return NNACL_OK;
}

int ElementOptSubRelu(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  float32x4_t vin0_opt = vdupq_n_f32(in0[0]);
  float32x4_t vin1_opt = vdupq_n_f32(in1[0]);
  float32x4_t zeros = vdupq_n_f32(0.0f);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= size - C4NUM; index += C4NUM) {
      float32x4_t vin1 = vld1q_f32(in1 + index);
      float32x4_t vout = vmaxq_f32(vsubq_f32(vin0_opt, vin1), zeros);
      vst1q_f32(out + index, vout);
    }
#endif
    for (; index < size; index++) {
      out[index] = MSMAX(in0[0] - in1[index], 0);
    }
  } else {
#ifdef ENABLE_NEON
    for (; index <= size - C4NUM; index += C4NUM) {
      float32x4_t vin0 = vld1q_f32(in0 + index);
      float32x4_t vout = vmaxq_f32(vsubq_f32(vin0, vin1_opt), zeros);
      vst1q_f32(out + index, vout);
    }
#endif
    for (; index < size; index++) {
      out[index] = MSMAX(in0[index] - in1[0], 0);
    }
  }
  return NNACL_OK;
}

int ElementOptSubRelu6(const float *in0, const float *in1, float *out, int size, const ArithmeticParameter *param) {
#ifdef ENABLE_NEON
  float32x4_t vin0_opt = vdupq_n_f32(in0[0]);
  float32x4_t vin1_opt = vdupq_n_f32(in1[0]);
  float32x4_t zeros = vdupq_n_f32(0.0f);
  float32x4_t bounds = vdupq_n_f32(6.0f);
#endif
  int index = 0;
  if (param->in_elements_num0_ == 1) {
#ifdef ENABLE_NEON
    for (; index <= size - C4NUM; index += C4NUM) {
      float32x4_t vin1 = vld1q_f32(in1 + index);
      float32x4_t vout = vminq_f32(vmaxq_f32(vsubq_f32(vin0_opt, vin1), zeros), bounds);
      vst1q_f32(out + index, vout);
    }
#endif
    for (; index < size; index++) {
      out[index] = MSMIN(MSMAX(in0[0] - in1[index], 0), 6);
    }
  } else {
#ifdef ENABLE_NEON
    for (; index <= size - C4NUM; index += C4NUM) {
      float32x4_t vin0 = vld1q_f32(in0 + index);
      float32x4_t vout = vminq_f32(vmaxq_f32(vsubq_f32(vin0, vin1_opt), zeros), bounds);
      vst1q_f32(out + index, vout);
    }
#endif
    for (; index < size; index++) {
      out[index] = MSMIN(MSMAX(in0[index] - in1[0], 0), 6);
    }
  }
  return NNACL_OK;
}

int ElementSub(const float *in0, const float *in1, float *out, int size) {
  int index = 0;
#ifdef ENABLE_NEON
  for (; index <= size - C4NUM; index += C4NUM) {
    float32x4_t vin0 = vld1q_f32(in0 + index);
    float32x4_t vin1 = vld1q_f32(in1 + index);
    float32x4_t vout = vsubq_f32(vin0, vin1);
    vst1q_f32(out + index, vout);
  }
#endif
#ifdef ENABLE_AVX
  for (; index <= size - C8NUM; index += C8NUM) {
    MS_FLOAT32X8 vin0 = MS_LD256_F32(in0 + index);
    MS_FLOAT32X8 vin1 = MS_LD256_F32(in1 + index);
    MS_FLOAT32X8 vout = MS_SUB256_F32(vin0, vin1);
    MS_ST256_F32(out + index, vout);
  }
#endif
  for (; index < size; index++) {
    out[index] = in0[index] - in1[index];
  }
  return NNACL_OK;
}

int ElementSubInt(const int *in0, const int *in1, int *out, int size) {
  int index = 0;
#ifdef ENABLE_NEON
  for (; index <= size - C4NUM; index += C4NUM) {
    int32x4_t vin0 = vld1q_s32(in0 + index);
    int32x4_t vin1 = vld1q_s32(in1 + index);
    int32x4_t vout = vsubq_s32(vin0, vin1);
    vst1q_s32(out + index, vout);
  }
#endif
  for (; index < size; index++) {
    out[index] = in0[index] - in1[index];
  }
  return NNACL_OK;
}

int ElementSubRelu(const float *in0, const float *in1, float *out, int size) {
  int index = 0;
#ifdef ENABLE_NEON
  float32x4_t zeros = vdupq_n_f32(0.0f);
  for (; index <= size - C4NUM; index += C4NUM) {
    float32x4_t vin0 = vld1q_f32(in0 + index);
    float32x4_t vin1 = vld1q_f32(in1 + index);
    float32x4_t vout = vsubq_f32(vin0, vin1);
    vout = vbslq_f32(vcleq_f32(vout, zeros), zeros, vout);
    vst1q_f32(out + index, vout);
  }
#endif
  for (; index < size; index++) {
    float res = in0[index] - in1[index];
    out[index] = res > 0 ? res : 0;
  }
  return NNACL_OK;
}

int ElementSubRelu6(const float *in0, const float *in1, float *out, int size) {
  int index = 0;
#ifdef ENABLE_NEON
  float32x4_t zeros = vdupq_n_f32(0.0f);
  float32x4_t bounds = vdupq_n_f32(6.0f);
  for (; index <= size - C4NUM; index += C4NUM) {
    float32x4_t vin0 = vld1q_f32(in0 + index);
    float32x4_t vin1 = vld1q_f32(in1 + index);
    float32x4_t vout = vminq_f32(vmaxq_f32(vsubq_f32(vin0, vin1), zeros), bounds);
    vst1q_f32(out + index, vout);
  }
#endif
  for (; index < size; index++) {
    out[index] = MSMIN(MSMAX(in0[index] - in1[index], 0), 6);
  }

  return NNACL_OK;
}
