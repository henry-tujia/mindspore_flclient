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
#include <float.h>
#include "nnacl/fp32/activation_fp32.h"
#include "nnacl/fp32/exp_fp32.h"
#include "nnacl/errorcode.h"

int Fp32Relu(const float *src, int length, float *dst) {
  int i = 0;
#if defined(ENABLE_AVX512)
  MS_FLOAT32X16 zero_16 = MS_MOV512_F32(0.0f);
  for (; i <= length - C16NUM; i += C16NUM) {
    MS_ST512_F32(dst + i, MS_MAX512_F32(MS_LD512_F32(src + i), zero_16));
  }
#endif

#if defined(ENABLE_AVX)
  MS_FLOAT32X8 zero_8 = MS_MOV256_F32(0.0f);
  for (; i <= length - 8; i += 8) {
    MS_ST256_F32(dst + i, MS_MAX256_F32(MS_LD256_F32(src + i), zero_8));
  }
#endif

#if defined(ENABLE_SSE) || defined(ENABLE_ARM)
  MS_FLOAT32X4 zero = MS_MOVQ_F32(0.0f);
  for (; i <= length - 4; i += 4) {
    MS_STQ_F32(dst + i, MS_MAXQ_F32(MS_LDQ_F32(src + i), zero));
  }
#endif
  for (; i < length; ++i) {
    dst[i] = src[i] > 0 ? src[i] : 0;
  }
  return NNACL_OK;
}

int Int32Relu(const int32_t *src, int length, int32_t *dst) {
  for (int i = 0; i < length; ++i) {
    dst[i] = src[i] > 0 ? src[i] : 0;
  }
  return NNACL_OK;
}

int Fp32Relu6(const float *src, int length, float *dst) {
  int i = 0;

#if defined(ENABLE_AVX512)
  MS_FLOAT32X16 zero_16 = MS_MOV512_F32(0.0f);
  MS_FLOAT32X16 six_16 = MS_MOV512_F32(6.0f);
  for (; i <= length - C16NUM; i += C16NUM) {
    MS_FLOAT32X16 dst_tmp = MS_MAX512_F32(MS_LD512_F32(src + i), zero_16);
    dst_tmp = MS_MIN512_F32(dst_tmp, six_16);
    MS_ST512_F32(dst + i, dst_tmp);
  }
#endif

#if defined(ENABLE_AVX)
  MS_FLOAT32X8 zero_8 = MS_MOV256_F32(0.0f);
  MS_FLOAT32X8 six_8 = MS_MOV256_F32(6.0f);
  for (; i <= length - 8; i += 8) {
    MS_FLOAT32X8 dst_tmp = MS_MAX256_F32(MS_LD256_F32(src + i), zero_8);
    dst_tmp = MS_MIN256_F32(dst_tmp, six_8);
    MS_ST256_F32(dst + i, dst_tmp);
  }
#endif

#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  MS_FLOAT32X4 zero = MS_MOVQ_F32(0.0f);
  MS_FLOAT32X4 six = MS_MOVQ_F32(6.0f);
  for (; i <= length - 4; i += 4) {
    MS_FLOAT32X4 dst_tmp = MS_MAXQ_F32(MS_LDQ_F32(src + i), zero);
    dst_tmp = MS_MINQ_F32(dst_tmp, six);
    MS_STQ_F32(dst + i, dst_tmp);
  }
#endif
  for (; i < length; ++i) {
    if (src[i] < 0) {
      dst[i] = 0;
    } else {
      dst[i] = src[i] > 6.0f ? 6.0f : src[i];
    }
  }
  return NNACL_OK;
}

int LRelu(const float *src, int length, float *dst, float alpha) {
  int i = 0;
#if defined(ENABLE_AVX)
  for (; i <= length - 8; i += 8) {
    MS_FLOAT32X8 src_tmp = MS_LD256_F32(src + i);
    MS_FLOAT32X8 mul_tmp = MS_MUL256_N_F32(src_tmp, alpha);
    MS_FLOAT32X8 mask = MS_CMP256_F32(src_tmp, MS_MOV256_F32(0.0f), 30);
    MS_ST256_F32(dst + i, MS_BLEND256_F32(mul_tmp, src_tmp, mask));
  }
#endif

#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  for (; i <= length - 4; i += 4) {
    MS_FLOAT32X4 src_tmp = MS_LDQ_F32(src + i);
    MS_FLOAT32X4 mul_tmp = MS_MULQ_N_F32(src_tmp, alpha);
#ifdef ENABLE_ARM
    MS_UINT32X4 mask = MS_CMPLEQ_F32(src_tmp, MS_MOVQ_F32(0.0f));
#else
    MS_FLOAT32X4 mask = MS_CMPLEQ_F32(src_tmp, MS_MOVQ_F32(0.0f));
#endif
    MS_STQ_F32(dst + i, MS_BLENDQ_F32(src_tmp, mul_tmp, mask));
  }
#endif
  for (; i < length; ++i) {
    dst[i] = src[i] > 0 ? src[i] : (src[i] * alpha);
  }
  return NNACL_OK;
}

int Sigmoid(const float *src, int length, float *dst) {
  int i = 0;
#if defined(ENABLE_AVX512)
  for (; i <= length - C16NUM; i += C16NUM) {
    simd_exp_avx512(MS_SUB512_F32(MS_MOV512_F32(0.0f), (MS_LD512_F32(src + i))), dst + i);
    MS_ST512_F32(dst + i,
                 MS_DIV512_F32(MS_MOV512_F32(1.0f), MS_ADD512_F32(MS_MOV512_F32(1.0f), MS_LD512_F32(dst + i))));
  }
#endif

#if defined(ENABLE_AVX)
  for (; i <= length - 8; i += 8) {
    simd_exp_avx(MS_SUB256_F32(MS_MOV256_F32(0.0f), (MS_LD256_F32(src + i))), dst + i);
    MS_ST256_F32(dst + i,
                 MS_DIV256_F32(MS_MOV256_F32(1.0f), MS_ADD256_F32(MS_MOV256_F32(1.0f), MS_LD256_F32(dst + i))));
  }
#endif

#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  for (; i <= length - 4; i += 4) {
    simd_exp(MS_SUBQ_F32(MS_MOVQ_F32(0.0f), MS_LDQ_F32(src + i)), dst + i);
    MS_STQ_F32(dst + i, MS_DIVQ_F32(MS_MOVQ_F32(1.0f), MS_ADDQ_F32(MS_MOVQ_F32(1.0f), MS_LDQ_F32(dst + i))));
  }
#endif
  for (; i < length; ++i) {
    single_exp(-src[i], dst + i);
    dst[i] = 1.0f / (1.0f + dst[i]);
  }
  return NNACL_OK;
}

float TanhOpt(float src) {
  if (src > 5.0) {
    return 1.0f;
  } else if (src < -5.0) {
    return -1.0f;
  } else {
    float square = src * src;
    float a = (((square + 378.0f) * square + 17325.0f) * square + 135135.0f) * src;
    float b = ((28.0f * square + 3150.0f) * square + 62370.0f) * square + 135135.0f;
    return a / b;
  }
}

int Tanh(const float *src, int length, float *dst) {
  int i = 0;
#if defined(ENABLE_AVX512)
  for (; i <= length - C16NUM; i += C16NUM) {
    MS_FLOAT32X16 input = MS_LD512_F32(src + i);
    MS_ST512_F32(dst + i, MS_TANHX16_F32(input));
  }
#endif

#if defined(ENABLE_AVX)
  for (; i <= length - 8; i += 8) {
    MS_FLOAT32X8 input = MS_LD256_F32(src + i);
    MS_ST256_F32(dst + i, MS_TANHX8_F32(input));
  }
#endif

#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  for (; i <= length - 4; i += 4) {
    MS_FLOAT32X4 input = MS_LDQ_F32(src + i);
    MS_STQ_F32(dst + i, MS_TANHX4_F32(input));
  }
#endif
  for (; i < length; ++i) {
    dst[i] = TanhOpt(src[i]);
  }
  return NNACL_OK;
}

int Swish(const float *src, int length, float *dst) {
  int ret = Sigmoid(src, length, dst);
  if (ret != NNACL_OK) {
    return NNACL_ERR;
  }
  int index = 0;
#if defined(ENABLE_AVX)
  for (; index <= length - 8; index += 8) {
    MS_FLOAT32X8 src_value = MS_LD256_F32(src + index);
    MS_FLOAT32X8 sigmoid_value = MS_LD256_F32(dst + index);
    MS_FLOAT32X8 result = MS_MUL256_F32(src_value, sigmoid_value);
    MS_ST256_F32(dst + index, result);
  }
#endif

#if defined(ENABLE_ARM) || defined(ENABLE_SSE)
  for (; index <= length - 4; index += 4) {
    MS_FLOAT32X4 src_value = MS_LDQ_F32(src + index);
    MS_FLOAT32X4 sigmoid_value = MS_LDQ_F32(dst + index);
    MS_FLOAT32X4 result = MS_MULQ_F32(src_value, sigmoid_value);
    MS_STQ_F32(dst + index, result);
  }
#endif
  for (; index < length; ++index) {
    dst[index] = src[index] * dst[index];
  }
  return NNACL_OK;
}

int HSwish(const float *src, int length, float *dst) {
  for (int i = 0; i < length; ++i) {
    float in = src[i];
    float relu6 = MSMIN(MSMAX(in + 3, 0), 6);
    dst[i] = in * relu6 / 6;
  }
  return NNACL_OK;
}

int HSigmoid(const float *src, int length, float *dst) {
  for (int i = 0; i < length; ++i) {
    float relu6 = MSMIN(MSMAX(src[i] + 3, 0), 6);
    dst[i] = relu6 / 6;
  }
  return NNACL_OK;
}

int HardTanh(const float *src, int length, float *dst, float min_val, float max_val) {
  if (max_val <= min_val) {
    return NNACL_ERR;
  }
  int i = 0;
  if (min_val == FLT_MIN) {
    for (i = 0; i < length; ++i) {
      dst[i] = src[i] > max_val ? max_val : src[i];
    }
  } else if (max_val == FLT_MAX) {
    for (i = 0; i < length; ++i) {
      dst[i] = src[i] < min_val ? min_val : src[i];
    }
  } else {
    for (i = 0; i < length; ++i) {
      dst[i] = src[i] < min_val ? min_val : (src[i] > max_val ? max_val : src[i]);
    }
  }
  return NNACL_OK;
}

int Gelu(const float *src, int length, float *dst, bool approximate) {
  if (src == NULL || dst == NULL) {
    return NNACL_ERR;
  }
  int i = 0;
  if (approximate) {
    // dst = 0.5 * x * (1 + tanh((2 / pi) ^ 0.5 * (x + 0.044715x^3)))
#if defined(ENABLE_AVX)
    MS_FLOAT32X8 para1 = MS_MOV256_F32(0.79788456080287f);
    MS_FLOAT32X8 para2 = MS_MOV256_F32(0.035677408136f);
    MS_FLOAT32X8 para3 = MS_MOV256_F32(1.0f);
    MS_FLOAT32X8 para4 = MS_MOV256_F32(0.5f);
    int C8 = DOWN_ROUND(length, C8NUM);
    for (; i < C8; i += C8NUM) {
      MS_FLOAT32X8 in = MS_LD256_F32(src + i);
      const MS_FLOAT32X8 res = MS_MUL256_F32(
        MS_MUL256_F32(para4, in),
        MS_ADD256_F32(
          para3, MS_TANHX8_F32(MS_MUL256_F32(MS_ADD256_F32(para1, MS_MUL256_F32(MS_MUL256_F32(para2, in), in)), in))));
      MS_ST256_F32(dst + i, res);
    }
#endif
#if defined(ENABLE_SSE) || defined(ENABLE_ARM)
    MS_FLOAT32X4 para5 = MS_MOVQ_F32(0.79788456080287f);
    MS_FLOAT32X4 para6 = MS_MOVQ_F32(0.035677408136f);
    MS_FLOAT32X4 para7 = MS_MOVQ_F32(1.0f);
    MS_FLOAT32X4 para8 = MS_MOVQ_F32(0.5f);
    int C4 = DOWN_ROUND(length, C4NUM);
    for (; i < C4; i += C4NUM) {
      MS_FLOAT32X4 in = MS_LDQ_F32(src + i);
      MS_FLOAT32X4 res = MS_MULQ_F32(
        MS_MULQ_F32(para8, in),
        MS_ADDQ_F32(para7,
                    MS_TANHX4_F32(MS_MULQ_F32(MS_ADDQ_F32(para5, MS_MULQ_F32(MS_MULQ_F32(para6, in), in)), in))));
      MS_STQ_F32(dst + i, res);
    }
#endif
    for (; i < length; i++) {
      dst[i] = 0.5 * src[i] * (1.0 + TanhOpt((0.79788456080287f + 0.035677408136f * src[i] * src[i]) * src[i]));
    }
  } else {
#if defined(ENABLE_AVX) || defined(ENABLE_SSE) || defined(ENABLE_ARM)
    MS_FLOAT32X4 para1 = MS_MOVQ_F32(1.4142135623730951f);
    MS_FLOAT32X4 para2 = MS_MOVQ_F32(1.0f);
    MS_FLOAT32X4 para3 = MS_MOVQ_F32(0.5f);
    int C4 = DOWN_ROUND(length, C4NUM);
    for (; i < C4; i += C4NUM) {
      MS_FLOAT32X4 in = MS_LDQ_F32(src + i);
      MS_FLOAT32X4 res = MS_MULQ_F32(MS_MULQ_F32(para3, in), MS_ADDQ_F32(para2, MS_ERFX4_F32(MS_DIVQ_F32(in, para1))));
      MS_STQ_F32(dst + i, res);
    }
#endif
    for (; i < length; i++) {
      dst[i] = 0.5 * src[i] * (1.0 + erf(src[i] / 1.4142135623730951f));
    }
  }
  return NNACL_OK;
}

int Softplus(const float *src, int length, float *dst) {
  int i = 0;
  for (; i < length; ++i) {
    single_exp(src[i], dst + i);
    dst[i] = log1p(dst[i]);
  }
  return NNACL_OK;
}

int Elu(const float *src, int length, float *dst, float alpha) {
  int i = 0;
#if defined(ENABLE_ARM)
  MS_FLOAT32X4 one = MS_MOVQ_F32(1.0f);
  for (; i <= length - 4; i += 4) {
    MS_FLOAT32X4 src_tmp = MS_LDQ_F32(src + i);
    MS_FLOAT32X4 exp_tmp;
    exp_tmp[0] = exp(src_tmp[0]);  // exp(x)
    exp_tmp[1] = exp(src_tmp[1]);
    exp_tmp[2] = exp(src_tmp[2]);
    exp_tmp[3] = exp(src_tmp[3]);
    exp_tmp = MS_SUBQ_F32(exp_tmp, one);  // exp(x) - 1
    MS_FLOAT32X4 elu_tmp = MS_MULQ_N_F32(exp_tmp, alpha);
    MS_UINT32X4 mask = MS_CMPLEQ_F32(src_tmp, MS_MOVQ_F32(0.0f));
    MS_STQ_F32(dst + i, MS_BLENDQ_F32(src_tmp, elu_tmp, mask));
  }
#endif
  for (; i < length; ++i) {
    dst[i] = src[i] > 0 ? src[i] : (expm1(src[i]) * alpha);
  }
  return NNACL_OK;
}
