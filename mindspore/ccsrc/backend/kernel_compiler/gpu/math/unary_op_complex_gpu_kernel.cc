/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "utils/complex.h"
#include "backend/kernel_compiler/gpu/math/unary_op_complex_gpu_kernel.h"

namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;
MS_REG_GPU_KERNEL_TWO(Real, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeFloat64),
                      UnaryOpComplexGpuKernel, Complex<double>, double)
MS_REG_GPU_KERNEL_TWO(Real, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeFloat32),
                      UnaryOpComplexGpuKernel, Complex<float>, float)

MS_REG_GPU_KERNEL_TWO(Real, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                      UnaryOpComplexGpuKernel, char, char)
MS_REG_GPU_KERNEL_TWO(Real, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                      UnaryOpComplexGpuKernel, int16_t, int16_t)
MS_REG_GPU_KERNEL_TWO(Real, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      UnaryOpComplexGpuKernel, int32_t, int32_t)
MS_REG_GPU_KERNEL_TWO(Real, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                      UnaryOpComplexGpuKernel, int64_t, int64_t)
MS_REG_GPU_KERNEL_TWO(Real, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                      UnaryOpComplexGpuKernel, unsigned char, unsigned char)
MS_REG_GPU_KERNEL_TWO(Real, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
                      UnaryOpComplexGpuKernel, uint16_t, uint16_t)
MS_REG_GPU_KERNEL_TWO(Real, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
                      UnaryOpComplexGpuKernel, uint32_t, uint32_t)
MS_REG_GPU_KERNEL_TWO(Real, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
                      UnaryOpComplexGpuKernel, uint64_t, uint64_t)

MS_REG_GPU_KERNEL_TWO(Real, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      UnaryOpComplexGpuKernel, float, float)
MS_REG_GPU_KERNEL_TWO(Real, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      UnaryOpComplexGpuKernel, double, double)
MS_REG_GPU_KERNEL_TWO(Real, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      UnaryOpComplexGpuKernel, half, half)
MS_REG_GPU_KERNEL_TWO(Real, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                      UnaryOpComplexGpuKernel, bool, bool)

MS_REG_GPU_KERNEL_TWO(Imag, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeFloat64),
                      UnaryOpComplexGpuKernel, Complex<double>, double)
MS_REG_GPU_KERNEL_TWO(Imag, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeFloat32),
                      UnaryOpComplexGpuKernel, Complex<float>, float)

MS_REG_GPU_KERNEL_TWO(Imag, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                      UnaryOpComplexGpuKernel, char, char)
MS_REG_GPU_KERNEL_TWO(Imag, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                      UnaryOpComplexGpuKernel, int16_t, int16_t)
MS_REG_GPU_KERNEL_TWO(Imag, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      UnaryOpComplexGpuKernel, int32_t, int32_t)
MS_REG_GPU_KERNEL_TWO(Imag, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                      UnaryOpComplexGpuKernel, int64_t, int64_t)
MS_REG_GPU_KERNEL_TWO(Imag, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                      UnaryOpComplexGpuKernel, unsigned char, unsigned char)
MS_REG_GPU_KERNEL_TWO(Imag, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
                      UnaryOpComplexGpuKernel, uint16_t, uint16_t)
MS_REG_GPU_KERNEL_TWO(Imag, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
                      UnaryOpComplexGpuKernel, uint32_t, uint32_t)
MS_REG_GPU_KERNEL_TWO(Imag, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
                      UnaryOpComplexGpuKernel, uint64_t, uint64_t)

MS_REG_GPU_KERNEL_TWO(Imag, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      UnaryOpComplexGpuKernel, float, float)
MS_REG_GPU_KERNEL_TWO(Imag, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      UnaryOpComplexGpuKernel, double, double)
MS_REG_GPU_KERNEL_TWO(Imag, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      UnaryOpComplexGpuKernel, half, half)
MS_REG_GPU_KERNEL_TWO(Imag, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                      UnaryOpComplexGpuKernel, bool, bool)

MS_REG_GPU_KERNEL_TWO(Conj, KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
                      UnaryOpComplexGpuKernel, Complex<double>, Complex<double>)

MS_REG_GPU_KERNEL_TWO(Conj, KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
                      UnaryOpComplexGpuKernel, Complex<float>, Complex<float>)

MS_REG_GPU_KERNEL_TWO(Conj, KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                      UnaryOpComplexGpuKernel, char, char)
MS_REG_GPU_KERNEL_TWO(Conj, KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                      UnaryOpComplexGpuKernel, int16_t, int16_t)
MS_REG_GPU_KERNEL_TWO(Conj, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      UnaryOpComplexGpuKernel, int32_t, int32_t)
MS_REG_GPU_KERNEL_TWO(Conj, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                      UnaryOpComplexGpuKernel, int64_t, int64_t)
MS_REG_GPU_KERNEL_TWO(Conj, KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                      UnaryOpComplexGpuKernel, unsigned char, unsigned char)
MS_REG_GPU_KERNEL_TWO(Conj, KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
                      UnaryOpComplexGpuKernel, uint16_t, uint16_t)
MS_REG_GPU_KERNEL_TWO(Conj, KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
                      UnaryOpComplexGpuKernel, uint32_t, uint32_t)
MS_REG_GPU_KERNEL_TWO(Conj, KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
                      UnaryOpComplexGpuKernel, uint64_t, uint64_t)

MS_REG_GPU_KERNEL_TWO(Conj, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      UnaryOpComplexGpuKernel, float, float)
MS_REG_GPU_KERNEL_TWO(Conj, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      UnaryOpComplexGpuKernel, double, double)
MS_REG_GPU_KERNEL_TWO(Conj, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                      UnaryOpComplexGpuKernel, half, half)
}  // namespace kernel
}  // namespace mindspore
