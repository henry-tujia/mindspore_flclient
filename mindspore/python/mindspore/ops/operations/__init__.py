# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
Primitive operator classes.

A collection of operators to build neural networks or to compute functions.
"""

from .image_ops import (CropAndResize, NonMaxSuppressionV3)
from .array_ops import (Argmax, Argmin, Cast, Concat, Pack, Stack, Unpack, Unstack,
                        Diag, DiagPart, DType, ExpandDims, Eye,
                        Fill, Ones, Zeros, GatherNd, GatherV2, Gather, SparseGatherV2, InvertPermutation,
                        IsInstance, IsSubClass, ArgMaxWithValue, OnesLike, ZerosLike,
                        Rank, Reshape, ResizeNearestNeighbor, ArgMinWithValue, Meshgrid,
                        SameTypeShape, ScatterAdd, ScatterSub, ScatterMul, ScatterDiv, ScatterMax, ScatterMin,
                        ScatterUpdate, ScalarToArray, ScalarToTensor, ScatterNd, ScatterNdUpdate, Select,
                        Shape, DynamicShape, Size, Slice, Split, SplitV, TransShape, ParallelConcat, Padding,
                        UniqueWithPad, ScatterNdAdd, ScatterNdSub, ScatterNonAliasingAdd, ReverseV2, Rint,
                        Squeeze, StridedSlice, Tile, TensorScatterUpdate, TensorScatterAdd, EditDistance, Sort,
                        Transpose, TruncatedNormal, TupleToArray, UnsortedSegmentMin, UnsortedSegmentMax,
                        UnsortedSegmentProd, UnsortedSegmentSum, SpaceToDepth, DepthToSpace, SpaceToBatch,
                        BatchToSpace, SpaceToBatchND, BatchToSpaceND, BroadcastTo, InplaceUpdate, ReverseSequence,
                        EmbeddingLookup, Unique, GatherD, Identity, Range, MaskedFill, MaskedSelect, SearchSorted,
                        TensorScatterMax, TensorScatterMin, TensorScatterSub, ScatterElements, ExtractVolumePatches)
from .comm_ops import (AllGather, AllReduce, NeighborExchange, NeighborExchangeV2, AlltoAll, _AllSwap, ReduceScatter, Broadcast,
                       _MirrorOperator, _MirrorMiniStepOperator, _MiniStepAllGather, ReduceOp, _VirtualDataset,
                       _VirtualOutput, _VirtualDiv, _GetTensorSlice, _VirtualAdd, _VirtualAssignAdd, _VirtualAccuGrad,
                       _HostAllGather, _HostReduceScatter, _MirrorMicroStepOperator, _MicroStepAllGather)
from .debug_ops import (ImageSummary, InsertGradientOf, HookBackward, ScalarSummary,
                        TensorSummary, HistogramSummary, Print, Assert)
from .control_ops import GeSwitch, Merge
from .inner_ops import (ScalarCast, Randperm, NoRepeatNGram, LambApplyOptimizerAssign, LambApplyWeightAssign,
                        MakeRefKey,
                        FusedWeightScaleApplyMomentum, FusedCastAdamWeightDecay)

from .math_ops import (Abs, ACos, Asin, Asinh, AddN, AccumulateNV2, AssignAdd, AssignSub, Atan2, BatchMatMul,
                       BitwiseAnd, BitwiseOr, Ger,
                       BitwiseXor, Inv, Invert, ApproximateEqual, InplaceAdd, InplaceSub,
                       ReduceMax, ReduceMin, ReduceMean, ReduceSum, ReduceAll, ReduceProd, CumProd, Cdist, ReduceAny,
                       Cos, Div, DivNoNan, Equal, EqualCount, Exp, Expm1, Erf, Erfc, Floor, FloorDiv, FloorMod, Ceil,
                       Acosh, Greater, GreaterEqual, Lerp, Less, LessEqual, Log, Log1p, LogicalAnd, Mod,
                       LogicalNot, LogicalOr, LpNorm, MatMul, Maximum, MulNoNan,
                       Minimum, Mul, Neg, NMSWithMask, NotEqual,
                       NPUAllocFloatStatus, NPUClearFloatStatus, LinSpace,
                       NPUGetFloatStatus, Pow, RealDiv, IsNan, IsInf, IsFinite, FloatStatus,
                       Reciprocal, CumSum, HistogramFixedWidth, SquaredDifference, Xdivy, Xlogy,
                       Sin, Sqrt, Rsqrt, BesselI0, BesselI1, BesselI0e, BesselI1e, TruncateDiv, TruncateMod, Addcdiv,
                       Addcmul, Square, Sub, TensorAdd, Add, Sign, Round, SquareSumAll, Atan, Atanh, Cosh, Sinh, Eps,
                       Tan, MatrixInverse, IndexAdd, Erfinv, Conj, Real, Imag, Complex, Trunc, IsClose)

from .random_ops import (RandomChoiceWithMask, StandardNormal, Gamma, Poisson, UniformInt, UniformReal,
                         RandomCategorical, StandardLaplace, Multinomial, UniformCandidateSampler,
                         LogUniformCandidateSampler)

from .nn_ops import (LSTM, SGD, Adam, AdamWeightDecay, FusedSparseAdam, FusedSparseLazyAdam, AdamNoUpdateParam,
                     ApplyMomentum, BatchNorm, BiasAdd, Conv2D, Conv3D, Conv2DTranspose, Conv3DTranspose,
                     DepthwiseConv2dNative,
                     DropoutDoMask, Dropout, Dropout2D, Dropout3D, DropoutGenMask, Flatten,
                     InstanceNorm, BNTrainingReduce, BNTrainingUpdate,
                     GeLU, Gelu, FastGeLU, FastGelu, Elu, CeLU,
                     GetNext, L2Normalize, LayerNorm, L2Loss, CTCLoss, CTCLossV2, CTCLossV2Grad, CTCGreedyDecoder,
                     LogSoftmax, MaxPool3D, AvgPool3D,
                     MaxPool, DataFormatDimMap,
                     AvgPool, Conv2DBackpropInput, ComputeAccidentalHits,
                     MaxPoolWithArgmax, OneHot, Pad, MirrorPad, Mish, PReLU, ReLU, ReLU6, ReLUV2, HSwish, HSigmoid,
                     ResizeBilinear, Sigmoid, SeLU, HShrink, ApplyKerasMomentum,
                     SigmoidCrossEntropyWithLogits, NLLLoss, BCEWithLogitsLoss,
                     SmoothL1Loss, SoftMarginLoss, Softmax, Softsign, Softplus, LRN, RNNTLoss, DynamicRNN, DynamicGRUV2,
                     SoftmaxCrossEntropyWithLogits, ROIAlign,
                     SparseSoftmaxCrossEntropyWithLogits, Tanh,
                     TopK, BinaryCrossEntropy, KLDivLoss, SparseApplyAdagrad, LARSUpdate, ApplyFtrl, SparseApplyFtrl,
                     ApplyProximalAdagrad, SparseApplyProximalAdagrad, SparseApplyAdagradV2, SparseApplyFtrlV2,
                     FusedSparseFtrl, FusedSparseProximalAdagrad, SparseApplyRMSProp, SparseApplyAdadelta,
                     ApplyAdaMax, ApplyAdadelta, ApplyAdagrad, ApplyAdagradV2, ApplyAdagradDA,
                     ApplyAddSign, ApplyPowerSign, ApplyGradientDescent, ApplyProximalGradientDescent,
                     ApplyRMSProp, ApplyCenteredRMSProp, BasicLSTMCell, InTopK, AdaptiveAvgPool2D, SoftShrink,
                     ApplyAdamWithAmsgrad)
from . import _quant_ops
from ._quant_ops import *
from .other_ops import (Assign, IOU, BoundingBoxDecode, BoundingBoxEncode,
                        ConfusionMatrix, PopulationCount, UpdateState, Load,
                        CheckValid, Partial, Depend, identity, CheckBprop, Push, Pull, PullWeight, PushWeight,
                        PushMetrics, StartFLJob, UpdateModel, GetModel, PyFunc, ExchangeKeys, GetKeys)
from ._thor_ops import (CusBatchMatMul, CusCholeskyTrsm, CusFusedAbsMax1, CusImg2Col, CusMatMulCubeDenseLeft,
                        CusMatMulCubeFraczRightMul, CusMatMulCube, CusMatrixCombine, CusTranspose02314,
                        CusMatMulCubeDenseRight, CusMatMulCubeFraczLeftCast, Im2Col, NewIm2Col,
                        LoadIm2Col, UpdateThorGradient, Cholesky, CholeskyTrsm,
                        DetTriangle, ProdForceSeA)
from .sparse_ops import (SparseToDense, SparseTensorDenseMatmul)
from ._embedding_cache_ops import (CacheSwapTable, UpdateCache, MapCacheIdx, SubAndFilter,
                                   MapUniform, DynamicAssign, PadAndShift)
from .sponge_ops import (BondForce, BondEnergy, BondAtomEnergy, BondForceWithAtomEnergy, BondForceWithAtomVirial,
                         DihedralForce, DihedralEnergy, DihedralAtomEnergy, DihedralForceWithAtomEnergy, AngleForce,
                         AngleEnergy, AngleAtomEnergy, AngleForceWithAtomEnergy, PMEReciprocalForce,
                         LJForce, LJEnergy, LJForceWithPMEDirectForce, PMEExcludedForce, PMEEnergy, Dihedral14LJForce,
                         Dihedral14LJForceWithDirectCF, Dihedral14LJEnergy, Dihedral14LJCFForceWithAtomEnergy,
                         Dihedral14LJAtomEnergy, Dihedral14CFEnergy, Dihedral14CFAtomEnergy,
                         MDTemperature, MDIterationLeapFrogLiujian,
                         CrdToUintCrd, MDIterationSetupRandState, TransferCrd, FFT3D, IFFT3D, NeighborListUpdate)
from .sponge_update_ops import (ConstrainForceCycleWithVirial, RefreshUintCrd, LastCrdToDr, RefreshCrdVel,
                                CalculateNowrapCrd, RefreshBoxmapTimes, Totalc6get, CrdToUintCrdQuarter,
                                MDIterationLeapFrogLiujianWithMaxVel, GetCenterOfMass, MapCenterOfMass,
                                NeighborListRefresh, MDIterationLeapFrog, MDIterationLeapFrogWithMaxVel,
                                MDIterationGradientDescent, BondForceWithAtomEnergyAndVirial, ConstrainForceCycle,
                                LJForceWithVirialEnergy, LJForceWithPMEDirectForceUpdate, PMEReciprocalForceUpdate,
                                PMEExcludedForceUpdate, LJForceWithVirialEnergyUpdate,
                                Dihedral14ForceWithAtomEnergyVirial, PMEEnergyUpdate,
                                ConstrainForceVirial, ConstrainForce, Constrain)
from .rl_ops import (BufferAppend, BufferGetItem, BufferSample)
from ._inner_ops import (MatmulDDS, DSDMatmul, NonZero)
from .custom_ops import (Custom)

__all__ = [
    'CeLU',
    'Ger',
    'Unique',
    'ReverseSequence',
    'Sort',
    'EditDistance',
    'CropAndResize',
    'Add',
    'TensorAdd',
    'Argmax',
    'Argmin',
    'MaxPool3D',
    'AvgPool3D',
    'ArgMaxWithValue',
    'ArgMinWithValue',
    'AddN',
    'AccumulateNV2',
    'Sub',
    'CumSum',
    'MatMul',
    'BatchMatMul',
    'Mul',
    'MaskedFill',
    'MaskedSelect',
    'Meshgrid',
    'Pow',
    'Exp',
    'Expm1',
    'Rsqrt',
    'Sqrt',
    'Square',
    'DynamicGRUV2',
    'SquaredDifference',
    'Xdivy',
    'Xlogy',
    'Conv2D',
    'Conv3D',
    'Conv2DTranspose',
    'Conv3DTranspose',
    'Flatten',
    'MaxPoolWithArgmax',
    'BNTrainingReduce',
    'BNTrainingUpdate',
    'BatchNorm',
    'MaxPool',
    'TopK',
    'LinSpace',
    'Adam',
    'AdamWeightDecay',
    'FusedCastAdamWeightDecay',
    'FusedSparseAdam',
    'FusedSparseLazyAdam',
    'AdamNoUpdateParam',
    'Softplus',
    'Softmax',
    'Softsign',
    'LogSoftmax',
    'SoftmaxCrossEntropyWithLogits',
    'BCEWithLogitsLoss',
    'ROIAlign',
    'SparseSoftmaxCrossEntropyWithLogits',
    'NLLLoss',
    'SGD',
    'ApplyMomentum',
    "ApplyKerasMomentum",
    'FusedWeightScaleApplyMomentum',
    'ExpandDims',
    'Cast',
    'IsSubClass',
    'IsInstance',
    'Reshape',
    'Squeeze',
    'Transpose',
    'OneHot',
    'GatherV2',
    'Gather',
    'SparseGatherV2',
    'EmbeddingLookup',
    'Padding',
    'GatherD',
    'Identity',
    'UniqueWithPad',
    'Concat',
    'Pack',
    'NonMaxSuppressionV3',
    'Stack',
    'Unpack',
    'Unstack',
    'Tile',
    'BiasAdd',
    'GeLU',
    'Gelu',
    'FastGeLU',
    'FastGelu',
    'Minimum',
    'Maximum',
    'StridedSlice',
    'ReduceSum',
    'ReduceMean',
    'LayerNorm',
    'Rank',
    'Lerp',
    'Less',
    'LessEqual',
    'RealDiv',
    'Div',
    'DivNoNan',
    'Inv',
    'Invert',
    'TruncatedNormal',
    'Fill',
    'Ones',
    'Zeros',
    'OnesLike',
    'ZerosLike',
    'Select',
    'Split',
    'SplitV',
    'Mish',
    'SeLU',
    'MulNoNan',
    'ReLU',
    'ReLU6',
    'Elu',
    'Erf',
    "Erfinv",
    'Erfc',
    'Sigmoid',
    'HSwish',
    'HSigmoid',
    'Tanh',
    'NoRepeatNGram',
    'Randperm',
    'RandomChoiceWithMask',
    'StandardNormal',
    'Multinomial',
    'Gamma',
    'Poisson',
    'UniformInt',
    'UniformReal',
    'StandardLaplace',
    'RandomCategorical',
    'ResizeBilinear',
    'ScalarSummary',
    'ImageSummary',
    'TensorSummary',
    'HistogramSummary',
    "Print",
    "Assert",
    'InsertGradientOf',
    'HookBackward',
    'InvertPermutation',
    'Shape',
    'DynamicShape',
    'DropoutDoMask',
    'DropoutGenMask',
    'Dropout',
    'Dropout2D',
    'Dropout3D',
    'Neg',
    'InplaceAdd',
    'InplaceSub',
    'Slice',
    'DType',
    'NPUAllocFloatStatus',
    'NPUGetFloatStatus',
    'NPUClearFloatStatus',
    'IsNan',
    'IsFinite',
    'IsInf',
    'Addcdiv',
    'Addcmul',
    'FloatStatus',
    'Reciprocal',
    'SmoothL1Loss',
    'SoftMarginLoss',
    'L2Loss',
    'CTCLoss',
    'CTCGreedyDecoder',
    'RNNTLoss',
    'DynamicRNN',
    'ReduceAll',
    'ReduceAny',
    'ScalarToArray',
    'ScalarToTensor',
    'TupleToArray',
    'GeSwitch',
    'Merge',
    'SameTypeShape',
    'CheckBprop',
    'CheckValid',
    'BoundingBoxEncode',
    'BoundingBoxDecode',
    'L2Normalize',
    'ScatterAdd',
    'ScatterSub',
    'ScatterMul',
    'ScatterDiv',
    'ScatterNd',
    'ScatterMax',
    'ScatterMin',
    'ScatterNdAdd',
    'ScatterNdSub',
    'ScatterNonAliasingAdd',
    'ReverseV2',
    'Rint',
    'ResizeNearestNeighbor',
    'HistogramFixedWidth',
    'Pad',
    'MirrorPad',
    'GatherNd',
    'TensorScatterUpdate',
    'TensorScatterAdd',
    'ScatterUpdate',
    'ScatterNdUpdate',
    'Floor',
    'NMSWithMask',
    'IOU',
    'Partial',
    'MakeRefKey',
    'Depend',
    'UpdateState',
    'identity',
    'AvgPool',
    # Back Primitive
    'Equal',
    'EqualCount',
    'NotEqual',
    'Greater',
    'GreaterEqual',
    'LogicalNot',
    'LogicalAnd',
    'LogicalOr',
    'Size',
    'DepthwiseConv2dNative',
    'UnsortedSegmentSum',
    'UnsortedSegmentMin',
    'UnsortedSegmentMax',
    'UnsortedSegmentProd',
    "AllGather",
    "AllReduce",
    "_AllSwap",
    "ReduceScatter",
    "Broadcast",
    "ReduceOp",
    'ScalarCast',
    'GetNext',
    'ReduceMax',
    'ReduceMin',
    'ReduceProd',
    'CumProd',
    'Cdist',
    'Log',
    'Log1p',
    'SigmoidCrossEntropyWithLogits',
    'FloorDiv',
    'FloorMod',
    'TruncateDiv',
    'TruncateMod',
    'Ceil',
    'Acosh',
    'Asinh',
    "PReLU",
    "Cos",
    "Cosh",
    "ACos",
    "Diag",
    "DiagPart",
    'Eye',
    'Assign',
    'AssignAdd',
    'AssignSub',
    "Sin",
    "Sinh",
    "Asin",
    "LSTM",
    "Abs",
    "BinaryCrossEntropy",
    "KLDivLoss",
    "SparseApplyAdagrad",
    "SparseApplyAdagradV2",
    "SpaceToDepth",
    "DepthToSpace",
    "Conv2DBackpropInput",
    "ComputeAccidentalHits",
    "Sign",
    "LpNorm",
    "LARSUpdate",
    "Round",
    "Eps",
    "ApplyFtrl",
    "SpaceToBatch",
    "SparseApplyFtrl",
    "SparseApplyFtrlV2",
    "FusedSparseFtrl",
    "ApplyProximalAdagrad",
    "SparseApplyProximalAdagrad",
    "FusedSparseProximalAdagrad",
    "SparseApplyRMSProp",
    "ApplyAdaMax",
    "ApplyAdadelta",
    "ApplyAdagrad",
    "ApplyAdagradV2",
    "ApplyAdagradDA",
    "ApplyAdamWithAmsgrad",
    "ApplyAddSign",
    "ApplyPowerSign",
    "ApplyGradientDescent",
    "ApplyProximalGradientDescent",
    "BatchToSpace",
    "Atan2",
    "ApplyRMSProp",
    "ApplyCenteredRMSProp",
    "SpaceToBatchND",
    "BatchToSpaceND",
    "SquareSumAll",
    "BitwiseAnd",
    "BitwiseOr",
    "BitwiseXor",
    "BesselI0",
    "BesselI1",
    "BesselI0e",
    "BesselI1e",
    "Atan",
    "Atanh",
    "Tan",
    "BasicLSTMCell",
    "BroadcastTo",
    "DataFormatDimMap",
    "ApproximateEqual",
    "InplaceUpdate",
    "InTopK",
    "UniformCandidateSampler",
    "LogUniformCandidateSampler",
    "LRN",
    "Mod",
    "ConfusionMatrix",
    "PopulationCount",
    "ParallelConcat",
    "Push",
    "Pull",
    "PullWeight",
    "PushWeight",
    "ReLUV2",
    "SparseToDense",
    "SparseTensorDenseMatmul",
    "SparseApplyAdadelta",
    "MatrixInverse",
    "Range",
    "SearchSorted",
    "IndexAdd",
    "AdaptiveAvgPool2D",
    "TensorScatterMax",
    "TensorScatterMin",
    "TensorScatterSub",
    "ScatterElements",
    "NonZero",
    "SoftShrink",
    "FFT3D",
    "IFFT3D",
    "HShrink",
    "PyFunc",
    "BufferAppend",
    "BufferGetItem",
    "BufferSample",
    "Erfinv",
    "Conj",
    "Real",
    "Imag",
    "Trunc",
    "Complex",
    "ExtractVolumePatches",
    "NeighborExchangeV2",
    "NeighborExchange",
    "AlltoAll",
    "Custom",
]

__sponge__ = [
    "BondForce",
    "BondEnergy",
    "BondAtomEnergy",
    "BondForceWithAtomEnergy",
    "BondForceWithAtomVirial",
    "DihedralForce",
    "DihedralEnergy",
    "DihedralAtomEnergy",
    "DihedralForceWithAtomEnergy",
    "AngleForce",
    "AngleEnergy",
    "AngleAtomEnergy",
    "AngleForceWithAtomEnergy",
    'PMEReciprocalForce',
    'LJForce',
    'LJForceWithPMEDirectForce',
    'LJEnergy',
    'PMEExcludedForce',
    'PMEEnergy',
    "Dihedral14LJForce",
    "Dihedral14LJEnergy",
    "Dihedral14LJForceWithDirectCF",
    "Dihedral14LJCFForceWithAtomEnergy",
    "Dihedral14LJAtomEnergy",
    "Dihedral14CFEnergy",
    "MDIterationLeapFrog",
    "Dihedral14CFAtomEnergy",
    "MDTemperature",
    "NeighborListUpdate",
    "MDIterationLeapFrogLiujian",
    "CrdToUintCrd",
    "MDIterationSetupRandState",
    "TransferCrd",
    # Update
    "ConstrainForceCycleWithVirial",
    "RefreshUintCrd",
    "LastCrdToDr",
    "RefreshCrdVel",
    "CalculateNowrapCrd",
    "RefreshBoxmapTimes",
    "Totalc6get",
    "CrdToUintCrdQuarter",
    "MDIterationLeapFrogLiujianWithMaxVel",
    "GetCenterOfMass",
    "MapCenterOfMass",
    "NeighborListRefresh",
    "MDIterationLeapFrog",
    "MDIterationLeapFrogWithMaxVel",
    "MDIterationGradientDescent",
    "BondForceWithAtomEnergyAndVirial",
    "ConstrainForceCycle",
    "LJForceWithVirialEnergy",
    "LJForceWithPMEDirectForceUpdate",
    "PMEReciprocalForceUpdate",
    "PMEExcludedForceUpdate",
    "LJForceWithVirialEnergyUpdate",
    "Dihedral14ForceWithAtomEnergyVirial",
    "PMEEnergyUpdate",
    "ConstrainForceVirial",
    "ConstrainForce",
    "Constrain",
]

__all__.extend(__sponge__)

__all__.sort()
