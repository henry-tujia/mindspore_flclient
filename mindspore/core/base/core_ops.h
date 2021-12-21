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

#ifndef MINDSPORE_CORE_BASE_CORE_OPS_H_
#define MINDSPORE_CORE_BASE_CORE_OPS_H_

#include <iostream>
#include <string>
#include <memory>
#include "ir/anf.h"
#include "ir/primitive.h"
#include "utils/flags.h"
#include "utils/hash_map.h"

namespace mindspore {
namespace prim {
inline const ValuePtr kValueOne = std::make_shared<Int64Imm>(1);
inline const mindspore::HashMap<std::string, ValuePtr> kSideEffectPropagate = {
  {mindspore::GRAPH_FLAG_SIDE_EFFECT_PROPAGATE, kValueOne},
};

constexpr auto kGetNext = "GetNext";
constexpr auto kGather = "Gather";
constexpr auto kCdist = "Cdist";
constexpr auto kCdistGrad = "CdistGrad";
// Arithmetic
constexpr auto kScalarAdd = "ScalarAdd";
constexpr auto kScalarSub = "ScalarSub";
constexpr auto kScalarMul = "ScalarMul";
constexpr auto kScalarDiv = "ScalarDiv";
constexpr auto kScalarFloordiv = "ScalarFloordiv";
constexpr auto kScalarMod = "ScalarMod";
constexpr auto kScalarPow = "ScalarPow";
constexpr auto kScalarTrunc = "ScalarTrunc";
constexpr auto kScalarFloor = "ScalarFloor";
constexpr auto kScalarUadd = "ScalarUadd";
constexpr auto kScalarUsub = "ScalarUsub";
constexpr auto kExp = "Exp";
constexpr auto kEqual = "Equal";
constexpr auto kNotEqual = "NotEqual";
constexpr auto kNeg = "Neg";
constexpr auto kSub = "Sub";
constexpr auto kMul = "Mul";
constexpr auto kMulNoNan = "MulNoNan";
constexpr auto kRealDiv = "RealDiv";
constexpr auto kReciprocal = "Reciprocal";
constexpr auto kLog = "Log";
constexpr auto kSelect = "Select";
constexpr auto kAdd = "Add";
constexpr auto kBiasAdd = "BiasAdd";
constexpr auto kTile = "Tile";
constexpr auto kBiasAddGrad = "BiasAddGrad";
constexpr auto kCos = "Cos";
constexpr auto kAbs = "Abs";
constexpr auto kTrunc = "Trunc";
constexpr auto kLpNorm = "LpNorm";
constexpr auto kSquare = "Square";
constexpr auto kRealInner = "RealInner";
constexpr auto kReal = "Real";
constexpr auto kImag = "Imag";
constexpr auto kConj = "Conj";
constexpr auto kGer = "Ger";

// Arrays
constexpr auto kDynamicShape = "DynamicShape";
constexpr auto kStack = "Stack";
constexpr auto kUnstack = "Unstack";
constexpr auto kTupleGetItem = "TupleGetItem";
constexpr auto kSliceGetItem = "SliceGetItem";
constexpr auto kGeLU = "GeLU";
constexpr auto kGLU = "GLU";
constexpr auto kReLU = "ReLU";
constexpr auto kReLU6 = "ReLU6";
constexpr auto kReLUV2 = "ReLUV2";
constexpr auto kReLUGrad = "ReluGrad";
constexpr auto kReLUGradV2 = "ReluGradV2";
constexpr auto kRint = "Rint";
constexpr auto kGeLUGrad = "GeLUGrad";
constexpr auto kFastGeLU = "FastGeLU";
constexpr auto kFastGeLUGrad = "FastGeLUGrad";
constexpr auto kStridedSlice = "StridedSlice";
constexpr auto kStridedSliceGrad = "StridedSliceGrad";
constexpr auto kZerosLike = "ZerosLike";
constexpr auto kOnes = "Ones";
constexpr auto kOnesLike = "OnesLike";
constexpr auto kDiag = "Diag";
constexpr auto kDiagPart = "DiagPart";
constexpr auto kDynamicBroadcastGradientArgs = "DynamicBroadcastGradientArgs";
constexpr auto kTranspose = "Transpose";
constexpr auto kSplitV = "SplitV";
constexpr auto kDynamicBroadcastTo = "DynamicBroadcastTo";
constexpr auto kDynamicReshape = "DynamicReshape";

// NN
constexpr auto kCTCLoss = "CTCLoss";
constexpr auto kLayerNorm = "LayerNorm";
constexpr auto kLayerNormGrad = "LayerNormGrad";
constexpr auto kDropoutGenMask = "DropoutGenMask";
constexpr auto kDropoutDoMask = "DropoutDoMask";
constexpr auto kDropout = "Dropout";
constexpr auto kDropoutGrad = "DropoutGrad";
constexpr auto kConv2DTranspose = "Conv2DTranspose";
constexpr auto kSparseApplyAdadelta = "SparseApplyAdadelta";
constexpr auto kRoll = "Roll";
constexpr auto kTanh = "Tanh";

// Here list all primitives used in backend or some special primitives used by core.
// GetNext
inline const PrimitivePtr kPrimGetNext = std::make_shared<Primitive>(kGetNext);

// Arithmetic
inline const PrimitivePtr kPrimScalarAdd = std::make_shared<Primitive>(kScalarAdd);
inline const PrimitivePtr kPrimScalarSub = std::make_shared<Primitive>(kScalarSub);
inline const PrimitivePtr kPrimScalarMul = std::make_shared<Primitive>(kScalarMul);
inline const PrimitivePtr kPrimScalarDiv = std::make_shared<Primitive>(kScalarDiv);
inline const PrimitivePtr kPrimScalarFloordiv = std::make_shared<Primitive>(kScalarFloordiv);
inline const PrimitivePtr kPrimScalarMod = std::make_shared<Primitive>(kScalarMod);
inline const PrimitivePtr kPrimScalarPow = std::make_shared<Primitive>(kScalarPow);
inline const PrimitivePtr kPrimScalarTrunc = std::make_shared<Primitive>(kScalarTrunc);
inline const PrimitivePtr kPrimScalarFloor = std::make_shared<Primitive>(kScalarFloor);
inline const PrimitivePtr kPrimScalarUadd = std::make_shared<Primitive>(kScalarUadd);
inline const PrimitivePtr kPrimScalarUsub = std::make_shared<Primitive>(kScalarUsub);
inline const PrimitivePtr kPrimScalarExp = std::make_shared<Primitive>("scalar_exp");
inline const PrimitivePtr kPrimScalarLog = std::make_shared<Primitive>("scalar_log");
inline const PrimitivePtr kPrimScalarSin = std::make_shared<Primitive>("scalar_sin");
inline const PrimitivePtr kPrimScalarCos = std::make_shared<Primitive>("scalar_cos");
inline const PrimitivePtr kPrimScalarTan = std::make_shared<Primitive>("scalar_tan");
inline const PrimitivePtr kPrimTrunc = std::make_shared<Primitive>(kTrunc);

// Comparisons
inline const PrimitivePtr kPrimScalarEq = std::make_shared<Primitive>("scalar_eq");
inline const PrimitivePtr kPrimScalarLt = std::make_shared<Primitive>("scalar_lt");
inline const PrimitivePtr kPrimScalarGt = std::make_shared<Primitive>("scalar_gt");
inline const PrimitivePtr kPrimScalarNe = std::make_shared<Primitive>("scalar_ne");
inline const PrimitivePtr kPrimScalarLe = std::make_shared<Primitive>("scalar_le");
inline const PrimitivePtr kPrimScalarGe = std::make_shared<Primitive>("scalar_ge");
inline const PrimitivePtr kPrimBoolNot = std::make_shared<Primitive>("bool_not");
inline const PrimitivePtr kPrimBoolAnd = std::make_shared<Primitive>("bool_and");
inline const PrimitivePtr kPrimBoolOr = std::make_shared<Primitive>("bool_or");
inline const PrimitivePtr kPrimBoolEq = std::make_shared<Primitive>("bool_eq");
inline const PrimitivePtr kPrimGreater = std::make_shared<Primitive>("Greater");
inline const PrimitivePtr kPrimGreaterEqual = std::make_shared<Primitive>("GreaterEqual");
inline const PrimitivePtr kPrimLess = std::make_shared<Primitive>("Less");
inline const PrimitivePtr kPrimLessEqual = std::make_shared<Primitive>("LessEqual");
inline const PrimitivePtr kPrimEqual = std::make_shared<Primitive>(kEqual);
inline const PrimitivePtr kPrimNotEqual = std::make_shared<Primitive>(kNotEqual);
inline const PrimitivePtr kPrimLogicalAnd = std::make_shared<Primitive>("LogicalAnd");
inline const PrimitivePtr kPrimLogicalOr = std::make_shared<Primitive>("LogicalOr");
inline const PrimitivePtr kPrimLogicalNot = std::make_shared<Primitive>("LogicalNot");
inline const PrimitivePtr kPrimEqualCount = std::make_shared<Primitive>("EqualCount");

inline const PrimitivePtr kPrimDistribute = std::make_shared<Primitive>("distribute");
inline const PrimitivePtr kPrimIm2Col = std::make_shared<Primitive>("im2col");
inline const PrimitivePtr kPrimCol2Im = std::make_shared<Primitive>("col2im");
inline const PrimitivePtr kPrimIm2ColV1 = std::make_shared<Primitive>("im2col_v1");
inline const PrimitivePtr kPrimCol2ImV1 = std::make_shared<Primitive>("col2im_v1");

inline const PrimitivePtr kPrimLabelGoto = std::make_shared<Primitive>("LabelGoto");
inline const PrimitivePtr kPrimLabelSwitch = std::make_shared<Primitive>("LabelSwitch");
inline const PrimitivePtr kPrimLabelSet = std::make_shared<Primitive>("LabelSet");

// Stack ops
inline const PrimitivePtr kPrimStackInit = std::make_shared<Primitive>("StackInit");
inline const PrimitivePtr kPrimStackDestroy = std::make_shared<Primitive>("StackDestroy");
inline const PrimitivePtr kPrimStackPush = std::make_shared<Primitive>("StackPush");
inline const PrimitivePtr kPrimStackPop = std::make_shared<Primitive>("StackPop");

// Arrays
inline const PrimitivePtr kPrimDynamicBroadcastTo = std::make_shared<Primitive>(kDynamicBroadcastTo);
inline const PrimitivePtr kPrimCummin = std::make_shared<Primitive>("Cummin");
inline const PrimitivePtr kPrimBroadcastTo = std::make_shared<Primitive>("BroadcastTo");
inline const PrimitivePtr kPrimScalarToArray = std::make_shared<Primitive>("scalar_to_array");
inline const PrimitivePtr kPrimTopK = std::make_shared<Primitive>("TopK");
inline const PrimitivePtr kPrimArrayToScalar = std::make_shared<Primitive>("array_to_scalar");
inline const PrimitivePtr kPrimBroadcastShape = std::make_shared<Primitive>("broadcast_shape");
inline const PrimitivePtr kPrimArrayMap = std::make_shared<Primitive>("array_map");
inline const PrimitivePtr kPrimArrayReduce = std::make_shared<Primitive>("array_reduce");
inline const PrimitivePtr kPrimCast = std::make_shared<Primitive>("Cast");
inline const PrimitivePtr kPrimConcat = std::make_shared<Primitive>("Concat");
inline const PrimitivePtr kPrimSqueeze = std::make_shared<Primitive>("Squeeze");
inline const PrimitivePtr kPrimUnsqueeze = std::make_shared<Primitive>("Unsqueeze");
inline const PrimitivePtr kPrimTranspose = std::make_shared<Primitive>(kTranspose);
inline const PrimitivePtr kPrimGatherV2 = std::make_shared<Primitive>("GatherV2");
inline const PrimitivePtr kPrimGatherD = std::make_shared<Primitive>("GatherD");
inline const PrimitivePtr kPrimGather = std::make_shared<Primitive>("Gather");
inline const PrimitivePtr kPrimGatherNd = std::make_shared<Primitive>("GatherNd");
inline const PrimitivePtr kPrimSparseGatherV2 = std::make_shared<Primitive>("SparseGatherV2");
inline const PrimitivePtr kPrimSparseToDense = std::make_shared<Primitive>("SparseToDense");
inline const PrimitivePtr kPrimShape = std::make_shared<Primitive>("Shape");
inline const PrimitivePtr kPrimStridedSlice = std::make_shared<Primitive>(kStridedSlice);
inline const PrimitivePtr kPrimStridedSliceGrad = std::make_shared<Primitive>(kStridedSliceGrad);
inline const PrimitivePtr kPrimDynamicShape = std::make_shared<Primitive>(kDynamicShape);
inline const PrimitivePtr kPrimEmbeddingLookup = std::make_shared<Primitive>("EmbeddingLookup");
inline const PrimitivePtr kPrimEmbeddingLookupCommGrad = std::make_shared<Primitive>("EmbeddingLookupCommGrad");
inline const PrimitivePtr kPrimSize = std::make_shared<Primitive>("Size");
inline const PrimitivePtr kPrimArgMax = std::make_shared<Primitive>("Argmax");
inline const PrimitivePtr kPrimArgMin = std::make_shared<Primitive>("Argmin");
inline const PrimitivePtr kPrimPack = std::make_shared<Primitive>("Pack");
inline const PrimitivePtr kPrimStack = std::make_shared<Primitive>(kStack);
inline const PrimitivePtr kPrimUnpack = std::make_shared<Primitive>("Unpack");
inline const PrimitivePtr kPrimUnstack = std::make_shared<Primitive>(kUnstack);
inline const PrimitivePtr kPrimUnsortedSegmentMax = std::make_shared<Primitive>("UnsortedSegmentMax");
inline const PrimitivePtr kPrimUnsortedSegmentSum = std::make_shared<Primitive>("UnsortedSegmentSum");
inline const PrimitivePtr kPrimUnsortedSegmentMin = std::make_shared<Primitive>("UnsortedSegmentMin");
inline const PrimitivePtr kPrimConcatOffset = std::make_shared<Primitive>("ConcatOffset");
inline const PrimitivePtr kPrimReshape = std::make_shared<Primitive>("Reshape");
inline const PrimitivePtr kPrimDynamicReshape = std::make_shared<Primitive>(kDynamicReshape);
inline const PrimitivePtr kPrimSubAndFilter = std::make_shared<Primitive>("SubAndFilter");
inline const PrimitivePtr kPrimMapCacheIdx = std::make_shared<Primitive>("MapCacheIdx");
inline const PrimitivePtr kPrimUpdateCache = std::make_shared<Primitive>("UpdateCache");
inline const PrimitivePtr kPrimComputeAccidentalHits = std::make_shared<Primitive>("ComputeAccidentalHits");
inline const PrimitivePtr kPrimCacheSwapTable = std::make_shared<Primitive>("CacheSwapTable");
inline const PrimitivePtr kPrimDynamicAssign = std::make_shared<Primitive>("DynamicAssign");
inline const PrimitivePtr kPrimPadAndShift = std::make_shared<Primitive>("PadAndShift");
inline const PrimitivePtr kPrimSlice = std::make_shared<Primitive>("Slice");
inline const PrimitivePtr kPrimSliceGrad = std::make_shared<Primitive>("SliceGrad");
inline const PrimitivePtr kPrimSliceFusion = std::make_shared<Primitive>("SliceFusion");
inline const PrimitivePtr kPrimTile = std::make_shared<Primitive>(kTile);
inline const PrimitivePtr kPrimAddN = std::make_shared<Primitive>("AddN");
inline const PrimitivePtr kPrimAccumulateNV2 = std::make_shared<Primitive>("AccumulateNV2");
inline const PrimitivePtr kPrimTransData = std::make_shared<Primitive>("TransData");
inline const PrimitivePtr kPrimTransDataRNN = std::make_shared<Primitive>("TransDataRNN");
inline const PrimitivePtr kPrimNMSWithMask = std::make_shared<Primitive>("NMSWithMask");
inline const PrimitivePtr kPrimPad = std::make_shared<Primitive>("Pad");
inline const PrimitivePtr kPrimArgMaxWithValue = std::make_shared<Primitive>("ArgMaxWithValue");
inline const PrimitivePtr kPrimUnique = std::make_shared<Primitive>("Unique");
inline const PrimitivePtr kPrimUniqueGrad = std::make_shared<Primitive>("UniqueGrad");
inline const PrimitivePtr kPrimExtractImagePatches = std::make_shared<Primitive>("ExtractImagePatches");
inline const PrimitivePtr kPrimDynamicRNN = std::make_shared<Primitive>("DynamicRNN");
inline const PrimitivePtr kPrimDynamicRNNGrad = std::make_shared<Primitive>("DynamicRNNGrad");
inline const PrimitivePtr kPrimDynamicGRUV2 = std::make_shared<Primitive>("DynamicGRUV2");
inline const PrimitivePtr kPrimDynamicGRUV2Grad = std::make_shared<Primitive>("DynamicGRUV2Grad");
inline const PrimitivePtr kPrimScatterAdd = std::make_shared<Primitive>("ScatterAdd");
inline const PrimitivePtr kPrimScatterSub = std::make_shared<Primitive>("ScatterSub");
inline const PrimitivePtr kPrimScatterMul = std::make_shared<Primitive>("ScatterMul");
inline const PrimitivePtr kPrimScatterDiv = std::make_shared<Primitive>("ScatterDiv");
inline const PrimitivePtr kPrimScatterMax = std::make_shared<Primitive>("ScatterMax");
inline const PrimitivePtr kPrimScatterMin = std::make_shared<Primitive>("ScatterMin");
inline const PrimitivePtr kPrimScatterNdAdd = std::make_shared<Primitive>("ScatterNdAdd");
inline const PrimitivePtr kPrimScatterUpdate = std::make_shared<Primitive>("ScatterUpdate");
inline const PrimitivePtr kPrimScatterElements = std::make_shared<Primitive>("ScatterElements");
inline const PrimitivePtr kPrimTensorCopySlices = std::make_shared<Primitive>("TensorCopySlices");
inline const PrimitivePtr kPrimMapUniform = std::make_shared<Primitive>("MapUniform");
inline const PrimitivePtr kPrimSplit = std::make_shared<Primitive>("Split");
inline const PrimitivePtr kPrimSplitV = std::make_shared<Primitive>(kSplitV);
inline const PrimitivePtr kPrimSequenceMask = std::make_shared<Primitive>("SequenceMask");
inline const PrimitivePtr kPrimRange = std::make_shared<Primitive>("Range");
inline const PrimitivePtr kPrimSpaceToBatchND = std::make_shared<Primitive>("SpaceToBatchND");
inline const PrimitivePtr kPrimBatchToSpaceND = std::make_shared<Primitive>("BatchToSpaceND");
inline const PrimitivePtr kPrimDepthToSpace = std::make_shared<Primitive>("DepthToSpace");
inline const PrimitivePtr kPrimBatchToSpace = std::make_shared<Primitive>("BatchToSpace");
inline const PrimitivePtr kPrimSpaceToBatch = std::make_shared<Primitive>("SpaceToBatch");
inline const PrimitivePtr kPrimScatterNd = std::make_shared<Primitive>("ScatterNd");
inline const PrimitivePtr kPrimScatterNdUpdate = std::make_shared<Primitive>("ScatterNdUpdate");
inline const PrimitivePtr kPrimScatterNonAliasingAdd = std::make_shared<Primitive>("ScatterNonAliasingAdd");
inline const PrimitivePtr kPrimConstantOfShape = std::make_shared<Primitive>("ConstantOfShape");
inline const PrimitivePtr kPrimSquaredDifference = std::make_shared<Primitive>("SquaredDifference");
inline const PrimitivePtr kPrimReverseV2 = std::make_shared<Primitive>("ReverseV2");
inline const PrimitivePtr kPrimReverseSequence = std::make_shared<Primitive>("ReverseSequence");
inline const PrimitivePtr kPrimRank = std::make_shared<Primitive>("Rank");
inline const PrimitivePtr kPrimResizeBilinear = std::make_shared<Primitive>("ResizeBilinear");
inline const PrimitivePtr kPrimResizeGrad = std::make_shared<Primitive>("ResizeGrad");
inline const PrimitivePtr kPrimResizeNearestNeighbor = std::make_shared<Primitive>("ResizeNearestNeighbor");
inline const PrimitivePtr kPrimSort = std::make_shared<Primitive>("Sort");
inline const PrimitivePtr kPrimMaskedFill = std::make_shared<Primitive>("MaskedFill");
inline const PrimitivePtr kPrimMaskedSelect = std::make_shared<Primitive>("MaskedSelect");
inline const PrimitivePtr kPrimDiag = std::make_shared<Primitive>(kDiag);
inline const PrimitivePtr kPrimDiagPart = std::make_shared<Primitive>(kDiagPart);
inline const PrimitivePtr kPrimNonZero = std::make_shared<Primitive>("NonZero");
inline const PrimitivePtr kPrimRealInner = std::make_shared<Primitive>(kRealInner);
inline const PrimitivePtr kPrimReal = std::make_shared<Primitive>(kReal);
inline const PrimitivePtr kPrimImag = std::make_shared<Primitive>(kImag);
inline const PrimitivePtr kPrimConj = std::make_shared<Primitive>(kConj);
inline const PrimitivePtr kPrimExtractVolumePatches = std::make_shared<Primitive>("ExtractVolumePatches");

// NN
inline const PrimitivePtr kPrimCeLU = std::make_shared<Primitive>("CeLU");
inline const PrimitivePtr kPrimAdam = std::make_shared<Primitive>("Adam");
inline const PrimitivePtr kPrimApplyAdaMax = std::make_shared<Primitive>("ApplyAdaMax");
inline const PrimitivePtr kPrimAudioSpectrogram = std::make_shared<Primitive>("AudioSpectrogram");
inline const PrimitivePtr kPrimFlatten = std::make_shared<Primitive>("Flatten");
inline const PrimitivePtr kPrimCrop = std::make_shared<Primitive>("Crop");
inline const PrimitivePtr kPrimFlattenGrad = std::make_shared<Primitive>("FlattenGrad");
inline const PrimitivePtr kPrimSoftmax = std::make_shared<Primitive>("Softmax");
inline const PrimitivePtr kPrimSparseSoftmaxCrossEntropy = std::make_shared<Primitive>("SparseSoftmaxCrossEntropy");
inline const PrimitivePtr kPrimLogSoftmax = std::make_shared<Primitive>("LogSoftmax");
inline const PrimitivePtr kPrimLogSoftmaxGrad = std::make_shared<Primitive>("LogSoftmaxGrad");
inline const PrimitivePtr kPrimLstm = std::make_shared<Primitive>("LSTM");
inline const PrimitivePtr kPrimTan = std::make_shared<Primitive>("Tan");
inline const PrimitivePtr kPrimAtan2 = std::make_shared<Primitive>("Atan2");
inline const PrimitivePtr kPrimAtan = std::make_shared<Primitive>("Atan");
inline const PrimitivePtr kPrimAsin = std::make_shared<Primitive>("Asin");
inline const PrimitivePtr kPrimSinh = std::make_shared<Primitive>("Sinh");
inline const PrimitivePtr kPrimCosh = std::make_shared<Primitive>("Cosh");
inline const PrimitivePtr kPrimTanh = std::make_shared<Primitive>(kTanh);
inline const PrimitivePtr kPrimAsinh = std::make_shared<Primitive>("Asinh");
inline const PrimitivePtr kPrimAcosh = std::make_shared<Primitive>("Acosh");
inline const PrimitivePtr kPrimAtanh = std::make_shared<Primitive>("Atanh");
inline const PrimitivePtr kPrimApplyGradientDescent = std::make_shared<Primitive>("ApplyGradientDescent");
inline const PrimitivePtr kPrimBesselI0e = std::make_shared<Primitive>("BesselI0e");
inline const PrimitivePtr kPrimBesselI1e = std::make_shared<Primitive>("BesselI1e");
inline const PrimitivePtr kPrimTanhGrad = std::make_shared<Primitive>("TanhGrad");
inline const PrimitivePtr kPrimPooling = std::make_shared<Primitive>("Pooling");
inline const PrimitivePtr kPrimPoolingGrad = std::make_shared<Primitive>("PoolingGrad");
inline const PrimitivePtr kPrimROIPooling = std::make_shared<Primitive>("ROIPooling");
inline const PrimitivePtr kPrimMaxPool = std::make_shared<Primitive>("MaxPool");
inline const PrimitivePtr kPrimMaxPoolGrad = std::make_shared<Primitive>("MaxPoolGrad");
inline const PrimitivePtr kPrimMaxPoolWithArgmax = std::make_shared<Primitive>("MaxPoolWithArgmax");
inline const PrimitivePtr kPrimMaxPoolGradWithArgmax = std::make_shared<Primitive>("MaxPoolGradWithArgmax");
inline const PrimitivePtr kPrimApplyCenteredRMSProp = std::make_shared<Primitive>("ApplyCenteredRMSProp");
inline const PrimitivePtr kPrimAvgPool = std::make_shared<Primitive>("AvgPool");
inline const PrimitivePtr kPrimAvgPool3D = std::make_shared<Primitive>("AvgPool3D");
inline const PrimitivePtr kPrimAvgPoolGrad = std::make_shared<Primitive>("AvgPoolGrad");
inline const PrimitivePtr kPrimAvgPool3DGrad = std::make_shared<Primitive>("AvgPool3DGrad");
inline const PrimitivePtr kPrimAvgPoolGradVm = std::make_shared<Primitive>("AvgPoolGradVm");
inline const PrimitivePtr kPrimFusedSparseAdam = std::make_shared<Primitive>("FusedSparseAdam");
inline const PrimitivePtr kPrimFusedBatchNorm = std::make_shared<Primitive>("FusedBatchNorm");
inline const PrimitivePtr kPrimConv2D = std::make_shared<Primitive>("Conv2D");
inline const PrimitivePtr kPrimConv3D = std::make_shared<Primitive>("Conv3D");
inline const PrimitivePtr kPrimCTCLossV2 = std::make_shared<Primitive>("CTCLossV2");
inline const PrimitivePtr kPrimCTCLossV2Grad = std::make_shared<Primitive>("CTCLossV2Grad");
inline const PrimitivePtr kPrimCTCLoss = std::make_shared<Primitive>(kCTCLoss);
inline const PrimitivePtr kPrimFullConnection = std::make_shared<Primitive>("FullConnection");
inline const PrimitivePtr kPrimConv2DTranspose = std::make_shared<Primitive>(kConv2DTranspose);
inline const PrimitivePtr kPrimConv3DTranspose = std::make_shared<Primitive>("Conv3DTranspose");
inline const PrimitivePtr kPrimRoll = std::make_shared<Primitive>(kRoll);
inline const PrimitivePtr kPrimGroupConv2DGradInput = std::make_shared<Primitive>("GroupConv2DGradInput");
inline const PrimitivePtr kPrimBatchNorm = std::make_shared<Primitive>("BatchNorm");
inline const PrimitivePtr kPrimBatchNormGrad = std::make_shared<Primitive>("BatchNormGrad");
inline const PrimitivePtr kPrimSyncBatchNorm = std::make_shared<Primitive>("SyncBatchNorm");
inline const PrimitivePtr kPrimSyncBatchNormGrad = std::make_shared<Primitive>("SyncBatchNormGrad");
inline const PrimitivePtr kPrimBNTrainingReduceGrad = std::make_shared<Primitive>("BNTrainingReduceGrad");
inline const PrimitivePtr kPrimReluGrad = std::make_shared<Primitive>(kReLUGrad);
inline const PrimitivePtr kPrimReluGradV2 = std::make_shared<Primitive>("ReluGradV2");
inline const PrimitivePtr kPrimRelu6Grad = std::make_shared<Primitive>("ReLU6Grad");
inline const PrimitivePtr kPrimConv2DBackpropInput = std::make_shared<Primitive>("Conv2DBackpropInput");
inline const PrimitivePtr kPrimConv2DBackpropFilter = std::make_shared<Primitive>("Conv2DBackpropFilter");
inline const PrimitivePtr kPrimConv3DBackpropInput = std::make_shared<Primitive>("Conv3DBackpropInput");
inline const PrimitivePtr kPrimConv3DBackpropFilter = std::make_shared<Primitive>("Conv3DBackpropFilter");
inline const PrimitivePtr kPrimCustomNormalize = std::make_shared<Primitive>("CustomNormalize");
inline const PrimitivePtr kPrimDepthwiseConv2dNative = std::make_shared<Primitive>("DepthwiseConv2dNative");
inline const PrimitivePtr kPrimCTCGreedyDecoder = std::make_shared<Primitive>("CTCGreedyDecoder");
inline const PrimitivePtr kPrimDynamicStitch = std::make_shared<Primitive>("DynamicStitch");
inline const PrimitivePtr kPrimDepthwiseConv2dNativeBackpropFilter =
  std::make_shared<Primitive>("DepthwiseConv2dNativeBackpropFilter");
inline const PrimitivePtr kPrimDepthwiseConv2dNativeBackpropInput =
  std::make_shared<Primitive>("DepthwiseConv2dNativeBackpropInput");
inline const PrimitivePtr kPrimDetectionPostProcess = std::make_shared<Primitive>("DetectionPostProcess");
inline const PrimitivePtr kPrimBiasAddGrad = std::make_shared<Primitive>(kBiasAddGrad);
inline const PrimitivePtr kPrimBiasAdd = std::make_shared<Primitive>(kBiasAdd);
inline const PrimitivePtr kPrimBiasSubGrad = std::make_shared<Primitive>("BiasSubGrad");
inline const PrimitivePtr kPrimBinaryCrossEntropy = std::make_shared<Primitive>("BinaryCrossEntropy");
inline const PrimitivePtr kPrimBinaryCrossEntropyGrad = std::make_shared<Primitive>("BinaryCrossEntropyGrad");
inline const PrimitivePtr kPrimSmoothL1Loss = std::make_shared<Primitive>("SmoothL1Loss");
inline const PrimitivePtr kPrimSmoothL1LossGrad = std::make_shared<Primitive>("SmoothL1LossGrad");
inline const PrimitivePtr kPrimSoftMarginLoss = std::make_shared<Primitive>("SoftMarginLoss");
inline const PrimitivePtr kPrimSoftMarginLossGrad = std::make_shared<Primitive>("SoftMarginLossGrad");
inline const PrimitivePtr kPrimSoftmaxCrossEntropyWithLogits =
  std::make_shared<Primitive>("SoftmaxCrossEntropyWithLogits");
inline const PrimitivePtr kPrimL2Loss = std::make_shared<Primitive>("L2Loss");
inline const PrimitivePtr kPrimSigmoidCrossEntropyWithLogits =
  std::make_shared<Primitive>("SigmoidCrossEntropyWithLogits");
inline const PrimitivePtr kPrimSigmoidCrossEntropyWithLogitsGrad =
  std::make_shared<Primitive>("SigmoidCrossEntropyWithLogitsGrad");
inline const PrimitivePtr kPrimSparseSoftmaxCrossEntropyWithLogits =
  std::make_shared<Primitive>("SparseSoftmaxCrossEntropyWithLogits");
inline const PrimitivePtr kPrimMomentum = std::make_shared<Primitive>("Momentum");
inline const PrimitivePtr kPrimApplyMomentum = std::make_shared<Primitive>("ApplyMomentum");
inline const PrimitivePtr kPrimApplyFtrl = std::make_shared<Primitive>("ApplyFtrl");
inline const PrimitivePtr kPrimLrn = std::make_shared<Primitive>("LRN");
inline const PrimitivePtr kPrimLayerNorm = std::make_shared<Primitive>(kLayerNorm);
inline const PrimitivePtr kPrimLayerNormGrad = std::make_shared<Primitive>(kLayerNormGrad);
inline const PrimitivePtr kPrimLayerNormXBackprop = std::make_shared<Primitive>("LayerNormXBackprop");
inline const PrimitivePtr kPrimLayerNormXBackpropV2 = std::make_shared<Primitive>("LayerNormXBackpropV2");
inline const PrimitivePtr kPrimLayerNormBetaGammaBackprop = std::make_shared<Primitive>("LayerNormBetaGammaBackprop");
inline const PrimitivePtr kPrimLayerNormBetaGammaBackpropV2 =
  std::make_shared<Primitive>("LayerNormBetaGammaBackpropV2");
inline const PrimitivePtr kPrimLog1p = std::make_shared<Primitive>("Log1p");
inline const PrimitivePtr kPrimDropoutGenMask = std::make_shared<Primitive>(kDropoutGenMask);
inline const PrimitivePtr kPrimDropoutDoMask = std::make_shared<Primitive>(kDropoutDoMask);
inline const PrimitivePtr kPrimDropoutGrad = std::make_shared<Primitive>(kDropoutGrad);
inline const PrimitivePtr kPrimDropout = std::make_shared<Primitive>(kDropout);
inline const PrimitivePtr kPrimUniformReal = std::make_shared<Primitive>("UniformReal");
inline const PrimitivePtr kPrimCudnnUniformReal = std::make_shared<Primitive>("CudnnUniformReal");
inline const PrimitivePtr kPrimOneHot = std::make_shared<Primitive>("OneHot");
inline const PrimitivePtr kPrimGeLU = std::make_shared<Primitive>(kGeLU);
inline const PrimitivePtr kPrimGeLUGrad = std::make_shared<Primitive>(kGeLUGrad);
inline const PrimitivePtr kPrimFastGeLU = std::make_shared<Primitive>(kFastGeLU);
inline const PrimitivePtr kPrimFastGeLUGrad = std::make_shared<Primitive>(kFastGeLUGrad);
inline const PrimitivePtr kPrimRelu = std::make_shared<Primitive>(kReLU);
inline const PrimitivePtr kPrimElu = std::make_shared<Primitive>("Elu");
inline const PrimitivePtr kPrimRelu6 = std::make_shared<Primitive>(kReLU6);
inline const PrimitivePtr kPrimReluV2 = std::make_shared<Primitive>(kReLUV2);
inline const PrimitivePtr kPrimPRelu = std::make_shared<Primitive>("PReLU");
inline const PrimitivePtr kPrimSoftplus = std::make_shared<Primitive>("Softplus");
inline const PrimitivePtr kPrimSoftplusGrad = std::make_shared<Primitive>("SoftplusGrad");
inline const PrimitivePtr kPrimZeros = std::make_shared<Primitive>("Zeros");
inline const PrimitivePtr kPrimZerosLike = std::make_shared<Primitive>(kZerosLike);
inline const PrimitivePtr kPrimOnes = std::make_shared<Primitive>(kOnes);
inline const PrimitivePtr kPrimOnesLike = std::make_shared<Primitive>(kOnesLike);
inline const PrimitivePtr kPrimBpropCut = std::make_shared<Primitive>("bprop_cut");
inline const PrimitivePtr kPrimFakeQuantPerLayer = std::make_shared<Primitive>("FakeQuantPerLayer");
inline const PrimitivePtr kPrimFakeQuantPerChannel = std::make_shared<Primitive>("FakeQuantPerChannel");
inline const PrimitivePtr kPrimFakeLearnedScaleQuantPerLayer =
  std::make_shared<Primitive>("FakeLearnedScaleQuantPerLayer");
inline const PrimitivePtr kPrimFakeLearnedScaleQuantPerChannel =
  std::make_shared<Primitive>("FakeLearnedScaleQuantPerChannel");
inline const PrimitivePtr kPrimFakeQuantWithMinMaxVars = std::make_shared<Primitive>("FakeQuantWithMinMaxVars");
inline const PrimitivePtr kPrimApplyRMSProp = std::make_shared<Primitive>("ApplyRMSProp");
inline const PrimitivePtr kPrimSparseApplyFtrl = std::make_shared<Primitive>("SparseApplyFtrl");
inline const PrimitivePtr kPrimSparseApplyProximalAdagrad = std::make_shared<Primitive>("SparseApplyProximalAdagrad");
inline const PrimitivePtr kPrimFusedAdam = std::make_shared<Primitive>("FusedAdam");
inline const PrimitivePtr kPrimFusedAdamWeightDecay = std::make_shared<Primitive>("FusedAdamWeightDecay");
inline const PrimitivePtr kPrimSGD = std::make_shared<Primitive>("SGD");
inline const PrimitivePtr kPrimBCEWithLogitsLoss = std::make_shared<Primitive>("BCEWithLogitsLoss");
inline const PrimitivePtr kPrimClipByNormNoDivSum = std::make_shared<Primitive>("ClipByNormNoDivSum");
inline const PrimitivePtr kPrimTensorMove = std::make_shared<Primitive>("TensorMove");
inline const PrimitivePtr kPrimL2Normalize = std::make_shared<Primitive>("L2Normalize");
inline const PrimitivePtr kPrimCustomExtractFeatures = std::make_shared<Primitive>("CustomExtractFeatures");
inline const PrimitivePtr kLambApplyOptimizerAssign = std::make_shared<Primitive>("LambApplyOptimizerAssign");
inline const PrimitivePtr kLambApplyWeightAssign = std::make_shared<Primitive>("LambApplyWeightAssign");
inline const PrimitivePtr kSoftmaxGradExt = std::make_shared<Primitive>("SoftmaxGradExt");
inline const PrimitivePtr kPrimSparseApplyAdadelta = std::make_shared<Primitive>(kSparseApplyAdadelta);
inline const PrimitivePtr kSquareSumV1 = std::make_shared<Primitive>("SquareSumV1");
inline const PrimitivePtr kFusedMulAdd = std::make_shared<Primitive>("FusedMulAdd");
inline const PrimitivePtr kPrimSoftShrink = std::make_shared<Primitive>("SoftShrink");
inline const PrimitivePtr kPrimSoftShrinkGrad = std::make_shared<Primitive>("SoftShrinkGrad");
inline const PrimitivePtr kPrimHShrink = std::make_shared<Primitive>("HShrink");
inline const PrimitivePtr kPrimHShrinkGrad = std::make_shared<Primitive>("HShrinkGrad");
inline const PrimitivePtr kPrimApplyAdagradDA = std::make_shared<Primitive>("ApplyAdagradDA");
inline const PrimitivePtr kPrimApplyAdagradV2 = std::make_shared<Primitive>("ApplyAdagradV2");
inline const PrimitivePtr kPrimSparseApplyRMSProp = std::make_shared<Primitive>("SparseApplyRMSProp");
inline const PrimitivePtr kPrimApplyKerasMomentum = std::make_shared<Primitive>("ApplyKerasMomentum");
inline const PrimitivePtr kPrimLARSUpdate = std::make_shared<Primitive>("LARSUpdate");
inline const PrimitivePtr kPrimApplyAddSign = std::make_shared<Primitive>("ApplyAddSign");
inline const PrimitivePtr kPrimApplyAdagrad = std::make_shared<Primitive>("ApplyAdagrad");
inline const PrimitivePtr kPrimApplyAdadelta = std::make_shared<Primitive>("ApplyAdadelta");
inline const PrimitivePtr kPrimApplyAdamWithAmsgrad = std::make_shared<Primitive>("ApplyAdamWithAmsgrad");

// Comm ops
inline const PrimitivePtr kPrimMirror = std::make_shared<Primitive>("_MirrorOperator");
inline const PrimitivePtr kPrimMirrorMiniStep = std::make_shared<Primitive>("_MirrorMiniStepOperator");
inline const PrimitivePtr kPrimMiniStepAllGather = std::make_shared<Primitive>("_MiniStepAllGather");
inline const PrimitivePtr kPrimMicroStepAllGather = std::make_shared<Primitive>("_MicroStepAllGather");
inline const PrimitivePtr kPrimVirtualDiv = std::make_shared<Primitive>("_VirtualDiv");
inline const PrimitivePtr kPrimVirtualAdd = std::make_shared<Primitive>("_VirtualAdd");
inline const PrimitivePtr kPrimVirtualDataset = std::make_shared<Primitive>("_VirtualDataset");
inline const PrimitivePtr kPrimVirtualOutput = std::make_shared<Primitive>("_VirtualOutput");
inline const PrimitivePtr kPrimSend = std::make_shared<Primitive>("Send");
inline const PrimitivePtr kPrimReceive = std::make_shared<Primitive>("Receive");
inline const PrimitivePtr kPrimAllReduce = std::make_shared<Primitive>("AllReduce");
inline const PrimitivePtr kPrimNeighborExchange = std::make_shared<Primitive>("NeighborExchange");
inline const PrimitivePtr kPrimNeighborExchangeV2 = std::make_shared<Primitive>("NeighborExchangeV2");
inline const PrimitivePtr kPrimNeighborExchangeV2Grad = std::make_shared<Primitive>("NeighborExchangeV2Grad");
inline const PrimitivePtr kPrimAllToAll = std::make_shared<Primitive>("AlltoAll");
inline const PrimitivePtr kPrimAllToAllv = std::make_shared<Primitive>("AllToAllv");
inline const PrimitivePtr kPrimAllSwap = std::make_shared<Primitive>("_AllSwap");
inline const PrimitivePtr kPrimBroadcast = std::make_shared<Primitive>("Broadcast");
inline const PrimitivePtr kPrimAllGather = std::make_shared<Primitive>("AllGather");
inline const PrimitivePtr kPrimReduceScatter = std::make_shared<Primitive>("ReduceScatter");
inline const PrimitivePtr kPrimMemCpyAsync = std::make_shared<Primitive>("memcpy_async");
inline const PrimitivePtr kPrimFill = std::make_shared<Primitive>("Fill");
inline const PrimitivePtr kPrimFusedPushWeight = std::make_shared<Primitive>("FusedPushWeight");
inline const PrimitivePtr kPrimFusedPullWeight = std::make_shared<Primitive>("FusedPullWeight");
inline const PrimitivePtr kPrimInitDataSetQueue = std::make_shared<Primitive>("InitDataSetQueue");
inline const PrimitivePtr kPrimVirtualAssignAdd = std::make_shared<Primitive>("_VirtualAssignAdd");
inline const PrimitivePtr kPrimVirtualAccuGrad = std::make_shared<Primitive>("_VirtualAccuGrad");
inline const PrimitivePtr kPrimMirrorMicroStep = std::make_shared<Primitive>("_MirrorMicroStepOperator");
inline const PrimitivePtr kPrimApplyProximalAdagrad = std::make_shared<Primitive>("ApplyProximalAdagrad");

// Quant ops
inline const PrimitivePtr kPrimBatchNormFold = std::make_shared<Primitive>("BatchNormFold");
inline const PrimitivePtr kPrimFakeQuantWithMinMaxVarsPerChannel =
  std::make_shared<Primitive>("FakeQuantWithMinMaxVarsPerChannel");
// Control ops
inline const PrimitivePtr kPrimMerge = std::make_shared<Primitive>("Merge");
// RowTensor
inline const PrimitivePtr kPrimMakeRowTensor = std::make_shared<Primitive>("MakeRowTensor");
inline const PrimitivePtr kPrimRowTensorGetValues = std::make_shared<Primitive>("RowTensorGetValues");
inline const PrimitivePtr kPrimRowTensorGetIndices = std::make_shared<Primitive>("RowTensorGetIndices");
inline const PrimitivePtr kPrimRowTensorGetDenseShape = std::make_shared<Primitive>("RowTensorGetDenseShape");
inline const PrimitivePtr kPrimRowTensorAdd = std::make_shared<Primitive>("RowTensorAdd");

// SparseTensor
inline const PrimitivePtr kPrimMakeSparseTensor = std::make_shared<Primitive>("MakeSparseTensor");
inline const PrimitivePtr kPrimSparseTensorGetValues = std::make_shared<Primitive>("SparseTensorGetValues");
inline const PrimitivePtr kPrimSparseTensorGetIndices = std::make_shared<Primitive>("SparseTensorGetIndices");
inline const PrimitivePtr kPrimSparseTensorGetDenseShape = std::make_shared<Primitive>("SparseTensorGetDenseShape");

// CSRTensor
inline const PrimitivePtr kPrimMakeCSRTensor = std::make_shared<Primitive>("MakeCSRTensor");
inline const PrimitivePtr kPrimCSRTensorGetValues = std::make_shared<Primitive>("CSRTensorGetValues");
inline const PrimitivePtr kPrimCSRTensorGetIndptr = std::make_shared<Primitive>("CSRTensorGetIndptr");
inline const PrimitivePtr kPrimCSRTensorGetIndices = std::make_shared<Primitive>("CSRTensorGetIndices");
inline const PrimitivePtr kPrimCSRTensorGetDenseShape = std::make_shared<Primitive>("CSRTensorGetDenseShape");

// Sparse ops
inline const PrimitivePtr kPrimSparseTensorDenseMatmul = std::make_shared<Primitive>("SparseTensorDenseMatmul");
inline const PrimitivePtr kPrimCSRDenseMul = std::make_shared<Primitive>("CSRDenseMul");
inline const PrimitivePtr kPrimCSRReduceSum = std::make_shared<Primitive>("CSRReduceSum");
inline const PrimitivePtr kPrimCSRMV = std::make_shared<Primitive>("CSRMV");
inline const PrimitivePtr kPrimCSRMul = std::make_shared<Primitive>("CSRMul");

// TensorList
inline const PrimitivePtr kPrimTensorListFromTensor = std::make_shared<Primitive>("TensorListFromTensor");
inline const PrimitivePtr kPrimTensorListReserve = std::make_shared<Primitive>("TensorListReserve");
inline const PrimitivePtr kPrimTensorListStack = std::make_shared<Primitive>("TensorListStack");
inline const PrimitivePtr kPrimTensorListSetItem = std::make_shared<Primitive>("TensorListSetItem");

// Maths
inline const PrimitivePtr kPrimGer = std::make_shared<Primitive>("Ger");
inline const PrimitivePtr kPrimCeil = std::make_shared<Primitive>("Ceil");
inline const PrimitivePtr kPrimTensorAdd = std::make_shared<Primitive>("TensorAdd");
inline const PrimitivePtr kPrimAdd = std::make_shared<Primitive>(kAdd);
inline const PrimitivePtr kPrimMatMul = std::make_shared<Primitive>("MatMul");
inline const PrimitivePtr kPrimMatMulV2 = std::make_shared<Primitive>("MatMulV2");
inline const PrimitivePtr kPrimMatrixDiag = std::make_shared<Primitive>("MatrixDiag");
inline const PrimitivePtr kPrimBatchMatMul = std::make_shared<Primitive>("BatchMatMul");
inline const PrimitivePtr kPrimBatchMatMulV2 = std::make_shared<Primitive>("BatchMatMulV2");
inline const PrimitivePtr kPrimMaximumGrad = std::make_shared<Primitive>("MaximumGrad");
inline const PrimitivePtr kPrimMinimumGrad = std::make_shared<Primitive>("MinimumGrad");
inline const PrimitivePtr kPrimReduce = std::make_shared<Primitive>("Reduce");
inline const PrimitivePtr kPrimReduceMean = std::make_shared<Primitive>("ReduceMean");
inline const PrimitivePtr kPrimReduceSum = std::make_shared<Primitive>("ReduceSum");
inline const PrimitivePtr kPrimReduceAll = std::make_shared<Primitive>("ReduceAll");
inline const PrimitivePtr kPrimReduceAny = std::make_shared<Primitive>("ReduceAny");
inline const PrimitivePtr kPrimReduceMax = std::make_shared<Primitive>("ReduceMax");
inline const PrimitivePtr kPrimReduceMin = std::make_shared<Primitive>("ReduceMin");
inline const PrimitivePtr kPrimCentralization = std::make_shared<Primitive>("Centralization");
inline const PrimitivePtr kPrimNeg = std::make_shared<Primitive>(kNeg);
inline const PrimitivePtr kPrimSin = std::make_shared<Primitive>("Sin");
inline const PrimitivePtr kPrimCos = std::make_shared<Primitive>(kCos);
inline const PrimitivePtr kPrimSub = std::make_shared<Primitive>(kSub);
inline const PrimitivePtr kPrimMul = std::make_shared<Primitive>(kMul);
inline const PrimitivePtr kPrimMulNoNan = std::make_shared<Primitive>(kMulNoNan);
inline const PrimitivePtr kPrimDiv = std::make_shared<Primitive>("Div");
inline const PrimitivePtr kPrimMod = std::make_shared<Primitive>("Mod");
inline const PrimitivePtr kPrimFloor = std::make_shared<Primitive>("Floor");
inline const PrimitivePtr kPrimInvert = std::make_shared<Primitive>("Invert");
inline const PrimitivePtr kPrimDivNoNan = std::make_shared<Primitive>("DivNoNan");
inline const PrimitivePtr kPrimMinimum = std::make_shared<Primitive>("Minimum");
inline const PrimitivePtr kPrimMaximum = std::make_shared<Primitive>("Maximum");
inline const PrimitivePtr kPrimSquare = std::make_shared<Primitive>(kSquare);
inline const PrimitivePtr kPrimCumSum = std::make_shared<Primitive>("CumSum");
inline const PrimitivePtr kPrimCumProd = std::make_shared<Primitive>("CumProd");
inline const PrimitivePtr kPrimSubscalar = std::make_shared<Primitive>("Subscalar");
inline const PrimitivePtr kPrimInplaceAdd = std::make_shared<Primitive>("InplaceAdd");
inline const PrimitivePtr kPrimLpNorm = std::make_shared<Primitive>(kLpNorm);
inline const PrimitivePtr kPrimInplaceSub = std::make_shared<Primitive>("InplaceSub");
inline const PrimitivePtr kPrimPow = std::make_shared<Primitive>("Pow");
inline const PrimitivePtr kPrimPower = std::make_shared<Primitive>("Power");
inline const PrimitivePtr kPrimRealDiv = std::make_shared<Primitive>(kRealDiv);
inline const PrimitivePtr kPrimFloorDiv = std::make_shared<Primitive>("FloorDiv");
inline const PrimitivePtr kPrimSqrt = std::make_shared<Primitive>("Sqrt");
inline const PrimitivePtr kPrimSqrtGrad = std::make_shared<Primitive>("SqrtGrad");
inline const PrimitivePtr kPrimReciprocal = std::make_shared<Primitive>(kReciprocal);
inline const PrimitivePtr kPrimReciprocalGrad = std::make_shared<Primitive>("ReciprocalGrad");
inline const PrimitivePtr kPrimExpandDims = std::make_shared<Primitive>("ExpandDims");
inline const PrimitivePtr kPrimAbs = std::make_shared<Primitive>(kAbs);
inline const PrimitivePtr kPrimAbsGrad = std::make_shared<Primitive>("AbsGrad");
inline const PrimitivePtr kPrimRint = std::make_shared<Primitive>("Rint");
inline const PrimitivePtr kPrimRound = std::make_shared<Primitive>("Round");
inline const PrimitivePtr kPrimExp = std::make_shared<Primitive>(kExp);
inline const PrimitivePtr kPrimExpm1 = std::make_shared<Primitive>("Expm1");
inline const PrimitivePtr kPrimLog = std::make_shared<Primitive>(kLog);
inline const PrimitivePtr kPrimRsqrt = std::make_shared<Primitive>("Rsqrt");
inline const PrimitivePtr kPrimRsqrtGrad = std::make_shared<Primitive>("RsqrtGrad");
inline const PrimitivePtr kPrimLinSpace = std::make_shared<Primitive>("LinSpace");
inline const PrimitivePtr kPrimNonMaxSuppression = std::make_shared<Primitive>("NonMaxSuppression");
inline const PrimitivePtr kPrimSign = std::make_shared<Primitive>("Sign");
inline const PrimitivePtr kPrimACos = std::make_shared<Primitive>("ACos");
inline const PrimitivePtr kPrimAsinGrad = std::make_shared<Primitive>("AsinGrad");
inline const PrimitivePtr kPrimACosGrad = std::make_shared<Primitive>("ACosGrad");
inline const PrimitivePtr kPrimAtanGrad = std::make_shared<Primitive>("AtanGrad");
inline const PrimitivePtr kPrimAsinhGrad = std::make_shared<Primitive>("AsinhGrad");
inline const PrimitivePtr kPrimAcoshGrad = std::make_shared<Primitive>("AcoshGrad");
inline const PrimitivePtr kPrimFloorMod = std::make_shared<Primitive>("FloorMod");
inline const PrimitivePtr kPrimCdist = std::make_shared<Primitive>(kCdist);
inline const PrimitivePtr kPrimCdistGrad = std::make_shared<Primitive>(kCdistGrad);
inline const PrimitivePtr kPrimWhere = std::make_shared<Primitive>("Where");
inline const PrimitivePtr kPrimIndexAdd = std::make_shared<Primitive>("IndexAdd");
inline const PrimitivePtr kPrimIdentityMath = std::make_shared<Primitive>("Identity", kSideEffectPropagate);
inline const PrimitivePtr kPrimErfinv = std::make_shared<Primitive>("Erfinv");
inline const PrimitivePtr kPrimIsNan = std::make_shared<Primitive>("IsNan");
inline const PrimitivePtr kPrimIsInf = std::make_shared<Primitive>("IsInf");
inline const PrimitivePtr kPrimIsFinite = std::make_shared<Primitive>("IsFinite");
inline const PrimitivePtr kPrimIsClose = std::make_shared<Primitive>("IsClose");
inline const PrimitivePtr kPrimLerp = std::make_shared<Primitive>("Lerp");
inline const PrimitivePtr kPrimSquareSumAll = std::make_shared<Primitive>("SquareSumAll");
inline const PrimitivePtr kPrimComplex = std::make_shared<Primitive>("Complex");
inline const PrimitivePtr kPrimXdivy = std::make_shared<Primitive>("Xdivy");
inline const PrimitivePtr kPrimInv = std::make_shared<Primitive>("Inv");

// Image
inline const PrimitivePtr kPrimNonMaxSuppressionV3 = std::make_shared<Primitive>("NonMaxSuppressionV3");

// Statements
inline const PrimitivePtr kPrimReturn = std::make_shared<Primitive>("Return");
inline const PrimitivePtr kPrimUnroll = std::make_shared<Primitive>("Unroll");
inline const PrimitivePtr kPrimSwitch = std::make_shared<Primitive>("Switch");
inline const PrimitivePtr kPrimSwitchLayer = std::make_shared<Primitive>("switch_layer");
inline const PrimitivePtr kPrimAssign = std::make_shared<Primitive>("Assign");
inline const PrimitivePtr kPrimAssignAdd = std::make_shared<Primitive>("AssignAdd");
inline const PrimitivePtr kPrimAssignSub = std::make_shared<Primitive>("AssignSub");
inline const PrimitivePtr kPrimSelect = std::make_shared<Primitive>(kSelect);
inline const PrimitivePtr kPrimCall = std::make_shared<Primitive>("call");

inline const PrimitivePtr kPrimMakeTuple = std::make_shared<Primitive>("MakeTuple");
inline const PrimitivePtr kPrimMakeSlice = std::make_shared<Primitive>("make_slice");
inline const PrimitivePtr kPrimTupleGetItem = std::make_shared<Primitive>(kTupleGetItem);
inline const PrimitivePtr kPrimSliceGetItem = std::make_shared<Primitive>(kSliceGetItem);
inline const PrimitivePtr kPrimArrayGetItem = std::make_shared<Primitive>("array_getitem");
inline const PrimitivePtr kPrimTupleSetItem = std::make_shared<Primitive>("tuple_setitem");
inline const PrimitivePtr kPrimArraySetItem = std::make_shared<Primitive>("array_setitem");
inline const PrimitivePtr kPrimGetAttr = std::make_shared<Primitive>("getattr");
inline const PrimitivePtr kPrimTupleLen = std::make_shared<Primitive>("tuple_len");
inline const PrimitivePtr kPrimArrayLen = std::make_shared<Primitive>("array_len");
inline const PrimitivePtr kPrimTileShape = std::make_shared<Primitive>("tile_shape");
inline const PrimitivePtr kPrimGenerateShapeIndex = std::make_shared<Primitive>("generate_shape_index");
inline const PrimitivePtr kPrimGenerateInverseIndex = std::make_shared<Primitive>("generate_inverse_index");

// Debug ops
inline const PrimitivePtr kPrimAssert = std::make_shared<Primitive>("Assert");
#ifndef ENABLE_SECURITY
inline const PrimitivePtr kPrimScalarSummary = std::make_shared<Primitive>("ScalarSummary");
inline const PrimitivePtr kPrimImageSummary = std::make_shared<Primitive>("ImageSummary");
inline const PrimitivePtr kPrimTensorSummary = std::make_shared<Primitive>("TensorSummary");
inline const PrimitivePtr kPrimHistogramSummary = std::make_shared<Primitive>("HistogramSummary");
#endif
inline const PrimitivePtr kPrimDebug = std::make_shared<Primitive>("Debug");

// Dynamic shape testing
inline const PrimitivePtr kPrimGpuConvertToDynamicShape = std::make_shared<Primitive>("GpuConvertToDynamicShape");
inline const PrimitivePtr kPrimErrorOnDynamicShapeInput = std::make_shared<Primitive>("ErrorOnDynamicShapeInput");

// Other miscellaneous
inline const PrimitivePtr kPrimDepend = std::make_shared<Primitive>("Depend", kSideEffectPropagate);
inline const PrimitivePtr kPrimIOU = std::make_shared<Primitive>("IOU");
inline const PrimitivePtr kPrimReformat = std::make_shared<Primitive>("Reformat");
inline const PrimitivePtr kPrimLoad = std::make_shared<Primitive>("Load");
inline const PrimitivePtr kPrimUpdateState = std::make_shared<Primitive>("UpdateState");
inline const PrimitivePtr kPrimPartial = std::make_shared<Primitive>("Partial", kSideEffectPropagate);
inline const PrimitivePtr kPrimIdentity = std::make_shared<Primitive>("identity", kSideEffectPropagate);
inline const PrimitivePtr kPrimHookBackward = std::make_shared<Primitive>("HookBackward");
inline const PrimitivePtr kPrimPrintShapeType = std::make_shared<Primitive>("PrintShapeType");
inline const PrimitivePtr kPrimSameTypeShape = std::make_shared<Primitive>("SameTypeShape");
inline const PrimitivePtr kPrimPrint = std::make_shared<Primitive>("Print");
inline const PrimitivePtr kPrimIs_ = std::make_shared<Primitive>("is_");
inline const PrimitivePtr kPrimIsNot = std::make_shared<Primitive>("is_not");
inline const PrimitivePtr kPrimInDict = std::make_shared<Primitive>("in_dict");
inline const PrimitivePtr kPrimNotInDict = std::make_shared<Primitive>("not_in_dict");
inline const PrimitivePtr kPrimIsConsant = std::make_shared<Primitive>("is_constant");
inline const PrimitivePtr kPrimEquivFormat = std::make_shared<Primitive>("EquivFormat");
inline const PrimitivePtr kPrimLshProjection = std::make_shared<Primitive>("LshProjection");
inline const PrimitivePtr kPrimHashtableLookup = std::make_shared<Primitive>("HashtableLookup");
inline const PrimitivePtr kPrimCustomPredict = std::make_shared<Primitive>("CustomPredict");
inline const PrimitivePtr kPrimPriorBox = std::make_shared<Primitive>("PriorBox");
inline const PrimitivePtr kPrimQuantDTypeCast = std::make_shared<Primitive>("QuantDTypeCast");
inline const PrimitivePtr kPrimWhile = std::make_shared<Primitive>("While");
inline const PrimitivePtr kPrimPull = std::make_shared<Primitive>("Pull");
inline const PrimitivePtr kPrimPush = std::make_shared<Primitive>("Push");
inline const PrimitivePtr kPrimNPUAllocFloatStatus = std::make_shared<Primitive>("NPUAllocFloatStatus");
inline const PrimitivePtr kPyFunc = std::make_shared<Primitive>("PyFunc");

// Structures
inline const PrimitivePtr kPrimMakeList = std::make_shared<Primitive>("make_list");
inline const PrimitivePtr kPrimMakeKeywordArg = std::make_shared<Primitive>("make_keyword_arg");
inline const PrimitivePtr kPrimListGetItem = std::make_shared<Primitive>("list_getitem");
inline const PrimitivePtr kPrimListSetItem = std::make_shared<Primitive>("list_setitem");
inline const PrimitivePtr kPrimDictGetItem = std::make_shared<Primitive>("dict_getitem");
inline const PrimitivePtr kPrimDictSetItem = std::make_shared<Primitive>("dict_setitem");
inline const PrimitivePtr kPrimDictGetKeys = std::make_shared<Primitive>("dict_getkeys");
inline const PrimitivePtr kPrimDictGetValues = std::make_shared<Primitive>("dict_getvalues");
inline const PrimitivePtr kPrimDictItems = std::make_shared<Primitive>("dict_items");
inline const PrimitivePtr kPrimListAppend = std::make_shared<Primitive>("list_append");
inline const PrimitivePtr kPrimListLen = std::make_shared<Primitive>("list_len");

// Other miscellaneous
inline const PrimitivePtr kPrimEnvSetItem = std::make_shared<Primitive>("env_setitem");
inline const PrimitivePtr kPrimEnvGetItem = std::make_shared<Primitive>("env_getitem");
inline const PrimitivePtr kPrimEnvAdd = std::make_shared<Primitive>("env_add");
inline const PrimitivePtr kPrimMakeRefKey = std::make_shared<Primitive>("MakeRefKey");
inline const PrimitivePtr kPrimGetRefKey = std::make_shared<Primitive>("get_ref_key");
inline const PrimitivePtr kPrimMakeRef = std::make_shared<Primitive>("make_ref");
inline const PrimitivePtr kPrimGetRefValue = std::make_shared<Primitive>("get_ref_value");

// Python interpreter runner
inline const PrimitivePtr kPrimPyInterpret = std::make_shared<Primitive>("PyInterpret");

// Other primitive not used by backend but used in core;
inline const PrimitivePtr kPrimStateSetItem = std::make_shared<Primitive>("state_setitem");
inline const PrimitivePtr kPrimJ = std::make_shared<Primitive>("J", kSideEffectPropagate);
inline const PrimitivePtr kPrimShard = std::make_shared<Primitive>("Shard", kSideEffectPropagate);

// Used to build graph which have keyword arguments
inline const PrimitivePtr kPrimExtractKeywordArg = std::make_shared<Primitive>("extract_keyword_arg");
inline const PrimitivePtr kPrimMakeDict = std::make_shared<Primitive>("make_dict");

// GraphKernel ops
inline const PrimitivePtr kPrimInplaceAssign = std::make_shared<Primitive>("InplaceAssign");

// Custom
inline const PrimitivePtr kPrimCustom = std::make_shared<Primitive>("Custom");

// Only used in lite
inline const PrimitivePtr kPrimLeakyRelu = std::make_shared<Primitive>("LeakyRelu");
inline const PrimitivePtr kPrimConstant = std::make_shared<Primitive>("Constant");
inline const PrimitivePtr kPrimLocalResponseNormalization = std::make_shared<Primitive>("LocalResponseNormalization");
inline const PrimitivePtr kPrimFftReal = std::make_shared<Primitive>("FftReal");
inline const PrimitivePtr kPrimMfcc = std::make_shared<Primitive>("Mfcc");
inline const PrimitivePtr kPrimRfft = std::make_shared<Primitive>("Rfft");
inline const PrimitivePtr kPrimFftImag = std::make_shared<Primitive>("FftImag");
inline const PrimitivePtr kPrimSkipGram = std::make_shared<Primitive>("SkipGram");
inline const PrimitivePtr kPrimConv2DFusion = std::make_shared<Primitive>("Conv2DFusion");
inline const PrimitivePtr kPrimConv2dTransposeFusion = std::make_shared<Primitive>("Conv2dTransposeFusion");
inline const PrimitivePtr kPrimDepthWiseConv2DFusion = std::make_shared<Primitive>("DepthWiseConv2DFusion");
inline const PrimitivePtr kPrimAddFusion = std::make_shared<Primitive>("AddFusion");
inline const PrimitivePtr kPrimScaleFusion = std::make_shared<Primitive>("ScaleFusion");
inline const PrimitivePtr kPrimSubFusion = std::make_shared<Primitive>("SubFusion");
inline const PrimitivePtr kPrimMulFusion = std::make_shared<Primitive>("MulFusion");
inline const PrimitivePtr kPrimSigmoid = std::make_shared<Primitive>("Sigmoid");
inline const PrimitivePtr kPrimSigmoidGrad = std::make_shared<Primitive>("SigmoidGrad");
inline const PrimitivePtr kPrimHSigmoid = std::make_shared<Primitive>("HSigmoid");
inline const PrimitivePtr kPrimHSigmoidGrad = std::make_shared<Primitive>("HSigmoidGrad");
inline const PrimitivePtr kPrimClip = std::make_shared<Primitive>("Clip");
inline const PrimitivePtr kPrimHardTanh = std::make_shared<Primitive>("HardTanh");
inline const PrimitivePtr kPrimDepthWiseConv2DTransposeFusion =
  std::make_shared<Primitive>("DepthWiseConv2DTransposeFusion");
inline const PrimitivePtr kPrimArgMinFusion = std::make_shared<Primitive>("ArgMinFusion");
inline const PrimitivePtr kPrimArgMaxFusion = std::make_shared<Primitive>("ArgMaxFusion");
inline const PrimitivePtr kPrimSpaceToDepth = std::make_shared<Primitive>("SpaceToDepth");
inline const PrimitivePtr kPrimPadFusion = std::make_shared<Primitive>("PadFusion");
inline const PrimitivePtr kPrimPowFusion = std::make_shared<Primitive>("PowFusion");
inline const PrimitivePtr kPrimResize = std::make_shared<Primitive>("Resize");
inline const PrimitivePtr kPrimArgMinWithValue = std::make_shared<Primitive>("ArgMinWithValue");
inline const PrimitivePtr kPrimIf = std::make_shared<Primitive>("If");
inline const PrimitivePtr kPrimAvgPoolFusion = std::make_shared<Primitive>("AvgPoolFusion");
inline const PrimitivePtr kPrimMaxPoolFusion = std::make_shared<Primitive>("MaxPoolFusion");
inline const PrimitivePtr kPrimActivation = std::make_shared<Primitive>("Activation");
inline const PrimitivePtr kPrimPReLUFusion = std::make_shared<Primitive>("PReLUFusion");
inline const PrimitivePtr kPrimTopKFusion = std::make_shared<Primitive>("TopKFusion");
inline const PrimitivePtr kPrimTileFusion = std::make_shared<Primitive>("TileFusion");
inline const PrimitivePtr kPrimReduceFusion = std::make_shared<Primitive>("ReduceFusion");
inline const PrimitivePtr kPrimLayerNormFusion = std::make_shared<Primitive>("LayerNormFusion");
inline const PrimitivePtr kPrimDType = std::make_shared<Primitive>("DType");
inline const PrimitivePtr kPrimDivFusion = std::make_shared<Primitive>("DivFusion");
inline const PrimitivePtr kPrimErf = std::make_shared<Primitive>("Erf");
inline const PrimitivePtr kPrimErfc = std::make_shared<Primitive>("Erfc");
inline const PrimitivePtr kPrimSplice = std::make_shared<Primitive>("Splice");
inline const PrimitivePtr kPrimAffine = std::make_shared<Primitive>("Affine");
inline const PrimitivePtr kPrimEltwise = std::make_shared<Primitive>("Eltwise");

// Type introspection
inline const PrimitivePtr kPrimTypeOf = std::make_shared<Primitive>("typeof");
inline const PrimitivePtr kPrimHasType = std::make_shared<Primitive>("hastype");

inline const PrimitivePtr kPrimResolve = std::make_shared<Primitive>("resolve");
inline const PrimitivePtr kPrimEmbed = std::make_shared<Primitive>("embed");
inline const PrimitivePtr kPrimRefToEmbed = std::make_shared<Primitive>("RefToEmbed");
inline const PrimitivePtr kPrimCreateInstance = std::make_shared<Primitive>("create_instance");

// Other miscellaneous
inline const PrimitivePtr kPrimGetRefOrigin = std::make_shared<Primitive>("get_ref_origin");
inline const PrimitivePtr kPrimInsertGradientOf = std::make_shared<Primitive>("InsertGradientOf");
inline const PrimitivePtr kPrimCheckBprop = std::make_shared<Primitive>("CheckBprop");
inline const PrimitivePtr kPrimMixedPrecisionCast = std::make_shared<Primitive>("mixed_precision_cast");
inline const PrimitivePtr kPrimMakeRecord = std::make_shared<Primitive>("make_record");

// Structures
inline const PrimitivePtr kPrimListMap = std::make_shared<Primitive>("list_map");
inline const PrimitivePtr kPrimListReduce = std::make_shared<Primitive>("list_reduce");
inline const PrimitivePtr kPrimTupleReversed = std::make_shared<Primitive>("tuple_reversed");
inline const PrimitivePtr kPrimReducedShape = std::make_shared<Primitive>("reduced_shape");
inline const PrimitivePtr kPrimTupleDiv = std::make_shared<Primitive>("tuple_div");
inline const PrimitivePtr kPrimTupleToArray = std::make_shared<Primitive>("tuple_to_array");
inline const PrimitivePtr kPrimShapeMul = std::make_shared<Primitive>("shape_mul");
inline const PrimitivePtr kPrimTupleEqual = std::make_shared<Primitive>("tuple_equal");
inline const PrimitivePtr kPrimListEqual = std::make_shared<Primitive>("list_equal");
inline const PrimitivePtr kPrimMakeRange = std::make_shared<Primitive>("make_range");
inline const PrimitivePtr kPrimStopGradient = std::make_shared<Primitive>("stop_gradient");
inline const PrimitivePtr kPrimStringEqual = std::make_shared<Primitive>("string_equal");
inline const PrimitivePtr kPrimStringConcat = std::make_shared<Primitive>("string_concat");
inline const PrimitivePtr kPrimDictLen = std::make_shared<Primitive>("dict_len");
inline const PrimitivePtr kPrimFakeBprop = std::make_shared<Primitive>("fake_bprop");
inline const PrimitivePtr kPrimBroadcastGradientArgs = std::make_shared<Primitive>("BroadcastGradientArgs");
inline const PrimitivePtr kPrimDynamicBroadcastGradientArgs =
  std::make_shared<Primitive>(kDynamicBroadcastGradientArgs);

// Random
inline const PrimitivePtr kPrimStandardNormal = std::make_shared<Primitive>("StandardNormal");

// RL Ops
inline const PrimitivePtr kPrimTensorArrayStack = std::make_shared<Primitive>("TensorArrayStack");

class DoSignaturePrimitive : public Primitive {
 public:
  explicit DoSignaturePrimitive(const std::string &name, const ValuePtr &function)
      : Primitive("S-Prim-" + name), function_(function) {}

  ~DoSignaturePrimitive() override = default;

  MS_DECLARE_PARENT(DoSignaturePrimitive, Primitive)

  const ValuePtr function() const { return function_; }

 private:
  ValuePtr function_;
};
using DoSignaturePrimitivePtr = std::shared_ptr<DoSignaturePrimitive>;
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_CORE_OPS_H_
