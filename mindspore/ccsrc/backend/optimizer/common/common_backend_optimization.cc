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
#include "backend/optimizer/common/common_backend_optimization.h"
#include <memory>
#include <string>
#include "backend/optimizer/common/optimizer.h"
#include "backend/optimizer/pass/eliminate_func_data_type.h"
#include "backend/optimizer/pass/convert_const_input_to_attr.h"
#include "backend/optimizer/pass/custom_op_const_input_to_attr.h"
#include "backend/optimizer/pass/custom_op_reg_info_to_attr.h"
#include "backend/optimizer/pass/convert_tuple_output_to_maketuple.h"
#include "backend/optimizer/pass/convert_const_input_to_tensor_input.h"
#include "backend/optimizer/pass/convert_tuple_input_to_dynamic_input.h"
#include "backend/optimizer/pass/const_to_attr_strided_slice_grad.h"
#include "backend/optimizer/pass/convert_const_scalar_to_tensor.h"
#include "backend/optimizer/pass/convert_attr_to_unify_mindir.h"
#include "backend/optimizer/pass/add_training_attr.h"
#include "backend/optimizer/pass/optimize_updatestate.h"
#include "backend/optimizer/pass/conv_transpose_to_conv_bp.h"
#include "backend/optimizer/pass/reduce_sum_optimizer.h"
#include "backend/optimizer/pass/add_dynamic_shape_attr.h"
#include "backend/optimizer/pass/add_akg_kernel_attrs.h"
#include "backend/optimizer/pass/sparse_process.h"
#include "utils/ms_context.h"
#include "debug/anf_ir_dump.h"

namespace mindspore {
namespace opt {
void BackendCommonOptimization(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  MS_LOG(INFO) << "Status record: start common optimization. graph id: " << kernel_graph->graph_id();
  MS_EXCEPTION_IF_NULL(kernel_graph);
#ifdef ENABLE_DUMP_IR
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  if (save_graphs) {
    std::string file_name = "hwopt_common_before_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
  auto optimizer = std::make_shared<GraphOptimizer>();
  auto common_pm = std::make_shared<PassManager>("common_pm");
  common_pm->AddPass(std::make_shared<AddDynamicShapeAttr>());
  common_pm->AddPass(std::make_shared<ReduceSumOptimizer>());
  common_pm->AddPass(std::make_shared<ConvertConstInputToAttr>());
  common_pm->AddPass(std::make_shared<CustomOpConstInputToAttr>());
  common_pm->AddPass(std::make_shared<SparseProcess>());
  common_pm->AddPass(std::make_shared<ConvertAttrToUnifyMindIR>());
  common_pm->AddPass(std::make_shared<ConstToAttrStridedSliceGradPass>());
  common_pm->AddPass(std::make_shared<ConvertConstInputToTensorInput>());
  common_pm->AddPass(std::make_shared<ConvertTupleOutputToMaketuple>());
  common_pm->AddPass(std::make_shared<ConvertConstScalarToTensor>());
  common_pm->AddPass(std::make_shared<ConvertTupleInputToDynamicInput>());
  common_pm->AddPass(std::make_shared<AddTrainingAttr>());
  optimizer->AddPassManager(common_pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
#ifdef ENABLE_DUMP_IR
  if (save_graphs) {
    std::string file_name = "hwopt_common_after_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
  MS_LOG(INFO) << "Status record: end common optimization. graph id: " << kernel_graph->graph_id();
}

void CommonFinalOptimization(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  // Run optimizer passes.
  auto optimizer = std::make_shared<GraphOptimizer>();
  auto pm = std::make_shared<PassManager>("final_opt");
  pm->AddPass(std::make_shared<OptimizeUpdateState>());
  pm->AddPass(std::make_shared<AddAkgKernelAttrs>());
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
#ifdef ENABLE_DUMP_IR
  // Dump IR if save_graphs is set.
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  const bool save_graphs = context->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  if (save_graphs) {
    std::string filename = "hwopt_common_final_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(filename, kernel_graph);
  }
#endif
}

void CommonUnifyMindIR(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(INFO) << "start common unify mindir opt graph:" << kernel_graph->graph_id();
#ifdef ENABLE_DUMP_IR
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  if (save_graphs) {
    std::string file_name =
      "hwopt_common_unify_mindir_before_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
  auto opt = std::make_shared<GraphOptimizer>();
  auto pm = std::make_shared<PassManager>("common_unify_mindir_pm");
  pm->AddPass(std::make_shared<ConvTransposeToConvBackpropInputPass>());
  pm->AddPass(std::make_shared<CustomOpRegInfoToAttr>());
  opt->AddPassManager(pm);
  (void)opt->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
#ifdef ENABLE_DUMP_IR
  if (save_graphs) {
    std::string file_name = "hwopt_common_unify_mindir_after_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
}

void AddDynamicShapeAttrPass(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  auto opt = std::make_shared<GraphOptimizer>();
  auto pm = std::make_shared<PassManager>("add_dynamic_shape_attr");
  pm->AddPass(std::make_shared<AddDynamicShapeAttr>());
  opt->AddPassManager(pm);
  (void)opt->Optimize(kernel_graph);
}

void EliminateIllegalDataTypePass(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(INFO) << "start eliminate illegal data type for kernel graph id:" << kernel_graph->graph_id();
#ifdef ENABLE_DUMP_IR
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  if (save_graphs) {
    std::string file_name =
      "hwopt_common_eliminate_illegal_data_type_before_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
  auto opt = std::make_shared<GraphOptimizer>();
  auto pm = std::make_shared<PassManager>("common_eliminate_illegal_data_type_pm");
  pm->AddPass(std::make_shared<EliminateFuncDataType>());
  opt->AddPassManager(pm);
  (void)opt->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
#ifdef ENABLE_DUMP_IR
  if (save_graphs) {
    std::string file_name =
      "hwopt_common_eliminate_illegal_data_type_after_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
}
}  // namespace opt
}  // namespace mindspore
