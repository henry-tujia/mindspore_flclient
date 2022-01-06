/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "pybind_api/ir/primitive_py.h"

#include <mutex>
#include <map>
#include <utility>
#include "ir/signature.h"
#include "pipeline/jit/parse/data_converter.h"
#include "pipeline/jit/parse/python_adapter.h"
#include "pybind11/pytypes.h"
#include "pybind_api/api_register.h"
#include "pybind_api/export_flags.h"
#include "pybind_api/ir/base_ref_py.h"
#include "utils/convert_utils_base.h"
#include "utils/convert_utils_py.h"
#include "utils/ms_context.h"
#include "utils/primitive_utils.h"
#include "utils/check_convert_utils.h"
#include "pipeline/jit/resource.h"
#include "pipeline/pynative/pynative_execute.h"

namespace mindspore {
namespace {
constexpr auto kBpropAttrName = "bprop";
constexpr auto kCellHookAttrName = "cell_hook";
constexpr auto kCellIDAttrName = "cell_id";
std::map<std::string, std::string> kOpAttrNameReplaceMap = {
  {"data_format", "format"},
};

void SyncData(const py::object &arg) {
  if (py::isinstance<py::tuple>(arg)) {
    py::tuple arg_list = py::cast<py::tuple>(arg);
    for (size_t i = 0; i < arg_list.size(); i++) {
      SyncData(arg_list[i]);
    }
  }
  if (py::isinstance<tensor::Tensor>(arg)) {
    auto tensor = py::cast<tensor::TensorPtr>(arg);
    tensor->data_sync();
  }
}
}  // namespace
std::map<std::string, py::object> PrimitivePy::hook_grad_;

PrimitivePy::PrimitivePy(const std::string &name) : Primitive(name, false), python_obj_(py::none()) {}

PrimitivePy::PrimitivePy(const py::object &python_obj, const PrimitivePyAdapterPtr &adapter)
    : Primitive(adapter->name_, false), python_obj_(python_obj), adapter_(adapter) {
  MS_LOG(DEBUG) << "New primitive:" << adapter->name_;
  set_signatures(adapter->signatures_);
  (void)Primitive::SetAttrs(adapter->attrs_);
  Primitive::set_prim_type(adapter->prim_type_);
  Primitive::set_const_prim(adapter->is_const_prim_);
  Primitive::set_const_input_indexes(adapter->const_input_indexes_);
  set_hook(adapter->hook_);
  set_instance_name(adapter->instance_name_);
}
PrimitivePy::~PrimitivePy() {}

void PrimitivePy::set_signatures(const std::vector<Signature> &signatures) {
  signatures_ = signatures;
  set_has_signature(!signatures.empty());
}

py::function PrimitivePy::GetBpropFunction() {
  static const char *const get_bprop_func_name = "get_bprop";
  if (py::hasattr(python_obj_, get_bprop_func_name)) {
    py::function fn = python_obj_.attr(get_bprop_func_name)().cast<py::function>();
    return fn;
  } else {
    auto fn = GetBpropFunctionByObj(python_obj_);
    return fn;
  }
}

py::tuple check_bprop_out(const py::object &grads_obj, const py::tuple &py_args, const std::string &bprop_cls_name) {
  py::tuple grads;
  if (!py::isinstance<py::tuple>(grads_obj)) {
    grads = py::make_tuple(grads_obj);
  } else {
    grads = py::cast<py::tuple>(grads_obj);
  }
  if (!MsContext::GetInstance()->get_param<bool>(MS_CTX_CHECK_BPROP_FLAG)) {
    return grads;
  }
  constexpr int filter_args_size = 2;
  if (grads.size() != py_args.size() - filter_args_size) {
    MS_EXCEPTION(TypeError) << "For user defined method 'bprop' of net '" << bprop_cls_name
                            << "', the number of return values(gradients) should be equal to the number of input "
                               "arguments except 'out' and 'dout', which is: "
                            << (py_args.size() - filter_args_size) << ", but got:" << grads.size() << ".";
  }
  for (size_t i = 0; i < grads.size(); i++) {
    if (py::isinstance<tensor::Tensor>(py_args[i])) {
      if (!py::isinstance<tensor::Tensor>(grads[i])) {
        MS_EXCEPTION(ValueError) << "For user defined method 'bprop' of net '" << bprop_cls_name << "', the " << i
                                 << "th return value(gradient of the " << i << "th argument) should be Tensor, but got "
                                 << py::cast<std::string>(grads[i].attr("__class__").attr("__name__"))
                                 << ", and the value is " << py::cast<py::str>(grads[i]) << ".";
      }

      py::object arg_dtype = py_args[i].attr("dtype");
      py::object grad_dtype = grads[i].attr("dtype");
      py::tuple arg_shape = py_args[i].attr("shape");
      py::tuple grad_shape = grads[i].attr("shape");
      if (!grad_dtype.equal(arg_dtype)) {
        MS_EXCEPTION(TypeError) << "For user defined method 'bprop' of net '" << bprop_cls_name << "', the " << i
                                << "th return value(gradient of the " << i
                                << "th argument) should have the same dtype as the " << i
                                << "th argument, which is:" << py::cast<py::str>(arg_dtype)
                                << ", but got: " << py::cast<py::str>(grad_dtype) << ".";
      }
      if (!grad_shape.equal(arg_shape)) {
        MS_EXCEPTION(ValueError) << "For user defined method 'bprop' of net '" << bprop_cls_name << "', the " << i
                                 << "th return value(gradient of the " << i
                                 << "th argument) should have the same shape as the " << i
                                 << "th argument, which is:" << py::cast<py::str>(arg_shape)
                                 << ", but got: " << py::cast<py::str>(grad_shape) << ".";
      }
    }
  }
  return grads;
}

void PrimitivePy::ConvertCTensorToPyTensor(const py::tuple &input_args, py::tuple *convert_args) const {
  MS_EXCEPTION_IF_NULL(convert_args);
  if (input_args.size() != (*convert_args).size()) {
    MS_LOG(EXCEPTION) << "The size of input_args: " << input_args.size()
                      << " should be equal to the size of convert_args: " << (*convert_args).size();
  }
  for (size_t i = 0; i < input_args.size(); ++i) {
    (*convert_args)[i] = py::isinstance<tensor::Tensor>(input_args[i])
                           ? parse::python_adapter::CallPyFn(parse::PYTHON_MOD_PARSE_MODULE,
                                                             parse::PYTHON_MOD_CONVERT_TO_MS_TENSOR, input_args[i])
                           : input_args[i];
  }
}

void PrimitivePy::CheckHookConsistency(const py::object &grad_out, const py::object &expected_grad_out) const {
  if (py::isinstance<py::tuple>(expected_grad_out)) {
    if (!py::isinstance<py::tuple>(grad_out)) {
      hook_grad_.clear();
      MS_EXCEPTION(TypeError) << "The output gradient should be a tuple!";
    }
    auto actual_out_tuple = py::cast<py::tuple>(grad_out);
    auto expected_out_tuple = py::cast<py::tuple>(expected_grad_out);
    if (actual_out_tuple.size() != expected_out_tuple.size()) {
      hook_grad_.clear();
      MS_EXCEPTION(ValueError) << "The tuple size of output gradient should be " << expected_out_tuple.size()
                               << ", but it is " << actual_out_tuple.size();
    }
    for (size_t i = 0; i < expected_out_tuple.size(); ++i) {
      CheckHookConsistency(actual_out_tuple[i], expected_out_tuple[i]);
    }
  }

  if (py::isinstance<tensor::Tensor>(expected_grad_out)) {
    if (!py::isinstance<tensor::Tensor>(grad_out)) {
      hook_grad_.clear();
      py::object code_obj = py::getattr(hook_, "__code__");
      py::object co_name = py::getattr(code_obj, "co_name");
      MS_EXCEPTION(TypeError) << "The output type of:" << py::str(co_name) << " should be a tensor but got "
                              << py::cast<std::string>(grad_out.attr("__class__").attr("__name__")) << ".";
    }
    auto actual_out_tensor = py::cast<tensor::TensorPtr>(grad_out);
    auto expected_out_tensor = py::cast<tensor::TensorPtr>(expected_grad_out);
    MS_EXCEPTION_IF_NULL(actual_out_tensor);
    MS_EXCEPTION_IF_NULL(expected_out_tensor);
    if (actual_out_tensor->GetShapeAndDataTypeInfo() != expected_out_tensor->GetShapeAndDataTypeInfo()) {
      hook_grad_.clear();
      py::object code_obj = py::getattr(hook_, "__code__");
      py::object co_name = py::getattr(code_obj, "co_name");
      MS_EXCEPTION(ValueError) << "The output type of " << py::str(co_name)
                               << " is not consistent with the expected, it should be "
                               << expected_out_tensor->GetShapeAndDataTypeInfo() << ", but got "
                               << actual_out_tensor->GetShapeAndDataTypeInfo();
    }
  }
}

BaseRef PrimitivePy::RunCellBpropFunction(const py::tuple &py_args) const {
  SyncData(py_args);
  auto size = py_args.size();
  constexpr size_t grad_param_nums = 2;
  py::tuple input_args(size - grad_param_nums);
  for (size_t i = 0; i < size - grad_param_nums; ++i) {
    input_args[i] = py_args[i];
  }
  py::tuple convert_args(py_args.size());
  ConvertCTensorToPyTensor(py_args, &convert_args);
  auto inst = pynative::PynativeExecutor::GetInstance();
  MS_EXCEPTION_IF_NULL(inst);
  try {
    MS_LOG(DEBUG) << "Run bprop function start";
    inst->NewGraph(hook_, input_args.cast<py::args>());
    py::object grads_obj = hook_(*convert_args);
    py::tuple grads = check_bprop_out(grads_obj, py_args, bprop_cls_name_);
    inst->EndGraph(hook_, grads_obj, input_args.cast<py::args>());
    MS_LOG(DEBUG) << "Run bprop function end";
    return std::make_shared<PyObjectRef>(grads);
  } catch (std::exception &bt) {
    inst->ClearRes();
    std::rethrow_exception(std::current_exception());
  }
}

BaseRef PrimitivePy::RunCellHookFunction(const py::tuple &py_args) const {
  constexpr size_t grad_input_index = 1;
  constexpr size_t grad_output_index = 2;
  constexpr size_t input_param_nums = 3;
  SyncData(py_args[grad_output_index]);

  py::object obj;
  auto cell_id = GetValue<std::string>(this->GetAttr(kCellIDAttrName));
  auto iter = hook_grad_.find(cell_id);
  if (iter != hook_grad_.end()) {
    py::object code_obj = py::getattr(hook_, "__code__");
    py::object co_name = py::getattr(code_obj, "co_name");
    if (std::string(py::str(co_name)) == "staging_specialize") {
      py::object name_obj = py::getattr(hook_, "__name__");
      MS_LOG(EXCEPTION) << "Decorating hook function " << py::str(name_obj) << " with '@ms_function' is not supported.";
    }

    py::tuple convert_args(input_param_nums - 1);
    py::tuple input_args(input_param_nums - 1);
    input_args[0] = iter->second;
    input_args[1] = py_args[grad_output_index];
    ConvertCTensorToPyTensor(input_args, &convert_args);
    auto hook_args = py::tuple(input_param_nums);
    hook_args[0] = cell_id;
    hook_args[grad_input_index] = py::make_tuple(convert_args[0]);
    hook_args[grad_output_index] = py::make_tuple(convert_args[1]);
    obj = hook_(*hook_args);
    if (py::isinstance<py::none>(obj)) {
      obj = py_args[grad_output_index];
    }
    CheckHookConsistency(obj, py_args[grad_output_index]);
    (void)hook_grad_.erase(cell_id);
  } else {
    hook_grad_[cell_id] = py_args[grad_output_index];
    obj = py_args[grad_output_index];
  }
  obj = py::make_tuple(obj);
  return std::make_shared<PyObjectRef>(obj);
}

BaseRef PrimitivePy::RunVariableHookFunction(const py::tuple &py_args) const {
  py::object code_obj = py::getattr(hook_, "__code__");
  py::object co_name = py::getattr(code_obj, "co_name");
  if (std::string(py::str(co_name)) == "staging_specialize") {
    py::object name_obj = py::getattr(hook_, "__name__");
    MS_LOG(EXCEPTION) << "Decorating hook function " << py::str(name_obj) << " with '@ms_function' is not supported.";
  }

  constexpr size_t grad_output_index = 2;
  SyncData(py_args[grad_output_index]);
  py::object obj = hook_(py::make_tuple(py_args[grad_output_index]));
  if (py::isinstance<py::none>(obj)) {
    obj = py_args[grad_output_index];
  }
  CheckHookConsistency(obj, py_args[grad_output_index]);
  obj = py::make_tuple(obj);
  return std::make_shared<PyObjectRef>(obj);
}

BaseRef PrimitivePy::RunHookFunction(const VectorRef &args) const {
  py::tuple py_args = ConvertDatatoPyTuple(args);
  bool is_bprop = this->HasAttr(kBpropAttrName);
  if (is_bprop) {
    return RunCellBpropFunction(py_args);
  }
  bool is_cell = this->HasAttr(kCellHookAttrName);
  if (is_cell) {
    return RunCellHookFunction(py_args);
  }
  return RunVariableHookFunction(py_args);
}

py::function PrimitivePy::GetComputeFunction() const {
  static const char *const compute_func_name = "vm_impl";

  if (py::hasattr(python_obj_, compute_func_name)) {
    MS_LOG(DEBUG) << name() << " compute_func_name";
    py::function fn = python_obj_.attr(compute_func_name).cast<py::function>();
    return fn;
  }

  static const std::string vm_module = "mindspore.ops.vm_impl_registry";
  static const std::string get_vm_impl_fn = "get_vm_impl_fn";
  MS_LOG(DEBUG) << name() << ": get_vm_impl_fn";
  py::function get_fn = parse::python_adapter::GetPyFn(vm_module, get_vm_impl_fn);
  py::function vm_fn = get_fn(python_obj_);
  if (py::isinstance<py::none>(vm_fn)) {
    MS_LOG(DEBUG) << "Cannot find " << python_obj_.attr("__class__").attr("__name__").cast<std::string>();
    vm_fn = mindspore::GetComputeFunction(Primitive::name());
  }
  return vm_fn;
}

py::dict PrimitivePy::GetAttrDict() {
  py::dict attr_dict;
  for (auto &attr : attrs_) {
    attr_dict[py::str(attr.first)] = ValueToPyData(attr.second);
  }
  return attr_dict;
}

void PrimitivePy::CopyHookFunction(const PrimitivePtr &primitive) {
  MS_EXCEPTION_IF_NULL(primitive);
  if (!primitive->isa<PrimitivePy>()) {
    MS_LOG(EXCEPTION) << "Cannot copy a primitive which is not python primitive hook function to python primitive!";
  }
  auto primitive_py = primitive->cast<PrimitivePyPtr>();
  MS_EXCEPTION_IF_NULL(primitive_py);
  this->set_hook(primitive_py->hook());
  if (primitive_py->HasAttr(kBpropAttrName)) {
    set_bprop_cls_name(primitive_py->bprop_cls_name_);
    (void)this->AddAttr(kBpropAttrName, primitive_py->GetAttr(kBpropAttrName));
  }
}

BaseRef PrimitivePy::RunComputeFunction(const VectorRef &args) const {
  auto py_args = ConvertDatatoPyTuple(args);
  auto result = this->RunPyComputeFunction(py_args);
  if (py::isinstance<py::none>(result)) {
    return std::make_shared<BaseRef>(nullptr);
  }
  return std::make_shared<PyObjectRef>(result);
}

py::object PrimitivePy::RunPyComputeFunction(const py::tuple &py_args) const {
  auto func = this->GetComputeFunction();
  if (py::isinstance<py::none>(func)) {
    return py::none();
  }
  auto result = func(*py_args);
  return result;
}

bool PrimitivePy::HasComputeFunction() const {
  auto func = GetComputeFunction();
  return !py::isinstance<py::none>(func);
}

PrimitivePtr PrimitivePy::Clone() {
  auto clone_fn = python_obj_.attr("_clone");
  py::object obj_adapter = clone_fn();
  auto prim_adapter = obj_adapter.cast<PrimitivePyAdapterPtr>();
  auto prim = std::make_shared<PrimitivePy>(obj_adapter, prim_adapter);
  prim_adapter->set_attached_primitive(prim);
  return prim;
}

py::dict PrimitivePy::RunInfer(const py::tuple &args) {
  if (!HasPyObj()) {
    MS_LOG(EXCEPTION) << "[" << this->ToString() << "]: pyobj is empty";
  }
  // Python obj could be replaced as None, so it will losed the original info when throw exception in python.
  if (!py::hasattr(python_obj_, PY_PRIM_METHOD_INFER)) {
    MS_LOG(EXCEPTION) << "prim:" << ToString() << " has no attr:" << PY_PRIM_METHOD_INFER;
  }
  auto infer_fuc = python_obj_.attr(PY_PRIM_METHOD_INFER);
  return infer_fuc(*args);
}

void PrimitivePy::RunCheck(const py::tuple &args) {
  if (!HasPyObj()) {
    MS_LOG(EXCEPTION) << "[" << this->ToString() << "]: pyobj is empty";
  }
  // Python obj could be replaced as None, so it will losed the original info when throw exception in python.
  if (!py::hasattr(python_obj_, PY_PRIM_METHOD_CHECK)) {
    MS_LOG(EXCEPTION) << "prim:" << ToString() << " has no attr:" << PY_PRIM_METHOD_CHECK;
  }
  auto check_func = python_obj_.attr(PY_PRIM_METHOD_CHECK);
  (void)check_func(*args);
}

py::object PrimitivePy::RunInferValue(const py::tuple &args) {
  if (!HasPyObj()) {
    MS_LOG(EXCEPTION) << "[" << this->ToString() << "]: pyobj is empty";
  }
  // Python obj could be replaced as None, so it will losed the original info when throw exception in python.
  if (!py::hasattr(python_obj_, PY_PRIM_METHOD_INFER_VALUE)) {
    MS_LOG(EXCEPTION) << "prim:" << ToString() << " has no attr:" << PY_PRIM_METHOD_INFER_VALUE;
  }
  auto infer_value = python_obj_.attr(PY_PRIM_METHOD_INFER_VALUE);
  return infer_value(*args);
}

PrimitivePyAdapter::PrimitivePyAdapter(const py::str &name) : name_(name) {}

void PrimitivePyAdapter::AddPyAttr(const py::str &name, const py::object &obj) {
  std::string attr_name = name;
  ValuePtr converted_ret = nullptr;
  if (py::isinstance<py::module>(obj)) {
    MS_LOG(EXCEPTION) << "Call 'add_attr' to add attribute to primitive failed,"
                      << " not support py::module to be attribute value; primitive name: " << this->name_
                      << ", attribute name: " << attr_name << " attribute value: " << py::str(obj);
  }
  bool converted = parse::ConvertData(obj, &converted_ret);
  if (!converted) {
    MS_LOG(EXCEPTION) << "Call 'add_attr' to add attribute to primitive failed,"
                      << " convert python obj to MindSpore obj failed; primitive name: " << this->name_
                      << ", attribute name:" << attr_name << ", attribute value:" << py::str(obj)
                      << ", attribute type:" << py::cast<std::string>(obj.attr("__class__").attr("__name__"));
  }
  if (kOpAttrNameReplaceMap.find(attr_name) != kOpAttrNameReplaceMap.end()) {
    attr_name = kOpAttrNameReplaceMap[attr_name];
  }
  (void)CheckAndConvertUtils::ConvertAttrValueToInt(this->name_, name, &converted_ret);
  if (attr_name == "primitive_target") {
    MS_EXCEPTION_IF_NULL(converted_ret);
    if (!converted_ret->isa<StringImm>()) {
      MS_LOG(EXCEPTION) << "Call 'add_attr' to add attribute to primitive '" << this->name_
                        << "' failed, value of attribute 'primitive_target' must be CPU|GPU|Ascend but got "
                        << py::str(obj);
    }
    auto target = GetValue<std::string>(converted_ret);
    if (!target.empty() && target != kCPUDevice && target != kGPUDevice && target != kAscendDevice &&
        target != "Device") {
      MS_LOG(EXCEPTION) << "Call 'add_attr' to add attribute to primitive '" << this->name_
                        << "' failed, value of attribute 'primitive_target' must be CPU|GPU|Ascend but got "
                        << py::str(obj);
    }
    if (target != kCPUDevice && target != kGPUDevice) {
      auto context_ptr = MsContext::GetInstance();
      MS_EXCEPTION_IF_NULL(context_ptr);
      context_ptr->set_param<bool>(MS_CTX_ALREADY_SET_ENABLE_MINDRT, true);
    }
  }

  attrs_[attr_name] = converted_ret;
  auto prim = attached_primitive_.lock();
  if (prim != nullptr) {
    (void)prim->AddAttr(attr_name, converted_ret);
  }
}

void PrimitivePyAdapter::DelPyAttr(const py::str &name) {
  (void)attrs_.erase(name);
  auto prim = attached_primitive_.lock();
  if (prim != nullptr) {
    (void)prim->DelAttr(name);
  }
}

py::dict PrimitivePyAdapter::GetAttrDict() {
  auto prim = attached_primitive_.lock();
  if (prim != nullptr) {
    return prim->GetAttrDict();
  }

  py::dict attr_dict;
  for (auto &attr : attrs_) {
    attr_dict[py::str(attr.first)] = ValueToPyData(attr.second);
  }
  return attr_dict;
}

void PrimitivePyAdapter::set_prim_type(const PrimType t) {
  prim_type_ = t;
  auto prim = attached_primitive_.lock();
  if (prim != nullptr) {
    prim->set_prim_type(t);
  }
}
void PrimitivePyAdapter::set_const_prim(bool is_const_prim) {
  is_const_prim_ = is_const_prim;
  auto prim = attached_primitive_.lock();
  if (prim != nullptr) {
    prim->set_const_prim(is_const_prim);
  }
}
void PrimitivePyAdapter::set_const_input_indexes(const std::vector<size_t> &const_input_indexes) {
  const_input_indexes_ = const_input_indexes;
  auto prim = attached_primitive_.lock();
  if (prim != nullptr) {
    prim->set_const_input_indexes(const_input_indexes);
  }
}

void PrimitivePyAdapter::set_signatures(const std::vector<Signature> &signatures) {
  signatures_ = signatures;
  auto prim = attached_primitive_.lock();
  if (prim != nullptr) {
    prim->set_signatures(signatures);
  }
}

void PrimitivePyAdapter::set_hook(const py::function &hook) {
  hook_ = hook;
  auto prim = attached_primitive_.lock();
  if (prim != nullptr) {
    prim->set_hook(hook);
  }
}

void PrimitivePyAdapter::set_instance_name(const std::string &s) {
  instance_name_ = s;
  auto prim = attached_primitive_.lock();
  if (prim != nullptr) {
    prim->set_instance_name(s);
  }
}

void PrimitivePyAdapter::set_attached_primitive(const PrimitivePyPtr &prim) {
  if (attached_primitive_.lock() != nullptr) {
    MS_LOG(EXCEPTION) << "PrimitivePyAdapter can't attach to multi Primitive.";
  }
  MS_EXCEPTION_IF_NULL(prim);
  attached_primitive_ = prim;
}

REGISTER_PYBIND_DEFINE(Primitive_, ([](const py::module *m) {
                         (void)py::enum_<PrimType>(*m, "prim_type", py::arithmetic())
                           .value("unknown", PrimType::kPrimTypeUnknown)
                           .value("builtin", PrimType::kPrimTypeBuiltIn)
                           .value("py_infer_shape", PrimType::kPrimTypePyInfer)
                           .value("user_custom", PrimType::kPrimTypeUserCustom)
                           .value("py_infer_check", PrimType::kPrimTypePyCheck);
                         (void)py::class_<PrimitivePyAdapter, std::shared_ptr<PrimitivePyAdapter>>(*m, "Primitive_")
                           .def_readonly(PYTHON_PRIMITIVE_FLAG, &PrimitivePyAdapter::parse_info_)
                           .def(py::init<py::str &>())
                           .def("add_attr", &PrimitivePyAdapter::AddPyAttr, "add primitive attr")
                           .def("del_attr", &PrimitivePyAdapter::DelPyAttr, "del primitive attr")
                           .def("get_attr_dict", &PrimitivePyAdapter::GetAttrDict, "get primitive attr")
                           .def("set_prim_type", &PrimitivePyAdapter::set_prim_type, "Set primitive type.")
                           .def("set_const_prim", &PrimitivePyAdapter::set_const_prim, "Set primitive is const.")
                           .def("set_const_input_indexes", &PrimitivePyAdapter::set_const_input_indexes,
                                "Set primitive const input indexes.")
                           .def("set_signatures", &PrimitivePyAdapter::set_signatures,
                                "Set primitive inputs signature.")
                           .def("register_hook", &PrimitivePyAdapter::set_hook, "Set primitive hook function.")
                           .def("set_instance_name", &PrimitivePyAdapter::set_instance_name,
                                "Set primitive instance name.");
                       }));
}  // namespace mindspore
