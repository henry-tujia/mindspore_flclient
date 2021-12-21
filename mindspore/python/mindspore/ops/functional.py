# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
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

"""The names of functional part are summarized here."""

from mindspore.common._register_for_tensor import tensor_operator_registry
from mindspore.common import ms_function
from mindspore.common import Tensor
from mindspore.nn.grad.cell_grad import _JvpInner
from mindspore.nn.grad.cell_grad import _VjpInner
from mindspore.ops import _constants
from mindspore.ops.primitive import constexpr
from .primitive import Primitive
from . import operations as P
from .operations import _grad_ops
from .operations import _csr_ops
from .composite import _Grad, Shard
from .._c_expression import security

typeof = Primitive('typeof')
hastype = Primitive('hastype')
cast = P.Cast()
dtype = P.DType()
isconstant = Primitive('is_constant')
isconstant.set_const_prim(True)

issubclass_ = P.IsSubClass()
isinstance_ = P.IsInstance()
eye = P.Eye()
fill = P.Fill()
tile = P.Tile()
select = P.Select()
size = P.Size()
ones_like = P.OnesLike()
shape = P.Shape()
dyn_shape = P.DynamicShape()
rank = P.Rank()
reshape = P.Reshape()

merge = P.Merge()
geswitch = P.GeSwitch()
addn = P.AddN()
absolute = P.Abs()
tensor_add = P.Add()
add = tensor_add
neg_tensor = P.Neg()
tensor_lt = P.Less()
less = tensor_lt
tensor_le = P.LessEqual()
le = tensor_le
tensor_gt = P.Greater()
gt = tensor_gt
tensor_ge = P.GreaterEqual()
ge = tensor_ge
tensor_sub = P.Sub()
sub = tensor_sub
tensor_mul = P.Mul()
mul = tensor_mul
tensor_div = P.RealDiv()
div = tensor_div
tensor_floordiv = P.FloorDiv()
floordiv = tensor_floordiv
tensor_pow = P.Pow()
pows = tensor_pow
tensor_mod = P.FloorMod()
floormod = tensor_mod
tensor_exp = P.Exp()
exp = tensor_exp
tensor_expm1 = P.Expm1()
tensor_slice = P.Slice()
strided_slice = P.StridedSlice()
same_type_shape = P.SameTypeShape()
check_bprop = P.CheckBprop()
equal = P.Equal()
not_equal = P.NotEqual()
isfinite = P.IsFinite()
isnan = P.IsNan()
assign_sub = P.AssignSub()
assign_add = P.AssignAdd()
assign = P.Assign()
square = P.Square()
sqrt = P.Sqrt()
log = P.Log()
reduce_sum = P.ReduceSum()
reduce_max = P.ReduceMax()
reduce_min = P.ReduceMin()
reduce_mean = P.ReduceMean()
reduce_prod = P.ReduceProd()
tensor_slice = P.Slice()
maximum = P.Maximum()
minimum = P.Minimum()
floor = P.Floor()
logical_not = P.LogicalNot()
logical_or = P.LogicalOr()
logical_and = P.LogicalAnd()
sin = P.Sin()
cos = P.Cos()
tan = P.Tan()
asin = P.Asin()
acos = P.ACos()
atan = P.Atan()
sinh = P.Sinh()
cosh = P.Cosh()
tanh = P.Tanh()
asinh = P.Asinh()
acosh = P.Acosh()
atanh = P.Atanh()
atan2 = P.Atan2()
bitwise_and = P.BitwiseAnd()
bitwise_or = P.BitwiseOr()
bitwise_xor = P.BitwiseXor()
invert = P.Invert()
erf = P.Erf()
erfc = P.Erfc()
sort = P.Sort()
tensor_range = P.Range()

scalar_to_array = P.ScalarToArray()
scalar_to_tensor = P.ScalarToTensor()
tuple_to_array = P.TupleToArray()
scalar_cast = P.ScalarCast()
if not security.enable_security():
    print_ = P.Print()
expand_dims = P.ExpandDims()
transpose = P.Transpose()
squeeze = P.Squeeze()
scatter_nd = P.ScatterNd()
gather = P.Gather()
gather_d = P.GatherD()
gather_nd = P.GatherNd()
scatter_update = P.ScatterUpdate()
tensor_scatter_update = P.TensorScatterUpdate()
scatter_nd_update = P.ScatterNdUpdate()
stack = P.Stack()

csr_mul = _csr_ops.CSRMul()
csr_mv = _csr_ops.CSRMV()
csr_reduce_sum = _csr_ops.CSRReduceSum()


def pack(x):
    """Call stack in this pack function."""
    print("WARNING: 'pack' is deprecated from version 1.1 and will be removed in a future version, use 'stack' instead"
          ".")
    return stack(x)


partial = P.Partial()
# depend: mount a node to another node
depend = P.Depend()
identity = P.identity()

@constexpr
def _convert_grad_position_type(grad_position):
    """Check and convert the type and size of grad position index."""
    if isinstance(grad_position, tuple):
        for gp in grad_position:
            if not isinstance(gp, int):
                raise TypeError(f"For 'F.grad', the element in 'grad_position' should be int, "
                                f"but got {type(gp).__name__}")
            if gp < 0:
                raise ValueError("The element in grad_position must be >= 0.")
    elif isinstance(grad_position, int):
        if grad_position < 0:
            raise ValueError("grad_position must be >= 0.")
        grad_position = (grad_position,)
    else:
        raise TypeError(f"For 'F.grad', the 'grad_position' should be int or tuple, "
                        f"but got {type(grad_position).__name__}")
    return grad_position


grad_by_position = _Grad(get_by_list=False, sens_param=False, get_by_position=True)
grad_by_position_with_sens = _Grad(get_by_list=False, sens_param=True, get_by_position=True)

def grad(fn, grad_position=0, sens_param=False):
    r"""
    A wrapper function to generate the gradient function for the input function.

    Args:
        fn (Union(Cell, function)): Function to do GradOperation.
        grad_position (Union(int, tuple[int])): If int, get the gradient with respect to single input.
            If tuple, get the gradients with respect to selected inputs. 'grad_position' begins with 0. Default: 0.
        sens_param (bool): Whether to append sensitivity (gradient with respect to output) as input.
            If sens_param is False, a 'ones_like(outputs)' sensitivity will be attached automatically. Default: False.

    Returns:
        Function, returns the gradient function for the input function or cell.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.nn as nn
        >>> import mindspore.context as context
        >>> from mindspore import Tensor
        >>> from mindspore.ops.functional import grad
        >>> context.set_context(mode=context.GRAPH_MODE)
        >>> class Net(nn.Cell):
        ...     def construct(self, x, y, z):
        ...         return x*y*z
        >>> x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
        >>> y = Tensor(np.array([[-2, 3], [-1, 2]]).astype(np.float32))
        >>> z = Tensor(np.array([[0, 3], [5, -1]]).astype(np.float32))
        >>> net = Net()
        >>> output = grad(net, grad_position=(1, 2))(x, y, z)
        >>> print(output)
        (Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 0.00000000e+00,  6.00000000e+00],
         [ 1.50000000e+01, -4.00000000e+00]]), Tensor(shape=[2, 2], dtype=Float32, value=
        [[-2.00000000e+00,  6.00000000e+00],
         [-3.00000000e+00,  8.00000000e+00]]))
    """
    grad_position = _convert_grad_position_type(grad_position)
    if sens_param:
        return grad_by_position_with_sens(fn, None, grad_position)
    return grad_by_position(fn, None, grad_position)


def jvp(fn, inputs, v):
    """
    Compute the jacobian-vector-product of the given network.

    Args:
        fn (Function or Cell): The function or net that takes Tensor inputs and returns a tensor or tuple of Tensors.
        inputs (Tensor or tuple or list): The inputs to `fn`.
        v (Tensor or tuple or list): The shape and type of v should be the same as inputs.

    Returns:
        Tuple, tuple of output and jvp.

        - **netout** (Tensors or Tuple of Tensors) - The output of "fn(inputs)".
        - **jvp** (Tensors or Tuple of Tensors) - The result of the dot product.

    Raises:
        TypeError: If the input is not a tensor or tuple or list of tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore.ops import functional as F
        >>> from mindspore import Tensor
        >>> class Net(nn.Cell):
        ...     def construct(self, x, y):
        ...         return x**3 + y
        >>> x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
        >>> y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
        >>> v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
        >>> output = F.jvp(Net(), (x, y), (v, v))
        >>> print(output[0])
        [[ 2. 10.]
         [30. 68.]]
        >>> print(output[1])
        [[ 4. 13.]
         [28. 49.]]
    """
    jvp_inner = _JvpInner()

    @ms_function
    def _wrap_container(*arg):
        args = arg[1:]
        vectors = arg[0]
        return jvp_inner(fn, vectors, *args)
    if not isinstance(inputs, (Tensor, tuple, list)):
        _raise_type_error()
    if isinstance(inputs, (tuple, list)):
        return _wrap_container(v, *inputs)
    return _wrap_container(v, inputs)


def vjp(fn, inputs, v):
    """
    Compute the vector-jacobian-product of the given network.

    Args:
        fn (Function or Cell): The function or net that takes Tensor inputs and returns a tensor or tuple of Tensors.
        inputs (Tensor or tuple or list): The inputs to `fn`.
        v (Tensor or tuple or list): The shape and type of v should be the same as outputs.

    Returns:
        Tuple, tuple of output and vjp.

        - **netout** (Tensors or Tuple of Tensors) - The output of "fn(inputs)".
        - **vjp** (Tensors or Tuple of Tensors) - The result of the dot product.

    Raises:
        TypeError: If the input is not a tensor or tuple or list of tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore.ops import functional as F
        >>> from mindspore import Tensor
        >>> class Net(nn.Cell):
        ...     def construct(self, x, y):
        ...         return x**3 + y
        >>> x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
        >>> y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
        >>> v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
        >>> output = F.vjp(Net(), (x, y), v)
        >>> print(output[0])
        [[ 2. 10.]
         [30. 68.]]
        >>> print(output[1])
        (Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 3.00000000e+00,  1.20000000e+01],
         [ 2.70000000e+01,  4.80000000e+01]]), Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 1.00000000e+00,  1.00000000e+00],
         [ 1.00000000e+00,  1.00000000e+00]]))
    """
    vjp_inner = _VjpInner()

    @ms_function
    def wrap_container(*arg):
        args = arg[:-1]
        vectors = arg[-1]
        return vjp_inner(fn, *args, vectors)
    if not isinstance(inputs, (Tensor, tuple, list)):
        _raise_type_error()
    if isinstance(inputs, (tuple, list)):
        return wrap_container(*inputs, v)
    return wrap_container(inputs, v)

shard_fn = Shard()
def shard(fn, in_axes, out_axes, device, level=0):
    return shard_fn(fn, in_axes, out_axes, device, level)


@constexpr
def _raise_type_error():
    raise TypeError("The inputs type should be a Tensor, tuple or list of Tensor.")


tuple_setitem = Primitive('tuple_setitem')
tuple_getitem = Primitive(_constants.kTupleGetItem)
list_getitem = Primitive('list_getitem')
list_setitem = Primitive('list_setitem')
dict_getitem = Primitive('dict_getitem')
dict_setitem = Primitive('dict_setitem')
tuple_div = Primitive("tuple_div")
tuple_len = Primitive("tuple_len")
list_len = Primitive("list_len")
tuple_reversed = Primitive("tuple_reversed")
make_range = Primitive("make_range")
make_tuple = Primitive('MakeTuple')
make_dict = Primitive('make_dict')
make_list = Primitive('make_list')
make_slice = Primitive('make_slice')
tuple_equal = Primitive("tuple_equal")
list_equal = Primitive("list_equal")
make_ref = Primitive("make_ref")

scalar_add = Primitive(_constants.kScalarAdd)
scalar_mul = Primitive(_constants.kScalarMul)
scalar_sub = Primitive(_constants.kScalarSub)
scalar_div = Primitive(_constants.kScalarDiv)
scalar_floordiv = Primitive(_constants.kScalarFloordiv)
scalar_log = Primitive('scalar_log')
scalar_pow = Primitive(_constants.kScalarPow)
scalar_gt = Primitive('scalar_gt')
scalar_ge = Primitive('scalar_ge')
scalar_le = Primitive('scalar_le')
scalar_lt = Primitive('scalar_lt')
scalar_eq = Primitive('scalar_eq')
scalar_ne = Primitive('scalar_ne')
scalar_uadd = Primitive(_constants.kScalarUadd)
scalar_usub = Primitive(_constants.kScalarUsub)
scalar_mod = Primitive(_constants.kScalarMod)
string_eq = Primitive('string_equal')
string_concat = Primitive('string_concat')
bool_not = Primitive("bool_not")
bool_or = Primitive("bool_or")
bool_and = Primitive("bool_and")
bool_eq = Primitive("bool_eq")
logical_and = P.LogicalAnd()
logical_or = P.LogicalOr()
logical_not = P.LogicalNot()
cumsum = P.CumSum()
cumprod = P.CumProd()
tensor_scatter_add = P.TensorScatterAdd()
array_to_scalar = Primitive('array_to_scalar')
is_ = Primitive("is_")
is_not = Primitive("is_not")
in_dict = Primitive("in_dict")
not_in_dict = Primitive("not_in_dict")
mixed_precision_cast = Primitive("mixed_precision_cast")
broadcast_gradient_args = Primitive('BroadcastGradientArgs')
array_reduce = Primitive('array_reduce')
zeros_like = P.ZerosLike()
distribute = Primitive('distribute')
embed = Primitive('embed')
ref_to_embed = _grad_ops.RefToEmbed()
env_setitem = Primitive('env_setitem')
env_getitem = Primitive('env_getitem')
env_add = Primitive('env_add')
J = Primitive('J')
SliceGetItem = Primitive("SliceGetItem")
switch = Primitive('Switch')
switch_layer = Primitive('switch_layer')
# for sum bprop
reduced_shape = Primitive("reduced_shape")
# shape_mul:input must be shape multiply elements in tuple(shape)
shape_mul = Primitive("shape_mul")
# a primitive to compare between tuple.
stop_gradient = Primitive("stop_gradient")

make_row_tensor = Primitive('MakeRowTensor')
row_tensor_get_values = Primitive('RowTensorGetValues')
row_tensor_get_indices = Primitive('RowTensorGetIndices')
row_tensor_get_dense_shape = Primitive('RowTensorGetDenseShape')
row_tensor_add = Primitive('RowTensorAdd')

make_sparse_tensor = Primitive('MakeSparseTensor')
sparse_tensor_get_values = Primitive('SparseTensorGetValues')
sparse_tensor_get_indices = Primitive('SparseTensorGetIndices')
sparse_tensor_get_dense_shape = Primitive('SparseTensorGetDenseShape')

make_csr_tensor = Primitive('MakeCSRTensor')
csr_tensor_get_values = Primitive('CSRTensorGetValues')
csr_tensor_get_indices = Primitive('CSRTensorGetIndices')
csr_tensor_get_indptr = Primitive('CSRTensorGetIndptr')
csr_tensor_get_shape = Primitive('CSRTensorGetDenseShape')

tensor_operator_registry.register('all', P.ReduceAll)
tensor_operator_registry.register('any', P.ReduceAny)
tensor_operator_registry.register('abs', P.Abs)
tensor_operator_registry.register('mean', P.ReduceMean)
tensor_operator_registry.register('reshape', P.Reshape)
tensor_operator_registry.register('transpose', P.Transpose)
tensor_operator_registry.register('broadcast_to', P.BroadcastTo)
tensor_operator_registry.register('matmul', P.MatMul)
tensor_operator_registry.register('argmax', P.Argmax)
tensor_operator_registry.register('cumsum', P.CumSum)
tensor_operator_registry.register('reduce_max', P.ReduceMax)
tensor_operator_registry.register('reduce_min', P.ReduceMin)
tensor_operator_registry.register('maximum', P.Maximum)
tensor_operator_registry.register('minimum', P.Minimum)
tensor_operator_registry.register('fill', P.Fill)
tensor_operator_registry.register('tile', P.Tile)
tensor_operator_registry.register('logical_not', P.LogicalNot)
tensor_operator_registry.register('sum', P.ReduceSum)
tensor_operator_registry.register('split', P.Split)
# ms cannot support Tensor(True) compare
tensor_operator_registry.register('__eq__', equal)
tensor_operator_registry.register('__ne__', not_equal)
tensor_operator_registry.register('__neg__', neg_tensor)
tensor_operator_registry.register('__lt__', tensor_lt)
tensor_operator_registry.register('__le__', tensor_le)
tensor_operator_registry.register('__gt__', tensor_gt)
tensor_operator_registry.register('__ge__', tensor_ge)
tensor_operator_registry.register('__logical_not__', logical_not)
tensor_operator_registry.register('shape', shape)
tensor_operator_registry.register('squeeze', squeeze)
# support GE backend for no compare operators
tensor_operator_registry.register('cast', cast)
tensor_operator_registry.register('shape_mul', shape_mul)
tensor_operator_registry.register('fill', fill)
tensor_operator_registry.register('concatenate', P.Concat)
tensor_operator_registry.register('eye', eye)
tensor_operator_registry.register('reduce_sum', reduce_sum)
tensor_operator_registry.register('tensor_slice', tensor_slice)
tensor_operator_registry.register('select', select)
tensor_operator_registry.register('gather_d', gather_d)
tensor_operator_registry.register('gather_nd', gather_nd)
tensor_operator_registry.register('stack', P.Stack)
tensor_operator_registry.register('log', log)
tensor_operator_registry.register('floor', floor)
# support sparse tensor operators
tensor_operator_registry.register('csr_mul', csr_mul)
__all__ = [name for name in dir() if name[0] != "_"]
__all__.remove('Primitive')
