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
""" test graph fallback """
import math
import numpy as np
from mindspore import ms_function, context, Tensor

context.set_context(mode=context.GRAPH_MODE)


def test_fallback_abs():
    """
    Feature: JIT Fallback
    Description: Test abs() in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = -1
        return abs(x)
    assert foo() == 1


def test_fallback_all():
    """
    Feature: JIT Fallback
    Description: Test all() in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = (0, 1, 2, 3)
        return all(x)
    assert not foo()


def test_fallback_any():
    """
    Feature: JIT Fallback
    Description: Test any() in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = (0, 1, 0, 0)
        return any(x)
    out = foo()
    assert out


def test_fallback_bin():
    """
    Feature: JIT Fallback
    Description: Test bin() in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = bin(3)
        return x
    assert foo() == '0b11'


def test_fallback_bool():
    """
    Feature: JIT Fallback
    Description: Test bool() in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = bool(1)
        return x
    assert foo()


def test_fallback_chr():
    """
    Feature: JIT Fallback
    Description: Test chr() in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = chr(0x61)
        return x
    assert foo() == 'a'


def test_fallback_complex():
    """
    Feature: JIT Fallback
    Description: Test complex() in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = complex(1, 2)
        return Tensor(x)
    res = foo()
    expect_res = np.array(1 + 2j)
    assert isinstance(res, Tensor)
    assert np.all(res.asnumpy() == expect_res)


def test_fallback_dict():
    """
    Feature: JIT Fallback
    Description: Test dict() in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        dict_x = dict(a=1, b=2, c=3)
        return dict_x
    print(foo())


def test_fallback_divmod():
    """
    Feature: JIT Fallback
    Description: Test divmod() in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = divmod(7, 2)
        return x
    assert foo() == (3, 1)


def test_fallback_float():
    """
    Feature: JIT Fallback
    Description: Test float() in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = float(1)
        return x

    out = foo()
    assert math.isclose(out, 1, abs_tol=1e-5)


def test_fallback_hash():
    """
    Feature: JIT Fallback
    Description: Test hash() in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = hash(1)
        return x
    assert foo() == 1


def test_fallback_hex():
    """
    Feature: JIT Fallback
    Description: Test hex() in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = hex(255)
        return x
    assert foo() == '0xff'


def test_fallback_int():
    """
    Feature: JIT Fallback
    Description: Test int() in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = int(5.0)
        return x
    assert foo() == 5


def test_fallback_list():
    """
    Feature: JIT Fallback
    Description: Test list() in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = list([1, 2, 3])
        return x
    print(foo())


def test_fallback_max():
    """
    Feature: JIT Fallback
    Description: Test max() in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = max([1, 2, 3])
        return x
    assert foo() == 3


def test_fallback_min():
    """
    Feature: JIT Fallback
    Description: Test min() in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = min([1, 2, 3])
        return x
    assert foo() == 1


def test_fallback_oct():
    """
    Feature: JIT Fallback
    Description: Test oct() in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = oct(8)
        return x
    assert foo() == '0o10'


def test_fallback_ord():
    """
    Feature: JIT Fallback
    Description: Test ord() in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = ord('a')
        return x
    assert foo() == 97


def test_fallback_reversed():
    """
    Feature: JIT Fallback
    Description: Test reversed() in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = reversed([1, 2, 3])
        return list(x)
    assert foo() == (3, 2, 1)


def test_fallback_round():
    """
    Feature: JIT Fallback
    Description: Test round() in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = round(0.6)
        return x
    assert foo() == 1


def test_fallback_set():
    """
    Feature: JIT Fallback
    Description: Test set() in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = set([1, 2, 1])
        return x
    assert list(foo()) == [1, 2]


def test_fallback_slice():
    """
    Feature: JIT Fallback
    Description: Test slice() in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        slice_x = slice(5)
        arr = range(10)
        return arr[slice_x]
    assert list(foo()) == [0, 1, 2, 3, 4]


def test_fallback_sorted():
    """
    Feature: JIT Fallback
    Description: Test sorted() in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = sorted([5, 3, 1, 4, 2])
        return x
    assert list(foo()) == [1, 2, 3, 4, 5]


def test_fallback_str():
    """
    Feature: JIT Fallback
    Description: Test str() in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = str(10)
        return x
    assert foo() == '10'


def test_fallback_sum():
    """
    Feature: JIT Fallback
    Description: Test sum() in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = sum([1, 2, 3])
        return x
    assert foo() == 6


def test_fallback_tuple():
    """
    Feature: JIT Fallback
    Description: Test tuple() in graph mode.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        tuple_x = tuple([1, 2, 3])
        return tuple_x
    print(foo())
