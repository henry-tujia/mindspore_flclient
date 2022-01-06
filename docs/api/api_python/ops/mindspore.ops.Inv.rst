mindspore.ops.Inv
=================

.. py:class:: mindspore.ops.Inv()

    按元素计算输入Tensor的倒数。

    .. math::
        out_i = \frac{1}{x_{i} }

    **输入：**

    **x** (Tensor) - 任意维度的Tensor。数据类型必须是float16、float32或int32。

    **输出：**

    Tensor，shape和数据类型与 `x` 相同。

    **异常：**

    - **TypeError** - `x` 的数据类型不是float16、float32或int32。

    **支持平台：**

    ``Ascend``

    **样例：**

    >>> inv = ops.Inv()
    >>> x = Tensor(np.array([0.25, 0.4, 0.31, 0.52]), mindspore.float32)
    >>> output = inv(x)
    >>> print(output)
    [4.        2.5       3.2258065 1.923077 ]
    