mindspore.ops.Add
=================

.. py:class:: mindspore.ops.Add(*args, **kwargs)

    两个输入Tensor按元素相加。

    输入 `x` 和 `y` 遵循隐式类型转换规则，使数据类型保持一致。
    输入必须是两个Tensor，或一个Tensor和一个Scalar。
    当输入是两个Tensor时，它们的数据类型不能同时是bool，它们的shape可以广播。
    当输入是一个Tensor和一个Scalar时，Scalar只能是一个常数。

    .. math::

        out_{i} = x_{i} + y_{i}

    **输入：**

    - **x** (Union[Tensor, Number, bool]) - 第一个输入，是一个Number、bool值或数据类型为Number或bool的Tensor。
    - **y** (Union[Tensor, Number, bool]) - 第二个输入，当第一个输入是Tensor时，第二个输入应该是一个Number或bool值，或数据类型为Number或bool的Tensor。

    **输出：**

    Tensor，shape与广播后的shape相同，数据类型为两个输入中精度较高的类型。

    **异常：**

    - **TypeError** - `x` 和 `y` 不是Tensor、Number或bool。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> # 用例1: x和y都是Tensor。
    >>> add = ops.Add()
    >>> x = Tensor(np.array([1, 2, 3]).astype(np.float32))
    >>> y = Tensor(np.array([4, 5, 6]).astype(np.float32))
    >>> output = add(x, y)
    >>> print(output)
    [5.7.9.]
    >>> # 用例2: x是Scalar Tensor，y是Tensor。
    >>> add = ops.Add()
    >>> x = Tensor(1, mindspore.int32)
    >>> y = Tensor(np.array([4, 5, 6]).astype(np.float32))
    >>> output = add(x, y)
    >>> print(output)
    [5. 6. 7.]
    >>> # x的数据类型为int32，y的数据类型为float32。
    >>> # 输出的数据类型为高精度float32。
    >>> print(output.dtype)
    Float32
