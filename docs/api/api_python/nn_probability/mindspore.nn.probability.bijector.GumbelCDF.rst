mindspore.nn.probability.bijector.GumbelCDF
============================================

.. py:class:: mindspore.nn.probability.bijector.GumbelCDF(loc=0.0, scale=1.0, name='GumbelCDF')

    GumbelCDF Bijector。
    此Bijector对应的映射函数为：

    .. math::
        Y = g(x) = \exp(-\exp(\frac{-(X - loc)}{scale}))

    **参数：**

    - **loc** (float, list, numpy.ndarray, Tensor) - 位移因子，即上述公式中的loc。默认值：0.0。
    - **scale** (float, list, numpy.ndarray, Tensor) - 比例因子，即上述公式中的scale。默认值：1.0。
    - **name** (str) - Bijector名称。默认值：'GumbelCDF'。

    **支持平台：**

    ``Ascend`` ``GPU``

    .. note::
        `scale` 中元素必须大于零。对于 `inverse` 和 `inverse_log_jacobian` ，输入应在(0, 1)范围内。`loc` 和 `scale` 中元素的数据类型必须为float。如果 `loc` 、 `scale` 作为numpy.ndarray或Tensor传入，则它们必须具有相同的数据类型，否则将引发错误。

    **异常：**

    - **TypeError** - `loc` 或 `scale` 中元素的数据类型不为float，或 `loc` 和 `scale` 中元素的数据类型不相同。

    **支持平台：**

    ``Ascend`` ``GPU``

    **样例：**

    >>> import mindspore
    >>> import mindspore.nn as nn
    >>> import mindspore.nn.probability.bijector as msb
    >>> from mindspore import Tensor
    >>>
    >>> # 初始化GumbelCDF Bijector，loc设置为1.0和scale设置为2.0。
    >>> gumbel_cdf = msb.GumbelCDF(1.0, 2.0)
    >>> # 在网络中使用ScalarAffinebijector。
    >>> x = Tensor([1, 2, 3], dtype=mindspore.float32)
    >>> y = Tensor([0.1, 0.2, 0.3], dtype=mindspore.float32)
    >>> ans1 = gumbel_cdf.forward(x)
    >>> print(ans1.shape)
    (3,)
    >>> ans2 = gumbel_cdf.inverse(y)
    >>> print(ans2.shape)
    (3,)
    >>> ans3 = gumbel_cdf.forward_log_jacobian(x)
    >>> print(ans3.shape)
    (3,)
    >>> ans4 = gumbel_cdf.inverse_log_jacobian(y)
    >>> print(ans4.shape)
    (3,)

    .. py:method:: forward(value)

        正映射，计算输入随机变量 :math:`X = value` 经过映射后的值 :math:`Y = g(value)`。

        **参数：**

        - **value** (Tensor) - 输入随机变量的值。

        **返回：**

        Tensor, 输入随机变量的值。

    .. py:method:: forward_log_jacobian(value)

        计算正映射导数的对数值，即 :math:`\log(dg(x) / dx)`。

        **参数：**

        - **value** (Tensor) - 输入随机变量的值。

        **返回：**

        Tensor, 正映射导数的对数值。

    .. py:method:: inverse(value)

        正映射，计算输出随机变量 :math:`Y = value` 时对应的输入随机变量的值 :math:`X = g(value)`。

        **参数：**

        - **value** (Tensor) - 输出随机变量的值。

        **返回：**

        Tensor, 输出随机变量的值。

    .. py:method:: inverse_log_jacobian(value)

        计算逆映射导数的对数值，即 :math:`\log(dg^{-1}(x) / dx)`。

        **参数：**

        - **value** (Tensor) - 输出随机变量的值。

        **返回：**

        Tensor, 逆映射导数的对数值。
