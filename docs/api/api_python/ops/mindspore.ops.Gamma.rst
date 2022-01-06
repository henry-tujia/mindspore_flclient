mindspore.ops.Gamma
===================

.. py:class:: mindspore.ops.Gamma(seed=0, seed2=0)

    根据概率密度函数分布生成随机正浮点数x。

    .. math::

        \text{P}(x|α,β) = \frac{\exp(-x/β)}{{β^α}\cdot{\Gamma(α)}}\cdot{x^{α-1}}

    **参数：**

    - **seed** (int) - 算子层的随机种子，用于生成随机数。必须是非负的。默认值：0。
    - **seed2** (int)：全局的随机种子，和算子层的随机种子共同决定最终生成的随机数。必须是非负的。默认值：0。

    **输入：**

    - **shape** (tuple) - 待生成的随机Tensor的shape。只支持常量值。
    - **alpha** (Tensor) - α为Gamma分布的shape parameter，主要决定了曲线的形状。其值必须大于0。数据类型为float32。
    - **beta** (Tensor) - β为Gamma分布的inverse scale parameter，主要决定了曲线有多陡。其值必须大于0。数据类型为float32。

    **输出：**

    Tensor。shape是输入 `shape` 以及alpha、beta广播后的shape。数据类型为float32。

    **异常：**

    - **TypeError** - `seed` 和 `seed2` 都不是int。
    - **TypeError** - `alpha` 和 `beta` 都不是Tensor。
    - **ValueError** - `shape` 不是常量值。

    **支持平台：**

    ``Ascend``

    **样例：**

    >>> shape = (3, 1, 2)
    >>> alpha = Tensor(np.array([[3, 4], [5, 6]]), mstype.float32)
    >>> beta = Tensor(np.array([1.0]), mstype.float32)
    >>> gamma = ops.Gamma(seed=3)
    >>> output = gamma(shape, alpha, beta)
    >>> result = output.shape
    >>> print(result)
    (3, 2, 2)
