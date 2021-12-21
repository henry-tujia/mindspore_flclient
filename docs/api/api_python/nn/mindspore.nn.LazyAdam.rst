mindspore.nn.LazyAdam
======================

.. py:class:: mindspore.nn.LazyAdam(*args, **kwargs)

    通过Adaptive Moment Estimation (Adam)算法更新梯度。请参阅论文 `Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_。

    当梯度稀疏时，此优化器将使用Lazy Adam算法。

    更新公式如下：

    .. math::
        \begin{array}{ll} \\
            m_{t+1} = \beta_1 * m_{t} + (1 - \beta_1) * g \\
            v_{t+1} = \beta_2 * v_{t} + (1 - \beta_2) * g * g \\
            l = \alpha * \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t} \\
            w_{t+1} = w_{t} - l * \frac{m_{t+1}}{\sqrt{v_{t+1}} + \epsilon}
        \end{array}

    :math:`m` 代表第一个矩向量 `moment1` ，:math:`v` 代表第二个矩向量 `moment2` ，:math:`g` 代表 `gradients` ，:math:`l` 代表缩放因子，:math:`\beta_1,\beta_2` 代表 `beta1` 和 `beta2` ，:math:`t` 代表当前step，:math:`beta_1^t` 和 :math:`beta_2^t` 代表 `beta1_power` 和 `beta2_power` ， :math:`\alpha` 代表 `learning_rate` ， :math:`w` 代表 `params` ， :math:`\epsilon` 代表 `eps`。


    .. note::
       .. include:: mindspore.nn.optim_note_sparse.rst

       需要注意的是，梯度稀疏时该优化器只更新网络参数的当前的索引位置，稀疏行为不等同于Adam算法。

       .. include:: mindspore.nn.optim_note_weight_decay.rst

    **参数：**

    - **param** (Union[list[Parameter], list[dict]]) - 必须是 `Parameter` 组成的列表或字典组成的列表。当列表元素是字典时，字典的键可以是"params"、"lr"、"weight_decay"、"grad_centralization"和"order_params"：

      .. include:: mindspore.nn.optim_group_param.rst
      .. include:: mindspore.nn.optim_group_lr.rst
      .. include:: mindspore.nn.optim_group_weight_decay.rst
      .. include:: mindspore.nn.optim_group_gc.rst
      .. include:: mindspore.nn.optim_group_order.rst

    - **learning_rate** (Union[float, Tensor, Iterable, LearningRateSchedule]): 默认值：1e-3。

      .. include:: mindspore.nn.optim_arg_dynamic_lr.rst

    - **beta1** (float)：`moment1` 的指数衰减率。参数范围（0.0,1.0）。默认值：0.9。
    - **beta2** (float)：`moment2` 的指数衰减率。参数范围（0.0,1.0）。默认值：0.999。
    - **eps** (float) - 将添加到分母中，以提高数值稳定性。必须大于0。默认值：1e-8。
    - **use_locking** (bool) - 是否对参数更新加锁保护。如果为True，则 `w` 、`m` 和 `v` 的Tensor更新将受到锁的保护。如果为False，则结果不可预测。默认值：False。
    - **use_nesterov** (bool) - 是否使用Nesterov Accelerated Gradient (NAG)算法更新梯度。如果为True，使用NAG更新梯度。如果为False，则在不使用NAG的情况下更新梯度。默认值：False。
    - **weight_decay** (Union[float, int]) - 权重衰减（L2 penalty）。必须大于等于0。默认值：0.0。

      .. include:: mindspore.nn.optim_arg_loss_scale.rst

    **输入：**

    - **gradients** (tuple[Tensor]) - `params` 的梯度，shape与 `params` 相同。

    **输出：**

    Tensor[bool]，值为True。

    **异常：**

    - **TypeError** - `learning_rate` 不是int、float、Tensor、Iterable或LearningRateSchedule。
    - **TypeError** - `parameters` 的元素不是Parameter或字典。
    - **TypeError** - `beta1`、`beta2`、`eps` 或 `loss_scale` 不是float。
    - **TypeError** - `weight_decay` 不是float或int。
    - **TypeError** - `use_locking` 或 `use_nesterov` 不是bool。
    - **ValueError** - `loss_scale` 或 `eps` 小于或等于0。
    - **ValueError** - `beta1`、`beta2` 不在（0.0,1.0）范围内。
    - **ValueError** - `weight_decay` 小于0。

    **支持平台：**

    ``Ascend`` ``GPU``

    **样例：**

    >>> net = Net()
    >>> #1) 所有参数使用相同的学习率和权重衰减
    >>> optim = nn.LazyAdam(params=net.trainable_params())
    >>>
    >>> #2) 使用参数分组并设置不同的值
    >>> conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
    >>> no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
    >>> group_params = [{'params': conv_params, 'weight_decay': 0.01, 'grad_centralization':True},
    ...                 {'params': no_conv_params, 'lr': 0.01},
    ...                 {'order_params': net.trainable_params()}]
    >>> optim = nn.LazyAdam(group_params, learning_rate=0.1, weight_decay=0.0)
    >>> # conv_params参数组将使用优化器中的学习率0.1、该组的权重衰减0.01、该组的梯度中心化配置True。
    >>> # no_conv_params参数组将使用该组的学习率0.01、优化器中的权重衰减0.0、梯度中心化使用默认值False。
    >>> # 优化器按照"order_params"配置的参数顺序更新参数。
    >>>
    >>> loss = nn.SoftmaxCrossEntropyWithLogits()
    >>> model = Model(net, loss_fn=loss, optimizer=optim)

.. include:: mindspore.nn.optim_target_unique_for_sparse.rst
