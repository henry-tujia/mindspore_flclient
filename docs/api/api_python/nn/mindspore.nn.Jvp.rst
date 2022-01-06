mindspore.nn.Jvp
=================

.. py:class:: mindspore.nn.Jvp(fn)

    �������������ſɱ�������(Jacobian-vector product, JVP)��JVP��Ӧ `ǰ��ģʽ�Զ�΢�� <https://mindspore.cn/docs/programming_guide/zh-CN/master/design/gradient.html#id3>`_��

    **������**

    - **fn** (Cell) - ����Cell�����磬���ڽ���Tensor���벢����Tensor����TensorԪ�顣

    **���룺**

    - **inputs** (Tensor) - �����������Σ���������Tensor��
    - **v** (Tensor or Tuple of Tensor) - ���ſɱȾ����˵�������Shape�����������һ�¡�

    **�����**

    2��Tensor��TensorԪ�鹹�ɵ�Ԫ�顣

    - **net_output** (Tensor or Tuple of Tensor) - ��������������������
    - **jvp** (Tensor or Tuple of Tensor) - �ſɱ��������Ľ����

    **֧��ƽ̨��**

    ``Ascend`` ``GPU`` ``CPU``

    **������**

    >>> from mindspore.nn import Jvp
    >>> class Net(nn.Cell):
    ...     def construct(self, x, y):
    ...         return x**3 + y
    >>> x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    >>> y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    >>> v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    >>> output = Jvp(Net())(x, y, (v, v))