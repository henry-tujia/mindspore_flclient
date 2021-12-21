mindspore.nn.Vjp
=================

.. py:class:: mindspore.nn.Vjp(fn)

    �����������������ſɱȻ�(vector-Jacobian product, VJP)��VJP��Ӧ����ģʽ�Զ�΢�֡�

    **������**

    - **fn** (Cell) - ����Cell�����磬���ڽ����������벢����������������Ԫ�顣

    **���룺**

    - **inputs** (Tensor) - �����������Σ���������������
    - **v** (Tensor or Tuple of Tensor) - ���ſɱȾ����˵���������״����������һ�¡�

    **�����**

    2������������Ԫ�鹹�ɵ�Ԫ�顣

    - **net_output** (Tensor or Tuple of Tensor) - ��������������������
    - **vjp** (Tensor or Tuple of Tensor) - �����ſɱȻ��Ľ����

    **֧��ƽ̨��**

    ``Ascend`` ``GPU`` ``CPU``

    **������**

    >>> from mindspore.nn import Vjp
    >>> class Net(nn.Cell):
    ...     def construct(self, x, y):
    ...         return x**3 + y
    >>> x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    >>> y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    >>> v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    >>> output = Vjp(Net())(x, y, v)