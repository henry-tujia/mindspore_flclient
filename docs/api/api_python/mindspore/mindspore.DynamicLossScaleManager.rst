mindspore.DynamicLossScaleManager
==================================

.. py:class:: mindspore.DynamicLossScaleManager(init_loss_scale=2**24, scale_factor=2, scale_window=2000)

    ��̬�����ݶȷŴ�ϵ���Ĺ��������̳��� :class:`mindspore.LossScaleManager` ��

    **������**

    - **init_loss_scale** (float) - ��ʼ�ݶȷŴ�ϵ����Ĭ��ֵ��2**24��
    - **scale_factor** (int) - �Ŵ�/��С������Ĭ��ֵ��2��
    - **scale_window** (int) - �����ʱ����������step�����������Ĭ��ֵ��2000��

    **������**

    >>> from mindspore import Model, nn, DynamicLossScaleManager
    >>>
    >>> net = Net()
    >>> loss_scale_manager = DynamicLossScaleManager()
    >>> optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
    >>> model = Model(net, loss_scale_manager=loss_scale_manager, optimizer=optim)

    .. py:method:: get_drop_overflow_update()

        ��ֵ��ʾ�Ƿ��ڷ������ʱ�������ֲ������¡�

        **���أ�**

        bool��ʼ��ΪTrue��

    .. py:method:: get_loss_scale()

        ���ص�ǰ�ݶȷŴ�ϵ����

        **���أ�**

        float���ݶȷŴ�ϵ����

    .. py:method:: get_update_cell()

        �������ڸ����ݶȷŴ�ϵ���� `Cell` ʵ����:class:`mindspore.TrainOneStepWithLossScaleCell` ����ø�ʵ����

        **���أ�**

        :class:`mindspore.DynamicLossScaleUpdateCell` ʵ�������ڸ����ݶȷŴ�ϵ����

    .. py:method:: update_loss_scale(overflow)

        �������״̬�����ݶȷŴ�ϵ������������������С�ݶȷŴ�ϵ�������������ݶȷŴ�ϵ����

        **������**

        **overflow** (bool) - ��ʾ�Ƿ������