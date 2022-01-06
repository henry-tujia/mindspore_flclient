mindspore.DatasetHelper
========================

.. py:class:: mindspore.DatasetHelper(dataset, dataset_sink_mode=True, sink_size=-1, epoch_num=1)

    DatasetHelper��һ������MindData���ݼ����࣬�ṩ���ݼ���Ϣ��

    ���ݲ�ͬ�������ģ��ı����ݼ��ĵ������ڲ�ͬ����������ʹ����ͬ�ĵ�����

    .. note::
        DatasetHelper�ĵ������ṩһ��epoch�����ݡ�

    **������**

    - **dataset** (Dataset) - ѵ�����ݼ������������ݼ����������ݼ�������API�� :class:`mindspore.dataset` �����ɣ����� :class:`mindspore.dataset.ImageFolderDataset` ��
    - **dataset_sink_mode** (bool) - ���ֵΪTrue��ʹ�� :class:`mindspore.ops.GetNext` ���豸��Device����ͨ������ͨ���л�ȡ���ݣ�����������ֱ�ӱ������ݼ���ȡ���ݡ�Ĭ��ֵ��True��
    - **sink_size** (int) - ����ÿ���³��е������������ `sink_size` Ϊ-1�����³�ÿ��epoch���������ݼ������ `sink_size` ����0�����³�ÿ��epoch�� `sink_size` ���ݡ�Ĭ��ֵ��-1��
    - **epoch_num** (int) - ���ƴ����͵�epoch��������Ĭ��ֵ��1��

    **������**

    >>> from mindspore import DatasetHelper
    >>>
    >>> train_dataset = create_custom_dataset()
    >>> set_helper = DatasetHelper(train_dataset, dataset_sink_mode=False)
    >>> # DatasetHelper�����ǿɵ�����
    >>> for next_element in set_helper:
    ...     next_element
    
    .. py:method:: continue_send()
        
        ��epoch��ʼʱ�������豸�������ݡ�

    .. py:method:: dynamic_min_max_shapes()
        
        ���ض�̬���ݵ���״(shape)��Χ����С��״(shape)�������״(shape)����

        **������**

        >>>from mindspore import DatasetHelper
        >>>
        >>>train_dataset = create_custom_dataset()
        >>># config dynamic shape
        >>>dataset.set_dynamic_columns(columns={"data1": [16, None, 83], "data2": [None]})
        >>>dataset_helper = DatasetHelper(train_dataset, dataset_sink_mode=True)
        >>>
        >>>min_shapes, max_shapes = dataset_helper.dynamic_min_max_shapes()


    .. py:method:: get_data_info()
        
        �³�ģʽ�£���ȡ��ǰ�������ݵ����ͺ���״(shape)��ͨ����������״(shape)��̬�仯�ĳ���ʹ�á�
    
        **������**

        >>> from mindspore import DatasetHelper
        >>>
        >>> train_dataset = create_custom_dataset()
        >>> dataset_helper = DatasetHelper(train_dataset, dataset_sink_mode=True)
        >>>
        >>> types, shapes = dataset_helper.get_data_info()

    .. py:method:: release()
        
        �ͷ������³���Դ��

    .. py:method:: sink_size()
        
        ��ȡÿ�ε����� `sink_size` ��

        **������**

        >>>from mindspore import DatasetHelper
        >>>
        >>>train_dataset = create_custom_dataset()
        >>>dataset_helper = DatasetHelper(train_dataset, dataset_sink_mode=True, sink_size=-1)
        >>>
        >>># if sink_size==-1, then will return the full size of source dataset.
        >>>sink_size = dataset_helper.sink_size()

    .. py:method:: stop_send()
        
        ֹͣ���������³����ݡ�

    .. py:method:: types_shapes()
        
        �ӵ�ǰ�����е����ݼ���ȡ���ͺ���״(shape)��

        **������**

        >>>from mindspore import DatasetHelper
        >>>
        >>>train_dataset = create_custom_dataset()
        >>>dataset_helper = DatasetHelper(train_dataset, dataset_sink_mode=True)
        >>>
        >>>types, shapes = dataset_helper.types_shapes()