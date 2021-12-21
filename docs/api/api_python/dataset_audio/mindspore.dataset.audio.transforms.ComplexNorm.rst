mindspore.dataset.audio.transforms.ComplexNorm
=================================================

.. py:class:: mindspore.dataset.audio.transforms.ComplexNorm(power=1.0)

    计算形如(..., complex=2)维度的复数序列的范数，其中第0维代表实部，第1维代表虚部。

    **参数：**

    - **power** (float, optional) - 范数的幂，取值非负（默认为1.0）。

    **样例：**

    >>> import numpy as np
    >>>
    >>> waveform = np.random.random([2, 4, 2])
    >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
    >>> transforms = [audio.ComplexNorm()]
    >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
