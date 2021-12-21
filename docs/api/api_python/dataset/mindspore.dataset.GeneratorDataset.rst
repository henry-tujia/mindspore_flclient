﻿mindspore.dataset.GeneratorDataset
===================================

.. py:class:: mindspore.dataset.GeneratorDataset(source, column_names=None, column_types=None, schema=None, num_samples=None, num_parallel_workers=1, shuffle=None, sampler=None, num_shards=None, shard_id=None, python_multiprocessing=True, max_rowsize=6)

    通过调用Python数据源从Python中生成数据作为源数据集。生成的数据集的列名和列类型取决于用户定义的Python数据源。

    **参数：**

    - **source** (Union[Callable, Iterable, Random Accessible]) -
      一个Python的可调用对象，可以是一个可迭代的Python对象，或支持随机访问的Python对象。
      要求传入的可调用对象，可以通过 `source().next()` 的方式返回一个由NumPy数组构成的元组。
      要求传入的可迭代对象，可以通过 `iter(source).next()` 的方式返回一个由NumPy数组构成的元组。
      要求传入的支持随机访问对象，可以通过 `source[idx]` 的方式返回一个由NumPy数组构成的元组。
    - **column_names** (Union[str, list[str]]，可选) - 指定数据集生成的列名（默认值为None），用户必须提供此参数或通过参数 `schema` 指定列名。
    - **column_types** ((list[mindspore.dtype]，可选) - 指定生成数据集各个数据列的数据类型（默认为None）。
      如果未指定该参数，则自动推断类型；如果指定了该参数，将在数据输出时做类型匹配检查。
    - **schema** (Union[Schema, str]，可选) - 读取模式策略，用于指定读取数据列的数据类型、数据维度等信息，支持传入JSON文件或 `schema` 对象的路径。
      对于数据集生成的列名，用户需要提供 `column_names` 或 `schema` 进行指定，如果同时指定两者，则将优先从 `schema` 获取列名信息。
    - **num_samples** (int，可选) - 指定从数据集中读取的样本数（默认为None）。
    - **num_parallel_workers** (int，可选) - 指定读取数据的工作线程数（默认值为1）。
    - **shuffle** (bool，可选) - 是否混洗数据集。只有输入的 `source` 参数带有可随机访问属性（__getitem__）时，才可以指定该参数。（默认值为None，下表中会展示不同配置的预期行为）。
    - **sampler** (Union[Sampler, Iterable]，可选) - 指定从数据集中选取样本的采样器。只有输入的 `source` 参数带有可随机访问属性（__getitem__）时，才可以指定该参数（默认值为None，下表中会展示不同配置的预期行为）。
    - **num_shards** (int, 可选): 分布式训练时，将数据集划分成指定的分片数（默认值None）。指定此参数后，`num_samples` 表示每个分片的最大样本数。需要输入 `data` 支持可随机访问才能指定该参数。
    - **shard_id** (int, 可选): 分布式训练时，指定使用的分片ID号（默认值None）。只有当指定了 `num_shards` 时才能指定此参数。
    - **python_multiprocessing** (bool，可选) - 启用Python多进程模式加速运算（默认为True）。当传入Python对象的计算量很大时，开启此选项可能会有较好效果。
    - **max_rowsize** (int，可选) - 指定在多进程之间复制数据时，共享内存分配的最大空间（数量级为MB，默认为6MB），仅当参数 `python_multiprocessing` 设为True时，此参数才会生效。

    **异常：**

    - **RuntimeError** - Python对象 `source` 在执行期间引发异常。
    - **RuntimeError** - 参数 `column_names` 指定的列名数量与 `source` 的输出数据数量不匹配。
    - **RuntimeError** - 参数 `num_parallel_workers` 超过最大线程数。
    - **RuntimeError** - 同时指定了 `sampler` 和 `shuffle` 。
    - **RuntimeError** - 同时指定了 `sampler` 和 `num_shards` 。
    - **RuntimeError** - 指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
    - **RuntimeError** - 指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
    - **ValueError** - `shard_id` 参数错误（小于0或者大于等于 `num_shards` ）。

    .. note::
        - `source` 参数接收用户自定义的Python函数（PyFuncs），不要将 `mindspore.nn` 和 `mindspore.ops` 目录下或其他的网络计算算子添加
          到 `source` 中。
        - 此数据集可以指定 `sampler` 参数，但 `sampler` 和 `shuffle` 是互斥的。下表展示了几种合法的输入参数及预期的行为。

    .. list-table:: 配置 `sampler` 和 `shuffle` 的不同组合得到的预期排序结果
       :widths: 25 25 50
       :header-rows: 1

       * - 参数 `sampler`
         - 参数 `shuffle`
         - 预期数据顺序
       * - None
         - None
         - 随机排列
       * - None
         - True
         - 随机排列
       * - None
         - False
         - 顺序排列
       * - 参数 `sampler`
         - None
         - 由 `sampler` 行为定义的顺序
       * - 参数 `sampler`
         - True
         - 不允许
       * - 参数 `sampler`
         - False
         - 不允许

    **样例：**

    >>> import numpy as np
    >>>
    >>> # 1）定义一个Python生成器作为GeneratorDataset的可调用对象。
    >>> def generator_multidimensional():
    ...     for i in range(64):
    ...         yield (np.array([[i, i + 1], [i + 2, i + 3]]),)
    >>>
    >>> dataset = ds.GeneratorDataset(source=generator_multidimensional, column_names=["multi_dimensional_data"])
    >>>
    >>> # 2）定义一个Python生成器返回多列数据，作为GeneratorDataset的可调用对象。
    >>> def generator_multi_column():
    ...     for i in range(64):
    ...         yield np.array([i]), np.array([[i, i + 1], [i + 2, i + 3]])
    >>>
    >>> dataset = ds.GeneratorDataset(source=generator_multi_column, column_names=["col1", "col2"])
    >>>
    >>> # 3）定义一个可迭代数据集对象，作为GeneratorDataset的可调用对象。
    >>> class MyIterable:
    ...     def __init__(self):
    ...         self._index = 0
    ...         self._data = np.random.sample((5, 2))
    ...         self._label = np.random.sample((5, 1))
    ...
    ...     def __next__(self):
    ...         if self._index >= len(self._data):
    ...             raise StopIteration
    ...         else:
    ...             item = (self._data[self._index], self._label[self._index])
    ...             self._index += 1
    ...             return item
    ...
    ...     def __iter__(self):
    ...         self._index = 0
    ...         return self
    ...
    ...     def __len__(self):
    ...         return len(self._data)
    >>>
    >>> dataset = ds.GeneratorDataset(source=MyIterable(), column_names=["data", "label"])
    >>>
    >>> # 4）定义一个支持随机访问数据集对象，作为GeneratorDataset的可调用对象。
    >>> class MyAccessible:
    ...     def __init__(self):
    ...         self._data = np.random.sample((5, 2))
    ...         self._label = np.random.sample((5, 1))
    ...
    ...     def __getitem__(self, index):
    ...         return self._data[index], self._label[index]
    ...
    ...     def __len__(self):
    ...         return len(self._data)
    >>>
    >>> dataset = ds.GeneratorDataset(source=MyAccessible(), column_names=["data", "label"])
    >>>
    >>> # 注意，Python的list、dict、tuple也是支持随机可访问的，同样可以作为GeneratorDataset的输入
    >>> dataset = ds.GeneratorDataset(source=[(np.array(0),), (np.array(1),), (np.array(2),)], column_names=["col"])

    .. include:: mindspore.dataset.Dataset.add_sampler.rst

    .. include:: mindspore.dataset.Dataset.rst

    .. include:: mindspore.dataset.Dataset.use_sampler.rst

    .. include:: mindspore.dataset.Dataset.zip.rst
