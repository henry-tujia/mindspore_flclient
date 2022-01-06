# Copyright 2019-2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This dataset module supports various formats of datasets, including ImageNet, TFData,
MNIST, Cifar10/100, Manifest, MindRecord, and more. This module loads data with
high performance and parses data precisely. Some of the operations that are
provided to users to preprocess data include shuffle, batch, repeat, map, and zip.
"""
import atexit
import builtins
import glob
import json
import math
import os
import signal
import stat
import time
import uuid
import multiprocessing
from multiprocessing.pool import RUN, TERMINATE
from multiprocessing.util import Finalize
import queue
from enum import Enum
from functools import partial
from importlib import import_module
import sys
import threading

import copy
import weakref
import platform
import psutil
import numpy as np
from scipy.io import loadmat
from PIL import Image

import mindspore._c_dataengine as cde
from mindspore._c_expression import typing

from mindspore.common import Tensor
from mindspore import log as logger
from mindspore.parallel._ps_context import _is_role_pserver, _is_role_sched
from mindspore.parallel._utils import _get_device_num
from mindspore.dataset.engine.offload import GetOffloadModel

import mindspore.dataset.transforms.py_transforms as py_transforms
from mindspore.dataset.text.utils import SentencePieceModel, DE_C_INTER_SENTENCEPIECE_MODE

from . import samplers
from .iterators import DictIterator, TupleIterator, DummyIterator, check_iterator_cleanup, _set_iterator_cleanup, \
    ITERATORS_LIST, _unset_iterator_cleanup
from .queue import _SharedQueue
from .validators import check_batch, check_shuffle, check_map, check_filter, check_repeat, check_skip, check_zip, \
    check_rename, check_numpyslicesdataset, check_device_send, check_take, check_project, check_imagefolderdataset, \
    check_mnist_cifar_dataset, check_manifestdataset, check_tfrecorddataset, check_vocdataset, check_cocodataset, \
    check_celebadataset, check_minddataset, check_generatordataset, check_sync_wait, check_zip_dataset, \
    check_add_column, check_textfiledataset, check_concat, check_random_dataset, check_split, \
    check_bucket_batch_by_length, check_cluedataset, check_save, check_csvdataset, check_paddeddataset, \
    check_tuple_iterator, check_dict_iterator, check_schema, check_to_device_send, check_flickr_dataset, \
    check_sb_dataset, check_flowers102dataset, check_cityscapes_dataset, check_usps_dataset, check_div2k_dataset, \
    check_sbu_dataset, check_qmnist_dataset, check_emnist_dataset, check_fake_image_dataset, check_places365_dataset, \
    check_photo_tour_dataset, check_ag_news_dataset, check_dbpedia_dataset, check_lj_speech_dataset, \
    check_yes_no_dataset, check_speech_commands_dataset, check_tedlium_dataset, check_svhn_dataset, \
    check_stl10_dataset, check_yelp_review_dataset, check_penn_treebank_dataset, check_iwslt2016_dataset, \
    check_iwslt2017_dataset, check_sogou_news_dataset, check_yahoo_answers_dataset, check_udpos_dataset, \
    check_conll2000_dataset, check_amazon_review_dataset, check_semeion_dataset, check_caltech101_dataset, \
    check_caltech256_dataset, check_wiki_text_dataset, check_imdb_dataset, check_wider_face_dataset, \
    check_en_wik9_dataset
from ..core.config import get_callback_timeout, _init_device_info, get_enable_shared_mem, get_num_parallel_workers, \
    get_prefetch_size
from ..core.datatypes import mstype_to_detype, mstypelist_to_detypelist
from ..core.validator_helpers import replace_none
from ..core.py_util_helpers import ExceptionHandler
from ..transforms.py_transforms_util import FuncWrapper

try:
    context = import_module("mindspore.context")
except ModuleNotFoundError:
    context = None

if platform.system().lower() == "darwin" and multiprocessing.get_start_method() != "fork":
    multiprocessing.set_start_method("fork", True)


OffloadToManualOffloadMode = {
    None: cde.ManualOffloadMode.UNSPECIFIED,
    False: cde.ManualOffloadMode.DISABLED,
    True: cde.ManualOffloadMode.ENABLED
}


class Shuffle(str, Enum):
    """Specify the shuffle mode.

    - GLOBAL: Shuffle both the files and samples.
    - FILES: Shuffle files only.
    - INFILE: Shuffle data within each file.
    """
    GLOBAL: str = "global"
    FILES: str = "files"
    INFILE: str = "infile"


ShuffleToShuffleMode = {Shuffle.FILES: cde.ShuffleMode.FILES,
                        Shuffle.GLOBAL: cde.ShuffleMode.GLOBAL,
                        Shuffle.INFILE: cde.ShuffleMode.INFILE}


def shuffle_to_shuffle_mode(shuffle):
    """
    Shuffle Enum to Shuffle Mode

    Args:
        shuffle (Shuffle): shuffle flag to shuffle mode in C layer

    Returns:
        ShuffleMode, shuffle mode
    """
    shuffle_mode = cde.ShuffleMode.GLOBAL  # Global shuffle
    if not isinstance(shuffle, Shuffle):
        if shuffle is None or shuffle:
            shuffle_mode = cde.ShuffleMode.GLOBAL  # Global shuffle
        else:
            shuffle_mode = cde.ShuffleMode.FALSE  # No shuffle
    else:
        shuffle_mode = ShuffleToShuffleMode[shuffle]
    return shuffle_mode


def shuffle_to_bool(shuffle):
    """
    Shuffle Enum to bool

    Args:
        shuffle (Shuffle): shuffle flag to bool

    Returns:
        bool, True / False
    """
    shuffle_bool = True
    if not isinstance(shuffle, Shuffle):
        if shuffle is None:
            shuffle_bool = None
        elif shuffle:
            shuffle_bool = True
        else:
            shuffle_bool = False
    else:
        shuffle_bool = True
    return shuffle_bool


@check_zip
def zip(datasets):
    """
    Zip the datasets in the input tuple of datasets.

    Args:
        datasets (tuple of class Dataset): A tuple of datasets to be zipped together.
            The number of datasets must be more than 1.

    Returns:
        Dataset, dataset zipped.

    Raises:
        ValueError: If the number of datasets is 1.
        TypeError: If datasets is not a tuple.

    Examples:
            >>> # Create a dataset which is the combination of dataset_1 and dataset_2
            >>> dataset = ds.zip((dataset_1, dataset_2))
    """
    if len(datasets) <= 1:
        raise ValueError(
            "Can't zip empty or just one dataset!")
    for dataset in datasets:
        if not isinstance(dataset, Dataset):
            raise TypeError("Invalid dataset, expected Dataset object, but got %s!" % type(dataset))
    return ZipDataset(datasets)


def _get_operator_process():
    """
    Inner implemented method, mainly for passing sub-process id in C layer

    Returns:
         dict, mapping dict of operator id and corresponding process id.
    """
    global _OP_PROCESS
    process_info = _OP_PROCESS
    op_process = dict()
    keys = process_info.keys()
    fetched_all = True
    for key in keys:
        op_process[key] = list(process_info[key][1])
        item_full = (len(process_info[key][1]) == process_info[key][0])
        fetched_all = fetched_all and item_full
    return op_process, fetched_all


def _set_dataset_permissions(file_name, num_files):
    """
    set saved dataset files' permissions to 600
    the rule of dataset filenames should be the same as those in C++.
    """
    num_digits = len(str(num_files - 1))
    if num_files == 1:
        paths = [file_name]
    else:
        paths = ["{}{}".format(file_name, str(x).rjust(num_digits, '0')) for x in range(num_files)]

    for item in paths:
        if os.path.exists(item):
            os.chmod(item, stat.S_IRUSR | stat.S_IWUSR)
            index_file = item + ".db"
            if os.path.exists(index_file):
                os.chmod(index_file, stat.S_IRUSR | stat.S_IWUSR)


class Dataset:
    """
    Abstract class to represent a dataset in DataEngine's data pipeline.

    This class is the base class of SourceDataset and Dataset, and represents
    a node in the data flow graph.

    Args:
        num_parallel_workers (int, optional): Number of workers to process the dataset in parallel
            (default=None).
    """

    def __init__(self, children=None, num_parallel_workers=None, cache=None):
        # Note: children and parent are internal variables, not recommended for external using.
        self.children = replace_none(children, [])
        if isinstance(self.children, tuple):
            self.children = list(self.children)
        if not isinstance(self.children, list):
            self.children = [self.children]

        self.parent = []
        for child in self.children:
            child.parent.append(weakref.ref(self))
        self.num_parallel_workers = num_parallel_workers
        self.cache = cache

        self._device_iter = 0
        self._input_indexs = ()
        self.saved_output_types = None
        self.saved_output_shapes = None
        self.dynamic_setting = [False, None]
        self.saved_min_shapes = None
        self.saved_max_shapes = None
        self._col_names = None
        self.dataset_size = None
        self._batch_size = None
        self._num_classes = None
        self._repeat_count = None
        self._class_indexing = None
        self._sync = False

    def create_ir_tree(self):
        """
        Internal method to build an IR tree.

        Returns:
            DatasetNode, the root node of the IR tree.
            Dataset, the root dataset of the IR tree.
        """
        parent = self.parent
        self.parent = []
        dataset = copy.deepcopy(self)
        global _OP_NAME
        _OP_NAME = Dataset._get_operator_id(dataset)
        ir_tree = dataset.parse_tree()
        self.parent = parent
        _init_device_info()
        return ir_tree, dataset

    def close_pool(self):
        """
        Close multiprocessing pool in dataset. If you are familiar with multiprocessing library, you can regard this
        as a destructor for a processingPool object.
        """
        if hasattr(self, 'process_pool') and self.process_pool is not None:
            self.process_pool.close()
        for child in self.children:
            child.close_pool()

    def notify_watchdog(self):
        """
        Close watchdog thread in dataset. Now GeneratorDataset/map/batch will use a thread named watch_dog to monitor
        multiprocess, for get_dataset_size/output_shapes/output_types/get_col_name/num_classes, we need notify_watchdog
        to close watch_dog thread manually.
        """
        if hasattr(self, 'sample_fn') and self.sample_fn is not None:
            if self.sample_fn.multi_process:
                self.sample_fn._abort_watchdog()  # pylint: disable=W0212
        if hasattr(self, 'watch_dog') and self.watch_dog is not None and hasattr(self, 'eot') and self.eot is not None:
            self._abort_watchdog()
        for child in self.children:
            child.notify_watchdog()

    @staticmethod
    def _get_operator_id(dataset):
        """
        Internal method to iterate the tree and obtain op_id of each operator.

        Returns:
            Dataset, the root dataset of the tree.
        """
        op_name = dict()
        generator_process = dict()
        op_name[str(dataset)] = 0
        op_id = 1

        def process_name(datasets, operator_id):
            if not datasets:
                return 0
            temp = []
            for item in datasets:
                for d in item.children:
                    temp.append(d)
                    op_name[str(d)] = operator_id
                    if isinstance(d, GeneratorDataset) and d.sample_fn and d.sample_fn.pids:
                        generator_process[operator_id] = [d.num_parallel_workers, set(d.sample_fn.pids)]

            operator_id = operator_id + 1
            return process_name(temp, operator_id)

        process_name([dataset], op_id)
        if generator_process:
            global _OP_PROCESS
            _OP_PROCESS.update(generator_process)
        return op_name

    def parse_tree(self):
        """
        Internal method to parse the API tree into an IR tree.

        Returns:
            DatasetNode, the root node of the IR tree.
        """
        if len(self.parent) > 1:
            raise ValueError("The data pipeline is not a tree (i.e., one node has 2 consumers)")
        ir_children = [d.parse_tree() for d in self.children]
        # Bootstrap can only be performed on a copy of the original dataset node.
        # Bootstrap on original dataset node will make all iterators share the same process pool
        self.iterator_bootstrap()
        ir_node = self.parse(ir_children)
        ir_node = self.post_parse(ir_node)
        return ir_node

    def __safe_deepcopy__(self, memodict, exclude=()):
        if id(self) in memodict:
            return memodict[id(self)]
        cls = self.__class__
        new_op = cls.__new__(cls)
        memodict[id(self)] = new_op
        for arg, value in self.__dict__.items():
            if arg in exclude:
                setattr(new_op, arg, value)
            else:
                try:
                    setattr(new_op, arg, copy.deepcopy(value, memodict))
                except TypeError:
                    setattr(new_op, arg, value)
        return new_op

    def iterator_bootstrap(self):
        pass

    @staticmethod
    def _noop_mode():
        if _is_role_sched() or _is_role_pserver():
            return True
        return False

    def __add__(self, datasets):
        return self.concat(datasets)

    def to_json(self, filename=""):
        """
        Serialize a pipeline into JSON string and dump into file if filename is provided.

        Args:
            filename (str): filename of JSON file to be saved as (default="").

        Returns:
            str, JSON string of the pipeline.
        """
        ir_tree, _ = self.create_ir_tree()
        return json.loads(ir_tree.to_json(filename))

    @check_bucket_batch_by_length
    def bucket_batch_by_length(self, column_names, bucket_boundaries, bucket_batch_sizes, element_length_function=None,
                               pad_info=None, pad_to_bucket_boundary=False, drop_remainder=False):
        """
        Bucket elements according to their lengths. Each bucket will be padded and batched when
        they are full.

        A length function is called on each row in the dataset. The row is then
        bucketed based on its length and bucket boundaries. When a bucket reaches its
        corresponding size specified in bucket_batch_sizes, the entire bucket will be
        padded according to pad_info, and then form a batch.
        Each batch will be full, except one special case: the last batch for each bucket may not be full.

        Args:
            column_names (list[str]): Columns passed to element_length_function.
            bucket_boundaries (list[int]): A list consisting of the upper boundaries
                of the buckets. Must be strictly increasing. If there are n boundaries,
                n+1 buckets are created: One bucket for [0, bucket_boundaries[0]), one
                bucket for [bucket_boundaries[i], bucket_boundaries[i+1]) for each
                0<i<n-1, and the last bucket for [bucket_boundaries[n-1], inf).
            bucket_batch_sizes (list[int]): A list consisting of the batch sizes for
                each bucket. Must contain len(bucket_boundaries)+1 elements.
            element_length_function (Callable, optional): A function that takes in
                M arguments where M = len(column_names) and returns an integer. If no value
                provided, parameter M the len(column_names) must be 1, and the size of the first
                dimension of that column will be taken as the length (default=None).
            pad_info (dict, optional): The information about how to batch each column. The key
                corresponds to the column name, and the value must be a tuple of 2 elements.
                The first element corresponds to the shape to pad to, and the second
                element corresponds to the value to pad with. If a column is not
                specified, then that column will be padded to the longest in the current
                batch, and 0 will be used as the padding value. Any None dimensions will
                be padded to the longest in the current batch, unless if
                pad_to_bucket_boundary is True. If no padding is wanted, set pad_info
                to None (default=None).
            pad_to_bucket_boundary (bool, optional): If True, will pad each None
                dimension in pad_info to the bucket_boundary minus 1. If there are any
                elements that fall into the last bucket, an error will occur
                (default=False).
            drop_remainder (bool, optional): If True, will drop the last batch for each
                bucket if it is not a full batch (default=False).

        Returns:
            Dataset, dataset bucketed and batched by length.

        Examples:
            >>> # Create a dataset where certain counts rows are combined into a batch
            >>> # and drops the last incomplete batch if there is one.
            >>> import numpy as np
            >>> def generate_2_columns(n):
            ...     for i in range(n):
            ...         yield (np.array([i]), np.array([j for j in range(i + 1)]))
            >>>
            >>> column_names = ["col1", "col2"]
            >>> dataset = ds.GeneratorDataset(generate_2_columns(8), column_names)
            >>> bucket_boundaries = [5, 10]
            >>> bucket_batch_sizes = [2, 1, 1]
            >>> element_length_function = (lambda col1, col2: max(len(col1), len(col2)))
            >>> # Will pad col2 to shape [bucket_boundaries[i]] where i is the
            >>> # index of the bucket that is currently being batched.
            >>> pad_info = {"col2": ([None], -1)}
            >>> pad_to_bucket_boundary = True
            >>> dataset = dataset.bucket_batch_by_length(column_names, bucket_boundaries,
            ...                                          bucket_batch_sizes,
            ...                                          element_length_function, pad_info,
            ...                                          pad_to_bucket_boundary)
        """
        return BucketBatchByLengthDataset(self, column_names, bucket_boundaries, bucket_batch_sizes,
                                          element_length_function, pad_info, pad_to_bucket_boundary, drop_remainder)

    @check_batch
    def batch(self, batch_size, drop_remainder=False, num_parallel_workers=None, per_batch_map=None,
              input_columns=None, output_columns=None, column_order=None, pad_info=None,
              python_multiprocessing=False, max_rowsize=16):
        """
        Combine batch_size number of consecutive rows into batches.

        For any child node, a batch is treated as a single row.
        For any column, all the elements within that column must have the same shape.
        If a per_batch_map callable is provided, it will be applied to the batches of tensors.

        Note:
            The order of using repeat and batch reflects the number of batches and per_batch_map.
            It is recommended that the repeat operation applied after the batch operation finished.

        Args:
            batch_size (int or function): The number of rows each batch is created with. An
                int or callable object which takes exactly 1 parameter, BatchInfo.
            drop_remainder (bool, optional): Determines whether or not to drop the last block
                whose data row number is less than batch size (default=False). If True, and if there are less
                than batch_size rows available to make the last batch, then those rows will
                be dropped and not propagated to the child node.
            num_parallel_workers (int, optional): Number of workers(threads) to process the dataset in parallel
                (default=None).
            per_batch_map (callable, optional): Per batch map callable (default=None). A callable which takes
                (list[Tensor], list[Tensor], ..., BatchInfo) as input parameters. Each list[Tensor] represents a batch
                of Tensors on a given column. The number of lists should match with the number of entries in
                input_columns. The last parameter of the callable should always be a BatchInfo object. Per_batch_map
                should return (list[Tensor], list[Tensor], ...). The length of each list in output should be the same as
                the input. output_columns is required if the number of output lists is different from input.
            input_columns (Union[str, list[str]], optional): List of names of the input columns. The size of the list
                should match with signature of per_batch_map callable (default=None).
            output_columns (Union[str, list[str]], optional): List of names assigned to the columns
                outputted by the last operation. This parameter is mandatory if len(input_columns) !=
                len(output_columns). The size of this list must match the number of output
                columns of the last operation. (default=None, output columns will have the same
                name as the input columns, i.e., the columns will be replaced).
            column_order (Union[str, list[str]], optional): Specifies the list of all the columns you need in the whole
                dataset (default=None). The parameter is required when len(input_column) != len(output_column).
                Caution: the list here is not just the columns specified in parameter input_columns and output_columns.
            pad_info (dict, optional): Whether to perform padding on selected columns. pad_info={"col1":([224,224],0)}
                would pad column with name "col1" to a tensor of size [224,224] and fill the missing with 0
                (default=None).
            python_multiprocessing (bool, optional): Parallelize Python function per_batch_map with multi-processing.
                This option could be beneficial if the function is computational heavy (default=False).
            max_rowsize(int, optional): Maximum size of row in MB that is used for shared memory allocation to copy
               data between processes.  This is only used if python_multiprocessing is set to True (default=16).

        Returns:
            BatchDataset, dataset batched.

        Examples:
            >>> # Create a dataset where every 100 rows are combined into a batch
            >>> # and drops the last incomplete batch if there is one.
            >>> dataset = dataset.batch(100, True)
            >>> # resize image according to its batch number, if it's 5-th batch, resize to (5^2, 5^2) = (25, 25)
            >>> def np_resize(col, batchInfo):
            ...     output = col.copy()
            ...     s = (batchInfo.get_batch_num() + 1) ** 2
            ...     index = 0
            ...     for c in col:
            ...         img = Image.fromarray(c.astype('uint8')).convert('RGB')
            ...         img = img.resize((s, s), Image.ANTIALIAS)
            ...         output[index] = np.array(img)
            ...         index += 1
            ...     return (output,)
            >>> dataset = dataset.batch(batch_size=8, input_columns=["image"], per_batch_map=np_resize)
        """
        return BatchDataset(self, batch_size, drop_remainder, num_parallel_workers, per_batch_map, input_columns,
                            output_columns, column_order, pad_info, python_multiprocessing, max_rowsize)

    @check_sync_wait
    def sync_wait(self, condition_name, num_batch=1, callback=None):
        """
        Add a blocking condition to the input Dataset. A synchronize action will be applied.

        Args:
            condition_name (str): The condition name that is used to toggle sending next row.
            num_batch (int): the number of batches without blocking at the start of each epoch (default=1).
            callback (function): The callback function that will be invoked when sync_update is called (default=None).

        Returns:
            SyncWaitDataset, dataset added a blocking condition.

        Raises:
            RuntimeError: If condition name already exists.

        Examples:
            >>> import numpy as np
            >>> def gen():
            ...     for i in range(100):
            ...         yield (np.array(i),)
            >>>
            >>> class Augment:
            ...     def __init__(self, loss):
            ...         self.loss = loss
            ...
            ...     def preprocess(self, input_):
            ...         return input_
            ...
            ...     def update(self, data):
            ...         self.loss = data["loss"]
            >>>
            >>> batch_size = 4
            >>> dataset = ds.GeneratorDataset(gen, column_names=["input"])
            >>>
            >>> aug = Augment(0)
            >>> dataset = dataset.sync_wait(condition_name="policy", callback=aug.update)
            >>> dataset = dataset.map(operations=[aug.preprocess], input_columns=["input"])
            >>> dataset = dataset.batch(batch_size)
            >>> count = 0
            >>> for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
            ...     assert data["input"][0] == count
            ...     count += batch_size
            ...     data = {"loss": count}
            ...     dataset.sync_update(condition_name="policy", data=data)
        """
        return SyncWaitDataset(self, condition_name, num_batch, callback)

    @check_shuffle
    def shuffle(self, buffer_size):
        """
        Randomly shuffles the rows of this dataset using the following policy:

        1. Make a shuffle buffer that contains the first buffer_size rows.
        2. Randomly select an element from the shuffle buffer to be the next row
           propagated to the child node.
        3. Get the next row (if any) from the parent node and put it in the shuffle buffer.
        4. Repeat steps 2 and 3 until there are no more rows left in the shuffle buffer.

        A random seed can be provided to be used on the first epoch. In every subsequent
        epoch, the seed is changed to a new one, randomly generated value.

        Args:
            buffer_size (int): The size of the buffer (must be larger than 1) for
                shuffling. Setting buffer_size equal to the number of rows in the entire
                dataset will result in a global shuffle.

        Returns:
            Dataset, dataset shuffled.

        Raises:
            RuntimeError: If exist sync operators before shuffle.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> # Optionally set the seed for the first epoch
            >>> ds.config.set_seed(58)
            >>> # Create a shuffled dataset using a shuffle buffer of size 4
            >>> dataset = dataset.shuffle(4)
        """
        return ShuffleDataset(self, buffer_size)

    def flat_map(self, func):
        """
        Map `func` to each row in dataset and flatten the result.

        The specified `func` is a function that must take one 'Ndarray' as input
        and return a 'Dataset'.

        Args:
            func (function): A function that must take one 'Ndarray' as an argument and
                return a 'Dataset'.

        Returns:
            Dataset, dataset applied by the function.

        Examples:
            >>> # use NumpySlicesDataset as an example
            >>> dataset = ds.NumpySlicesDataset([[0, 1], [2, 3]])
            >>>
            >>> def flat_map_func(array):
            ...     # create a NumpySlicesDataset with the array
            ...     dataset = ds.NumpySlicesDataset(array)
            ...     # repeat the dataset twice
            ...     dataset = dataset.repeat(2)
            ...     return dataset
            >>>
            >>> dataset = dataset.flat_map(flat_map_func)
            >>> # [[0, 1], [0, 1], [2, 3], [2, 3]]

        Raises:
            TypeError: If `func` is not a function.
            TypeError: If `func` doesn't return a Dataset.
        """
        dataset = None
        if not hasattr(func, '__call__'):
            logger.critical("func must be a function.")
            raise TypeError("func must be a function.")

        for row_data in self.create_tuple_iterator(output_numpy=True):
            if dataset is None:
                dataset = func(row_data)
            else:
                dataset += func(row_data)

        if not isinstance(dataset, Dataset):
            logger.critical("flat_map must return a Dataset object.")
            raise TypeError("flat_map must return a Dataset object.")
        return dataset

    @check_map
    def map(self, operations, input_columns=None, output_columns=None, column_order=None,
            num_parallel_workers=None, python_multiprocessing=False, cache=None, callbacks=None,
            max_rowsize=16, offload=None):
        """
        Apply each operation in operations to this dataset.

        The order of operations is determined by the position of each operation in the operations parameter.
        operations[0] will be applied first, then operations[1], then operations[2], etc.

        Each operation will be passed one or more columns from the dataset as input, and zero or
        more columns will be outputted. The first operation will be passed the columns specified
        in input_columns as input. If there is more than one operator in operations, the outputted
        columns of the previous operation are used as the input columns for the next operation.
        The columns outputted by the very last operation will be assigned names specified by
        output_columns.

        Only the columns specified in column_order will be propagated to the child node. These
        columns will be in the same order as specified in column_order.

        Args:
            operations (Union[list[TensorOp], list[functions]]): List of operations to be
                applied on the dataset. Operations are applied in the order they appear in this list.
            input_columns (Union[str, list[str]], optional): List of the names of the columns that will be passed to
                the first operation as input. The size of this list must match the number of
                input columns expected by the first operator. (default=None, the first
                operation will be passed however many columns that are required, starting from
                the first column).
            output_columns (Union[str, list[str]], optional): List of names assigned to the columns outputted by
                the last operation. This parameter is mandatory if len(input_columns) !=
                len(output_columns). The size of this list must match the number of output
                columns of the last operation. (default=None, output columns will have the same
                name as the input columns, i.e., the columns will be replaced).
            column_order (list[str], optional): Specifies the list of all the columns you need in the whole
                dataset (default=None). The parameter is required when len(input_column) != len(output_column).
                Caution: the list here is not just the columns specified in parameter input_columns and output_columns.
            num_parallel_workers (int, optional): Number of threads used to process the dataset in
                parallel (default=None, the value from the configuration will be used).
            python_multiprocessing (bool, optional): Parallelize Python operations with multiple worker processes. This
                option could be beneficial if the Python operation is computational heavy (default=False).
            cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
                (default=None, which means no cache is used).
            callbacks (DSCallback, list[DSCallback], optional): List of Dataset callbacks to be called (Default=None).
            max_rowsize (int, optional): Maximum size of row in MB that is used for shared memory allocation to copy
               data between processes.  This is only used if python_multiprocessing is set to True (Default=16).
            offload (bool, optional): Flag to indicate whether offload is used (Default=None).

        Note:
            - Input `operations` mainly accept c_transforms, py_transforms operator in mindspore.dataset part, plus user
              defined Python function(PyFuncs).
            - Do not add network computing operators from mindspore.nn and mindspore.ops or others into this
              `operations`.

        Returns:
            Dataset, dataset after mapping operation.

        Examples:
            >>> # dataset is an instance of Dataset which has 2 columns, "image" and "label".
            >>>
            >>> # Define two operations, where each operation accepts 1 input column and outputs 1 column.
            >>> decode_op = c_vision.Decode(rgb=True)
            >>> random_jitter_op = c_vision.RandomColorAdjust(brightness=(0.8, 0.8), contrast=(1, 1),
            ...                                               saturation=(1, 1), hue=(0, 0))
            >>>
            >>> # 1) Simple map example.
            >>>
            >>> # Apply decode_op on column "image". This column will be replaced by the outputted
            >>> # column of decode_op. Since column_order is not provided, both columns "image"
            >>> # and "label" will be propagated to the child node in their original order.
            >>> dataset = dataset.map(operations=[decode_op], input_columns=["image"])
            >>>
            >>> # Decode and rename column "image" to "decoded_image".
            >>> dataset = dataset.map(operations=[decode_op], input_columns=["image"], output_columns=["decoded_image"])
            >>>
            >>> # Specify the order of the output columns.
            >>> dataset = dataset.map(operations=[decode_op], input_columns=["image"],
            ...                       output_columns=None, column_order=["label", "image"])
            >>>
            >>> # Rename column "image" to "decoded_image" and also specify the order of the output columns.
            >>> dataset = dataset.map(operations=[decode_op], input_columns=["image"],
            ...                       output_columns=["decoded_image"], column_order=["label", "decoded_image"])
            >>>
            >>> # Rename column "image" to "decoded_image" and keep only this column.
            >>> dataset = dataset.map(operations=[decode_op], input_columns=["image"],
            ...                       output_columns=["decoded_image"], column_order=["decoded_image"])
            >>>
            >>> # A simple example for mapping pyfunc. Renaming columns and specifying column order
            >>> # work in the same way as the previous examples.
            >>> dataset = ds.NumpySlicesDataset(data=[[0, 1, 2]], column_names=["data"])
            >>> dataset = dataset.map(operations=[(lambda x: x + 1)], input_columns=["data"])
            >>>
            >>> # 2) Map example with more than one operation.
            >>>
            >>> # Create a dataset where the images are decoded, then randomly color jittered.
            >>> # decode_op takes column "image" as input and outputs one column. The column
            >>> # outputted by decode_op is passed as input to random_jitter_op.
            >>> # random_jitter_op will output one column. Column "image" will be replaced by
            >>> # the column outputted by random_jitter_op (the very last operation). All other
            >>> # columns are unchanged. Since column_order is not specified, the order of the
            >>> # columns will remain the same.
            >>> dataset = dataset.map(operations=[decode_op, random_jitter_op], input_columns=["image"])
            >>>
            >>> # Rename the column outputted by random_jitter_op to "image_mapped".
            >>> # Specifying column order works in the same way as examples in 1).
            >>> dataset = dataset.map(operations=[decode_op, random_jitter_op], input_columns=["image"],
            ...                       output_columns=["image_mapped"])
            >>>
            >>> # Map with multiple operations using pyfunc. Renaming columns and specifying column order
            >>> # work in the same way as examples in 1).
            >>> dataset = ds.NumpySlicesDataset(data=[[0, 1, 2]], column_names=["data"])
            >>> dataset = dataset.map(operations=[(lambda x: x * x), (lambda x: x - 1)], input_columns=["data"],
            ...                                   output_columns=["data_mapped"])
            >>>
            >>> # 3) Example where number of input columns is not equal to number of output columns.
            >>>
            >>> # operations[0] is a lambda that takes 2 columns as input and outputs 3 columns.
            >>> # operations[1] is a lambda that takes 3 columns as input and outputs 1 column.
            >>> # operations[2] is a lambda that takes 1 column as input and outputs 4 columns.
            >>> #
            >>> # Note: The number of output columns of operation[i] must equal the number of
            >>> # input columns of operation[i+1]. Otherwise, this map call will also result
            >>> # in an error.
            >>> operations = [(lambda x, y: (x, x + y, x + y + 1)),
            ...               (lambda x, y, z: x * y * z),
            ...               (lambda x: (x % 2, x % 3, x % 5, x % 7))]
            >>>
            >>> # Note: Since the number of input columns is not the same as the number of
            >>> # output columns, the output_columns and column_order parameters must be
            >>> # specified. Otherwise, this map call will also result in an error.
            >>>
            >>> dataset = ds.NumpySlicesDataset(data=([[0, 1, 2]], [[3, 4, 5]]), column_names=["x", "y"])
            >>>
            >>> # Propagate all columns to the child node in this order:
            >>> dataset = dataset.map(operations, input_columns=["x", "y"],
            ...                       output_columns=["mod2", "mod3", "mod5", "mod7"],
            ...                       column_order=["mod2", "mod3", "mod5", "mod7"])
            >>>
            >>> # Propagate some columns to the child node in this order:
            >>> dataset = dataset.map(operations, input_columns=["x", "y"],
            ...                       output_columns=["mod2", "mod3", "mod5", "mod7"],
            ...                       column_order=["mod7", "mod3", "col2"])
        """
        if hasattr(self, 'operator_mixed') and getattr(self, 'operator_mixed') is True:
            num_parallel_workers = 1
            logger.warning(
                "Input 'operations' of 'map' includes network computing operators like in mindspore.nn, mindspore.ops, "
                "mindspore.numpy module and etc, which do not support multi-thread compiling, recommend to replace it "
                "with python implemented operator like numpy etc. Here decrease 'num_parallel_workers' into 1.")

        return MapDataset(self, operations, input_columns, output_columns, column_order, num_parallel_workers,
                          python_multiprocessing, cache, callbacks, max_rowsize, offload)

    @check_filter
    def filter(self, predicate, input_columns=None, num_parallel_workers=None):
        """
        Filter dataset by prediction.

        Note:
             If input_columns not provided or provided with empty, all columns will be used.

        Args:
            predicate (callable): Python callable which returns a boolean value. If False then filter the element.
            input_columns (Union[str, list[str]], optional): List of names of the input columns, when
                default=None, the predicate will be applied on all columns in the dataset.
            num_parallel_workers (int, optional): Number of workers to process the dataset
                in parallel (default=None).

        Returns:
            Dataset, dataset filtered.

        Examples:
            >>> # generator data(0 ~ 63)
            >>> # filter the data that greater than or equal to 11
            >>> dataset = dataset.filter(predicate=lambda data: data < 11, input_columns = ["data"])
        """
        return FilterDataset(self, predicate, input_columns, num_parallel_workers)

    @check_repeat
    def repeat(self, count=None):
        """
        Repeat this dataset `count` times. Repeat infinitely if the count is None or -1.

        Note:
            The order of using repeat and batch reflects the number of batches. It is recommended that
            the repeat operation is used after the batch operation.

        Args:
            count (int): Number of times the dataset is going to be repeated (default=None).

        Returns:
            Dataset, dataset repeated.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>>
            >>> # Create a dataset where the dataset is repeated for 50 epochs
            >>> dataset = dataset.repeat(50)
            >>>
            >>> # Create a dataset where each epoch is shuffled individually
            >>> dataset = dataset.shuffle(10)
            >>> dataset = dataset.repeat(50)
            >>>
            >>> # Create a dataset where the dataset is first repeated for
            >>> # 50 epochs before shuffling. The shuffle operator will treat
            >>> # the entire 50 epochs as one big dataset.
            >>> dataset = dataset.repeat(50)
            >>> dataset = dataset.shuffle(10)
        """
        return RepeatDataset(self, count)

    @check_skip
    def skip(self, count):
        """
        Skip the first N elements of this dataset.

        Args:
            count (int): Number of elements in the dataset to be skipped.

        Returns:
            Dataset, dataset that containing rows like origin rows subtract skipped rows.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> # Create a dataset which skips first 3 elements from data
            >>> dataset = dataset.skip(3)
        """
        return SkipDataset(self, count)

    @check_take
    def take(self, count=-1):
        """
        Takes at most given numbers of elements from the dataset.

        Note:
            1. If count is greater than the number of elements in the dataset or equal to -1,
               all the elements in dataset will be taken.
            2. The order of using take and batch matters. If take is before batch operation,
               then take the given number of rows; otherwise take the given number of batches.

        Args:
            count (int, optional): Number of elements to be taken from the dataset (default=-1).

        Returns:
            Dataset, dataset taken.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> # Create a dataset where the dataset includes 50 elements.
            >>> dataset = dataset.take(50)
        """
        return TakeDataset(self, count)

    def _get_absolute_split_sizes(self, sizes):
        """
        Internal method called by split to calculate absolute split sizes and to
        do some error checking after calculating absolute split sizes.

        Returns:
            int, absolute split sizes of the dataset.
        """
        # Call get_dataset_size here and check input here because
        # don't want to call this once in check_split and another time in
        # here again
        dataset_size = self.get_dataset_size()

        if dataset_size is None or dataset_size <= 0:
            raise RuntimeError("dataset_size is unknown, unable to split.")

        if not isinstance(sizes, list):
            raise RuntimeError("sizes must be a list.")

        all_int = all(isinstance(item, int) for item in sizes)
        if all_int:
            sizes_sum = sum(sizes)
            if sizes_sum != dataset_size:
                raise RuntimeError("Sum of split sizes {} is not equal to dataset size {}."
                                   .format(sizes_sum, dataset_size))
            return sizes

        absolute_sizes = []
        for item in sizes:
            absolute_size = int(round(item * dataset_size))
            if absolute_size == 0:
                raise RuntimeError("Split percentage {} is too small.".format(item))
            absolute_sizes.append(absolute_size)

        absolute_sizes_sum = sum(absolute_sizes)

        # if we still need more rows, give them to the first split.
        # if we have too many rows, remove the extras from the first split that has
        # enough rows.
        size_difference = int(dataset_size - absolute_sizes_sum)
        if size_difference > 0:
            absolute_sizes[0] += size_difference
        else:
            for i, _ in enumerate(absolute_sizes):
                if absolute_sizes[i] + size_difference > 0:
                    absolute_sizes[i] += size_difference
                    break

        if sum(absolute_sizes) != dataset_size:
            raise RuntimeError("Sum of calculated split sizes {} is not equal to dataset size {}."
                               .format(absolute_sizes_sum, dataset_size))

        return absolute_sizes

    @check_split
    def split(self, sizes, randomize=True):
        """
        Split the dataset into smaller, non-overlapping datasets.

        This is a general purpose split function which can be called from any operator in the pipeline.
        There is another, optimized split function, which will be called automatically if ds.split is
        called where ds is a MappableDataset.

        Args:
            sizes (Union[list[int], list[float]]): If a list of integers [s1, s2, …, sn] is
                provided, the dataset will be split into n datasets of size s1, size s2, …, size sn
                respectively. If the sum of all input sizes does not equal the original dataset size, an
                error will throw.
                If a list of floats [f1, f2, …, fn] is provided, all floats must be between 0 and 1
                and must sum to 1, otherwise an error will throw. The dataset will be split into n
                Datasets of size round(f1*K), round(f2*K), …, round(fn*K) where K is the size of the
                original dataset.
                If after rounding:

                - Any size equals 0, an error will occur.
                - The sum of split sizes < K, the difference of K - sigma(round(fi * k)) will be added to the first
                  split.
                - The sum of split sizes > K, the difference of sigma(round(fi * K)) - K will be removed from the first
                  large enough split such that it will have at least 1 row after removing the difference.

            randomize (bool, optional): Determines whether or not to split the data randomly (default=True).
                If True, the data will be randomly split. Otherwise, each split will be created with
                consecutive rows from the dataset.

        Note:
            1. Dataset cannot be sharded if split is going to be called.
            2. It is strongly recommended to not shuffle the dataset, but use randomize=True instead.
               Shuffling the dataset may not be deterministic, which means the data in each split
               will be different in each epoch.

        Raises:
            RuntimeError: If get_dataset_size returns None or is not supported for this dataset.
            RuntimeError: If `sizes` is list of integers and sum of all elements in sizes does not
                equal the dataset size.
            RuntimeError: If `sizes` is list of float and there is a split with size 0 after calculations.
            RuntimeError: If the dataset is sharded prior to calling split.
            ValueError: If `sizes` is list of float and not all floats are between 0 and 1, or if the
                floats don't sum to 1.

        Returns:
            tuple(Dataset), a tuple of datasets that have been split.

        Examples:
            >>> # TextFileDataset is not a mappable dataset, so this non-optimized split will be called.
            >>> # Since many datasets have shuffle on by default, set shuffle to False if split will be called!
            >>> dataset = ds.TextFileDataset(text_file_dataset_dir, shuffle=False)
            >>> train_dataset, test_dataset = dataset.split([0.9, 0.1])
        """
        if self.is_shuffled():
            logger.warning("Dataset is shuffled before split.")

        if self.is_sharded():
            raise RuntimeError("Dataset should not be sharded before split.")

        absolute_sizes = self._get_absolute_split_sizes(sizes)
        splits = []
        rows_to_skip = 0
        for size in absolute_sizes:
            ds = copy.deepcopy(self)
            if randomize:
                # want to shuffle the same way every epoch before split
                # in alter_tree, shuffle buffer is minimum 10000, so use 10000 here
                ds = ds.shuffle(10000)
                ds.reshuffle_each_epoch = False

            if rows_to_skip > 0:
                ds = ds.skip(rows_to_skip)

            ds = ds.take(size)
            splits.append(ds)

            rows_to_skip += size

        return tuple(splits)

    @check_zip_dataset
    def zip(self, datasets):
        """
        Zip the datasets in the sense of input tuple of datasets. Columns in the input datasets must have different
        name.

        Args:
            datasets (Union[tuple, class Dataset]): A tuple of datasets or a single class Dataset
                to be zipped together with this dataset.

        Returns:
            Dataset, dataset zipped.

        Examples:
            >>> # Create a dataset which is the combination of dataset and dataset_1
            >>> dataset = dataset.zip(dataset_1)
        """
        if isinstance(datasets, tuple):
            datasets = (self, *datasets)
        elif isinstance(datasets, Dataset):
            datasets = (self, datasets)
        else:
            raise TypeError("Invalid datasets, expected Dataset object or tuple of Dataset, but got %s!" % datasets)
        return ZipDataset(datasets)

    @check_concat
    def concat(self, datasets):
        """
        Concatenate the dataset objects in the input list.
        Performing "+" operation on dataset objects can achieve the same effect.

        Note:
            The column name, and rank and type of the column data must be the same in the input datasets.

        Args:
            datasets (Union[list, class Dataset]): A list of datasets or a single class Dataset
                to be concatenated together with this dataset.

        Returns:
            Dataset, dataset concatenated.

        Examples:
            >>> # Create a dataset by concatenating dataset_1 and dataset_2 with "+" operator
            >>> dataset = dataset_1 + dataset_2
            >>> # Create a dataset by concatenating dataset_1 and dataset_2 with concat operation
            >>> dataset = dataset_1.concat(dataset_2)
        """
        if isinstance(datasets, Dataset):
            datasets = [self] + [datasets]
        elif isinstance(datasets, list):
            datasets = [self] + datasets
        else:
            raise TypeError("Invalid datasets, expected Dataset object or list of Dataset, but got %s!" % datasets)
        return ConcatDataset(datasets)

    @check_rename
    def rename(self, input_columns, output_columns):
        """
        Rename the columns in input datasets.

        Args:
            input_columns (Union[str, list[str]]): List of names of the input columns.
            output_columns (Union[str, list[str]]): List of names of the output columns.

        Returns:
            Dataset, dataset renamed.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> input_columns = ["input_col1", "input_col2", "input_col3"]
            >>> output_columns = ["output_col1", "output_col2", "output_col3"]
            >>>
            >>> # Create a dataset where input_col1 is renamed to output_col1, and
            >>> # input_col2 is renamed to output_col2, and input_col3 is renamed
            >>> # to output_col3.
            >>> dataset = dataset.rename(input_columns=input_columns, output_columns=output_columns)
        """

        return RenameDataset(self, input_columns, output_columns)

    @check_project
    def project(self, columns):
        """
        Project certain columns in input dataset.

        The specified columns will be selected from the dataset and passed into
        the pipeline with the order specified. The other columns are discarded.

        Args:
            columns(Union[str, list[str]]): List of names of the columns to project.

        Returns:
            Dataset, dataset projected.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> columns_to_project = ["column3", "column1", "column2"]
            >>>
            >>> # Create a dataset that consists of column3, column1, column2
            >>> # in that order, regardless of the original order of columns.
            >>> dataset = dataset.project(columns=columns_to_project)
        """

        return ProjectDataset(self, columns)

    def apply(self, apply_func):
        """
        Apply a function in this dataset.

        Args:
            apply_func (function): A function that must take one 'Dataset' as an argument and
                                   return a preprocessed 'Dataset'.

        Returns:
            Dataset, dataset applied by the function.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>>
            >>> # Declare an apply_func function which returns a Dataset object
            >>> def apply_func(data):
            ...     data = data.batch(2)
            ...     return data
            >>>
            >>> # Use apply to call apply_func
            >>> dataset = dataset.apply(apply_func)

        Raises:
            TypeError: If apply_func is not a function.
            TypeError: If apply_func doesn't return a Dataset.
        """

        if not hasattr(apply_func, '__call__'):
            raise TypeError("apply_func must be a function.")

        dataset = apply_func(self)
        if not isinstance(dataset, Dataset):
            raise TypeError("apply_func must return a dataset.")
        return dataset

    @check_device_send
    def device_que(self, send_epoch_end=True, create_data_info_queue=False):
        """
        Return a transferred Dataset that transfers data through a device.

        Args:
            send_epoch_end (bool, optional): Whether to send end of sequence to device or not (default=True).
            create_data_info_queue (bool, optional): Whether to create queue which stores
                types and shapes of data or not(default=False).

        Note:
            If device is Ascend, features of data will be transferred one by one. The limitation
            of data transmission per time is 256M.

        Returns:
            Dataset, dataset for transferring.
        """
        return self.to_device(send_epoch_end=send_epoch_end, create_data_info_queue=create_data_info_queue)

    @check_device_send
    def to_device(self, send_epoch_end=True, create_data_info_queue=False):
        """
        Transfer data from CPU to GPU or Ascend or other devices.

        Args:
            send_epoch_end (bool, optional): Whether to send the end of sequence to device or not (default=True).
            create_data_info_queue (bool, optional): Whether to create queue which stores
                types and shapes of data or not(default=False).

        Note:
            If device is Ascend, features of data will be transferred one by one. The limitation
            of data transmission per second is 256M.

        Returns:
            TransferDataset, dataset for transferring.

        Raises:
            RuntimeError: If distribution file path is given but failed to read.
        """
        return TransferDataset(self, send_epoch_end, create_data_info_queue)

    @check_save
    def save(self, file_name, num_files=1, file_type='mindrecord'):
        """
        Save the dynamic data processed by the dataset pipeline in common dataset format.
        Supported dataset formats: 'mindrecord' only

        Implicit type casting exists when saving data as 'mindrecord'. The transform table shows how to do type casting.

        .. list-table:: Implicit Type Casting when Saving as 'mindrecord'
           :widths: 25 25 50
           :header-rows: 1

           * - Type in 'dataset'
             - Type in 'mindrecord'
             - Details
           * - bool
             - None
             - Not supported
           * - int8
             - int32
             -
           * - uint8
             - bytes(1D uint8)
             - Drop dimension
           * - int16
             - int32
             -
           * - uint16
             - int32
             -
           * - int32
             - int32
             -
           * - uint32
             - int64
             -
           * - int64
             - int64
             -
           * - uint64
             - None
             - Not supported
           * - float16
             - float32
             -
           * - float32
             - float32
             -
           * - float64
             - float64
             -
           * - string
             - string
             - Multi-dimensional string not supported

        Note:
            1. To save the samples in order, set dataset's shuffle to False and num_files to 1.
            2. Before calling the function, do not use batch operator, repeat operator or data augmentation operators
               with random attribute in map operator.
            3. When array dimension is variable, one-dimensional arrays or
               multi-dimensional arrays with variable dimension 0 are supported.
            4. Mindrecord does not support uint64, multi-dimensional uint8(drop dimension) nor
               multi-dimensional string.

        Args:
            file_name (str): Path to dataset file.
            num_files (int, optional): Number of dataset files (default=1).
            file_type (str, optional): Dataset format (default='mindrecord').

        """
        ir_tree, api_tree = self.create_ir_tree()

        runtime_context = cde.PythonRuntimeContext()
        runtime_context.Init()
        consumer = cde.PythonSaveToDisk(file_name, num_files, file_type)
        consumer.Init(ir_tree)
        runtime_context.AssignConsumer(consumer)

        consumer.Save()
        _set_dataset_permissions(file_name, num_files)
        del api_tree

    @check_tuple_iterator
    def create_tuple_iterator(self, columns=None, num_epochs=-1, output_numpy=False, do_copy=True):
        """
        Create an iterator over the dataset. The datatype retrieved back will be a list of ndarrays.

        To specify which columns to list and the order needed, use columns_list. If columns_list
        is not provided, the order of the columns will remain unchanged.

        Args:
            columns (list[str], optional): List of columns to be used to specify the order of columns
                (default=None, means all columns).
            num_epochs (int, optional): Maximum number of epochs that iterator can be iterated.
                (default=-1, iterator can be iterated infinite number of epochs)
            output_numpy (bool, optional): Whether or not to output NumPy datatype.
                If output_numpy=False, iterator will output MSTensor (default=False).
            do_copy (bool, optional): when output data type is mindspore.Tensor,
                use this param to select the conversion method, only take False for better performance (default=True).

        Returns:
            Iterator, tuple iterator over the dataset.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> iterator = dataset.create_tuple_iterator()
            >>> for item in iterator:
            ...     # item is a list
            ...     print(type(item))
            ...     break
            <class 'list'>
        """
        if output_numpy is None:
            output_numpy = False

        if Dataset._noop_mode():
            return DummyIterator(self, 'tuple')
        return TupleIterator(self, columns, num_epochs, output_numpy, do_copy)

    @check_dict_iterator
    def create_dict_iterator(self, num_epochs=-1, output_numpy=False):
        """
        Create an iterator over the dataset. The data retrieved will be a dictionary datatype.

        The order of the columns in the dictionary may not be the same as the original order.

        Args:
            num_epochs (int, optional): Maximum number of epochs that iterator can be iterated
                (default=-1, iterator can be iterated infinite number of epochs).
            output_numpy (bool, optional): Whether or not to output NumPy datatype,
                if output_numpy=False, iterator will output MSTensor (default=False).

        Returns:
            Iterator, dictionary iterator over the dataset.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> iterator = dataset.create_dict_iterator()
            >>> for item in iterator:
            ...     # item is a dict
            ...     print(type(item))
            ...     break
            <class 'dict'>
        """
        if output_numpy is None:
            output_numpy = False

        if Dataset._noop_mode():
            return DummyIterator(self, 'dict')
        return DictIterator(self, num_epochs, output_numpy)

    def __iter__(self):
        """Create an iterator over the dataset."""
        return self.create_tuple_iterator(num_epochs=1)

    @property
    def input_indexs(self):
        """
        Get Input Index Information

        Returns:
            tuple, tuple of the input index information.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> # set input_indexs
            >>> dataset.input_indexs = 10
            >>> print(dataset.input_indexs)
            10
        """
        if self._input_indexs != ():
            return self._input_indexs

        # find input_indexes of children
        children_input_index = [child.input_indexs for child in self.children]

        # in case of more than one child, return the first input_indexes
        for cix in children_input_index:
            if cix != ():
                return cix

        # if all children's input_indexes are () or the node is a leaf
        return self._input_indexs

    @input_indexs.setter
    def input_indexs(self, value):
        self._input_indexs = value

    def copy_batch_size(self, value):
        self._batch_size = value

    def _init_tree_getters(self):
        """
        Get pipeline information.
        """
        ir_tree, api_tree = self.create_ir_tree()

        runtime_context = cde.PythonRuntimeContext()
        runtime_context.Init()
        getter = cde.TreeGetters()
        getter.Init(ir_tree)
        runtime_context.AssignConsumer(getter)
        return getter, runtime_context, api_tree

    def __init_size_getter(self):
        """
        Get pipeline information.
        """
        ir_tree, api_tree = self.create_ir_tree()

        runtime_context = cde.PythonRuntimeContext()
        runtime_context.Init()
        getter = cde.DatasetSizeGetters()
        getter.Init(ir_tree)
        runtime_context.AssignConsumer(getter)
        return getter, runtime_context, api_tree

    def get_col_names(self):
        """
        Return the names of the columns in dataset.

        Returns:
            list, list of column names in the dataset.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> col_names = dataset.get_col_names()
        """
        if self._col_names is None:
            runtime_getter = self._init_tree_getters()
            self._col_names = runtime_getter[0].GetColumnNames()
            self.close_pool()
            runtime_getter[2].notify_watchdog()
        return self._col_names

    def output_shapes(self):
        """
        Get the shapes of output data.

        Returns:
            list, list of shapes of each column.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> output_shapes = dataset.output_shapes()
        """
        if self.saved_output_shapes is None:
            runtime_getter = self._init_tree_getters()
            self.saved_output_shapes = runtime_getter[0].GetOutputShapes()
            self.saved_output_types = runtime_getter[0].GetOutputTypes()
            self.close_pool()
            runtime_getter[2].notify_watchdog()
        if self.dynamic_setting[0]:
            self.saved_output_shapes, self.saved_min_shapes, self.saved_max_shapes = self._dynamic_output_shapes()
        return self.saved_output_shapes

    def output_types(self):
        """
        Get the types of output data.

        Returns:
            list, list of data types.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> output_types = dataset.output_types()
        """
        if self.saved_output_types is None:
            runtime_getter = self._init_tree_getters()
            self.saved_output_shapes = runtime_getter[0].GetOutputShapes()
            self.saved_output_types = runtime_getter[0].GetOutputTypes()
            self.close_pool()
            runtime_getter[2].notify_watchdog()
        if self.dynamic_setting[0]:
            self.saved_output_shapes, self.saved_min_shapes, self.saved_max_shapes = self._dynamic_output_shapes()
        return self.saved_output_types

    def get_dataset_size(self):
        """
        Return the number of batches in an epoch.

        Returns:
            int, number of batches.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> dataset_size = dataset.get_dataset_size()
        """
        if self.dataset_size is None:
            runtime_getter = self.__init_size_getter()
            self.dataset_size = runtime_getter[0].GetDatasetSize(False)
            self.close_pool()
            runtime_getter[2].notify_watchdog()
        return self.dataset_size

    def set_dynamic_columns(self, columns=None):
        """
        Set dynamic shape information of source data, it should be set after the pipeline is defined.

        Args:
            columns (dict): A dict contains shape information of each column in dataset.
                The value of shape[i] is :py:obj:`None` indicates that the data length of shape[i] is dynamic.

        Examples:
            >>> import numpy as np
            >>>
            >>> def generator1():
            ...     for i in range(1, 100):
            ...         yield np.ones((16, i, 83)), np.array(i)
            >>>
            >>> dataset = ds.GeneratorDataset(generator1, ["data1", "data2"])
            >>> dataset.set_dynamic_columns(columns={"data1": [16, None, 83], "data2": []})
        """
        if not isinstance(columns, dict):
            raise TypeError("Pass a dict to set dynamic shape, example: {\"data1\": [16, None, 256]}")
        self.dynamic_setting[0] = True
        self.dynamic_setting[1] = columns

    def dynamic_min_max_shapes(self):
        """
        Get minimum and maximum data length of dynamic source data, for dynamic graph compilation.

        Returns:
            lists, min_shapes, max_shapes of source data.

        Examples:
            >>> import numpy as np
            >>>
            >>> def generator1():
            ...     for i in range(1, 100):
            ...         yield np.ones((16, i, 83)), np.array(i)
            >>>
            >>> dataset = ds.GeneratorDataset(generator1, ["data1", "data2"])
            >>> dataset.set_dynamic_columns(columns={"data1": [16, None, 83], "data2": []})
            >>> min_shapes, max_shapes = dataset.dynamic_min_max_shapes()
        """
        if self.saved_min_shapes is None or self.saved_max_shapes is None:
            self.saved_output_shapes, self.saved_min_shapes, self.saved_max_shapes = self._dynamic_output_shapes()
        return self.saved_min_shapes, self.saved_max_shapes

    @staticmethod
    def __check_dynamic_column_name(dynamic_columns, dataset_columns):
        for column in dynamic_columns:
            if column not in dataset_columns:
                raise RuntimeError("dynamic column [" + column + "] does not match any column in dataset: " +
                                   str(dataset_columns))

    @staticmethod
    def __check_dynamic_column_shape(data, col, dynamic_columns):
        shape_mismatch = "dynamic column [" + col + "] with shape " + str(dynamic_columns[col]) + \
                         " does not match dataset column [" + col + "] with shape " + str(list(data[col].shape))
        if data[col].ndim != len(dynamic_columns[col]):
            raise RuntimeError(shape_mismatch)
        for dim in range(len(dynamic_columns[col])):
            if dynamic_columns[col][dim] is not None and dynamic_columns[col][dim] != data[col].shape[dim]:
                raise RuntimeError(shape_mismatch)

    def _dynamic_output_shapes(self):
        """
        Get dynamic information of source data.

        Returns:
            lists, dynamic_shapes, min_shapes, max_shapes of source data.
        """
        if not self.dynamic_setting[1]:
            raise RuntimeError("dynamic_columns is not set, call set_dynamic_columns() by final Dataset Op.")

        if self.saved_output_shapes is not None and self.saved_min_shapes is not None and \
                self.saved_max_shapes is not None:
            return self.saved_output_shapes, self.saved_min_shapes, self.saved_max_shapes

        logger.warning("Calculating dynamic shape of input data, this will take a few minutes...")
        # Assume data1 shape is dynamic, data2 shape is fix
        # {"data1": [batch_size, None, feat_len], "data2": [batch_size, feat_len]}
        dynamic_columns = self.dynamic_setting[1]
        # ["data1", "data2"]
        dataset_columns = self.get_col_names()
        Dataset.__check_dynamic_column_name(dynamic_columns, dataset_columns)

        # Shape[1] of data1 is variable
        # {"data1": {(batch_size, 100, feat_len), (16, 200, 83)}, "data2": {(batch_size, feat_len)}}
        column_shape_set = {col: set() for col in dataset_columns}
        dataset_size_counter = 0
        for data in self.create_dict_iterator(num_epochs=1, output_numpy=True):
            dataset_size_counter += 1
            for col in data.keys():
                if col in dynamic_columns:
                    Dataset.__check_dynamic_column_shape(data, col, dynamic_columns)
                column_shape_set[col].add(tuple(data[col].shape))

        # we get dataset_size after dryrun
        self.dataset_size = dataset_size_counter

        min_shapes, max_shapes, dynamic_shapes = list(), list(), list()
        for col, shape_set in column_shape_set.items():
            if len(shape_set) > 1:
                if col not in dynamic_columns:
                    raise RuntimeError("column [" + col + "] has dynamic shape but not set by set_dynamic_columns()" +
                                       ", shapes of [" + col + "]: " + str(list(shape_set)))
                shape_npy = np.array(list(shape_set))
                max_shape = shape_npy.max(axis=0)
                min_shape = shape_npy.min(axis=0)

                # Set min shape to 1 due to unknown shuffle
                min_shape = np.where(np.equal(dynamic_columns[col], None), 1, min_shape)
                # Set dynamic dim to -1 for ME
                dynamic_shape = np.where(np.equal(dynamic_columns[col], None), -1, dynamic_columns[col])

                max_shapes.append(max_shape.tolist())
                min_shapes.append(min_shape.tolist())
                dynamic_shapes.append(dynamic_shape.tolist())
            else:
                # Also append fix shape to keep order of column shape
                fix_shape = list(list(shape_set)[0])
                max_shapes.append(fix_shape)
                min_shapes.append(fix_shape)
                dynamic_shapes.append(fix_shape)
                if col in dynamic_columns:
                    logger.warning("column [" + col + "] has no dynamic shape but set by set_dynamic_columns()")
                    # Set min shape to 1 due to unknown shuffle
                    min_shapes[-1] = np.where(np.equal(dynamic_columns[col], None), 1, fix_shape).tolist()
                    # Set dynamic dim to -1 for ME
                    dynamic_shapes[-1] = np.where(np.equal(dynamic_columns[col], None), -1, fix_shape).tolist()
        return dynamic_shapes, min_shapes, max_shapes

    def num_classes(self):
        """
        Get the number of classes in a dataset.

        Returns:
            int, number of classes.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> num_classes = dataset.num_classes()
        """
        if self._num_classes is None:
            runtime_getter = self._init_tree_getters()
            self._num_classes = runtime_getter[0].GetNumClasses()
            self.close_pool()
            runtime_getter[2].notify_watchdog()
        if self._num_classes == -1:
            return None
        return self._num_classes

    def get_sync_notifiers(self):
        if self.children:
            return self.children[0].get_sync_notifiers()
        return {}

    def disable_sync(self):
        if self.children:
            return self.children[0].disable_sync()
        return {}

    def is_sync(self):
        if self.children:
            return self.children[0].is_sync()
        return False

    def sync_update(self, condition_name, num_batch=None, data=None):
        """
        Release a blocking condition and trigger callback with given data.

        Args:
            condition_name (str): The condition name that is used to toggle sending next row.
            num_batch (Union[int, None]): The number of batches (rows) that are released.
                When num_batch is None, it will default to the number specified by the
                sync_wait operator (default=None).
            data (Any): The data passed to the callback, user defined (default=None).
        """
        if (not isinstance(num_batch, int) and num_batch is not None) or \
                (isinstance(num_batch, int) and num_batch <= 0):
            # throwing exception, disable all sync_wait in pipeline
            self.disable_sync()
            raise RuntimeError("Sync_update batch size can only be positive integer, got : {}.".format(num_batch))
        notifiers_dict = self.get_sync_notifiers()
        if not isinstance(condition_name, str):
            raise TypeError("Argument condition_name with value {} is not of type str, but got {}."
                            .format(condition_name, type(condition_name)))
        if condition_name not in notifiers_dict:
            # throwing exception, disable all sync_wait in pipeline
            self.disable_sync()
            raise RuntimeError("Condition name not found.")
        if num_batch is not None:
            num_batch *= self.get_batch_size()
        notifiers_dict[condition_name](num_batch, data)

    def get_batch_size(self):
        """
        Return the size of batch.

        Returns:
            int, the number of data in a batch.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> batch_size = dataset.get_batch_size()
        """
        if self._batch_size is None:
            runtime_getter = self._init_tree_getters()
            self._batch_size = runtime_getter[0].GetBatchSize()
        if self._batch_size is None:
            self._batch_size = 1
        return self._batch_size

    def get_repeat_count(self):
        """
        Get the replication times in RepeatDataset (default is 1).

        Returns:
            int, the count of repeat.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> repeat_count = dataset.get_repeat_count()
        """
        if self._repeat_count is None:
            runtime_getter = self._init_tree_getters()
            self._repeat_count = runtime_getter[0].GetRepeatCount()
        if self._repeat_count is None:
            self._repeat_count = 1
        return self._repeat_count

    def get_class_indexing(self):
        """
        Return the class index.

        Returns:
            dict, a str-to-int mapping from label name to index.
            dict, a str-to-list<int> mapping from label name to index for Coco ONLY. The second number
            in the list is used to indicate the super category.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> class_indexing = dataset.get_class_indexing()
        """
        if self.children:
            return self.children[0].get_class_indexing()
        return {}

    def reset(self):
        """Reset the dataset for next epoch."""

    def is_shuffled(self):
        """Returns True if the dataset or its children is shuffled."""
        for input_dataset in self.children:
            if input_dataset.is_shuffled():
                return True

        return False

    def is_sharded(self):
        """Returns True if the dataset or its children is sharded."""
        for input_dataset in self.children:
            if input_dataset.is_sharded():
                return True

        return False

    def parse(self, children=None):
        raise NotImplementedError("Dataset has to implement parse method.")

    def post_parse(self, ir_node):
        if self.cache:
            ir_node = ir_node.set_cache_client(self.cache.cache_client)
        if self.num_parallel_workers:
            ir_node = ir_node.set_num_workers(self.num_parallel_workers)

        return ir_node


class TextBaseDataset(Dataset):
    """
    Abstract class to represent a text source dataset which produces content to the data pipeline.
    """

    def __init__(self, children=None, num_parallel_workers=None, cache=None):
        super().__init__(children=children, num_parallel_workers=num_parallel_workers, cache=cache)

    def parse(self, children=None):
        raise NotImplementedError("Dataset has to implement parse method.")

    def build_vocab(self, columns, freq_range, top_k, special_tokens, special_first):
        """
        Function to create a Vocab from source dataset

        Build a vocab from a dataset. This would collect all the unique words in a dataset and return a vocab
        which contains top_k most frequent words (if top_k is specified)

        Args:

            columns(Union[str, list[str]]): Column names to get words from.
            freq_range(tuple[int]): A tuple of integers (min_frequency, max_frequency). Words within the frequency
                range will be stored.
                Naturally 0 <= min_frequency <= max_frequency <= total_words. min_frequency/max_frequency
                can be set to default, which corresponds to 0/total_words separately.
            top_k(int): Number of words to be built into vocab. top_k most frequent words are
                taken. The top_k is taken after freq_range. If not enough top_k, all words will be taken
            special_tokens(list[str]): A list of strings, each one is a special token.
            special_first(bool): Whether special_tokens will be prepended/appended to vocab, If special_tokens
                is specified and special_first is set to default, special_tokens will be prepended.

        Returns:
            Vocab, vocab built from the dataset.

        Examples:
            >>> import numpy as np
            >>>
            >>> def gen_corpus():
            ...     # key: word, value: number of occurrences, reason for using letters is so their order is apparent
            ...     corpus = {"Z": 4, "Y": 4, "X": 4, "W": 3, "U": 3, "V": 2, "T": 1}
            ...     for k, v in corpus.items():
            ...         yield (np.array([k] * v, dtype='S'),)
            >>> column_names = ["column1"]
            >>> dataset = ds.GeneratorDataset(gen_corpus, column_names)
            >>> dataset = dataset.build_vocab(columns=["column1"],
            ...                               freq_range=(1, 10), top_k=5,
            ...                               special_tokens=["<pad>", "<unk>"],
            ...                               special_first=True)

        """
        vocab = cde.Vocab()
        columns = replace_none(columns, [])
        if not isinstance(columns, list):
            columns = [columns]

        freq_range = replace_none(freq_range, (0, 9223372036854775807))
        if freq_range[0] is None:
            freq_range = (0, freq_range[1])
        if freq_range[1] is None:
            freq_range = (freq_range[0], 9223372036854775807)
        special_tokens = replace_none(special_tokens, [])
        top_k = replace_none(top_k, 9223372036854775807)

        ir_tree, api_tree = self.create_ir_tree()

        # vocab node
        vocab_node = cde.BuildVocabNode(ir_tree, vocab, columns, freq_range, top_k, special_tokens, special_first)

        runtime_context = cde.PythonRuntimeContext()
        runtime_context.Init()

        # build vocab
        consumer = cde.PythonBuildVocabConsumer()
        consumer.Init(vocab_node)
        runtime_context.AssignConsumer(consumer)

        consumer.Start()
        del api_tree

        return vocab

    def build_sentencepiece_vocab(self, columns, vocab_size, character_coverage, model_type, params):
        """
        Function to create a SentencePieceVocab from source dataset

        Args:

            columns(list[str]): Column names to get words from.
            vocab_size(int): Vocabulary size.
            character_coverage(int): Percentage of characters covered by the model, must be between
                0.98 and 1.0 Good defaults are: 0.9995 for languages with rich character sets like
                Japanese or Chinese character sets, and 1.0 for other languages with small character sets
                like English or Latin.
            model_type(SentencePieceModel): Model type. Choose from unigram (default), bpe, char, or word.
                The input sentence must be pretokenized when using word type.
            params(dict): Any extra optional parameters of sentencepiece library according to your raw data

        Returns:
            SentencePieceVocab, vocab built from the dataset.

        Examples:
            >>> from mindspore.dataset.text import SentencePieceModel
            >>>
            >>> dataset = ds.TextFileDataset("/path/to/sentence/piece/vocab/file", shuffle=False)
            >>> dataset = dataset.build_sentencepiece_vocab(["text"], 5000, 0.9995, SentencePieceModel.UNIGRAM, {})
        """
        if not isinstance(model_type, SentencePieceModel):
            raise TypeError("Argument model_type with value {0} is not of type SentencePieceModel, but got {1}."\
                            .format(model_type, type(model_type)))
        model_type = DE_C_INTER_SENTENCEPIECE_MODE[model_type]
        vocab = cde.SentencePieceVocab()

        ir_tree, api_tree = self.create_ir_tree()

        # vocab node
        vocab_node = cde.BuildSentenceVocabNode(ir_tree, vocab, columns, vocab_size, character_coverage, model_type,
                                                params)

        runtime_context = cde.PythonRuntimeContext()
        runtime_context.Init()

        # build vocab
        consumer = cde.PythonBuildVocabConsumer()
        consumer.Init(vocab_node)
        runtime_context.AssignConsumer(consumer)

        consumer.Start()
        del api_tree

        return vocab


class SourceDataset(Dataset):
    """
    Abstract class to represent a source dataset which produces content to the data pipeline.
    """

    def __init__(self, num_parallel_workers=None, num_samples=None, shuffle=True, num_shards=None, shard_id=None,
                 cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, cache=cache)
        self.num_samples = replace_none(num_samples, 0)
        self.num_shards = replace_none(num_shards, 1)
        self.shard_id = replace_none(shard_id, 0)

        if shuffle is not None and not isinstance(shuffle, (bool, Shuffle)):
            raise TypeError("shuffle must be of boolean or enum of 'Shuffle' values like 'Shuffle.GLOBAL' or "
                            "'Shuffle.FILES' or 'Shuffle.INFILE'.")

        self.shuffle_flag = 2  # Global shuffle
        if not isinstance(shuffle, Shuffle):
            if shuffle is None or shuffle:
                self.shuffle_flag = 2  # Global shuffle
            else:
                self.shuffle_flag = 0  # No shuffle
        else:
            if shuffle == Shuffle.GLOBAL:
                self.shuffle_flag = 2  # Global shuffle
            elif shuffle == Shuffle.FILES:
                self.shuffle_flag = 1  # Files shuffle
            elif shuffle == Shuffle.INFILE:
                self.shuffle_flag = 3  # Infile shuffle

    def parse(self, children=None):
        raise NotImplementedError("Dataset has to implement parse method.")

    @staticmethod
    def _find_files(patterns):
        """
        Utility function to search for files with the given glob patterns.

        Args:
            patterns (Union[str, list[str]]): String or list of patterns to be searched.

        Returns:
            list, list of files.
        """

        if not isinstance(patterns, list):
            patterns = [patterns]

        file_list = []
        unmatched_patterns = []
        for pattern in patterns:
            matches = [match for match in glob.glob(pattern, recursive=True) if os.path.isfile(match)]

            if matches:
                file_list.extend(matches)
            else:
                unmatched_patterns.append(pattern)

        if unmatched_patterns:
            raise ValueError("The following patterns did not match any files: {}.".format(unmatched_patterns))

        if file_list:  # not empty
            return file_list
        raise ValueError("The list of path names matching the patterns is empty.")

    def is_shuffled(self):
        return self.shuffle_flag > 0

    def is_sharded(self):
        if self.num_shards is not None:
            return self.num_shards > 1
        return False


class MappableDataset(SourceDataset):
    """
    Abstract class to represent a source dataset which supports use of samplers.
    """

    def parse(self, children=None):
        raise NotImplementedError("Dataset has to implement parse method.")

    def __init__(self, num_parallel_workers=None, sampler=None, num_samples=None, shuffle=None, num_shards=None,
                 shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.shuffle_flag = replace_none(shuffle, True)
        self.sampler = samplers.select_sampler(num_samples, sampler, shuffle, num_shards, shard_id)

    def add_sampler(self, new_sampler):
        """
        Add a sampler for current dataset.

        Args:
            new_sampler (Sampler): The sampler to be added as the parent sampler for current dataset.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> # use a DistributedSampler instead
            >>> new_sampler = ds.DistributedSampler(10, 2)
            >>> dataset.add_sampler(new_sampler)
        """
        # note: By adding a sampler, the sampled IDs will flow to new_sampler
        # after first passing through the current samplers attached to this dataset.
        self.dataset_size = None
        new_sampler.add_child(self.sampler)
        self.sampler = new_sampler

    def use_sampler(self, new_sampler):
        """
        Make the current dataset use the new_sampler provided by other API.

        Args:
            new_sampler (Sampler): The sampler to use for the current dataset.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> # use a DistributedSampler instead
            >>> new_sampler = ds.DistributedSampler(10, 2)
            >>> dataset.use_sampler(new_sampler)
        """
        if new_sampler is None:
            raise TypeError("Input sampler can not be None.")
        if not isinstance(new_sampler, (samplers.BuiltinSampler, samplers.Sampler)):
            raise TypeError("Input sampler is not an instance of a sampler.")
        self.dataset_size = None

        self.sampler = self.sampler.child_sampler
        self.add_sampler(new_sampler)

    def is_shuffled(self):
        return self.sampler.is_shuffled()

    def is_sharded(self):
        return self.sampler.is_sharded()

    @check_split
    def split(self, sizes, randomize=True):
        """
        Split the dataset into smaller, non-overlapping datasets.

        Args:
            sizes (Union[list[int], list[float]]): If a list of integers [s1, s2, …, sn] is
                provided, the dataset will be split into n datasets of size s1, size s2, …, size sn
                respectively. If the sum of all sizes does not equal the original dataset size, an
                error will occur.
                If a list of floats [f1, f2, …, fn] is provided, all floats must be between 0 and 1
                and must sum to 1, otherwise an error will occur. The dataset will be split into n
                Datasets of size round(f1*K), round(f2*K), …, round(fn*K) where K is the size of the
                original dataset.
                If after rounding:

                - Any size equals 0, an error will occur.
                - The sum of split sizes < K, the difference will be added to the first split.
                - The sum of split sizes > K, the difference will be removed from the first large
                  enough split such that it will have at least 1 row after removing the difference.

            randomize (bool, optional): Determines whether or not to split the data randomly (default=True).
                If True, the data will be randomly split. Otherwise, each split will be created with
                consecutive rows from the dataset.

        Note:
            1. There is an optimized split function, which will be called automatically when the dataset
               that calls this function is a MappableDataset.
            2. Dataset should not be sharded if split is going to be called. Instead, create a
               DistributedSampler and specify a split to shard after splitting. If the dataset is
               sharded after a split, it is strongly recommended setting the same seed in each instance
               of execution, otherwise each shard may not be part of the same split (see Examples).
            3. It is strongly recommended to not shuffle the dataset, but use randomize=True instead.
               Shuffling the dataset may not be deterministic, which means the data in each split
               will be different in each epoch. Furthermore, if sharding occurs after split, each
               shard may not be part of the same split.

        Raises:
            RuntimeError: If get_dataset_size returns None or is not supported for this dataset.
            RuntimeError: If `sizes` is list of integers and sum of all elements in sizes does not
                equal the dataset size.
            RuntimeError: If `sizes` is list of float and there is a split with size 0 after calculations.
            RuntimeError: If the dataset is sharded prior to calling split.
            ValueError: If `sizes` is list of float and not all floats are between 0 and 1, or if the
                floats don't sum to 1.

        Returns:
            tuple(Dataset), a tuple of datasets that have been split.

        Examples:
            >>> # Since many datasets have shuffle on by default, set shuffle to False if split will be called!
            >>> dataset = ds.ImageFolderDataset(image_folder_dataset_dir, shuffle=False)
            >>>
            >>> # Set the seed, and tell split to use this seed when randomizing.
            >>> # This is needed because sharding will be done later
            >>> ds.config.set_seed(58)
            >>> train_dataset, test_dataset = dataset.split([0.9, 0.1])
            >>>
            >>> # To shard the train dataset, use a DistributedSampler
            >>> train_sampler = ds.DistributedSampler(10, 2)
            >>> train_dataset.use_sampler(train_sampler)
        """
        if self.is_shuffled():
            logger.warning("Dataset is shuffled before split.")

        if self.is_sharded():
            raise RuntimeError("Dataset should not be sharded before split.")

        absolute_sizes = self._get_absolute_split_sizes(sizes)
        splits = []
        current_split_start_index = 0
        for size in absolute_sizes:
            ds = copy.deepcopy(self)
            ds.dataset_size = None
            if randomize:
                # want to shuffle the same way every epoch before split, we are assuming
                # that the user will call set_seed
                random_sampler = samplers.RandomSampler()
                random_sampler.reshuffle_each_epoch = False
                ds.add_sampler(random_sampler)

            subset_sampler = samplers.SequentialSampler(current_split_start_index, size)
            ds.add_sampler(subset_sampler)

            # add sequential sampler, so that if user calls use_sampler, we will
            # get rid of the sequential sampler instead of something we need
            ds.add_sampler(samplers.SequentialSampler())

            splits.append(ds)

            current_split_start_index += size

        return tuple(splits)


class BucketBatchByLengthDataset(Dataset):
    """
    The result of applying BucketBatchByLength operator to the input dataset.
    """

    def __init__(self, input_dataset, column_names, bucket_boundaries, bucket_batch_sizes, element_length_function,
                 pad_info, pad_to_bucket_boundary, drop_remainder):
        super().__init__(children=input_dataset)

        self.column_names = to_list(column_names)
        self.bucket_boundaries = replace_none(bucket_boundaries, [])
        self.bucket_batch_sizes = replace_none(bucket_batch_sizes, [])
        self.element_length_function = element_length_function
        self.pad_info = replace_none(pad_info, {})
        self.pad_to_bucket_boundary = replace_none(pad_to_bucket_boundary, False)
        self.drop_remainder = replace_none(drop_remainder, False)

    def parse(self, children=None):
        return cde.BucketBatchByLengthNode(children[0], self.column_names, self.bucket_boundaries,
                                           self.bucket_batch_sizes, self.element_length_function, self.pad_info,
                                           self.pad_to_bucket_boundary, self.drop_remainder)


class BatchDataset(Dataset):
    """
    The result of applying Batch operator to the input dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be batched.
        batch_size (Union[int, function]): The number of rows each batch is created with. An
            int or callable which takes exactly 1 parameter, BatchInfo.
        drop_remainder (bool, optional): Determines whether or not to drop the last
            possibly incomplete batch (default=False). If True, and if there are less
            than batch_size rows available to make the last batch, then those rows will
            be dropped and not propagated to the child node.
        num_parallel_workers (int, optional): Number of workers to process the dataset in parallel (default=None).
        per_batch_map (callable, optional): Per batch map callable. A callable which takes
            (list[Tensor], list[Tensor], ..., BatchInfo) as input parameters. Each list[Tensor] represents a batch of
            Tensors on a given column. The number of lists should match with number of entries in input_columns. The
            last parameter of the callable must always be a BatchInfo object.
        input_columns (Union[str, list[str]], optional): List of names of the input columns. The size of the list must
            match with signature of per_batch_map callable.
        output_columns (Union[str, list[str]], optional): List of names assigned to the columns outputted by
            the last operation. This parameter is mandatory if len(input_columns) !=
            len(output_columns). The size of this list must match the number of output
            columns of the last operation. (default=None, output columns will have the same
            name as the input columns, i.e., the columns will be replaced).
        column_order (Union[str, list[str]], optional): Specifies the list of all the columns you need in the whole
                dataset. The parameter is required when len(input_column) != len(output_column). Caution: the list here
                is not just the columns specified in parameter input_columns and output_columns.
        pad_info (dict, optional): Whether to perform padding on selected columns. pad_info={"col1":([224,224],0)}
            will pad column with name "col1" to a tensor of size [224,224] and fill the missing with 0.
        max_rowsize(int, optional): Maximum size of row in MB that is used for shared memory allocation to copy
            data between processes.  This is only used if python_multiprocessing is set to True (default=16).

    """

    def __init__(self, input_dataset, batch_size, drop_remainder=False, num_parallel_workers=None, per_batch_map=None,
                 input_columns=None, output_columns=None, column_order=None, pad_info=None,
                 python_multiprocessing=False, max_rowsize=16):
        super().__init__(children=input_dataset, num_parallel_workers=num_parallel_workers)

        if BatchDataset._is_ancestor_of_repeat(input_dataset):
            logger.warning("Repeat is located before batch, data from two epochs can be batched together.")

        BatchDataset._update_batch_size_for_syncwait(input_dataset, batch_size)

        # if batch_size is callable, set batch_size to 1 and batch_size_func to that callable function
        self.batch_size = batch_size if not callable(batch_size) else 1
        self.batch_size_func = None if not callable(batch_size) else batch_size

        self.drop_remainder = replace_none(drop_remainder, False)

        self.per_batch_map = per_batch_map

        self.input_columns = to_list(input_columns)
        self.output_columns = to_list(output_columns)
        self.column_order = to_list(column_order)

        self.pad = bool(pad_info is not None)
        self.pad_info = replace_none(pad_info, dict())

        self.python_multiprocessing = python_multiprocessing
        self.process_pool = None
        self.hook = None
        self.eot = None
        self.watch_dog = None
        self.workers = []
        self.max_rowsize = max_rowsize

    def parse(self, children=None):
        return cde.BatchNode(children[0], self.batch_size, self.drop_remainder, self.pad, self.input_columns,
                             self.output_columns, self.column_order, self.batch_size_func, self.per_batch_map,
                             self.pad_info)

    @staticmethod
    def _is_ancestor_of_repeat(dataset):
        """
        Utility function to find the case where repeat is used before batch.

        Args:
             dataset (Dataset): Dataset to be checked.

        Returns:
            bool, whether repeat is used before batch.
        """
        if isinstance(dataset, RepeatDataset):
            return True
        flag = False
        for input_dataset in dataset.children:
            flag = flag | BatchDataset._is_ancestor_of_repeat(input_dataset)
        return flag

    @staticmethod
    def _update_batch_size_for_syncwait(dataset, batch_size):
        """
        Utility function to notify batch size to sync_wait.

        Args:
             dataset (Dataset): Dataset to be checked.
             batch_size (int): batch size to notify.
        """
        if isinstance(dataset, SyncWaitDataset):
            dataset.update_sync_batch_size(batch_size)
        for input_dataset in dataset.children:
            BatchDataset._update_batch_size_for_syncwait(input_dataset, batch_size)

    def __deepcopy__(self, memodict):
        return self.__safe_deepcopy__(memodict, exclude=("per_batch_map", "batch_size_func", "__transfer_dataset__"))

    # Iterator bootstrap will be called on iterator construction.
    # A deep copy of Dataset object is created prior of iterator_bootstrap.
    # This method will create per iterator process pool and bind pyfunc execution to the pool.
    def iterator_bootstrap(self):
        """
        Per iterator bootstrap callback.
        """
        if self.python_multiprocessing:
            if self.per_batch_map is None:
                logger.warning("per_batch_map is None so python_multiprocessing does not work.")
                return
            arg_q_list = []
            res_q_list = []

            # If user didn't specify num_parallel_workers, set it to default
            if self.num_parallel_workers is not None:
                num_parallel = self.num_parallel_workers
            else:
                num_parallel = get_num_parallel_workers()

            if get_enable_shared_mem():
                _check_shm_usage(num_parallel, 1, self.max_rowsize * self.batch_size, 2)
                for _ in range(num_parallel):
                    arg_q_list.append(_SharedQueue(1, max_rowsize=self.max_rowsize * self.batch_size))
                    res_q_list.append(_SharedQueue(1, max_rowsize=self.max_rowsize * self.batch_size))

            # Construct pool with the callable list
            # The callable list and _pyfunc_worker_init are used to pass lambda function in to subprocesses
            self.process_pool = multiprocessing.Pool(processes=num_parallel,
                                                     initializer=_pyfunc_worker_init,
                                                     initargs=([self.per_batch_map], arg_q_list, res_q_list))

            idx = 0
            global _OP_NAME, _OP_PROCESS, _LOCK
            op_id = _OP_NAME[str(self)]
            process_id = {op_id: [self.num_parallel_workers, set()]}
            # obtain process id from multiprocessing.pool
            for pool in self.process_pool._pool:  # pylint: disable=W0212
                process_id[op_id][1].add(pool.pid)
                self.workers.append(pool)
            with _LOCK:
                _OP_PROCESS.update(process_id)

            # Wrap per_batch_map into _PythonCallable
            self.per_batch_map = _PythonCallable(self.per_batch_map, idx, self.process_pool, arg_q_list, res_q_list)
            self.hook = _ExceptHookHandler()

            # batch will launch a watch dog thread to monitoring sub processes
            self._launch_watch_dog()

            atexit.register(_mp_pool_exit_preprocess)
            # If Python version greater than 3.8, we need to close ThreadPool in atexit for unclean pool teardown.
            if sys.version_info >= (3, 8):
                atexit.register(self.process_pool.close)
        else:
            if self.per_batch_map is not None:
                self.per_batch_map = FuncWrapper(self.per_batch_map)

    def _launch_watch_dog(self):
        if platform.system().lower() != 'windows':
            self.eot = threading.Event()
            self.watch_dog = threading.Thread(target=_watch_dog, args=(self.eot, self.workers, self.process_pool))
            self.watch_dog.daemon = True
            self.watch_dog.start()

    def _abort_watchdog(self):
        if not self.eot.is_set():
            self.eot.set()

    def __del__(self):
        if hasattr(self, 'process_pool') and self.process_pool is not None:
            self.process_pool.close()
        if hasattr(self, 'watch_dog') and self.watch_dog is not None and hasattr(self, 'eot') and self.eot is not None:
            self._abort_watchdog()


class BatchInfo(cde.CBatchInfo):
    """
    Only the batch size function and per_batch_map of the batch operator can dynamically adjust parameters
    based on the number of batches and epochs during training.
    """

    def get_batch_num(self):
        """
        Return the batch number of the current batch.
        """
        return

    def get_epoch_num(self):
        """
        Return the epoch number of the current batch.
        """
        return


class BlockReleasePair:
    """
    The blocking condition class used by SyncWaitDataset.

    Args:
        init_release_rows (int): Number of lines to allow through the pipeline.
        callback (function): The callback function that will be called when release is called (default=None).
    """

    def __init__(self, init_release_rows, callback=None):
        if isinstance(init_release_rows, int) and init_release_rows <= 0:
            raise ValueError("release_rows need to be greater than 0.")
        self.row_count = -init_release_rows
        self.cv = threading.Condition()
        self.callback = callback
        self.default_rows = init_release_rows
        self.disable = False

    def __deepcopy__(self, memodict):
        return self

    def reset(self):
        with self.cv:
            self.row_count = -self.default_rows
            self.cv.notify_all()

    def update_batched_size(self, batch_size):
        # sanity check
        if isinstance(batch_size, int) and batch_size <= 0:
            raise ValueError("batch_size need to be greater than 0.")

        # should only use before the pipeline creates
        self.row_count *= batch_size
        self.default_rows *= batch_size

    def block_func(self):
        """
        Function for handing blocking condition.

        Returns:
            bool, True.
        """
        with self.cv:
            # if disable is true, the always evaluate to true
            not_time_out = self.cv.wait_for(lambda: (self.row_count < 0 or self.disable),
                                            timeout=get_callback_timeout())
            # time_out will be False if time out occurs
            if not not_time_out:
                logger.warning("Timeout happened in sync_wait, maybe dataset.sync_update(condition=...) "
                               "is not added after dataset.create_dict_iterator(...), now disabling lock.")
                self.disable = True
            self.row_count += 1
        return True

    def release_func(self, pass_rows=None, data=None):
        with self.cv:
            if pass_rows is None:
                pass_rows = self.default_rows
            self.row_count -= pass_rows
            if self.callback is not None:
                self.callback(data)
            self.cv.notify_all()

    def disable_lock(self):
        with self.cv:
            self.disable = True
            self.cv.notify_all()


class SyncWaitDataset(Dataset):
    """
    The result of adding a blocking condition to the input Dataset.

    Args:
        input_dataset (Dataset): Input dataset to apply flow control.
        num_batch (int): Number of batches without blocking at the start of each epoch.
        condition_name (str): Condition name that is used to toggle sending next row.
        callback (function): Callback function that will be invoked when sync_update is called (default=None).

    Raises:
        RuntimeError: If condition name already exists.
    """

    def __init__(self, input_dataset, condition_name, num_batch, callback=None):
        super().__init__(children=input_dataset)

        # set to the default value, waiting for the batch to update it
        self._condition_name = condition_name
        if isinstance(num_batch, int) and num_batch <= 0:
            raise ValueError("num_batch need to be greater than 0.")

        self._pair = BlockReleasePair(num_batch, callback)
        if self._condition_name in self.children[0].get_sync_notifiers():
            raise RuntimeError("Condition name is already in use.")
        logger.info("Please remember to add dataset.sync_update(condition=%s), otherwise hanging will result. "
                    "If dataset.sync_update(condition=%s) has already been added, you can ignore the info.",
                    condition_name, condition_name)

    def parse(self, children=None):
        return cde.SyncWaitNode(children[0], self._condition_name, self._pair.block_func)

    def get_sync_notifiers(self):
        return {**self.children[0].get_sync_notifiers(), **{self._condition_name: self._pair.release_func}}

    def is_sync(self):
        return True

    def update_sync_batch_size(self, batch_size):
        if isinstance(batch_size, int) and batch_size <= 0:
            raise ValueError("num_batch need to be greater than 0.")
        self._pair.update_batched_size(batch_size)

    def disable_sync(self):
        logger.info("Disabling Sync")
        self._pair.disable_lock()

    @staticmethod
    def _is_ancestor_of_batch(dataset):
        """
        Utility function to find the case where sync_wait is used before batch.

        Args:
             dataset (Dataset): Dataset to be checked.

        Returns:
            bool, whether sync_wait is used before batch.
        """
        if isinstance(dataset, BatchDataset):
            return True
        flag = False
        for input_dataset in dataset.children:
            flag = flag | SyncWaitDataset._is_ancestor_of_batch(input_dataset)
        return flag

    def iterator_bootstrap(self):
        self._pair.reset()


class ShuffleDataset(Dataset):
    """
    The result of applying Shuffle operator to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be shuffled.
        buffer_size (int): Size of the buffer.

    Raises:
        RuntimeError: If exist sync operators before shuffle.
    """

    def __init__(self, input_dataset, buffer_size):
        super().__init__(children=input_dataset)
        self.buffer_size = buffer_size
        self.reshuffle_each_epoch = True

        if self.is_sync():
            raise RuntimeError("No shuffle after sync operators.")

    def parse(self, children=None):
        return cde.ShuffleNode(children[0], self.buffer_size, self.reshuffle_each_epoch)

    def is_shuffled(self):
        return True


# This wait function is for cleaning zombie subprocesses
def wait_pid():
    """
    This function is used by the main process to release subprocess resources.
    """
    try:
        while True:
            child_pid, _ = os.waitpid(-1, os.WNOHANG)
            if child_pid == 0:
                break
    except OSError:
        # waitpid may be failed for some reasons so we ignore this error
        pass


# Terminate subprocess launched by multiprocessing.pool
def _terminate_process(workers):
    for w in workers:
        if w.exitcode is None:
            w.terminate()
    for w in workers:
        if w._closed is False:  # pylint: disable=W0212
            w.join()


# Monitor the exit number of subprocesses
def _monitor_subprocess_exit(workers):
    subprocess_exit_num = 0
    for w in workers:
        if w.exitcode is not None:
            subprocess_exit_num += 1
    return subprocess_exit_num


# Dataset need _watch_dog thread to monitoring fork multi-processing,
# and thread can't be a member function otherwise python won't collect and release resources.
def _watch_dog(eot, workers, pool=None):
    """
    This thread is for monitoring subprocesses forked by GeneratorDataset/map/batch
    """
    if not isinstance(workers, list):
        raise TypeError("[Internal Error] The 2rd parameter of watch dog thread should be list of process, "\
                        "but got {}.".format(type(workers)))
    if pool is not None and not isinstance(pool, multiprocessing.pool.Pool):
        raise TypeError("[Internal Error] The 3rd parameter of watch dog thread should be multiprocessing.Pool, "\
                        "but got {}".format(type(pool)))
    while not eot.is_set():
        subprocess_exit_num = 0
        # Monitoring and count how many subprocesses already exit
        subprocess_exit_num = _monitor_subprocess_exit(workers)
        # If find subprocess exit, we will wait for 30s and do some waitpid operations
        if subprocess_exit_num > 0:
            if pool is not None:
                # Python multiprocessing.pool has a bug, if sub process of pool is killed, pool will launch
                # a new sub process, so we have to set worker_handler._state to TERMINATE to stop relaunching.
                if pool._state == RUN:  # pylint: disable=W0212
                    pool._state = TERMINATE  # pylint: disable=W0212
                    pool._worker_handler._state = TERMINATE  # pylint: disable=W0212
            start = time.time()
            while time.time() - start < 30:
                # We need to distinguishing get_dataset_size or train finished normally and hang scenario.
                # If get_dataset_size or train finished normally, _stop_subprocess can be execute and
                # self.need_abort can be set to True. If main process is hang in get(), self.need_abort
                # will never set to True, then we wait for 30s and kill main process
                if eot.is_set():
                    return
                # Sometimes subprocess may be zombie, so in 30s we can wait and do some useful tasks(waitpid).
                wait_pid()
            # multiprocessing.queue may hang in .get() forever when put() process was killed.
            # We have to exit main process otherwise main process will hang.
            if pool is not None:
                _terminate_process(pool._pool)  # pylint: disable=W0212
            else:
                _terminate_process(workers)
            logger.critical("The subprocess of dataset may exit unexpected or be killed, "
                            "main process will exit.")
            os.kill(os.getpid(), signal.SIGTERM)


# Pyfunc collection for multiprocess pyfunc
# This global variable will only be used within subprocesses
_GLOBAL_PYFUNC_LIST = []
_ARGS_QUEUE = []
_RET_QUEUE = []
_OP_NAME = dict()
_OP_PROCESS = dict()
_LOCK = threading.Lock()


# Pyfunc worker init function
# Python multiprocessing library forbid sending lambda function through pipe.
# This init function allow us to add all Python function to a global collection and then fork afterwards.
def _pyfunc_worker_init(pyfunc_list, args_queue, ret_queue):
    global _GLOBAL_PYFUNC_LIST
    global _ARGS_QUEUE
    global _RET_QUEUE
    _GLOBAL_PYFUNC_LIST = pyfunc_list
    _ARGS_QUEUE = args_queue
    _RET_QUEUE = ret_queue


# Pyfunc worker execution function
# All exceptions will be raised to main processes
def _pyfunc_worker_exec(index, qid, *args):
    """
    Internal function for call certain pyfunc in Python process.
    """
    # Some threads in multiprocess.pool can't process sigint signal,
    # and will occur hang problem, so ctrl+c will pass to parent process.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    if qid != -1:
        # Pass arguments through the Queue instead of directly to remote process
        args = _ARGS_QUEUE[qid].get()
        try:
            r = _GLOBAL_PYFUNC_LIST[index](*args)
        except Exception:
            return ExceptionHandler(where="in map(or batch) worker and execute python function")
        if isinstance(r, tuple):
            _RET_QUEUE[qid].put(r)
        else:
            _RET_QUEUE[qid].put((r,))
        return [qid]
    # not using shared memory for passing arguments, call function directly
    result = None
    try:
        result = _GLOBAL_PYFUNC_LIST[index](*args)
    except Exception:
        result = ExceptionHandler(where="in map(or batch) worker and execute python function")
    return result


# PythonCallable wrapper for multiprocess pyfunc
class _PythonCallable:
    """
    Internal Python function wrapper for multiprocessing pyfunc.
    """

    def __init__(self, py_callable, idx, pool=None, arg_q=None, res_q=None):
        # Original Python callable from user.
        self.py_callable = py_callable
        # Process pool created for current iterator.
        self.pool = pool
        # Python callable index for subprocess _GLOBAL_PYFUNC_LIST
        self.idx = idx

        if pool is not None:
            self.queuemap = {}
            self.arg_q = arg_q
            self.res_q = res_q
            self.next_queue = 0

    def __call__(self, *args):
        if self._pool_is_running() and check_iterator_cleanup() is False:
            result, qid, ret = self._send(*args)
            if ret:
                return result

            # todo this check might be wrong
            while check_iterator_cleanup() is False:
                try:
                    return self._receive(result, qid)
                except multiprocessing.TimeoutError:
                    continue
                except KeyboardInterrupt:
                    _set_iterator_cleanup()
                    self.pool.close()
                    self.pool.join()
                    raise Exception("Multiprocess MapOp worker receives KeyboardInterrupt.")
            return (None,)
        # Invoke original Python callable in master process in case the pool is gone.
        return self.py_callable(*args)

    def to_json(self):
        return self.py_callable.to_json()

    def _send(self, *args):
        """
        The map/batch operator will use multiprocessing-pool apply_async interface to execute python function
        in a sub process, apply_async will release GIL temporarily. For better performance, we use shared memory
        feature and pass shared queue instead of multiprocess args.
        """
        ret = False
        qid = None
        if self.arg_q != []:
            tid = threading.get_ident()
            # Need to register each thread to use a different queue to send data to pool
            if not tid in self.queuemap:
                qid = self.next_queue
                self.next_queue = self.next_queue + 1
                self.queuemap[tid] = qid
            else:
                qid = self.queuemap[tid]
            self.arg_q[qid].put(args)

            # This call will send the tensors along with Python callable index to the process pool.
            # Block, yield GIL. Current thread will reacquire GIL once result is returned.
            if self._pool_is_running() and check_iterator_cleanup() is False:
                result = self.pool.apply_async(_pyfunc_worker_exec, [self.idx, qid, []])
            else:
                ret = True
                result = self.py_callable(*args)
        else:
            result = self.pool.apply_async(_pyfunc_worker_exec, [self.idx, -1, *args])
        return result, qid, ret

    def _receive(self, result, qid):
        """
        The map/batch operator will use multiprocessing-pool get interface to sync output data from a sub process,
        get interface will reacquire GIL. For better performance, we use shared memory feature and get data from
        shared queue directly.
        """
        if self.arg_q != []:
            r = result.get(30)
            if isinstance(r, ExceptionHandler):
                r.reraise()
            if r[0] != qid:
                raise Exception("In PyCallable, got results from wrong thread")
            r = self.res_q[qid].get()
            return r
        r = result.get(30)
        if isinstance(r, ExceptionHandler):
            r.reraise()
        return r

    def _pool_is_running(self):
        # note here: the RUN state of python3.7 and python3.8 is different:
        # python3.7: RUN = 0
        # python3.8: RUN = "RUN"
        # so we use self.pool._state == RUN instead and we can't use _state == 0 any more.
        if self.pool is not None and self.pool._state == RUN:  # pylint: disable=W0212
            return True
        return False


def _mp_pool_exit_preprocess():
    if check_iterator_cleanup() is False:
        # Set the iterator_cleanup flag to True before exiting, and wait 3s for all apply_async
        # applied to the multiprocessing task to prevent multiprocessing from hang when exiting
        _set_iterator_cleanup()
        time.sleep(3)


class _ExceptHookHandler:
    def __init__(self):
        sys.excepthook = self.__handler_exception

    def __handler_exception(self, ex_type, value, tb):
        logger.critical("Uncaught exception: ", exc_info=(ex_type, value, tb))
        _mp_pool_exit_preprocess()


class MapDataset(TextBaseDataset, Dataset):
    """
    The result of applying the Map operator to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be mapped.
        operations (TensorOp): A function mapping a nested structure of tensors
            to another nested structure of tensor (default=None).
        input_columns (Union[str, list[str]]): List of names of the input columns
            (default=None, the operations will be applied on the first columns in the dataset).
            The size of the list should match the number of inputs of the first operator.
        output_columns (Union[str, list[str]], optional): List of names of the output columns.
            The size of the list should match the number of outputs of the last operator
            (default=None, output columns will be the input columns, i.e., the columns will
            be replaced).
        column_order (list[str], optional): Specifies the list of all the columns you need in the whole
            dataset. The parameter is required when len(input_column) != len(output_column). Caution: the list here
            is not just the columns specified in parameter input_columns and output_columns.
        num_parallel_workers (int, optional): Number of workers to process the dataset
            in parallel (default=None).
        python_multiprocessing (bool, optional): Parallelize Python operations with multiple worker process. This
            option could be beneficial if the Python operation is computational heavy (default=False).
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).
        callbacks (DSCallback, list[DSCallback], optional): List of Dataset callbacks to be called (Default=None)
        max_rowsize(int, optional): Maximum size of row in MB that is used for shared memory allocation to copy
            data between processes.  This is only used if python_multiprocessing is set to True (default=16).
        offload (bool, optional): Flag to indicate whether offload is used (Default=None).

    Raises:
        ValueError: If len(input_columns) != len(output_columns) and column_order is not specified.
    """

    def __init__(self, input_dataset, operations=None, input_columns=None, output_columns=None, column_order=None,
                 num_parallel_workers=None, python_multiprocessing=False, cache=None, callbacks=None, max_rowsize=16,
                 offload=None):
        super().__init__(children=input_dataset, num_parallel_workers=num_parallel_workers, cache=cache)
        self.operations = to_list(operations)
        self.operations = py_transforms.Compose.reduce(self.operations)
        self.input_columns = to_list(input_columns)
        self.output_columns = to_list(output_columns)
        self.column_order = replace_none(column_order, [])

        #  If output_columns were not provided then use input_columns
        self.output_columns = self.input_columns if not self.output_columns else self.output_columns

        if self.input_columns and self.output_columns \
                and len(self.input_columns) != len(self.output_columns) \
                and not self.column_order:
            raise ValueError("When length of input_columns and output_columns are not equal,"
                             " column_order must be specified.")

        self.python_multiprocessing = python_multiprocessing
        self.process_pool = None
        self.hook = None
        self.eot = None
        self.watch_dog = None
        self.workers = []

        self.callbacks = to_list(callbacks)
        self.max_rowsize = max_rowsize
        self.offload = offload

    def parse(self, children=None):
        operations = []
        for op in self.operations:
            if op and getattr(op, 'parse', None):
                operations.append(op.parse())
            else:
                operations.append(op)

        callbacks = [cb.create_runtime_obj() for cb in self.callbacks]
        return cde.MapNode(children[0], operations, self.input_columns, self.output_columns, self.column_order,
                           callbacks, self.max_rowsize, OffloadToManualOffloadMode[self.offload])

    def __deepcopy__(self, memodict):
        return self.__safe_deepcopy__(memodict, exclude=("operations", "callbacks", "__transfer_dataset__"))

    # Iterator bootstrap will be called on iterator construction.
    # A deep copy of Dataset object is created prior of iterator_bootstrap.
    # This method will create per iterator process pool and bind pyfunc execution to the pool.
    def iterator_bootstrap(self):
        """
        Per iterator bootstrap callback.
        """

        if self.python_multiprocessing:
            iter_specific_operations = []
            callable_list = []
            arg_q_list = []
            res_q_list = []

            # If user didn't specify num_parallel_workers, set it to default
            num_parallel = get_num_parallel_workers()
            if self.num_parallel_workers is not None:
                num_parallel = self.num_parallel_workers

            if get_enable_shared_mem():
                _check_shm_usage(num_parallel, 1, self.max_rowsize, 2)
                for _ in range(num_parallel):
                    arg_q_list.append(_SharedQueue(1, max_rowsize=self.max_rowsize))
                    res_q_list.append(_SharedQueue(1, max_rowsize=self.max_rowsize))

            # Pass #1, look for Python callables and build list
            for op in self.operations:
                # our c transforms is now callable and should not be run in Python multithreading
                if MapDataset.__operation_valid_for_multiprocessing(op):
                    callable_list.append(op)

            if callable_list:
                # Construct pool with the callable list
                # The callable list and _pyfunc_worker_init are used to pass lambda function in to subprocesses
                self.process_pool = multiprocessing.Pool(processes=num_parallel,
                                                         initializer=_pyfunc_worker_init,
                                                         initargs=(callable_list, arg_q_list, res_q_list))

                # Pass #2
                idx = 0
                global _OP_NAME, _OP_PROCESS, _LOCK
                op_id = _OP_NAME[str(self)]
                # obtain process id from multiprocessing.pool
                process_id = {op_id: [self.num_parallel_workers, set()]}
                for pool in self.process_pool._pool:  # pylint: disable=W0212
                    process_id[op_id][1].add(pool.pid)
                    self.workers.append(pool)
                with _LOCK:
                    _OP_PROCESS.update(process_id)
                for op in self.operations:
                    # our c transforms is now callable and should not be run in Python multithreading
                    if MapDataset.__operation_valid_for_multiprocessing(op):
                        # Wrap Python callable into _PythonCallable
                        iter_specific_operations.append(_PythonCallable(op, idx, self.process_pool,
                                                                        arg_q_list, res_q_list))
                        idx += 1
                    else:
                        # CPP ops remain the same
                        iter_specific_operations.append(op)
                self.operations = iter_specific_operations
                self.hook = _ExceptHookHandler()

                # Map multiprocessing will launch a watch dog thread for monitoring sub processes
                self._launch_watch_dog()

                atexit.register(_mp_pool_exit_preprocess)
                # If Python version greater than 3.8, we need to close ThreadPool in atexit for unclean pool teardown.
                if sys.version_info >= (3, 8):
                    atexit.register(self.process_pool.close)

    @staticmethod
    def __operation_valid_for_multiprocessing(op):
        if callable(op) and str(op).find("c_transform") < 0:
            return True
        return False

    def _launch_watch_dog(self):
        if platform.system().lower() != 'windows':
            self.eot = threading.Event()
            self.watch_dog = threading.Thread(target=_watch_dog, args=(self.eot, self.workers, self.process_pool))
            self.watch_dog.daemon = True
            self.watch_dog.start()

    def _abort_watchdog(self):
        if not self.eot.is_set():
            self.eot.set()

    def __del__(self):
        if hasattr(self, 'process_pool') and self.process_pool is not None:
            self.process_pool.close()
            self.process_pool.join()
        if hasattr(self, 'watch_dog') and self.watch_dog is not None and hasattr(self, 'eot') and self.eot is not None:
            self._abort_watchdog()


class FilterDataset(Dataset):
    """
    The result of applying filter predicate to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be mapped.
        predicate (callable): Python callable which returns a boolean value. If False then filter the element.
        input_columns (Union[str, list[str]], optional): List of names of the input columns
        (default=None, the predicate will be applied to all columns in the dataset).
        num_parallel_workers (int, optional): Number of workers to process the dataset
            in parallel (default=None).
    """

    def __init__(self, input_dataset, predicate, input_columns=None, num_parallel_workers=None):
        super().__init__(children=input_dataset, num_parallel_workers=num_parallel_workers)
        self.predicate = lambda *args: bool(predicate(*args))
        self.input_columns = to_list(input_columns)

    def parse(self, children=None):
        return cde.FilterNode(children[0], self.predicate, self.input_columns)


class RepeatDataset(Dataset):
    """
    The result of applying Repeat operator to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be repeated.
        count (int): Number of times the dataset will be repeated (default=-1, repeat indefinitely).
    """

    def __init__(self, input_dataset, count):
        super().__init__(children=input_dataset)
        self.count = replace_none(count, -1)

    def parse(self, children=None):
        return cde.RepeatNode(children[0], self.count)


class SkipDataset(Dataset):
    """
    The result of applying Skip operator to the input Dataset.

    Args:
        input_dataset (Dataset): Input dataset to have elements skipped.
        count (int): Number of elements to be skipped in the dataset.
    """

    def __init__(self, input_dataset, count):
        super().__init__(input_dataset)
        self.count = count

    def parse(self, children=None):
        return cde.SkipNode(children[0], self.count)


class TakeDataset(Dataset):
    """
    The result of applying Take operator to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to have elements taken from.
        count (int): Number of elements to be taken from the dataset.
    """

    def __init__(self, input_dataset, count):
        super().__init__(children=input_dataset)
        self.count = count

    def parse(self, children=None):
        return cde.TakeNode(children[0], self.count)


class ZipDataset(Dataset):
    """
    The result of applying Zip operator to the input Dataset.

    Args:
        datasets (tuple): A tuple of datasets to be zipped together.

    Raises:
        TypeError: If dataset is not an instance of Dataset.
    """

    def __init__(self, datasets):
        super().__init__(children=datasets)

    def parse(self, children=None):
        return cde.ZipNode(children)

    def is_sync(self):
        return any([c.is_sync() for c in self.children])


class ConcatDataset(Dataset):
    """
    The result of applying concat dataset operator to the input Dataset.

    Args:
        datasets (list): A list of datasets to be concatenated together.

    Raises:
        TypeError: If dataset is not an instance of Dataset.
        ValueError: If there is no samples in the one of the datasets.
    """

    def __init__(self, datasets):
        super().__init__(children=datasets)
        for dataset in datasets:
            if not isinstance(dataset, Dataset):
                raise TypeError("Invalid dataset, expected Dataset object, but got %s!" % type(dataset))
        self.datasets = datasets
        self._sampler = samplers.SequentialSampler(num_samples=None)

        self.children_sizes_ = [c.get_dataset_size() for c in self.children]
        child_index = 0
        for item in self.children_sizes_:
            if item == 0:
                raise ValueError("There are no samples in the dataset number %d. Please make sure there are "
                                 "valid samples in the dataset." % child_index)
            child_index += 1

        # _children_flag_and_nums: A list of pair<int ,int>.The first element of pair is flag that characterizes
        # whether the data set is mappable. The second element of pair is length of the dataset
        self._children_flag_and_nums = []

        # _children_start_end_index_: A list of pair<int ,int>.The elements of pair are used to characterize
        # the valid position of the dataset corresponding to the subscript when sampling
        self._children_start_end_index_ = []
        for index, child in enumerate(self.children):
            tem_list = [-1, -1]
            self._children_start_end_index_.append(tem_list)
            dataset_len = self.children_sizes_[index]
            if isinstance(child, GeneratorDataset) and not hasattr(child.source, "__getitem__"):
                dataset_len = 0
                self.children_sizes_[index] = 0

            if isinstance(child, MappableDataset):
                self._children_flag_and_nums.append((0, dataset_len))
            else:
                self._children_flag_and_nums.append((1, dataset_len))

    def parse(self, children=None):
        return cde.ConcatNode(children, self._sampler, self._children_flag_and_nums, self._children_start_end_index_)

    def use_sampler(self, sampler):
        """
        Set the distributedSampler to concat dataset

        Args:
            sampler (Sampler): The sampler to use for the current dataset.
                Currently supported: DistributedSampler.

        Raises:
            TypeError: If the sampler is not an instance of DistributedSampler
            ValueError: If the parameter shuffle of sampler is True
            ValueError: If the parameter NumSamples of sampler is not None.
            ValueError: If num_shards <=0.
        """
        if not isinstance(sampler, samplers.DistributedSampler):
            raise TypeError("The parameter %s of concat must be DistributedSampler!" % sampler)

        if sampler.is_shuffled():
            raise ValueError("The parameter shuffle of DistributedSampler must be False!")

        if sampler.num_shards <= 0:
            raise ValueError("The parameter num_shards of DistributedSampler must be positive int!")

        if sampler.get_num_samples() is not None:
            raise ValueError("The parameter num_samples of DistributedSampler is not support to be set!")

        self.dataset_size = None

        self._sampler = sampler
        cumulative_samples_nums = 0
        for index, child in enumerate(self.children):
            if hasattr(child, 'sampler') and child.sampler.get_num_samples() is not None:
                raise ValueError("The parameter NumSamples of %s is not support to be set!" % child)

            if isinstance(child, BatchDataset):
                raise TypeError("The parameter %s of concat must not be BatchDataset!" % child)

            # if child is mappable and the length is greater than 0
            if not self._children_flag_and_nums[index][0] and self._children_flag_and_nums[index][1]:

                tem_value = cumulative_samples_nums + self._children_flag_and_nums[index][1]

                if not self._children_flag_and_nums[index][1] >= sampler.num_shards:
                    if tem_value < sampler.num_shards:
                        self._children_start_end_index_[index][0] = cumulative_samples_nums
                        self._children_start_end_index_[index][1] = tem_value
                    else:
                        self._children_start_end_index_[index][0] = cumulative_samples_nums
                        self._children_start_end_index_[index][1] = tem_value % sampler.num_shards

                tem_sampler = copy.deepcopy(sampler)
                tem_sampler.set_offset(cumulative_samples_nums)
                child.use_sampler(tem_sampler)

            cumulative_samples_nums += self.children_sizes_[index]
            cumulative_samples_nums %= sampler.num_shards


class RenameDataset(Dataset):
    """
    The result of applying Rename operator to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be Renamed.
        input_columns (Union[str, list[str]]): List of names of the input columns.
        output_columns (Union[str, list[str]]): List of names of the output columns.
    """

    def __init__(self, input_dataset, input_columns, output_columns):
        super().__init__(children=input_dataset)
        self.input_column_names = to_list(input_columns)
        self.output_column_names = to_list(output_columns)

    def parse(self, children=None):
        return cde.RenameNode(children[0], self.input_column_names, self.output_column_names)


def to_list(items):
    if items is None:
        return []
    if isinstance(items, tuple):
        return list(items)
    if not isinstance(items, list):
        return [items]
    return items


class ProjectDataset(Dataset):
    """
    The result of applying Project operator to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be Projected.
        columns (Union[str, list[str]]): List of names of the columns to project.
    """

    def __init__(self, input_dataset, columns):
        super().__init__(children=input_dataset)
        self.columns = to_list(columns)

    def parse(self, children=None):
        return cde.ProjectNode(children[0], self.columns)


class _ToDevice:
    """
    Internal class to handle sending data to device.
    """

    def __init__(self, dataset, num_epochs):
        ir_tree, self.api_tree = dataset.create_ir_tree()

        self._runtime_context = cde.PythonRuntimeContext()
        self._runtime_context.Init()
        self._to_device = cde.ToDevice(num_epochs)
        self._to_device.Init(ir_tree)
        self._runtime_context.AssignConsumer(self._to_device)

        ITERATORS_LIST.append(weakref.ref(self))
        _unset_iterator_cleanup()

    def send(self):
        self._to_device.Send()

    def stop_send(self):
        """
        send stop send signal to pipeline, it is used when end of sequence is sent at the epoch end.
        """
        self._to_device.StopSend()

    def continue_send(self):
        """
        send continue send signal to pipeline, it is used when end of sequence is sent at the epoch end.
        """
        self._to_device.ContinueSend()

    def get_data_info(self):
        """
        Get type and shape of current batch.
        """
        return self._to_device.GetDataInfo()

    def release(self):
        """
        Manually terminate Device Queue instead of relying on out of scope destruction.
        """
        if hasattr(self, '_runtime_context') and self._runtime_context:
            if hasattr(self, '_to_device') and self._to_device:
                self._runtime_context.Terminate()
                del self._to_device
            del self._runtime_context

    def __deepcopy__(self, memodict):
        return self

    def get_offload_model(self):
        """
        Get offload model containing removed offload ops from pipeline.
        """
        offload_model = GetOffloadModel(self._to_device)
        return offload_model


class TransferDataset(Dataset):
    """
    The result of applying TDT operator to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be transferred.
        send_epoch_end (bool, optional): Whether to send end of sequence to device or not (default=True).
        create_data_info_queue (bool, optional): Whether to create queue which stores
            types and shapes of data or not (default=False).

    Raises:
        TypeError: If device_type is empty.
        ValueError: If device_type is not 'Ascend', 'GPU' or 'CPU'.
        RuntimeError: If dataset is unknown.
    """

    def __init__(self, input_dataset, send_epoch_end=True, create_data_info_queue=False):
        super().__init__(children=input_dataset)
        self.queue_name = str(uuid.uuid1())
        self.device_type = context.get_context("device_target") if context else "CPU"
        self.device_id = context.get_context("device_id") if context else 0

        self._send_epoch_end = replace_none(send_epoch_end, True)
        self._create_data_info_queue = create_data_info_queue
        self._to_device = None

    def parse(self, children=None):
        total_batch = 0
        if hasattr(self.children[0], "__total_batch__"):
            total_batch = self.children[0].__total_batch__
        return cde.TransferNode(children[0], self.queue_name, self.device_type, self.device_id, self._send_epoch_end,
                                total_batch, self._create_data_info_queue)

    def create_dict_iterator(self, num_epochs=-1, output_numpy=False):
        raise RuntimeError("TransferDataset is not iterable.")

    def create_tuple_iterator(self, columns=None, num_epochs=-1, output_numpy=False, do_copy=True):
        raise RuntimeError("TransferDataset is not iterable.")

    def __iter__(self):
        raise RuntimeError("TransferDataset is not iterable.")

    def output_shapes(self):
        raise RuntimeError("TransferDataset does not support obtaining output_shapes.")

    def output_types(self):
        raise RuntimeError("TransferDataset does not support obtaining output_types.")

    @check_to_device_send
    def send(self, num_epochs=-1):
        """
        Send to device
        """
        if Dataset._noop_mode():
            return
        if self._to_device is not None:
            del self._to_device
        self._to_device = _ToDevice(self, num_epochs)
        self._to_device.send()

    def stop_send(self):
        if self._to_device is not None:
            self._to_device.stop_send()

    def continue_send(self):
        if self._to_device is not None:
            self._to_device.continue_send()

    def get_data_info(self):
        """
        Get type and shape of current batch
        """
        if self._to_device is not None:
            return self._to_device.get_data_info()
        raise RuntimeError("Calling get_data_info with bad state.")

    def get_offload_model(self):
        if self._to_device is not None:
            return self._to_device.get_offload_model()

        raise RuntimeError("get_offload_model, _to_device is None")

    def release(self):
        """
        Manually terminate Device Queue instead of relying on out of scope destruction.
        """
        if self._to_device is not None:
            self._to_device.release()


class RangeDataset(MappableDataset):
    """
    A source dataset that reads and parses datasets stored on disk in a range.

    Args:
        start (int): Starting index.
        stop (int): Ending index.
        step (int): Step size in the range specified by start and stop.
    """

    def __init__(self, start, stop, step):
        super().__init__()
        self.start = start
        self.stop = stop
        self.step = step

    def parse(self, children=None):
        raise NotImplementedError("Dataset has to implement parse method.")

    def is_shuffled(self):
        return False

    def is_sharded(self):
        return False

    def get_dataset_size(self):
        if self.dataset_size is None:
            self.dataset_size = math.ceil((self.stop - self.start) / self.step)
        return self.dataset_size


class FashionMnistDataset(MappableDataset):
    """
    A source dataset for reading and parsing the FASHION-MNIST dataset.

    The generated dataset has two columns :py:obj:`[image, label]`.
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`label` is a scalar of the uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be `train`, `test` or `all`. `train` will read from 60,000
            train samples, `test` will read from 10,000 test samples, `all` will read from all 70,000 samples.
            (default=None, will read all samples)
        num_samples (int, optional): The number of images to be included in the dataset
            (default=None, will read all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, will use value set in the config).
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset
            (default=None, expected order behavior shown in the table).
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, `num_samples` reflects the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` (default=None). This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_dir does not contain data files.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Note:
        - This dataset can take in a `sampler`. `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> fashion_mnist_dataset_dir = "/path/to/fashion_mnist_dataset_directory"
        >>>
        >>> # Read 3 samples from FASHIONMNIST dataset
        >>> dataset = ds.FashionMnistDataset(dataset_dir=fashion_mnist_dataset_dir, num_samples=3)
        >>>
        >>> # Note: In FASHIONMNIST dataset, each dictionary has keys "image" and "label"

    About Fashion-MNIST dataset:

    Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and
    a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.
    We intend Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking
    machine learning algorithms. It shares the same image size and structure of training and testing splits.

    You can unzip the dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── fashionmnist_dataset_dir
             ├── t10k-images-idx3-ubyte
             ├── t10k-labels-idx1-ubyte
             ├── train-images-idx3-ubyte
             └── train-labels-idx1-ubyte

    Citation:

    .. code-block::

        @online{xiao2017/online,
          author       = {Han Xiao and Kashif Rasul and Roland Vollgraf},
          title        = {Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms},
          date         = {2017-08-28},
          year         = {2017},
          eprintclass  = {cs.LG},
          eprinttype   = {arXiv},
          eprint       = {cs.LG/1708.07747},
        }
    """

    @check_mnist_cifar_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None,
                 sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.FashionMnistNode(self.dataset_dir, self.usage, self.sampler)


class ImageFolderDataset(MappableDataset):
    """
    A source dataset that reads images from a tree of directories.
    All images within one folder have the same label.

    The generated dataset has two columns: :py:obj:`[image, label]`.
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`label` is of a scalar of uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        num_samples (int, optional): The number of images to be included in the dataset
            (default=None, all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, set in the config).
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset
            (default=None, expected order behavior shown in the table).
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, expected order behavior shown in the table).
        extensions (list[str], optional): List of file extensions to be
            included in the dataset (default=None).
        class_indexing (dict, optional): A str-to-int mapping from folder name to index
            (default=None, the folder names will be sorted
            alphabetically and each class will be given a
            unique index starting from 0).
        decode (bool, optional): Decode the images after reading (default=False).
        num_shards (int, optional): Number of shards that the dataset will be divided
            into (default=None). When this argument is specified, `num_samples` reflects
            the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_dir does not contain data files.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        RuntimeError: If class_indexing is not a dictionary.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Note:
        - The shape of the image column is [image_size] if decode flag is False, or [H,W,C] otherwise.
        - This dataset can take in a `sampler`. `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> image_folder_dataset_dir = "/path/to/image_folder_dataset_directory"
        >>>
        >>> # 1) Read all samples (image files) in image_folder_dataset_dir with 8 threads
        >>> dataset = ds.ImageFolderDataset(dataset_dir=image_folder_dataset_dir,
        ...                                 num_parallel_workers=8)
        >>>
        >>> # 2) Read all samples (image files) from folder cat and folder dog with label 0 and 1
        >>> dataset = ds.ImageFolderDataset(dataset_dir=image_folder_dataset_dir,
        ...                                 class_indexing={"cat":0, "dog":1})
        >>>
        >>> # 3) Read all samples (image files) in image_folder_dataset_dir with extensions .JPEG and .png (case sensitive)
        >>> dataset = ds.ImageFolderDataset(dataset_dir=image_folder_dataset_dir,
        ...                                 extensions=[".JPEG", ".png"])

    About ImageFolderDataset:

    You can construct the following directory structure from your dataset files and read by MindSpore's API.

    .. code-block::

        .
        └── image_folder_dataset_directory
             ├── class1
             │    ├── 000000000001.jpg
             │    ├── 000000000002.jpg
             │    ├── ...
             ├── class2
             │    ├── 000000000001.jpg
             │    ├── 000000000002.jpg
             │    ├── ...
             ├── class3
             │    ├── 000000000001.jpg
             │    ├── 000000000002.jpg
             │    ├── ...
             ├── classN
             ├── ...
    """

    @check_imagefolderdataset
    def __init__(self, dataset_dir, num_samples=None, num_parallel_workers=None, shuffle=None, sampler=None,
                 extensions=None, class_indexing=None, decode=False, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.extensions = replace_none(extensions, [])
        self.class_indexing = replace_none(class_indexing, {})
        self.decode = replace_none(decode, False)

    def parse(self, children=None):
        return cde.ImageFolderNode(self.dataset_dir, self.decode, self.sampler, self.extensions, self.class_indexing)


class IMDBDataset(MappableDataset):
    """
    A source dataset for reading and parsing Internet Movie Database (IMDb).

    The generated dataset has two columns: :py:obj:`[text, label]`.
    The tensor of column :py:obj:`text` is of the string type.
    The tensor of column :py:obj:`label` is of a scalar of uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be `train`, `test` or `all`
            (default=None, will read all samples).
        num_samples (int, optional): The number of images to be included in the dataset
            (default=None, will read all samples).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, set in the config).
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset
            (default=None, expected order behavior shown in the table).
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided
            into (default=None). When this argument is specified, `num_samples` reflects
            the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_dir does not contain data files.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Note:
        - The shape of the test column.
        - This dataset can take in a `sampler`. `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> imdb_dataset_dir = "/path/to/imdb_dataset_directory"
        >>>
        >>> # 1) Read all samples (text files) in imdb_dataset_dir with 8 threads
        >>> dataset = ds.IMDBDataset(dataset_dir=imdb_dataset_dir, num_parallel_workers=8)
        >>>
        >>> # 2) Read train samples (text files).
        >>> dataset = ds.IMDBDataset(dataset_dir=imdb_dataset_dir, usage="train")

    About IMDBDataset:

    The IMDB dataset contains 50, 000 highly polarized reviews from the Internet Movie Database (IMDB). The data set
    was divided into 25 000 comments for training and 25 000 comments for testing, with both the training set and test
    set containing 50% positive and 50% negative comments. Train labels and test labels are all lists of 0 and 1, where
    0 stands for negative and 1 for positive.

    You can unzip the dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── imdb_dataset_directory
             ├── train
             │    ├── pos
             │    │    ├── 0_9.txt
             │    │    ├── 1_7.txt
             │    │    ├── ...
             │    ├── neg
             │    │    ├── 0_3.txt
             │    │    ├── 1_1.txt
             │    │    ├── ...
             ├── test
             │    ├── pos
             │    │    ├── 0_10.txt
             │    │    ├── 1_10.txt
             │    │    ├── ...
             │    ├── neg
             │    │    ├── 0_2.txt
             │    │    ├── 1_3.txt
             │    │    ├── ...

    Citation:

    .. code-block::

        @InProceedings{maas-EtAl:2011:ACL-HLT2011,
          author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan
                        and  Ng, Andrew Y.  and  Potts, Christopher},
          title     = {Learning Word Vectors for Sentiment Analysis},
          booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics:
                        Human Language Technologies},
          month     = {June},
          year      = {2011},
          address   = {Portland, Oregon, USA},
          publisher = {Association for Computational Linguistics},
          pages     = {142--150},
          url       = {http://www.aclweb.org/anthology/P11-1015}
        }
    """

    @check_imdb_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None, sampler=None,
                 num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.IMDBNode(self.dataset_dir, self.usage, self.sampler)


class IWSLT2016Dataset(SourceDataset, TextBaseDataset):
    """
    A source dataset that reads and parses IWSLT2016 datasets.

    The generated dataset has two columns: :py:obj:`[text, translation]`.
    The tensor of column :py:obj: `text` is of the string type.
    The tensor of column :py:obj: `translation` is of the string type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Acceptable usages include "train", "valid", "test" and "all" (default=None, all samples).
        language_pair (sequence, optional): Sequence containing source and target language, supported values are
            (`en`, `fr`), ("en", "de"), ("en", "cs"), ("en", "ar"), ("fr", "en"), ("de", "en"), ("cs", "en"),
            ("ar", "en") (default=("de", "en")).
        valid_set (str, optional): A string to identify validation set, when usage is valid or all, the validation set
            of valid_set type will be read, supported values are "dev2010", "tst2010", "tst2011", "tst2012", "tst2013"
            and "tst2014" (default="tst2013").
        test_set (str, optional): A string to identify test set, when usage is test or all, the test set of test_set
            type will be read, supported values are "dev2010", "tst2010", "tst2011", "tst2012", "tst2013" and "tst2014"
            (default="tst2014").
        num_samples (int, optional): Number of samples (rows) to read (default=None, reads the full dataset).
        shuffle (Union[bool, Shuffle level], optional): Perform reshuffling of the data every epoch
            (default=Shuffle.GLOBAL).
            If shuffle is False, no shuffling will be performed;
            If shuffle is True, the behavior is the same as setting shuffle to be Shuffle.GLOBAL
            Otherwise, there are two levels of shuffling:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.
        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, `num_samples` reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_dir does not contain data files.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.

    Examples:
        >>> iwslt2016_dataset_dir = "/path/to/iwslt2016_dataset_dir"
        >>> dataset = ds.IWSLT2016Dataset(dataset_files=iwslt2016_dataset_dir, usage='all',
        ...                               language_pair=('de', 'en'), valid_set='tst2013', test_set='tst2014')

    About IWSLT2016 dataset:

    IWSLT is an international oral translation conference, a major annual scientific conference dedicated to all aspects
    of oral translation. The MT task of the IWSLT evaluation activity constitutes a data set, which can be publicly
    obtained through the WIT3 website wit3.fbk.eu. The IWSLT2016 data set includes translations from English to Arabic,
    Czech, French, and German, and translations from Arabic, Czech, French, and German to English.

    You can unzip the original IWSLT2016 dataset files into this directory structure and read by MindSpore's API. After
    decompression, you also need to decompress the data set to be read in the specified folder. For example, if you want
    to read the data set of de-en, you need to unzip the tgz file in the de/en directory, the data set is in the
    unzipped folder.

    .. code-block::

        .
        └── iwslt2016_dataset_directory
             ├── subeval_files
             └── texts
                  ├── ar
                  │    └── en
                  │        └── ar-en
                  ├── cs
                  │    └── en
                  │        └── cs-en
                  ├── de
                  │    └── en
                  │        └── de-en
                  │            ├── IWSLT16.TED.dev2010.de-en.de.xml
                  │            ├── train.tags.de-en.de
                  │            ├── ...
                  ├── en
                  │    ├── ar
                  │    │   └── en-ar
                  │    ├── cs
                  │    │   └── en-cs
                  │    ├── de
                  │    │   └── en-de
                  │    └── fr
                  │        └── en-fr
                  └── fr
                       └── en
                           └── fr-en

    Citation:

    .. code-block::

        @inproceedings{cettoloEtAl:EAMT2012,
        Address = {Trento, Italy},
        Author = {Mauro Cettolo and Christian Girardi and Marcello Federico},
        Booktitle = {Proceedings of the 16$^{th}$ Conference of the European Association for Machine Translation
                     (EAMT)},
        Date = {28-30},
        Month = {May},
        Pages = {261--268},
        Title = {WIT$^3$: Web Inventory of Transcribed and Translated Talks},
        Year = {2012}}
    """

    @check_iwslt2016_dataset
    def __init__(self, dataset_dir, usage=None, language_pair=None, valid_set=None, test_set=None,
                 num_samples=None, shuffle=Shuffle.GLOBAL, num_shards=None, shard_id=None, num_parallel_workers=None,
                 cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, 'all')
        self.language_pair = replace_none(language_pair, ["de", "en"])
        self.valid_set = replace_none(valid_set, 'tst2013')
        self.test_set = replace_none(test_set, 'tst2014')

    def parse(self, children=None):
        return cde.IWSLT2016Node(self.dataset_dir, self.usage, self.language_pair, self.valid_set, self.test_set,
                                 self.num_samples, self.shuffle_flag, self.num_shards, self.shard_id)


class IWSLT2017Dataset(SourceDataset, TextBaseDataset):
    """
    A source dataset that reads and parses IWSLT2017 datasets.

    The generated dataset has two columns: :py:obj:`[text, translation]`.
    The tensor of column :py:obj:`text` is of the string type.
    The tensor of column :py:obj:`translation` is of the string type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Acceptable usages include "train", "valid", "test" and "all" (default=None, all samples).
        language_pair (list, optional): List containing src and tgt language, supported values are ("en", "nl"),
            ("en", "de"), ("en", "it"), ("en", "ro"), ("nl", "en"), ("nl", "de"), ("nl", "it"), ("nl", "ro"),
            ("de", "en"), ("de", "nl"), ("de", "it"), ("de", "ro"), ("it", "en"), ("it", "nl"), ("it", "de"),
            ("it", "ro"), (`ro`, `en`), (`ro`, `nl`), (`ro`, `de`), (`ro`, `it`) (default=(`de`, `en`)).
        num_samples (int, optional): Number of samples (rows) to read (default=None, reads the full dataset).
        shuffle (Union[bool, Shuffle level], optional): Perform reshuffling of the data every epoch
            (default=Shuffle.GLOBAL).
            If shuffle is False, no shuffling will be performed;
            If shuffle is True, the behavior is the same as setting shuffle to be Shuffle.GLOBAL
            Otherwise, there are two levels of shuffling:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.
        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, `num_samples` reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_dir does not contain data files.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.

    Examples:
        >>> iwslt2017_dataset_dir = "/path/to/iwslt207_dataset_dir"
        >>> dataset = ds.IWSLT2017Dataset(dataset_files=iwslt2017_dataset_dir, usage='all', language_pair=('de', 'en'))

    About IWSLT2017 dataset:

    IWSLT is an international oral translation conference, a major annual scientific conference dedicated to all aspects
    of oral translation. The MT task of the IWSLT evaluation activity constitutes a data set, which can be publicly
    obtained through the WIT3 website wit3.fbk.eu. The IWSLT2017 data set involves German, English, Italian, Dutch, and
    Romanian. The data set includes translations in any two different languages.

    You can unzip the original IWSLT2017 dataset files into this directory structure and read by MindSpore's API. You
    need to decompress the dataset package in texts/DeEnItNlRo/DeEnItNlRo directory to get the DeEnItNlRo-DeEnItNlRo
    subdirectory.

    .. code-block::

        .
        └── iwslt2017_dataset_directory
            └── DeEnItNlRo
                └── DeEnItNlRo
                    └── DeEnItNlRo-DeEnItNlRo
                        ├── IWSLT17.TED.dev2010.de-en.de.xml
                        ├── train.tags.de-en.de
                        ├── ...

    Citation:

    .. code-block::

        @inproceedings{cettoloEtAl:EAMT2012,
        Address = {Trento, Italy},
        Author = {Mauro Cettolo and Christian Girardi and Marcello Federico},
        Booktitle = {Proceedings of the 16$^{th}$ Conference of the European Association for Machine Translation
                     (EAMT)},
        Date = {28-30},
        Month = {May},
        Pages = {261--268},
        Title = {WIT$^3$: Web Inventory of Transcribed and Translated Talks},
        Year = {2012}}
    """

    @check_iwslt2017_dataset
    def __init__(self, dataset_dir, usage=None, language_pair=None, num_samples=None, shuffle=Shuffle.GLOBAL,
                 num_shards=None, shard_id=None, num_parallel_workers=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, 'all')
        self.language_pair = replace_none(language_pair, ["de", "en"])

    def parse(self, children=None):
        return cde.IWSLT2017Node(self.dataset_dir, self.usage, self.language_pair, self.num_samples,
                                 self.shuffle_flag, self.num_shards, self.shard_id)


class KMnistDataset(MappableDataset):
    """
    A source dataset for reading and parsing the KMNIST dataset.

    The generated dataset has two columns :py:obj:`[image, label]`.
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`label` is a scalar of the uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be `train`, `test` or `all` . `train` will read from 60,000
            train samples, `test` will read from 10,000 test samples, `all` will read from all 70,000 samples.
            (default=None, will read all samples)
        num_samples (int, optional): The number of images to be included in the dataset
            (default=None, will read all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, will use value set in the config).
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset
            (default=None, expected order behavior shown in the table).
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, `num_samples` reflects the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` (default=None). This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If `dataset_dir` does not contain data files.
        RuntimeError: If `num_parallel_workers` exceeds the max thread numbers.
        RuntimeError: If `sampler` and `shuffle` are specified at the same time.
        RuntimeError: If `sampler` and sharding are specified at the same time.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `shard_id` is invalid (out of range [0, `num_shards`]).

    Note:
        - This dataset can take in a `sampler`. `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> kmnist_dataset_dir = "/path/to/kmnist_dataset_directory"
        >>>
        >>> # Read 3 samples from KMNIST dataset
        >>> dataset = ds.KMnistDataset(dataset_dir=kmnist_dataset_dir, num_samples=3)
        >>>
        >>> # Note: In kmnist_dataset dataset, each dictionary has keys "image" and "label"

    About KMNIST dataset:

    KMNIST is a dataset, adapted from Kuzushiji Dataset, as a drop-in replacement for MNIST dataset,
    which is the most famous dataset in the machine learning community.

    Here is the original KMNIST dataset structure.
    You can unzip the dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── kmnist_dataset_dir
             ├── t10k-images-idx3-ubyte
             ├── t10k-labels-idx1-ubyte
             ├── train-images-idx3-ubyte
             └── train-labels-idx1-ubyte

    Citation:

    .. code-block::

        @online{clanuwat2018deep,
          author       = {Tarin Clanuwat and Mikel Bober-Irizar and Asanobu Kitamoto and
                           Alex Lamb and Kazuaki Yamamoto and David Ha},
          title        = {Deep Learning for Classical Japanese Literature},
          date         = {2018-12-03},
          year         = {2018},
          eprintclass  = {cs.CV},
          eprinttype   = {arXiv},
          eprint       = {cs.CV/1812.01718},
        }
    """

    @check_mnist_cifar_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None,
                 sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.KMnistNode(self.dataset_dir, self.usage, self.sampler)


class MnistDataset(MappableDataset):
    """
    A source dataset for reading and parsing the MNIST dataset.

    The generated dataset has two columns :py:obj:`[image, label]`.
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`label` is a scalar of the uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be `train`, `test` or `all` . `train` will read from 60,000
            train samples, `test` will read from 10,000 test samples, `all` will read from all 70,000 samples.
            (default=None, will read all samples)
        num_samples (int, optional): The number of images to be included in the dataset
            (default=None, will read all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, will use value set in the config).
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset
            (default=None, expected order behavior shown in the table).
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, `num_samples` reflects the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` (default=None). This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_dir does not contain data files.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Note:
        - This dataset can take in a `sampler`. `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> mnist_dataset_dir = "/path/to/mnist_dataset_directory"
        >>>
        >>> # Read 3 samples from MNIST dataset
        >>> dataset = ds.MnistDataset(dataset_dir=mnist_dataset_dir, num_samples=3)
        >>>
        >>> # Note: In mnist_dataset dataset, each dictionary has keys "image" and "label"

    About MNIST dataset:

    The MNIST database of handwritten digits has a training set of 60,000 examples,
    and a test set of 10,000 examples. It is a subset of a larger set available from
    NIST. The digits have been size-normalized and centered in a fixed-size image.

    Here is the original MNIST dataset structure.
    You can unzip the dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── mnist_dataset_dir
             ├── t10k-images-idx3-ubyte
             ├── t10k-labels-idx1-ubyte
             ├── train-images-idx3-ubyte
             └── train-labels-idx1-ubyte

    Citation:

    .. code-block::

        @article{lecun2010mnist,
        title        = {MNIST handwritten digit database},
        author       = {LeCun, Yann and Cortes, Corinna and Burges, CJ},
        journal      = {ATT Labs [Online]},
        volume       = {2},
        year         = {2010},
        howpublished = {http://yann.lecun.com/exdb/mnist}
        }
    """

    @check_mnist_cifar_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None,
                 sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.MnistNode(self.dataset_dir, self.usage, self.sampler)


class PennTreebankDataset(SourceDataset, TextBaseDataset):
    """
    A source dataset that reads and parses PennTreebank datasets.

    The generated dataset has one column :py:obj:`[text]`.
    The tensor of column :py:obj:`text` is of the string type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Acceptable usages include `train`, `test`, 'valid' and `all`.
            'train' will read from 42,068 train samples of string type,
            'test' will read from 3,370 test samples of string type,
            'valid' will read from 3,761 test samples of string type,
            'all' will read from all 49,199 samples of string type (default=None, all samples).
        num_samples (int, optional): Number of samples (rows) to read (default=None, reads the full dataset).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        shuffle (Union[bool, Shuffle level], optional): Perform reshuffling of the data every epoch
            (default=Shuffle.GLOBAL).
            If shuffle is False, no shuffling will be performed;
            If shuffle is True, the behavior is the same as setting shuffle to be Shuffle.GLOBAL
            Otherwise, there are two levels of shuffling:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, 'num_samples' reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Examples:
        >>> penn_treebank_dataset_dir = "path/to/penn_treebank_dataset_directory"
        >>> dataset = ds.PennTreebankDataset(dataset_dir=penn_treebank_dataset_dir, usage='all')

    About PennTreebank dataset:

    Penn Treebank (PTB) dataset, is widely used in machine learning for NLP (Natural Language Processing)
    research. Word-level PTB does not contain capital letters, numbers, and punctuations, and the vocabulary
    is capped at 10k unique words, which is relatively small in comparison to most modern datasets which
    can result in a larger number of out of vocabulary tokens.

    Here is the original PennTreebank dataset structure.
    You can unzip the dataset files into this directory structure and read by MindSpore's API.

    .. code-block::
        .
        └── PennTreebank_dataset_dir
             ├── ptb.test.txt
             ├── ptb.train.txt
             └── ptb.valid.txt

    Citation:

    .. code-block::

        @techreport{Santorini1990,
          added-at = {2014-03-26T23:25:56.000+0100},
          author = {Santorini, Beatrice},
          biburl = {https://www.bibsonomy.org/bibtex/234cdf6ddadd89376090e7dada2fc18ec/butonic},
          file = {:Santorini - Penn Treebank tag definitions.pdf:PDF},
          institution = {Department of Computer and Information Science, University of Pennsylvania},
          interhash = {818e72efd9e4b5fae3e51e88848100a0},
          intrahash = {34cdf6ddadd89376090e7dada2fc18ec},
          keywords = {dis pos tagging treebank},
          number = {MS-CIS-90-47},
          timestamp = {2014-03-26T23:25:56.000+0100},
          title = {Part-of-speech tagging guidelines for the {P}enn {T}reebank {P}roject},
          url = {ftp://ftp.cis.upenn.edu/pub/treebank/doc/tagguide.ps.gz},
          year = 1990
        }
    """

    @check_penn_treebank_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=Shuffle.GLOBAL,
                 num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.PennTreebankNode(self.dataset_dir, self.usage, self.num_samples, self.shuffle_flag, self.num_shards,
                                    self.shard_id)


class PhotoTourDataset(MappableDataset):
    """
    A source dataset for reading and parsing the PhotoTour dataset.

    The generated dataset with different usage has different output columns.
    If train, the generated dataset has one column :py:obj:`[image]`,
    else three columns :py:obj:`[image1, image2, matches]`.
    The tensor of column :py:obj:`image`, :py:obj:`image1` and :py:obj:`image2` is of the uint8 type.
    The tensor of column :py:obj:`matches` is a scalar of the uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        name (str): Name of the dataset to load,
            should be one of 'notredame', 'yosemite', 'liberty', 'notredame_harris',
            'yosemite_harris' or 'liberty_harris'.
        usage (str, optional): Usage of the dataset, can be `train` or `test` (Default=None, will be set to 'train').
            When usage is `train`, number of samples for each `name` is
            {'notredame': 468159, 'yosemite': 633587, 'liberty': 450092, 'liberty_harris': 379587,
             'yosemite_harris': 450912, 'notredame_harris': 325295}.
            When usage is `test`, will read 100,000 samples for testing.
        num_samples (int, optional): The number of images to be included in the dataset
            (default=None, will read all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, will use value set in the config).
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset
            (default=None, expected order behavior shown in the table).
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, `num_samples` reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` (default=None). This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_dir does not contain data files.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If dataset_dir is not exist.
        ValueError: If usage is not in ["train", "test"].
        ValueError: If name is not in ["notredame", "yosemite", "liberty",
            "notredame_harris", "yosemite_harris", "liberty_harris"].
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Note:
        - This dataset can take in a sampler. `sampler` and `shuffle` are mutually exclusive. The table
          below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using 'sampler' and 'shuffle'
       :widths: 64 64 1
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> # Read 3 samples from PhotoTour dataset.
        >>> dataset = ds.PhotoTourDataset(dataset_dir="/path/to/photo_tour_dataset_directory",
        ...                               name='liberty', usage='train', num_samples=3)
        >>>
        >>> # In PhotoTourDataset dataset, if usage is 'train', each dictionary has key "image",
        >>> # else has keys "image1" "image2" and "matches".

    About PhotoTour dataset:

    The data is taken from Photo Tourism reconstructions from Trevi Fountain (Rome), Notre Dame (Paris) and Half
    Dome (Yosemite). Each dataset consists of a series of corresponding patches, which are obtained by projecting
    3D points from Photo Tourism reconstructions back into the original images.

    The dataset consists of 1024 x 1024 bitmap (.bmp) images, each containing a 16 x 16 array of image patches.
    Each patch is sampled as 64 x 64 grayscale, with a canonical scale and orientation. For details of how the scale
    and orientation is established, please see the paper. An associated metadata file info.txt contains the match
    information. Each row of info.txt corresponds to a separate patch, with the patches ordered from left to right and
    top to bottom in each bitmap image. The first number on each row of info.txt is the 3D point ID from which that
    patch was sampled -- patches with the same 3D point ID are projected from the same 3D point (into different images).
    The second number in info.txt corresponds to the image from which the patch was sampled, and is not used at present.

    You can unzip the original PhotoTour dataset files into this directory structure and read by MindSpore's API.

    .. code-block::
        .
        └── photo_tour_dataset_directory
            ├── liberty/
            │    ├── info.txt                 // two columns: 3D_point_ID, unused
            │    ├── m50_100000_100000_0.txt  // seven columns: patch_ID1, 3D_point_ID1, unused1,
            │    │                            // patch_ID2, 3D_point_ID2, unused2, unused3
            │    ├── patches0000.bmp          // 1024*1024 pixels, with 16 * 16 patches.
            │    ├── patches0001.bmp
            │    ├── ...
            ├── yosemite/
            │    ├── ...
            ├── notredame/
            │    ├── ...
            ├── liberty_harris/
            │    ├── ...
            ├── yosemite_harris/
            │    ├── ...
            ├── notredame_harris/
            │    ├── ...

    Citation:

    .. code-block::

        @INPROCEEDINGS{4269996,
            author={Winder, Simon A. J. and Brown, Matthew},
            booktitle={2007 IEEE Conference on Computer Vision and Pattern Recognition},
            title={Learning Local Image Descriptors},
            year={2007},
            volume={},
            number={},
            pages={1-8},
            doi={10.1109/CVPR.2007.382971}
        }
    """

    @check_photo_tour_dataset
    def __init__(self, dataset_dir, name, usage=None, num_samples=None, num_parallel_workers=None,
                 shuffle=None, sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.name = name
        self.usage = replace_none(usage, "train")

    def parse(self, children=None):
        return cde.PhotoTourNode(self.dataset_dir, self.name, self.usage, self.sampler)


class Places365Dataset(MappableDataset):
    """
    A source dataset for reading and parsing the Places365 dataset.

    The generated dataset has two columns :py:obj:`[image, label]`.
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`label` is a scalar of the uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be `train-standard`, `train-challenge` or `val`
            (default=None, will be set to 'train-standard').
        small (bool, optional): Use 256 * 256 images (True) or high resolution images (False) (default=False).
        decode (bool, optional): Decode the images after reading (default=True).
        num_samples (int, optional): The number of images to be included in the dataset
            (default=None, will read all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, will use value set in the config).
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset
            (default=None, expected order behavior shown in the table).
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, `num_samples` reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` (default=None). This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_dir does not contain data files.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).
        ValueError: If usage is not in ["train-standard", "train-challenge", "val"].
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Note:
        - This dataset can take in a sampler. 'sampler' and 'shuffle' are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using 'sampler' and 'shuffle'
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> place365_dataset_dir = "/path/to/place365_dataset_directory"
        >>>
        >>> # Read 3 samples from Places365 dataset
        >>> dataset = ds.Places365Dataset(dataset_dir=place365_dataset_dir, usage='train-standard',
        ...                               small=True, decode=True, num_samples=3)
        >>>
        >>> # In places365 dataset, each dictionary has keys "image" and "label".

    About Places365 dataset:

    Convolutional neural networks (CNNs) trained on the Places2 Database can be used for scene recognition as well as
    generic deep scene features for visual recognition.

    The author releases the data of Places365-Standard and the data of Places365-Challenge to the public.
    Places365-Standard is the core set of Places2 Database, which has been used to train the Places365-CNNs. The author
    will add other kinds of annotation on the Places365-Standard in the future. Places365-Challenge is the competition
    set of Places2 Database, which has 6.2 million extra images compared to the Places365-Standard.
    The Places365-Challenge will be used for the Places Challenge 2016.

    You can unzip the original Places365 dataset files into this directory structure and read by MindSpore's API.

    .. code-block::
        .
        └─├── categories_places365.txt
            ├── places365_train-standard.txt
            ├── places365_train-challenge.txt
            ├── val_large/
            │    ├── Places365_val_00000001.jpg
            │    ├── Places365_val_00000002.jpg
            │    ├── Places365_val_00000003.jpg
            │    ├── ...
            ├── val_256/
            │    ├── ...
            ├── data_large_standard/
            │    ├── ...
            ├── data_256_standard/
            │    ├── ...
            ├── data_large_challenge/
            │    ├── ...
            ├── data_256_challenge /
            │    ├── ...

    Citation:

    .. code-block::

        article{zhou2017places,
            title={Places: A 10 million Image Database for Scene Recognition},
            author={Zhou, Bolei and Lapedriza, Agata and Khosla, Aditya and Oliva, Aude and Torralba, Antonio},
            journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
            year={2017},
            publisher={IEEE}
        }
    """

    @check_places365_dataset
    def __init__(self, dataset_dir, usage=None, small=True, decode=False, num_samples=None, num_parallel_workers=None,
                 shuffle=None, sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = os.path.abspath(dataset_dir)
        self.usage = replace_none(usage, "train-standard")
        self.small = small
        self.decode = decode

    def parse(self, children=None):
        return cde.Places365Node(self.dataset_dir, self.usage, self.small, self.decode, self.sampler)


class QMnistDataset(MappableDataset):
    """
    A source dataset for reading and parsing the QMNIST dataset.

    The generated dataset has two columns :py:obj:`[image, label]`.
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`label` is a scalar when `compat` is True else a tensor both of the uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be `train`, `test`, `test10k`, `test50k`, `nist`
            or `all` (default=None, will read all samples).
        compat (bool, optional): Whether the label for each example is class number (compat=True) or the full QMNIST
            information (compat=False) (default=True).
        num_samples (int, optional): The number of images to be included in the dataset
            (default=None, will read all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, will use value set in the config).
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset
            (default=None, expected order behavior shown in the table).
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, `num_samples` reflects the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` (default=None). This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_dir does not contain data files.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Note:
        - This dataset can take in a `sampler`. `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> qmnist_dataset_dir = "/path/to/qmnist_dataset_directory"
        >>>
        >>> # Read 3 samples from QMNIST train dataset
        >>> dataset = ds.QMnistDataset(dataset_dir=qmnist_dataset_dir, num_samples=3)
        >>>
        >>> # Note: In QMNIST dataset, each dictionary has keys "image" and "label"

    About QMNIST dataset:

    The QMNIST dataset was generated from the original data found in the NIST Special Database 19 with the goal to
    match the MNIST preprocessing as closely as possible.
    Through an iterative process, researchers tried to generate an additional 50k images of MNIST-like data.
    They started with a reconstruction process given in the paper and used the Hungarian algorithm to find the best
    matches between the original MNIST samples and their reconstructed samples.

    Here is the original QMNIST dataset structure.
    You can unzip the dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── qmnist_dataset_dir
             ├── qmnist-train-images-idx3-ubyte
             ├── qmnist-train-labels-idx2-int
             ├── qmnist-test-images-idx3-ubyte
             ├── qmnist-test-labels-idx2-int
             ├── xnist-images-idx3-ubyte
             └── xnist-labels-idx2-int

    Citation:

    .. code-block::

        @incollection{qmnist-2019,
           title = "Cold Case: The Lost MNIST Digits",
           author = "Chhavi Yadav and L\'{e}on Bottou",\
           booktitle = {Advances in Neural Information Processing Systems 32},
           year = {2019},
           publisher = {Curran Associates, Inc.},
        }
    """

    @check_qmnist_dataset
    def __init__(self, dataset_dir, usage=None, compat=True, num_samples=None, num_parallel_workers=None,
                 shuffle=None, sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")
        self.compat = compat

    def parse(self, children=None):
        return cde.QMnistNode(self.dataset_dir, self.usage, self.compat, self.sampler)


class MindDataset(MappableDataset, TextBaseDataset):
    """
    A source dataset for reading and parsing MindRecord dataset.

    The columns of generated dataset depend on the source MindRecord files.

    Args:
        dataset_files (Union[str, list[str]]): If dataset_file is a str, it represents for
            a file name of one component of a mindrecord source, other files with identical source
            in the same path will be found and loaded automatically. If dataset_file is a list,
            it represents for a list of dataset files to be read directly.
        columns_list (list[str], optional): List of columns to be read (default=None).
        num_parallel_workers (int, optional): The number of readers (default=None).
        shuffle (Union[bool, Shuffle level], optional): Perform reshuffling of the data every epoch
            (default=None, performs global shuffle).
            If shuffle is False, no shuffling will be performed;
            If shuffle is True, the behavior is the same as setting shuffle to be Shuffle.GLOBAL
            Otherwise, there are three levels of shuffling:

            - Shuffle.GLOBAL: Global shuffle of all rows of data in dataset.

            - Shuffle.FILES: Shuffle the file sequence but keep the order of data within each file.

            - Shuffle.INFILE: Keep the file sequence the same but shuffle the data within each file.

        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, 'num_samples' reflects the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, sampler is exclusive
            with shuffle and block_reader). Support list: SubsetRandomSampler,
            PkSampler, RandomSampler, SequentialSampler, DistributedSampler.
        padded_sample (dict, optional): Samples will be appended to dataset, where
            keys are the same as column_list.
        num_padded (int, optional): Number of padding samples. Dataset size
            plus num_padded should be divisible by num_shards.
        num_samples (int, optional): The number of samples to be included in the dataset
            (default=None, all samples).
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_files are not valid or do not exist.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Note:
        - This dataset can take in a `sampler`. `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> mind_dataset_dir = ["/path/to/mind_dataset_file"] # contains 1 or multiple MindRecord files
        >>> dataset = ds.MindDataset(dataset_files=mind_dataset_dir)
    """

    def parse(self, children=None):
        return cde.MindDataNode(self.dataset_files, self.columns_list, self.sampler, self.new_padded_sample,
                                self.num_padded, shuffle_to_shuffle_mode(self.shuffle_option))

    @check_minddataset
    def __init__(self, dataset_files, columns_list=None, num_parallel_workers=None, shuffle=None, num_shards=None,
                 shard_id=None, sampler=None, padded_sample=None, num_padded=None, num_samples=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle_to_bool(shuffle), num_shards=num_shards, shard_id=shard_id, cache=cache)
        if shuffle is not None and not isinstance(shuffle, (bool, Shuffle)):
            raise TypeError("shuffle must be of boolean or enum of 'Shuffle' values like 'Shuffle.GLOBAL' or "
                            "'Shuffle.FILES' or 'Shuffle.INFILE'.")
        if num_samples and shuffle in (Shuffle.FILES, Shuffle.INFILE):
            raise ValueError("'Shuffle.FILES' or 'Shuffle.INFILE' and 'num_samples' "
                             "cannot be specified at the same time.")
        self.shuffle_option = shuffle
        if isinstance(dataset_files, list):
            self.load_dataset = False
        else:
            self.load_dataset = True
        self.dataset_files = dataset_files
        self.columns_list = replace_none(columns_list, [])

        if shuffle is False:
            logger.warning("WARN: global shuffle is not used.")

        if sampler is not None:
            if isinstance(sampler, (
                    samplers.SubsetRandomSampler, samplers.SubsetSampler, samplers.PKSampler,
                    samplers.DistributedSampler,
                    samplers.RandomSampler, samplers.SequentialSampler)) is False:
                raise ValueError("The sampler is not supported yet.")

        self.padded_sample = padded_sample
        self.num_padded = replace_none(num_padded, 0)

        self.new_padded_sample = {}
        if padded_sample:
            for k, v in padded_sample.items():
                if isinstance(v, np.ndarray):
                    self.new_padded_sample[k] = v.tobytes()
                else:
                    self.new_padded_sample[k] = v


def _iter_fn(dataset, num_samples):
    """
    Generator function wrapper for iterable dataset.
    """
    if num_samples is not None and num_samples != 0:
        ds_iter = iter(dataset)
        for _ in range(num_samples):
            try:
                val = next(ds_iter)
            except StopIteration:
                return
            # convert output tensors to ndarrays
            yield _convert_row(val)
    else:
        for val in dataset:
            # convert output tensors to ndarrays
            yield _convert_row(val)


def _generator_fn(generator, num_samples):
    """
    Generator function wrapper for generator function dataset.
    """
    if num_samples is not None and num_samples != 0:
        gen_iter = generator()
        for _ in range(num_samples):
            try:
                val = next(gen_iter)
            except StopIteration:
                return
            yield val
    else:
        gen_iter = generator()
        for val in gen_iter:
            yield val


def _cpp_sampler_fn(sample_ids, dataset):
    """
    Generator function wrapper for mappable dataset with cpp sampler.
    """
    if not isinstance(sample_ids, np.ndarray):
        raise RuntimeError("Sample IDs are not in a numpy array.")
    if sample_ids.size == 0:
        raise RuntimeError("Sampler passed an empty sample IDs list.")

    for i in sample_ids:
        val = dataset[i]
        # convert output tensors to ndarrays
        yield _convert_row(val)


def _cpp_sampler_fn_mp(sample_ids, sample_fn):
    """
    Multiprocessing generator function wrapper for mappable dataset with cpp sampler.
    """
    if not isinstance(sample_ids, np.ndarray):
        raise RuntimeError("Sample IDs are not in a numpy array.")
    if sample_ids.size == 0:
        raise RuntimeError("Sampler passed an empty sample IDs list.")

    return sample_fn.process(sample_ids)


def _fill_worker_indices(workers, indices, idx):
    """
    Worker index queue filler, fill worker index queue in round robin order.
    """
    num_worker = len(workers)
    while idx < len(indices):
        try:
            workers[idx % num_worker].put(indices[idx])
            idx += 1
        except queue.Full:
            break
    return idx


def _check_shm_usage(num_worker, queue_size, max_rowsize, num_queues=1):
    """
    Check sufficient shared memory is available for shared memory queues
    when training in parallel mode.
    """
    threshold_ratio = 0.8
    if platform.system().lower() not in {"windows", "darwin"}:
        device_num = _get_device_num()
        # In the cluster, _get_device_num indicates the number of the entire cluster. The maximum number of cards
        # on the ascend server is 8.
        if device_num > 1 and context.get_context("device_target") == "Ascend":
            device_num = min(device_num, 8)
        shm_estimate_usage = device_num * num_worker * num_queues * \
                             (queue_size + 2) * max_rowsize * 1024 * 1024
        try:
            shm_available = psutil.disk_usage('/dev/shm').free
            if shm_estimate_usage >= threshold_ratio * shm_available:
                raise RuntimeError(
                    "Insufficient shared memory available. Required: {}, Available: {}. "
                    "The required memory can't exceed 80% of the available shared memory. "
                    "Recommend to set_enable_shared_mem to False, reduce max_rowsize or reduce num_parallel_workers."
                    .format(shm_estimate_usage, shm_available))
        except FileNotFoundError:
            raise RuntimeError("Expected /dev/shm to exist.")


def _convert_row(row):
    """
    Convert Op return value to numpy
    """
    value = []
    if isinstance(row, dict):
        raise ValueError("Return value in user defined python function should be numpy array, but got dict.")

    # convert each column in row into numpy array
    for x in row:
        if isinstance(x, bytes):         # got image bytes from a file
            value.append(np.frombuffer(x, np.uint8))
        elif isinstance(x, Tensor):      # got mindspore.Tensor
            value.append(x.asnumpy())
        elif isinstance(x, dict):
            raise ValueError("Return value in user defined python function should be numpy array, but got dict.")
        else:
            value.append(np.array(x, copy=False))
    return tuple(value)


class SamplerFn:
    """
    Multiprocessing or multithread generator function wrapper master process.
    """

    def __init__(self, dataset, num_worker, multi_process, max_rowsize):
        self.workers = []
        self.num_worker = num_worker
        self.multi_process = multi_process
        self.need_join = False
        self.ppid = os.getpid()
        self.pids = []
        self.check_interval = 300  # the interval of check queue's size
        self._final_join = True

        # Event for end of epoch
        if multi_process is True:
            try:
                self.eof = multiprocessing.Event()
            except Exception:
                raise RuntimeError("Init multiprocessing.Event() failed, This might be caused by insufficient shm,"
                                   + " and the recommended shm size is at least 5 GB.")
        else:
            self.eof = threading.Event()
        # Create workers

        # get default queue size and adjust queuesize per worker if there are large # workers
        queue_size = get_prefetch_size()
        queue_size = min(queue_size, queue_size * 4 // num_worker)
        queue_size = max(2, queue_size)

        if multi_process and get_enable_shared_mem():
            _check_shm_usage(num_worker, queue_size, max_rowsize)
        for _ in range(num_worker):
            if multi_process is True:
                try:
                    worker = _GeneratorWorkerMp(dataset, self.eof, max_rowsize, queue_size)
                except Exception:
                    raise RuntimeError("Init multiprocessing.Queue() failed, This might be caused by insufficient shm,"
                                       + " and the recommended shm size is at least 5 GB.")
                worker.daemon = True
                # When multi processes fork a subprocess, the lock of the main process is copied to the subprocess,
                # which may cause deadlock. Therefore, the subprocess startup is performed in che initialization phase.
                # In this phase, the main process is not locked.
                worker.start()
                self.pids.append(worker.pid)
                self.need_join = True
            else:
                worker = _GeneratorWorkerMt(dataset, self.eof)
                worker.daemon = True
            self.workers.append(worker)
        if multi_process is True and platform.system().lower() != 'windows':
            self.eot = threading.Event()
            self.watch_dog = threading.Thread(target=_watch_dog, args=(self.eot, self.workers))
            self.watch_dog.daemon = True
            self.watch_dog.start()

            if self._final_join is True:
                self._jointhread = Finalize(
                    self.watch_dog, self._finalize_join,
                    args=(weakref.ref(self.watch_dog), self.eot),
                    exitpriority=-5
                )

    def process(self, indices):
        """
        The main process, start the child process or child thread, and fill the index queue.
        Get the result and return.
        """
        for w in self.workers:
            # Check whether the queue of the subprocess is empty.
            if not w.queue_empty():
                raise Exception("The queue of the subprocess is not empty.")
            # Start all workers
            if not w.is_alive():
                w.start()

        # Fill initial index queues
        idx_cursor = 0
        idx_cursor = _fill_worker_indices(self.workers, indices, idx_cursor)

        # Fetch results
        for i in range(len(indices)):
            if self.eof.is_set():
                self._stop_subprocess()
                return
            if self.multi_process is True and not psutil.pid_exists(self.workers[i % self.num_worker].pid):
                self._stop_subprocess()
                return
            # Fetch result and put index
            try:
                # To avoid get timeout from queue, check the res_queue size.
                start_time = int(time.time())
                wait_count = 1
                while self.workers[i % self.num_worker].res_queue.empty():
                    time.sleep(0.1)
                    cost_time = int(time.time()) - start_time
                    if cost_time / self.check_interval >= wait_count:
                        wait_count += 1
                        logger.warning("It has been waiting for " + str(cost_time) + "s because the multi "
                                       "thread/process of the generator generates data had been hung by gil lock.")

                result = self.workers[i % self.num_worker].get()
                if isinstance(result, ExceptionHandler):
                    result.reraise()
            except queue.Empty:
                self._stop_subprocess()
                raise Exception("Generator worker process timeout.")
            except KeyboardInterrupt:
                self._stop_subprocess()
                raise Exception("Generator worker receives KeyboardInterrupt.")
            if self.eof.is_set():
                self._stop_subprocess()
                return
            if idx_cursor < len(indices):
                idx_cursor = _fill_worker_indices(self.workers, indices, idx_cursor)
            yield _convert_row(result)

    def _stop_subprocess(self):
        """Only the main process can call join."""
        if self.need_join is True and self.ppid == os.getpid():
            self.eof.set()
            self.need_join = False
            for w in self.workers:
                if self.multi_process is True and hasattr(w, '_closed') and w._closed is False:  # pylint: disable=W0212
                    w.join()
            self._abort_watchdog()

    def _abort_watchdog(self):
        if hasattr(self, 'eot') and self.eot is not None and not self.eot.is_set():
            self.eot.set()

    @classmethod
    def _finalize_join(cls, twr, eot):
        thread = twr()
        if thread is not None:
            if eot is not None and not eot.is_set():
                eot.set()
            thread.join()

    def __del__(self):
        self._stop_subprocess()


def _subprocess_handle(eof, signum, frame):
    threading.Thread(target=eof.set()).start()


def _generator_worker_loop(dataset, idx_queue, result_queue, eof, is_multiprocessing):
    """
    Multithread or multiprocess generator worker process loop.
    """
    if is_multiprocessing:
        signal.signal(signal.SIGTERM, partial(_subprocess_handle, eof))
    while True:
        # Fetch index, block
        try:
            idx = idx_queue.get(timeout=1)
        except KeyboardInterrupt:
            if is_multiprocessing:
                eof.set()
                idx_queue.cancel_join_thread()
                result_queue.cancel_join_thread()
            raise Exception("Generator worker receives KeyboardInterrupt.")
        except queue.Empty:
            if eof.is_set():
                if is_multiprocessing:
                    idx_queue.cancel_join_thread()
                    result_queue.cancel_join_thread()
                return
            # If end-of-file (eof) is not set, continue to get data from idx_queue
            continue
        if idx is None:
            # When the queue is out of scope from master process, a None item can be fetched from the queue.
            # Upon receiving None, worker process should check if eof is set.
            if not eof.is_set():
                raise Exception("")
            return
        if eof.is_set():
            if is_multiprocessing:
                idx_queue.cancel_join_thread()
                result_queue.cancel_join_thread()
            return
        # Fetch data, any exception from __getitem__ will terminate worker and timeout master process
        try:
            result = dataset[idx]
        except Exception:
            result = ExceptionHandler(where="in GeneratorDataset worker process")
        # Send data, block
        while True:
            try:
                result_queue.put(result, timeout=5)
            except KeyboardInterrupt:
                if is_multiprocessing:
                    eof.set()
                    idx_queue.cancel_join_thread()
                    result_queue.cancel_join_thread()
                raise Exception("Generator worker receives KeyboardInterrupt.")
            except queue.Full:
                if eof.is_set():
                    if is_multiprocessing:
                        idx_queue.cancel_join_thread()
                        result_queue.cancel_join_thread()
                    return
                # If eof is not set, continue to put data to result_queue
                continue
            break
        del result, idx


class _GeneratorWorkerMt(threading.Thread):
    """
    Worker process for multi-thread Generator.
    """

    def __init__(self, dataset, eof):
        self.idx_queue = queue.Queue(16)
        self.res_queue = queue.Queue(16)
        super().__init__(target=_generator_worker_loop, args=(dataset, self.idx_queue, self.res_queue, eof, False))

    def put(self, item):
        """
        Put function for worker index queue. Never block. Raise queue.Full on failure.
        """
        self.idx_queue.put_nowait(item)

    def get(self):
        """
        Get function for worker result queue. Block with timeout.
        """
        return self.res_queue.get(timeout=30)

    def queue_empty(self):
        if not self.idx_queue.empty():
            logger.warning("idx_queue is not empty")
            return False
        if not self.res_queue.empty():
            logger.warning("res_queue is not empty")
            return False
        return True


class _GeneratorWorkerMp(multiprocessing.Process):
    """
    Worker process for multiprocess Generator.
    """

    def __init__(self, dataset, eof, max_rowsize, queue_size):
        self.idx_queue = multiprocessing.Queue(queue_size)
        if get_enable_shared_mem():
            self.res_queue = _SharedQueue(queue_size, max_rowsize=max_rowsize)
        else:
            self.res_queue = multiprocessing.Queue(queue_size)
        self.idx_queue._joincancelled = True  # pylint: disable=W0212
        self.res_queue._joincancelled = True  # pylint: disable=W0212
        super().__init__(target=_generator_worker_loop, args=(dataset, self.idx_queue, self.res_queue, eof, True))

    def put(self, item):
        """
        Put function for worker index queue. Never block. Raise queue.Full on failure.
        """
        self.idx_queue.put_nowait(item)

    def get(self):
        """
        Get function for worker result queue. Block with timeout.
        """
        # Relax 10s to 30s, since it sometimes will cause "Generator worker process timeout"
        # when we run too many iterators with infinite epoch(num_epoch=-1)
        return self.res_queue.get(timeout=30)

    def queue_empty(self):
        if not self.idx_queue.empty():
            logger.warning("idx_queue is not empty.")
            return False
        if not self.res_queue.empty():
            logger.warning("res_queue is not empty.")
            return False
        return True


class GeneratorDataset(MappableDataset, TextBaseDataset):
    """
    A source dataset that generates data from Python by invoking Python data source each epoch.

    The column names and column types of generated dataset depend on Python data defined by users.

    Args:
        source (Union[Callable, Iterable, Random Accessible]):
            A generator callable object, an iterable Python object or a random accessible Python object.
            Callable source is required to return a tuple of NumPy arrays as a row of the dataset on source().next().
            Iterable source is required to return a tuple of NumPy arrays as a row of the dataset on
            iter(source).next().
            Random accessible source is required to return a tuple of NumPy arrays as a row of the dataset on
            source[idx].
        column_names (Union[str, list[str]], optional): List of column names of the dataset (default=None). Users are
            required to provide either column_names or schema.
        column_types (list[mindspore.dtype], optional): List of column data types of the dataset (default=None).
            If provided, sanity check will be performed on generator output.
        schema (Union[Schema, str], optional): Path to the JSON schema file or schema object (default=None). Users are
            required to provide either column_names or schema. If both are provided, schema will be used.
        num_samples (int, optional): The number of samples to be included in the dataset
            (default=None, all images).
        num_parallel_workers (int, optional): Number of subprocesses used to fetch the dataset in parallel (default=1).
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Random accessible input is required.
            (default=None, expected order behavior shown in the table).
        sampler (Union[Sampler, Iterable], optional): Object used to choose samples from the dataset. Random accessible
            input is required (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            Random accessible input is required. When this argument is specified, `num_samples` reflects the maximum
            sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This argument must be specified only
            when num_shards is also specified. Random accessible input is required.
        python_multiprocessing (bool, optional): Parallelize Python operations with multiple worker process. This
            option could be beneficial if the Python operation is computational heavy (default=True).
        max_rowsize(int, optional): Maximum size of row in MB that is used for shared memory allocation to copy
            data between processes.  This is only used if python_multiprocessing is set to True (default 6 MB).

    Raises:
        RuntimeError: If source raises an exception during execution.
        RuntimeError: If len of column_names does not match output len of source.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Note:
        - Input `source` accept user defined Python function(PyFuncs), Do not add network computing operators from
          mindspore.nn and mindspore.ops or others into this `source`.
        - This dataset can take in a `sampler`. `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> import numpy as np
        >>>
        >>> # 1) Multidimensional generator function as callable input.
        >>> def generator_multidimensional():
        ...     for i in range(64):
        ...         yield (np.array([[i, i + 1], [i + 2, i + 3]]),)
        >>>
        >>> dataset = ds.GeneratorDataset(source=generator_multidimensional, column_names=["multi_dimensional_data"])
        >>>
        >>> # 2) Multi-column generator function as callable input.
        >>> def generator_multi_column():
        ...     for i in range(64):
        ...         yield np.array([i]), np.array([[i, i + 1], [i + 2, i + 3]])
        >>>
        >>> dataset = ds.GeneratorDataset(source=generator_multi_column, column_names=["col1", "col2"])
        >>>
        >>> # 3) Iterable dataset as iterable input.
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
        >>> # 4) Random accessible dataset as random accessible input.
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
        >>> # list, dict, tuple of Python is also random accessible
        >>> dataset = ds.GeneratorDataset(source=[(np.array(0),), (np.array(1),), (np.array(2),)], column_names=["col"])
    """

    @check_generatordataset
    def __init__(self, source, column_names=None, column_types=None, schema=None, num_samples=None,
                 num_parallel_workers=1, shuffle=None, sampler=None, num_shards=None, shard_id=None,
                 python_multiprocessing=True, max_rowsize=6):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id)
        if isinstance(source, builtins.zip):
            # Although zip is iteratable, it does not have the feature of repeated iteration, so pass it to the array.
            self.source = [item for item in source]
        else:
            self.source = source
        self.prepared_source = None  # source to be sent to C++
        if hasattr(self, 'operator_mixed') and getattr(self, 'operator_mixed') is True:
            self.num_parallel_workers = 1
            logger.warning(
                "Input 'source' of 'GeneratorDataset' includes network computing operators like in mindspore.nn, "
                "mindspore.ops, mindspore.numpy module and etc, which do not support multi-thread compiling, recommend"
                " to replace it with python implemented operator like numpy etc. Here decrease 'num_parallel_workers' "
                "into 1.")

        self.python_multiprocessing = python_multiprocessing

        self.column_names = to_list(column_names)

        if column_types is not None:
            self.column_types = mstypelist_to_detypelist(column_types)
        else:
            self.column_types = []

        self.schema = schema
        if schema is not None:
            self.schema = schema
            if not isinstance(schema, Schema):
                self.schema = Schema(schema)
        # Move get dataset_size by len from parse to here, because self.source will
        # lose attribution of '__len__' after deepcopy.
        self.source_len = -1  # unknown
        if hasattr(self.source, "__len__"):
            self.source_len = len(self.source)

        self.max_rowsize = max_rowsize
        self.sample_fn = None

    def __deepcopy__(self, memodict):
        if id(self) in memodict:
            return memodict[id(self)]
        new_op = self.__safe_deepcopy__(memodict, exclude=("source", "__transfer_dataset__"))

        sample_fn = None
        if new_op.sampler is not None and hasattr(self.source, "__getitem__"):
            # The reason why there is a try catch here is because when the new op is being constructed with shared
            # memory enabled, there will be an exception thrown if there is not enough shared memory available
            if self.source_len == -1:
                raise RuntimeError("Attempt to construct a random access dataset, '__len__' method is required!")
            try:
                if new_op.num_parallel_workers > 1:
                    self.__validate_memory_usage()

                    sample_fn = SamplerFn(self.source, new_op.num_parallel_workers, self.python_multiprocessing,
                                          self.max_rowsize)
                    new_op.prepared_source = (lambda sample_ids: _cpp_sampler_fn_mp(sample_ids, sample_fn))
                else:
                    new_op.prepared_source = (lambda sample_ids: _cpp_sampler_fn(sample_ids, self.source))
                new_op.sample_fn = sample_fn
            except RuntimeError as e:
                raise Exception(str(e))
        else:
            try:
                new_op.sampler = None
                new_op.sample_fn = sample_fn
                new_op.source_len = min(new_op.source_len,
                                        new_op.num_samples) if new_op.num_samples != 0 else new_op.source_len
                iter(self.source)
            except TypeError:
                # Use generator function if input callable
                new_op.prepared_source = (lambda: _generator_fn(self.source, new_op.num_samples))
            else:
                # Use iterator function if input is iterable
                # Random accessible input is also iterable
                new_op.prepared_source = (lambda: _iter_fn(self.source, new_op.num_samples))

        return new_op

    def is_shuffled(self):
        return self.sampler.is_shuffled()

    def is_sharded(self):
        return self.sampler.is_sharded()

    def parse(self, children=None):
        if self.schema is None:
            return cde.GeneratorNode(self.prepared_source, self.column_names, self.column_types, self.source_len,
                                     self.sampler, self.num_parallel_workers)
        schema = self.schema
        if isinstance(schema, Schema):
            schema = self.schema.cpp_schema
        return cde.GeneratorNode(self.prepared_source, schema, self.source_len, self.sampler,
                                 self.num_parallel_workers)

    def __validate_memory_usage(self):
        """
        Check memory usage when mulit-processing mode, when 85% prompt warning and 100% raise error.
        """
        if self.python_multiprocessing:
            # if use num_parallel_workers is to large when python_multiprocessing=True which would cause
            # OOM error get the num_shards
            valid_num_shards = 1
            if isinstance(self.sampler, samplers.DistributedSampler):
                valid_num_shards = self.sampler.num_shards
            elif self.num_shards is not None:
                valid_num_shards = self.num_shards

            # get process memory usage
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info().rss
            sys_memory_free = psutil.virtual_memory().free

            total_memory_maybe_used = process_memory * self.num_parallel_workers * valid_num_shards
            if total_memory_maybe_used / sys_memory_free > 0.85:
                valid_num_worker = math.floor(sys_memory_free * 0.85 / valid_num_shards / process_memory)
                valid_num_worker = 1 if valid_num_worker <= 0 else valid_num_worker
                info = "GeneratorDataset num_parallel_workers: " + str(self.num_parallel_workers) + \
                       " is too large which maybe cause a lot of memory occupation (>85%) or out of memory(OOM) " \
                       "during multi process running. Therefore, it is recommended to reduce num_parallel_workers to " \
                       + str(valid_num_worker) + " or smaller."
                logger.warning(info)


class TFRecordDataset(SourceDataset, TextBaseDataset):
    """
    A source dataset for reading and parsing datasets stored on disk in TFData format.

    The columns of generated dataset depend on the source TFRecord files.

    Args:
        dataset_files (Union[str, list[str]]): String or list of files to be read or glob strings to search for a
            pattern of files. The list will be sorted in a lexicographical order.
        schema (Union[str, Schema], optional): Path to the JSON schema file or schema object (default=None).
            If the schema is not provided, the meta data from the TFData file is considered the schema.
        columns_list (list[str], optional): List of columns to be read (default=None, read all columns).
        num_samples (int, optional): The number of samples (rows) to be included in the dataset (default=None).
            If num_samples is None and numRows(parsed from schema) does not exist, read the full dataset;
            If num_samples is None and numRows(parsed from schema) is greater than 0, read numRows rows;
            If both num_samples and numRows(parsed from schema) are greater than 0, read num_samples rows.
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        shuffle (Union[bool, Shuffle level], optional): Perform reshuffling of the data every epoch
            (default=Shuffle.GLOBAL).
            If shuffle is False, no shuffling will be performed;
            If shuffle is True, the behavior is the same as setting shuffle to be Shuffle.GLOBAL
            Otherwise, there are two levels of shuffling:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided
            into (default=None). When this argument is specified, `num_samples` reflects
            the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        shard_equal_rows (bool, optional): Get equal rows for all shards(default=False). If shard_equal_rows
            is false, number of rows of each shard may be not equal, and may lead to a failure in distributed training.
            When the number of samples of per TFRecord file are not equal, it is suggested to set to true.
            This argument should only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_files are not valid or do not exist.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Examples:
        >>> from mindspore import dtype as mstype
        >>>
        >>> tfrecord_dataset_dir = ["/path/to/tfrecord_dataset_file"] # contains 1 or multiple TFRecord files
        >>> tfrecord_schema_file = "/path/to/tfrecord_schema_file"
        >>>
        >>> # 1) Get all rows from tfrecord_dataset_dir with no explicit schema.
        >>> # The meta-data in the first row will be used as a schema.
        >>> dataset = ds.TFRecordDataset(dataset_files=tfrecord_dataset_dir)
        >>>
        >>> # 2) Get all rows from tfrecord_dataset_dir with user-defined schema.
        >>> schema = ds.Schema()
        >>> schema.add_column(name='col_1d', de_type=mstype.int64, shape=[2])
        >>> dataset = ds.TFRecordDataset(dataset_files=tfrecord_dataset_dir, schema=schema)
        >>>
        >>> # 3) Get all rows from tfrecord_dataset_dir with schema file.
        >>> dataset = ds.TFRecordDataset(dataset_files=tfrecord_dataset_dir, schema=tfrecord_schema_file)
    """

    @check_tfrecorddataset
    def __init__(self, dataset_files, schema=None, columns_list=None, num_samples=None, num_parallel_workers=None,
                 shuffle=Shuffle.GLOBAL, num_shards=None, shard_id=None, shard_equal_rows=False, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_files = self._find_files(dataset_files)
        self.dataset_files.sort()

        self.schema = schema
        self.columns_list = replace_none(columns_list, [])
        self.shard_equal_rows = replace_none(shard_equal_rows, False)

        if self.schema is not None and (self.num_samples is None or self.num_samples == 0):
            self.num_samples = Schema.get_num_rows(self.schema)

    def parse(self, children=None):
        schema = self.schema.cpp_schema if isinstance(self.schema, Schema) else self.schema
        return cde.TFRecordNode(self.dataset_files, schema, self.columns_list, self.num_samples, self.shuffle_flag,
                                self.num_shards, self.shard_id, self.shard_equal_rows)


class ManifestDataset(MappableDataset):
    """
    A source dataset for reading images from a Manifest file.

    The generated dataset has two columns: :py:obj:`[image, label]`.
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`label` is of a scalar of uint64 type.

    Args:
        dataset_file (str): File to be read.
        usage (str, optional): Acceptable usages include `train`, `eval` and `inference` (default= `train`).
        num_samples (int, optional): The number of images to be included in the dataset.
            (default=None, will include all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, will use value set in the config).
        shuffle (bool, optional): Whether to perform shuffle on the dataset (default=None, expected
            order behavior shown in the table).
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, expected order behavior shown in the table).
        class_indexing (dict, optional): A str-to-int mapping from label name to index
            (default=None, the folder names will be sorted alphabetically and each
            class will be given a unique index starting from 0).
        decode (bool, optional): decode the images after reading (default=False).
        num_shards (int, optional): Number of shards that the dataset will be divided
            into (default=None). When this argument is specified, `num_samples` reflects
            the max number of samples per shard.
        shard_id (int, optional): The shard ID within `num_shards` (default=None). This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_files are not valid or do not exist.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        RuntimeError: If class_indexing is not a dictionary.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Note:
        - The shape of the image column is [image_size] if decode flag is False, or [H,W,C] otherwise.
        - This dataset can take in a `sampler`. `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> manifest_dataset_dir = "/path/to/manifest_dataset_file"
        >>>
        >>> # 1) Read all samples specified in manifest_dataset_dir dataset with 8 threads for training
        >>> dataset = ds.ManifestDataset(dataset_file=manifest_dataset_dir, usage="train", num_parallel_workers=8)
        >>>
        >>> # 2) Read samples (specified in manifest_file.manifest) for shard 0 in a 2-way distributed training setup
        >>> dataset = ds.ManifestDataset(dataset_file=manifest_dataset_dir, num_shards=2, shard_id=0)
    """

    @check_manifestdataset
    def __init__(self, dataset_file, usage="train", num_samples=None, num_parallel_workers=None, shuffle=None,
                 sampler=None, class_indexing=None, decode=False, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_file = dataset_file
        self.decode = replace_none(decode, False)
        self.usage = replace_none(usage, "train")
        self.class_indexing = replace_none(class_indexing, {})

    def parse(self, children=None):
        return cde.ManifestNode(self.dataset_file, self.usage, self.sampler, self.class_indexing, self.decode)

    def get_class_indexing(self):
        """
        Get the class index.

        Returns:
            dict, a str-to-int mapping from label name to index.

        Examples:
            >>> manifest_dataset_dir = "/path/to/manifest_dataset_file"
            >>>
            >>> dataset = ds.ManifestDataset(dataset_file=manifest_dataset_dir)
            >>> class_indexing = dataset.get_class_indexing()
        """
        if self.class_indexing is None or not self.class_indexing:
            if self._class_indexing is None:
                runtime_getter = self._init_tree_getters()
                self._class_indexing = runtime_getter[0].GetClassIndexing()
            self.class_indexing = {}
            for pair in self._class_indexing:
                self.class_indexing[pair[0]] = pair[1][0]
        return self.class_indexing


class AGNewsDataset(SourceDataset, TextBaseDataset):
    """
    A source dataset that reads and parses AG News datasets.

    The generated dataset has three columns: :py:obj:`[index, title, description]`.
    The tensor of column :py:obj:`index` is of the string type.
    The tensor of column :py:obj:`title` is of the string type.
    The tensor of column :py:obj:`description` is of the string type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Acceptable usages include `train`, `test` and `all` (default=None, all samples).
        num_samples (int, optional): Number of samples (rows) to read (default=None, reads the full dataset).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        shuffle (Union[bool, Shuffle level], optional): Perform reshuffling of the data every epoch
            (default=Shuffle.GLOBAL).
            If shuffle is False, no shuffling will be performed;
            If shuffle is True, the behavior is the same as setting shuffle to be Shuffle.GLOBAL
            Otherwise, there are two levels of shuffling:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, 'num_samples' reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Examples:
        >>> ag_news_dataset_dir = "/path/to/ag_news_dataset_file"
        >>> dataset = ds.AGNewsDataset(dataset_dir=ag_news_dataset_dir, usage='all')

    About AGNews dataset:

    AG is a collection of over 1 million news articles. The news articles were collected
    by ComeToMyHead from over 2,000 news sources in over 1 year of activity. ComeToMyHead
    is an academic news search engine that has been in operation since July 2004.
    The dataset is provided by academics for research purposes such as data mining
    (clustering, classification, etc.), information retrieval (ranking, searching, etc.),
    xml, data compression, data streaming, and any other non-commercial activities.
    AG's news topic classification dataset was constructed by selecting the four largest
    classes from the original corpus. Each class contains 30,000 training samples and
    1,900 test samples. The total number of training samples in train.csv is 120,000
    and the number of test samples in test.csv is 7,600.

    You can unzip the dataset files into the following structure and read by MindSpore's API:

    .. code-block::

        .
        └── ag_news_dataset_dir
            ├── classes.txt
            ├── train.csv
            ├── test.csv
            └── readme.txt

    Citation:

    .. code-block::

        @misc{zhang2015characterlevel,
        title={Character-level Convolutional Networks for Text Classification},
        author={Xiang Zhang and Junbo Zhao and Yann LeCun},
        year={2015},
        eprint={1509.01626},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
        }
    """

    @check_ag_news_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None,
                 num_parallel_workers=None, shuffle=Shuffle.GLOBAL, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.AGNewsNode(self.dataset_dir, self.usage, self.num_samples, self.shuffle_flag, self.num_shards,
                              self.shard_id)


class AmazonReviewDataset(SourceDataset):
    """
    A source dataset that reads and parses Amazon Review Polarity and Amazon Review Full datasets.

    The generated dataset has three columns: :py:obj:`[label, title, content]`.
    The tensor of column :py:obj:`label` is of the string type.
    The tensor of column :py:obj:`title` is of the string type.
    The tensor of column :py:obj:`content` is of the string type.

    Args:
        dataset_dir (str): Path to the root directory that contains the Amazon Review Polarity dataset
            or the Amazon Review Full dataset.
        usage (str, optional): Usage of this dataset, can be `train`, `test` or `all` (default= `all`).
            For Polarity dataset, `train` will read from 3,600,000 train samples,
            `test` will read from 400,000 test samples,
            `all` will read from all 4,000,000 samples.
             For Full dataset, `train` will read from 3,000,000 train samples,
            `test` will read from 650,000 test samples,
            `all` will read from all 3,650,000 samples (default=None, all samples).
        num_samples (int, optional): Number of samples (rows) to be read (default=None, reads the full dataset).
        shuffle (Union[bool, Shuffle level], optional): Perform reshuffling of the data every epoch
            (default=Shuffle.GLOBAL).
            If shuffle is False, no shuffling will be performed;
            If shuffle is True, the behavior is the same as setting shuffle to be Shuffle.GLOBAL
            Otherwise, there are two levels of shuffling:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.
        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, `num_samples` reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the  mindspore.dataset.config).
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_dir does not contain data files.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.

    Examples:
        >>> amazon_review_dataset_dir = "/path/to/amazon_review_dataset_dir"
        >>> dataset = ds.AmazonReviewDataset(dataset_dir=amazon_review_dataset_dir, usage='all')

    About AmazonReview Dataset:

    The Amazon reviews full dataset consists of reviews from Amazon. The data span a period of 18 years, including ~35
    million reviews up to March 2013. Reviews include product and user information, ratings, and a plaintext review.
    The dataset is mainly used for text classification, given the content and title, predict the correct star rating.

    The Amazon reviews polarity dataset is constructed by taking review score 1 and 2 as negative, 4 and 5 as positive.
    Samples of score 3 is ignored. In the dataset, class 1 is the negative and class 2 is the positive.

    The Amazon Reviews Polarity and Amazon Reviews Full datasets have the same directory structures.
    You can unzip the dataset files into the following structure and read by MindSpore's API:

    .. code-block::

        .
        └── amazon_review_dir
             ├── train.csv
             ├── test.csv
             └── readme.txt

   Citation:

    .. code-block::

        @article{zhang2015character,
          title={Character-level convolutional networks for text classification},
          author={Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
          journal={Advances in neural information processing systems},
          volume={28},
          pages={649--657},
          year={2015}
        }
    """

    @check_amazon_review_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=Shuffle.GLOBAL,
                 num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, 'all')

    def parse(self, children=None):
        return cde.AmazonReviewNode(self.dataset_dir, self.usage, self.num_samples, self.shuffle_flag, self.num_shards,
                                    self.shard_id)


class Cifar10Dataset(MappableDataset):
    """
    A source dataset for reading and parsing Cifar10 dataset.
    This api only supports parsing Cifar10 file in binary version now.

    The generated dataset has two columns :py:obj:`[image, label]`.
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`label` is a scalar of the uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be `train`, `test` or `all` . `train` will read from 50,000
            train samples, `test` will read from 10,000 test samples, `all` will read from all 60,000 samples
            (default=None, all samples).
        num_samples (int, optional): The number of images to be included in the dataset
            (default=None, all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        shuffle (bool, optional): Whether to perform shuffle on the dataset (default=None, expected
            order behavior shown in the table).
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided
            into (default=None). When this argument is specified, `num_samples` reflects
            the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_dir does not contain data files.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Note:
        - This dataset can take in a `sampler`. `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> cifar10_dataset_dir = "/path/to/cifar10_dataset_directory"
        >>>
        >>> # 1) Get all samples from CIFAR10 dataset in sequence
        >>> dataset = ds.Cifar10Dataset(dataset_dir=cifar10_dataset_dir, shuffle=False)
        >>>
        >>> # 2) Randomly select 350 samples from CIFAR10 dataset
        >>> dataset = ds.Cifar10Dataset(dataset_dir=cifar10_dataset_dir, num_samples=350, shuffle=True)
        >>>
        >>> # 3) Get samples from CIFAR10 dataset for shard 0 in a 2-way distributed training
        >>> dataset = ds.Cifar10Dataset(dataset_dir=cifar10_dataset_dir, num_shards=2, shard_id=0)
        >>>
        >>> # In CIFAR10 dataset, each dictionary has keys "image" and "label"

    About CIFAR-10 dataset:

    The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes,
    with 6000 images per class. There are 50000 training images and 10000 test images.
    The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

    Here is the original CIFAR-10 dataset structure.
    You can unzip the dataset files into the following directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── cifar-10-batches-bin
             ├── data_batch_1.bin
             ├── data_batch_2.bin
             ├── data_batch_3.bin
             ├── data_batch_4.bin
             ├── data_batch_5.bin
             ├── test_batch.bin
             ├── readme.html
             └── batches.meta.txt

    Citation:

    .. code-block::

        @techreport{Krizhevsky09,
        author       = {Alex Krizhevsky},
        title        = {Learning multiple layers of features from tiny images},
        institution  = {},
        year         = {2009},
        howpublished = {http://www.cs.toronto.edu/~kriz/cifar.html}
        }
    """

    @check_mnist_cifar_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None,
                 sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.Cifar10Node(self.dataset_dir, self.usage, self.sampler)


class Cifar100Dataset(MappableDataset):
    """
    A source dataset for reading and parsing Cifar100 dataset.

    The generated dataset has three columns :py:obj:`[image, coarse_label, fine_label]`.
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`coarse_label` and :py:obj:`fine_labels` are each a scalar of uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be `train`, `test` or `all` . `train` will read from 50,000
            train samples, `test` will read from 10,000 test samples, `all` will read from all 60,000 samples
            (default=None, all samples).
        num_samples (int, optional): The number of images to be included in the dataset
            (default=None, all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        shuffle (bool, optional): Whether to perform shuffle on the dataset (default=None, expected
            order behavior shown in the table).
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided
            into (default=None). When this argument is specified, 'num_samples' reflects
            the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_dir does not contain data files.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Note:
        - This dataset can take in a `sampler`. `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and shuffle
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> cifar100_dataset_dir = "/path/to/cifar100_dataset_directory"
        >>>
        >>> # 1) Get all samples from CIFAR100 dataset in sequence
        >>> dataset = ds.Cifar100Dataset(dataset_dir=cifar100_dataset_dir, shuffle=False)
        >>>
        >>> # 2) Randomly select 350 samples from CIFAR100 dataset
        >>> dataset = ds.Cifar100Dataset(dataset_dir=cifar100_dataset_dir, num_samples=350, shuffle=True)
        >>>
        >>> # In CIFAR100 dataset, each dictionary has 3 keys: "image", "fine_label" and "coarse_label"

    About CIFAR-100 dataset:

    This dataset is just like the CIFAR-10, except it has 100 classes containing 600 images
    each. There are 500 training images and 100 testing images per class. The 100 classes in
    the CIFAR-100 are grouped into 20 superclasses. Each image comes with a "fine" label (the
    class to which it belongs) and a "coarse" label (the superclass to which it belongs).

    Here is the original CIFAR-100 dataset structure.
    You can unzip the dataset files into the following directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── cifar-100-binary
            ├── train.bin
            ├── test.bin
            ├── fine_label_names.txt
            └── coarse_label_names.txt

    Citation:

    .. code-block::

        @techreport{Krizhevsky09,
        author       = {Alex Krizhevsky},
        title        = {Learning multiple layers of features from tiny images},
        institution  = {},
        year         = {2009},
        howpublished = {http://www.cs.toronto.edu/~kriz/cifar.html}
        }
    """

    @check_mnist_cifar_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None,
                 sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.Cifar100Node(self.dataset_dir, self.usage, self.sampler)


class RandomDataset(SourceDataset):
    """
    A source dataset that generates random data.

    Args:
        total_rows (int, optional): Number of samples for the dataset to generate
            (default=None, number of samples is random).
        schema (Union[str, Schema], optional): Path to the JSON schema file or schema object (default=None).
            If the schema is not provided, the random dataset generates a random schema.
        columns_list (list[str], optional): List of columns to be read (default=None, read all columns)
        num_samples (int, optional): The number of samples to be included in the dataset
            (default=None, all samples).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset
            (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided
            into (default=None). When this argument is specified, 'num_samples' reflects
            the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
    """

    @check_random_dataset
    def __init__(self, total_rows=None, schema=None, columns_list=None, num_samples=None, num_parallel_workers=None,
                 cache=None, shuffle=None, num_shards=None, shard_id=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.total_rows = total_rows
        if schema is not None:
            self.total_rows = replace_none(total_rows, Schema.get_num_rows(schema))
        self.schema = schema
        self.columns_list = replace_none(columns_list, [])

    def parse(self, children=None):
        schema = self.schema.cpp_schema if isinstance(self.schema, Schema) else self.schema
        return cde.RandomNode(self.total_rows, schema, self.columns_list)


class Schema:
    """
    Class to represent a schema of a dataset.

    Args:
        schema_file(str): Path of the schema file (default=None).

    Returns:
        Schema object, schema info about dataset.

    Raises:
        RuntimeError: If schema file failed to load.

    Examples:
        >>> from mindspore import dtype as mstype
        >>>
        >>> # Create schema; specify column name, mindspore.dtype and shape of the column
        >>> schema = ds.Schema()
        >>> schema.add_column(name='col1', de_type=mstype.int64, shape=[2])
    """

    @check_schema
    def __init__(self, schema_file=None):
        self.schema_file = replace_none(schema_file, "")
        self.cpp_schema = cde.SchemaObj(self.schema_file)

    @check_add_column
    def add_column(self, name, de_type, shape=None):
        """
        Add new column to the schema.

        Args:
            name (str): The new name of the column.
            de_type (str): Data type of the column.
            shape (list[int], optional): Shape of the column
                (default=None, [-1] which is an unknown shape of rank 1).

        Raises:
            ValueError: If column type is unknown.
        """
        if isinstance(de_type, typing.Type):
            de_type = mstype_to_detype(de_type)
            col_type = str(de_type)
        else:
            col_type = str(cde.DataType(de_type))
        if shape is None:
            self.cpp_schema.add_column(name, col_type)
        else:
            self.cpp_schema.add_column(name, col_type, shape)

    def parse_columns(self, columns):
        """
        Parse the columns and add it to self.

        Args:
            columns (Union[dict, list[dict], tuple[dict]]): Dataset attribute information, decoded from schema file.

                - list[dict], 'name' and 'type' must be in keys, 'shape' optional.

                - dict, columns.keys() as name, columns.values() is dict, and 'type' inside, 'shape' optional.

        Raises:
            RuntimeError: If failed to parse columns.
            RuntimeError: If column's name field is missing.
            RuntimeError: If column's type field is missing.

        Examples:
            >>> from mindspore.dataset import Schema
            >>> schema = Schema()
            >>> columns1 = [{'name': 'image', 'type': 'int8', 'shape': [3, 3]},
            ...             {'name': 'label', 'type': 'int8', 'shape': [1]}]
            >>> schema.parse_columns(columns1)
            >>> columns2 = {'image': {'shape': [3, 3], 'type': 'int8'}, 'label': {'shape': [1], 'type': 'int8'}}
            >>> schema.parse_columns(columns2)
        """
        self.cpp_schema.parse_columns(json.dumps(columns, indent=2))

    def to_json(self):
        """
        Get a JSON string of the schema.

        Returns:
            str, JSON string of the schema.
        """
        return self.cpp_schema.to_json()

    def from_json(self, json_obj):
        """
        Get schema file from JSON object.

        Args:
            json_obj(dictionary): Object of JSON parsed.

        Raises:
            RuntimeError: if there is unknown item in the object.
            RuntimeError: if dataset type is missing in the object.
            RuntimeError: if columns are missing in the object.
        """
        self.cpp_schema.from_string(json.dumps(json_obj, indent=2))

    def __str__(self):
        return self.to_json()

    @staticmethod
    def get_num_rows(schema):
        schema_obj = schema
        if not isinstance(schema_obj, Schema):
            schema_obj = Schema(schema_obj)
        return schema_obj.cpp_schema.get_num_rows()


class UDPOSDataset(SourceDataset):
    """
    A source dataset that reads and parses UDPOS dataset.

    The generated dataset has three columns: :py:obj:`[word, universal, stanford]`.
    The tensor of column :py:obj:`word` is of the string type.
    The tensor of column :py:obj:`universal` is of the string type.
    The tensor of column :py:obj:`stanford` is of the string type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be `train`, `test`, `valid` or `all`. `train` will read from
            12,543 train samples, `test` will read from 2,077 test samples, `valid` will read from 2,002 test samples,
            `all` will read from all 16,622 samples (default=None, all samples).
        num_samples (int, optional): Number of samples (rows) to read (default=None, reads the full dataset).
        shuffle (Union[bool, Shuffle level], optional): Perform reshuffling of the data every epoch
            (default=Shuffle.GLOBAL).
            If shuffle is False, no shuffling will be performed;
            If shuffle is True, the behavior is the same as setting shuffle to be Shuffle.GLOBAL
            Otherwise, there are two levels of shuffling:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, `num_samples` reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_dir does not contain data files.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.

    Examples:
        >>> udpos_dataset_dir = "/path/to/udpos_dataset_dir"
        >>> dataset = ds.UDPOSDataset(dataset_files=udpos_dataset_dir, usage='all')
    """

    @check_udpos_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, shuffle=Shuffle.GLOBAL, num_shards=None,
                 shard_id=None, num_parallel_workers=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, 'all')

    def parse(self, children=None):
        return cde.UDPOSNode(self.dataset_dir, self.usage, self.num_samples, self.shuffle_flag, self.num_shards,
                             self.shard_id)


class USPSDataset(SourceDataset):
    """
    A source dataset for reading and parsing the USPS dataset.

    The generated dataset has two columns: :py:obj:`[image, label]`.
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`label` is of a scalar of uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be "train", "test" or "all". "train" will read from 7,291
            train samples, "test" will read from 2,007 test samples, "all" will read from all 9,298 samples.
            (default=None, will read all samples)
        num_samples (int, optional): The number of images to be included in the dataset
            (default=None, will read all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, will use value set in the config).
        shuffle (Union[bool, Shuffle level], optional): Perform reshuffling of the data every epoch
            (default=Shuffle.GLOBAL).
            If shuffle is False, no shuffling will be performed;
            If shuffle is True, the behavior is the same as setting shuffle to be Shuffle.GLOBAL
            Otherwise, there are two levels of shuffling:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, `num_samples` reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` (default=None). This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_dir is not valid or does not exist or does not contain data files.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If usage is invalid.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Examples:
        >>> usps_dataset_dir = "/path/to/usps_dataset_directory"
        >>>
        >>> # Read 3 samples from USPS dataset
        >>> dataset = ds.USPSDataset(dataset_dir=usps_dataset_dir, num_samples=3)
        >>>
        >>> # Note: In USPS dataset, each dictionary has keys "image" and "label"

    About USPS dataset:

    USPS is a digit dataset automatically scanned from envelopes by the U.S. Postal Service
    containing a total of 9,298 16×16 pixel grayscale samples.
    The images are centered, normalized and show a broad range of font styles.

    Here is the original USPS dataset structure.
    You can download and unzip the dataset files into this directory structure and read by MindSpore's API.

    .. code-block::
        .
        └── usps_dataset_dir
             ├── usps
             ├── usps.t

    Citation:

    .. code-block::

        @article{hull1994database,
          title={A database for handwritten text recognition research},
          author={Hull, Jonathan J.},
          journal={IEEE Transactions on pattern analysis and machine intelligence},
          volume={16},
          number={5},
          pages={550--554},
          year={1994},
          publisher={IEEE}
        }
    """

    @check_usps_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=Shuffle.GLOBAL,
                 num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.USPSNode(self.dataset_dir, self.usage, self.num_samples, self.shuffle_flag, self.num_shards,
                            self.shard_id)


class WikiTextDataset(SourceDataset):
    """
    A source dataset that reads and parses WikiText2 and WikiText103 datasets.

    The generated dataset has one column :py:obj:`[text]`.
    The tensor of column :py:obj:`text` is of the string type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Acceptable usages include `train`, `test`, 'valid' and `all`(default=None, all samples).
        num_samples (int, optional): Number of samples (rows) to read (default=None, reads the full dataset).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        shuffle (Union[bool, Shuffle level], optional): Perform reshuffling of the data every epoch
            (default=Shuffle.GLOBAL).
            If shuffle is False, no shuffling will be performed;
            If shuffle is True, the behavior is the same as setting shuffle to be Shuffle.GLOBAL
            Otherwise, there are two levels of shuffling:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, 'num_samples' reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Examples:
        >>> wiki_text_dataset_dir = "/path/to/wiki_text_dataset_directory"
        >>> dataset = ds.WikiTextDataset(dataset_dir=wiki_text_dataset_dir, usage='all')

    About WikiTextDataset dataset:

    The WikiText Long Term Dependency Language Modeling Dataset is an English lexicon containing 100 million words.
    These terms are drawn from Wikipedia's premium and benchmark articles, including versions of Wikitext2 and
    Wikitext103. For WikiText2, it has 36718 lines in wiki.train.tokens, 4358 lines in wiki.test.tokens and
    3760 lines in wiki.valid.tokens. For WikiText103, it has 1801350 lines in wiki.train.tokens, 4358 lines in
    wiki.test.tokens and 3760 lines in wiki.valid.tokens.

    Here is the original WikiText dataset structure.
    You can unzip the dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── WikiText2/WikiText103
             ├── wiki.train.tokens
             ├── wiki.test.tokens
             ├── wiki.valid.tokens

    Citation:

    .. code-block::

        @article{merity2016pointer,
          title={Pointer sentinel mixture models},
          author={Merity, Stephen and Xiong, Caiming and Bradbury, James and Socher, Richard},
          journal={arXiv preprint arXiv:1609.07843},
          year={2016}
        }
    """

    @check_wiki_text_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=Shuffle.GLOBAL,
                 num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.WikiTextNode(self.dataset_dir, self.usage, self.num_samples, self.shuffle_flag, self.num_shards,
                                self.shard_id)


class VOCDataset(MappableDataset):
    """
    A source dataset for reading and parsing VOC dataset.

    The generated dataset with different task setting has different output columns:

    - task = :py:obj:`Detection`, output columns: :py:obj:`[image, dtype=uint8]`, :py:obj:`[bbox, dtype=float32]`, \
        :py:obj:`[label, dtype=uint32]`, :py:obj:`[difficult, dtype=uint32]`, :py:obj:`[truncate, dtype=uint32]`.
    - task = :py:obj:`Segmentation`, output columns: :py:obj:`[image, dtype=uint8]`, :py:obj:`[target,dtype=uint8]`.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        task (str, optional): Set the task type of reading voc data, now only support `Segmentation` or `Detection`
            (default= `Segmentation`).
        usage (str, optional): Set the task type of ImageSets(default= `train`). If task is `Segmentation`, image and
            annotation list will be loaded in ./ImageSets/Segmentation/usage + ".txt"; If task is `Detection`, image and
            annotation list will be loaded in ./ImageSets/Main/usage + ".txt"; if task and usage are not set, image and
            annotation list will be loaded in ./ImageSets/Segmentation/train.txt as default.
        class_indexing (dict, optional): A str-to-int mapping from label name to index, only valid in
            `Detection` task (default=None, the folder names will be sorted alphabetically and each
            class will be given a unique index starting from 0).
        num_samples (int, optional): The number of images to be included in the dataset
            (default=None, all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        shuffle (bool, optional): Whether to perform shuffle on the dataset (default=None, expected
            order behavior shown in the table).
        decode (bool, optional): Decode the images after reading (default=False).
        sampler (Sampler, optional): Object used to choose samples from the dataset
            (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided
            into (default=None). When this argument is specified, `num_samples` reflects
            the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).
        extra_metadata(bool, optional): Flag to add extra meta-data to row. If True, an additional column named
            :py:obj:`[_meta-filename, dtype=string]` will be output at the end (default=False).

    Raises:
        RuntimeError: If dataset_dir does not contain data files.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If xml of Annotations is an invalid format.
        RuntimeError: If xml of Annotations loss attribution of `object`.
        RuntimeError: If xml of Annotations loss attribution of `bndbox`.
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If task is not equal 'Segmentation' or 'Detection'.
        ValueError: If task equal 'Segmentation' but class_indexing is not None.
        ValueError: If txt related to mode is not exist.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Note:
        - Column '[_meta-filename, dtype=string]' won't be output unless an explicit rename dataset op
          is added to remove the prefix('_meta-').
        - This dataset can take in a `sampler`. `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> voc_dataset_dir = "/path/to/voc_dataset_directory"
        >>>
        >>> # 1) Read VOC data for segmentation training
        >>> dataset = ds.VOCDataset(dataset_dir=voc_dataset_dir, task="Segmentation", usage="train")
        >>>
        >>> # 2) Read VOC data for detection training
        >>> dataset = ds.VOCDataset(dataset_dir=voc_dataset_dir, task="Detection", usage="train")
        >>>
        >>> # 3) Read all VOC dataset samples in voc_dataset_dir with 8 threads in random order
        >>> dataset = ds.VOCDataset(dataset_dir=voc_dataset_dir, task="Detection", usage="train",
        ...                         num_parallel_workers=8)
        >>>
        >>> # 4) Read then decode all VOC dataset samples in voc_dataset_dir in sequence
        >>> dataset = ds.VOCDataset(dataset_dir=voc_dataset_dir, task="Detection", usage="train",
        ...                         decode=True, shuffle=False)
        >>>
        >>> # In VOC dataset, if task='Segmentation', each dictionary has keys "image" and "target"
        >>> # In VOC dataset, if task='Detection', each dictionary has keys "image" and "annotation"

    About VOC dataset.

    The PASCAL Visual Object Classes (VOC) challenge is a benchmark in visual
    object category recognition and detection, providing the vision and machine
    learning communities with a standard dataset of images and annotation, and
    standard evaluation procedures.

    You can unzip the original VOC-2012 dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── voc2012_dataset_dir
            ├── Annotations
            │    ├── 2007_000027.xml
            │    ├── 2007_000032.xml
            │    ├── ...
            ├── ImageSets
            │    ├── Action
            │    ├── Layout
            │    ├── Main
            │    └── Segmentation
            ├── JPEGImages
            │    ├── 2007_000027.jpg
            │    ├── 2007_000032.jpg
            │    ├── ...
            ├── SegmentationClass
            │    ├── 2007_000032.png
            │    ├── 2007_000033.png
            │    ├── ...
            └── SegmentationObject
                 ├── 2007_000032.png
                 ├── 2007_000033.png
                 ├── ...

    Citation:

    .. code-block::

        @article{Everingham10,
        author       = {Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.},
        title        = {The Pascal Visual Object Classes (VOC) Challenge},
        journal      = {International Journal of Computer Vision},
        volume       = {88},
        year         = {2012},
        number       = {2},
        month        = {jun},
        pages        = {303--338},
        biburl       = {http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.html#bibtex},
        howpublished = {http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html}
        }
    """

    @check_vocdataset
    def __init__(self, dataset_dir, task="Segmentation", usage="train", class_indexing=None, num_samples=None,
                 num_parallel_workers=None, shuffle=None, decode=False, sampler=None, num_shards=None, shard_id=None,
                 cache=None, extra_metadata=False):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.task = replace_none(task, "Segmentation")
        self.usage = replace_none(usage, "train")
        self.class_indexing = replace_none(class_indexing, {})
        self.decode = replace_none(decode, False)
        self.extra_metadata = extra_metadata

    def parse(self, children=None):
        return cde.VOCNode(self.dataset_dir, self.task, self.usage, self.class_indexing, self.decode, self.sampler,
                           self.extra_metadata)

    def get_class_indexing(self):
        """
        Get the class index.

        Returns:
            dict, a str-to-int mapping from label name to index.

        Examples:
            >>> voc_dataset_dir = "/path/to/voc_dataset_directory"
            >>>
            >>> dataset = ds.VOCDataset(dataset_dir=voc_dataset_dir, task="Detection")
            >>> class_indexing = dataset.get_class_indexing()
        """
        if self.task != "Detection":
            raise NotImplementedError("Only 'Detection' support get_class_indexing.")
        if self.class_indexing is None or not self.class_indexing:
            if self._class_indexing is None:
                runtime_getter = self._init_tree_getters()
                self._class_indexing = runtime_getter[0].GetClassIndexing()
            self.class_indexing = {}
            for pair in self._class_indexing:
                self.class_indexing[pair[0]] = pair[1][0]
        return self.class_indexing


class _Caltech101Dataset:
    """
    Mainly for loading Caltech101 Dataset, and return two rows each time.
    """

    def __init__(self, dataset_dir, target_type="category", decode=False):
        self.dataset_dir = os.path.realpath(dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "101_ObjectCategories")
        self.annotation_dir = os.path.join(self.dataset_dir, "Annotations")
        self.target_type = target_type
        if self.target_type == "category":
            self.column_names = ["image", "category"]
        elif self.target_type == "annotation":
            self.column_names = ["image", "annotation"]
        else:
            self.column_names = ["image", "category", "annotation"]
        self.decode = decode
        self.classes = sorted(os.listdir(self.image_dir))
        if "BACKGROUND_Google" in self.classes:
            self.classes.remove("BACKGROUND_Google")
        name_map = {"Faces": "Faces_2",
                    "Faces_easy": "Faces_3",
                    "Motorbikes": "Motorbikes_16",
                    "airplanes": "Airplanes_Side_2"}
        self.annotation_classes = [name_map[class_name] if class_name in name_map else class_name
                                   for class_name in self.classes]
        self.image_index = []
        self.image_label = []
        for i, image_class in enumerate(self.classes):
            sub_dir = os.path.join(self.image_dir, image_class)
            if not os.path.isdir(sub_dir) or not os.access(sub_dir, os.R_OK):
                continue
            num_images = len(os.listdir(sub_dir))
            self.image_index.extend(range(1, num_images + 1))
            self.image_label.extend(num_images * [i])

    def __getitem__(self, index):
        image_file = os.path.join(self.image_dir, self.classes[self.image_label[index]],
                                  "image_{:04d}.jpg".format(self.image_index[index]))
        if not os.path.exists(image_file):
            raise ValueError("The image file {} does not exist or permission denied!".format(image_file))
        if self.decode:
            image = np.asarray(Image.open(image_file).convert("RGB"))
        else:
            image = np.fromfile(image_file, dtype=np.uint8)

        if self.target_type == "category":
            return image, self.image_label[index]
        annotation_file = os.path.join(self.annotation_dir, self.annotation_classes[self.image_label[index]],
                                       "annotation_{:04d}.mat".format(self.image_index[index]))
        if not os.path.exists(annotation_file):
            raise ValueError("The annotation file {} does not exist or permission denied!".format(annotation_file))
        annotation = loadmat(annotation_file)["obj_contour"]

        if self.target_type == "annotation":
            return image, annotation
        return image, self.image_label[index], annotation

    def __len__(self):
        return len(self.image_index)


class Caltech101Dataset(GeneratorDataset):
    """
    A source dataset that reads and parses Caltech101 dataset.

    The columns of the generated dataset depend on the value of `target_type`.
    When `target_type` is `category`, the columns are :py:obj:`[image, category]`.
    When `target_type` is `annotation`, the columns are :py:obj:`[image, annotation]`.
    When `target_type` is `all`, the columns are :py:obj:`[image, category, annotation]`.
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`category` is of the uint32 type.
    The tensor of column :py:obj:`annotation` is a 2-dimensional ndarray that stores the contour of the image
    and consists of a series of points.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset. This root directory contains two
            subdirectories, one is called 101_ObjectCategories, which stores images,
            and the other is called Annotations, which stores annotations.
        target_type (str, optional): Target of the image. If target_type is "category", return category represents
            the target class. If target_type is "annotation", return annotation.
            If target_type is "all", return category and annotation (default=None, means "category").
        num_samples (int, optional): The number of images to be included in the dataset
            (default=None, all images).
        num_parallel_workers (int, optional): Number of workers to read the data (default=1).
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset
            (default=None, expected order behavior shown in the table).
        decode (bool, optional): Whether or not to decode the images after reading (default=False).
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided
            into (default=None). When this argument is specified, `num_samples` reflects
            the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.

    Raises:
        RuntimeError: If dataset_dir does not contain data files.
        RuntimeError: If target_type is not set correctly.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Note:
        - This dataset can take in a `sampler`. `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> caltech101_dataset_directory = "/path/to/caltech101_dataset_directory"
        >>>
        >>> # 1) Read all samples (image files) in caltech101_dataset_directory with 8 threads
        >>> dataset = ds.Caltech101Dataset(dataset_dir=caltech101_dataset_directory, num_parallel_workers=8)
        >>>
        >>> # 2) Read all samples (image files) with the target_type "annotation"
        >>> dataset = ds.Caltech101Dataset(dataset_dir=caltech101_dataset_directory, target_type="annotation")

    About Caltech101Dataset:

    Pictures of objects belonging to 101 categories. About 40 to 800 images per category.
    Most categories have about 50 images. Collected in September 2003 by Fei-Fei Li, Marco Andreetto,
    and Marc 'Aurelio Ranzato. The size of each image is roughly 300 x 200 pixels.
    The official provides the contour data of each object in each picture, which is the annotation.

    .. code-block::

        .
        └── caltech101_dataset_directory
            ├── 101_ObjectCategories
            │    ├── Faces
            │    │    ├── image_0001.jpg
            │    │    ├── image_0002.jpg
            │    │    ...
            │    ├── Faces_easy
            │    │    ├── image_0001.jpg
            │    │    ├── image_0002.jpg
            │    │    ...
            │    ├── ...
            └── Annotations
                 ├── Airplanes_Side_2
                 │    ├── annotation_0001.mat
                 │    ├── annotation_0002.mat
                 │    ...
                 ├── Faces_2
                 │    ├── annotation_0001.mat
                 │    ├── annotation_0002.mat
                 │    ...
                 ├── ...

    Citation:

    .. code-block::

        @article{FeiFei2004LearningGV,
        author    = {Li Fei-Fei and Rob Fergus and Pietro Perona},
        title     = {Learning Generative Visual Models from Few Training Examples:
                    An Incremental Bayesian Approach Tested on 101 Object Categories},
        journal   = {Computer Vision and Pattern Recognition Workshop},
        year      = {2004},
        url       = {http://www.vision.caltech.edu/Image_Datasets/Caltech101/},
        }
    """

    @check_caltech101_dataset
    def __init__(self, dataset_dir, target_type=None, num_samples=None, num_parallel_workers=1,
                 shuffle=None, decode=False, sampler=None, num_shards=None, shard_id=None):
        self.dataset_dir = dataset_dir
        self.target_type = replace_none(target_type, "category")
        self.decode = replace_none(decode, False)
        dataset = _Caltech101Dataset(self.dataset_dir, self.target_type, self.decode)
        super().__init__(dataset, column_names=dataset.column_names, num_samples=num_samples,
                         num_parallel_workers=num_parallel_workers, shuffle=shuffle, sampler=sampler,
                         num_shards=num_shards, shard_id=shard_id)

    def get_class_indexing(self):
        """
        Get the class index.

        Returns:
            dict, a str-to-int mapping from label name to index.
        """
        class_dict = {'Faces': 0, 'Faces_easy': 1, 'Leopards': 2, 'Motorbikes': 3, 'accordion': 4, 'airplanes': 5,
                      'anchor': 6, 'ant': 7, 'barrel': 8, 'bass': 9, 'beaver': 10, 'binocular': 11, 'bonsai': 12,
                      'brain': 13, 'brontosaurus': 14, 'buddha': 15, 'butterfly': 16, 'camera': 17, 'cannon': 18,
                      'car_side': 19, 'ceiling_fan': 20, 'cellphone': 21, 'chair': 22, 'chandelier': 23,
                      'cougar_body': 24, 'cougar_face': 25, 'crab': 26, 'crayfish': 27, 'crocodile': 28,
                      'crocodile_head': 29, 'cup': 30, 'dalmatian': 31, 'dollar_bill': 32, 'dolphin': 33,
                      'dragonfly': 34, 'electric_guitar': 35, 'elephant': 36, 'emu': 37, 'euphonium': 38, 'ewer': 39,
                      'ferry': 40, 'flamingo': 41, 'flamingo_head': 42, 'garfield': 43, 'gerenuk': 44, 'gramophone': 45,
                      'grand_piano': 46, 'hawksbill': 47, 'headphone': 48, 'hedgehog': 49, 'helicopter': 50, 'ibis': 51,
                      'inline_skate': 52, 'joshua_tree': 53, 'kangaroo': 54, 'ketch': 55, 'lamp': 56, 'laptop': 57,
                      'llama': 58, 'lobster': 59, 'lotus': 60, 'mandolin': 61, 'mayfly': 62, 'menorah': 63,
                      'metronome': 64, 'minaret': 65, 'nautilus': 66, 'octopus': 67, 'okapi': 68, 'pagoda': 69,
                      'panda': 70, 'pigeon': 71, 'pizza': 72, 'platypus': 73, 'pyramid': 74, 'revolver': 75,
                      'rhino': 76, 'rooster': 77, 'saxophone': 78, 'schooner': 79, 'scissors': 80, 'scorpion': 81,
                      'sea_horse': 82, 'snoopy': 83, 'soccer_ball': 84, 'stapler': 85, 'starfish': 86,
                      'stegosaurus': 87, 'stop_sign': 88, 'strawberry': 89, 'sunflower': 90, 'tick': 91,
                      'trilobite': 92, 'umbrella': 93, 'watch': 94, 'water_lilly': 95, 'wheelchair': 96, 'wild_cat': 97,
                      'windsor_chair': 98, 'wrench': 99, 'yin_yang': 100}
        return class_dict


class Caltech256Dataset(MappableDataset):
    """
    A source dataset that reads and parses Caltech256 dataset.

    The generated dataset has two columns: :py:obj:`[image, label]`.
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`label` is of the uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        num_samples (int, optional): The number of images to be included in the dataset
            (default=None, all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, set in the config).
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset
            (default=None, expected order behavior shown in the table).
        decode (bool, optional): Whether or not to decode the images after reading (default=False).
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided
            into (default=None). When this argument is specified, `num_samples` reflects
            the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_dir does not contain data files.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Note:
        - This dataset can take in a `sampler`. `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> caltech256_dataset_dir = "/path/to/caltech256_dataset_directory"
        >>>
        >>> # 1) Read all samples (image files) in caltech256_dataset_dir with 8 threads
        >>> dataset = ds.Caltech256Dataset(dataset_dir=caltech256_dataset_dir, num_parallel_workers=8)

    About Caltech256Dataset:

    Caltech-256 is an object recognition dataset containing 30,607 real-world images, of different sizes,
    spanning 257 classes (256 object classes and an additional clutter class).
    Each class is represented by at least 80 images. The dataset is a superset of the Caltech-101 dataset.

    .. code-block::

        .
        └── caltech256_dataset_directory
             ├── 001.ak47
             │    ├── 001_0001.jpg
             │    ├── 001_0002.jpg
             │    ...
             ├── 002.american-flag
             │    ├── 002_0001.jpg
             │    ├── 002_0002.jpg
             │    ...
             ├── 003.backpack
             │    ├── 003_0001.jpg
             │    ├── 003_0002.jpg
             │    ...
             ├── ...

    Citation:

    .. code-block::

        @article{griffin2007caltech,
        title     = {Caltech-256 object category dataset},
        added-at  = {2021-01-21T02:54:42.000+0100},
        author    = {Griffin, Gregory and Holub, Alex and Perona, Pietro},
        biburl    = {https://www.bibsonomy.org/bibtex/21f746f23ff0307826cca3e3be45f8de7/s364315},
        interhash = {bfe1e648c1778c04baa60f23d1223375},
        intrahash = {1f746f23ff0307826cca3e3be45f8de7},
        publisher = {California Institute of Technology},
        timestamp = {2021-01-21T02:54:42.000+0100},
        year      = {2007}
        }
    """

    @check_caltech256_dataset
    def __init__(self, dataset_dir, num_samples=None, num_parallel_workers=None, shuffle=None, decode=False,
                 sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.decode = replace_none(decode, False)

    def parse(self, children=None):
        return cde.Caltech256Node(self.dataset_dir, self.decode, self.sampler)


class CocoDataset(MappableDataset):
    """
    A source dataset for reading and parsing COCO dataset.

    CocoDataset supports four kinds of tasks, which are Object Detection, Keypoint Detection, Stuff Segmentation and
    Panoptic Segmentation of 2017 Train/Val/Test dataset.

    The generated dataset with different task setting has different output columns:

    - task = :py:obj:`Detection`, output columns: :py:obj:`[image, dtype=uint8]`, :py:obj:`[bbox, dtype=float32]`, \
        :py:obj:`[category_id, dtype=uint32]`, :py:obj:`[iscrowd, dtype=uint32]`.
    - task = :py:obj:`Stuff`, output columns: :py:obj:`[image, dtype=uint8]`, :py:obj:`[segmentation,dtype=float32]`, \
        :py:obj:`[iscrowd,dtype=uint32]`.
    - task = :py:obj:`Keypoint`, output columns: :py:obj:`[image, dtype=uint8]`, \
        :py:obj:`[keypoints, dtype=float32]`, :py:obj:`[num_keypoints, dtype=uint32]`.
    - task = :py:obj:`Panoptic`, output columns: :py:obj:`[image, dtype=uint8]`, :py:obj:`[bbox, dtype=float32]`, \
        :py:obj:`[category_id, dtype=uint32]`, :py:obj:`[iscrowd, dtype=uint32]`, :py:obj:`[area, dtype=uint32]`.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        annotation_file (str): Path to the annotation JSON file.
        task (str, optional): Set the task type for reading COCO data. Supported task types:
            `Detection`, `Stuff`, `Panoptic` and `Keypoint` (default= `Detection`).
        num_samples (int, optional): The number of images to be included in the dataset
            (default=None, all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the configuration file).
        shuffle (bool, optional): Whether to perform shuffle on the dataset (default=None, expected
            order behavior shown in the table).
        decode (bool, optional): Decode the images after reading (default=False).
        sampler (Sampler, optional): Object used to choose samples from the dataset
            (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided
            into (default=None). When this argument is specified, `num_samples` reflects
            the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).
        extra_metadata(bool, optional): Flag to add extra meta-data to row. If True, an additional column will be
            output at the end :py:obj:`[_meta-filename, dtype=string]` (default=False).

    Raises:
        RuntimeError: If dataset_dir does not contain data files.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        RuntimeError: If parse JSON file failed.
        ValueError: If task is not in [`Detection`, `Stuff`, `Panoptic`, `Keypoint`].
        ValueError: If annotation_file is not exist.
        ValueError: If dataset_dir is not exist.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Note:
        - Column '[_meta-filename, dtype=string]' won't be output unless an explicit rename dataset op is added
          to remove the prefix('_meta-').
        - CocoDataset doesn't support PKSampler.
        - This dataset can take in a `sampler`. `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> coco_dataset_dir = "/path/to/coco_dataset_directory/images"
        >>> coco_annotation_file = "/path/to/coco_dataset_directory/annotation_file"
        >>>
        >>> # 1) Read COCO data for Detection task
        >>> dataset = ds.CocoDataset(dataset_dir=coco_dataset_dir,
        ...                          annotation_file=coco_annotation_file,
        ...                          task='Detection')
        >>>
        >>> # 2) Read COCO data for Stuff task
        >>> dataset = ds.CocoDataset(dataset_dir=coco_dataset_dir,
        ...                          annotation_file=coco_annotation_file,
        ...                          task='Stuff')
        >>>
        >>> # 3) Read COCO data for Panoptic task
        >>> dataset = ds.CocoDataset(dataset_dir=coco_dataset_dir,
        ...                          annotation_file=coco_annotation_file,
        ...                          task='Panoptic')
        >>>
        >>> # 4) Read COCO data for Keypoint task
        >>> dataset = ds.CocoDataset(dataset_dir=coco_dataset_dir,
        ...                          annotation_file=coco_annotation_file,
        ...                          task='Keypoint')
        >>>
        >>> # In COCO dataset, each dictionary has keys "image" and "annotation"

    About COCO dataset:

    COCO(Microsoft Common Objects in Context) is a large-scale object detection, segmentation, and captioning dataset
    with several features: Object segmentation, Recognition in context, Superpixel stuff segmentation,
    330K images (>200K labeled), 1.5 million object instances, 80 object categories, 91 stuff categories,
    5 captions per image, 250,000 people with keypoints. In contrast to the popular ImageNet dataset, COCO has fewer
    categories but more instances in per category.

    You can unzip the original COCO-2017 dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── coco_dataset_directory
             ├── train2017
             │    ├── 000000000009.jpg
             │    ├── 000000000025.jpg
             │    ├── ...
             ├── test2017
             │    ├── 000000000001.jpg
             │    ├── 000000058136.jpg
             │    ├── ...
             ├── val2017
             │    ├── 000000000139.jpg
             │    ├── 000000057027.jpg
             │    ├── ...
             └── annotations
                  ├── captions_train2017.json
                  ├── captions_val2017.json
                  ├── instances_train2017.json
                  ├── instances_val2017.json
                  ├── person_keypoints_train2017.json
                  └── person_keypoints_val2017.json

    Citation:

    .. code-block::

        @article{DBLP:journals/corr/LinMBHPRDZ14,
        author        = {Tsung{-}Yi Lin and Michael Maire and Serge J. Belongie and
                        Lubomir D. Bourdev and  Ross B. Girshick and James Hays and
                        Pietro Perona and Deva Ramanan and Piotr Doll{\'{a}}r and C. Lawrence Zitnick},
        title         = {Microsoft {COCO:} Common Objects in Context},
        journal       = {CoRR},
        volume        = {abs/1405.0312},
        year          = {2014},
        url           = {http://arxiv.org/abs/1405.0312},
        archivePrefix = {arXiv},
        eprint        = {1405.0312},
        timestamp     = {Mon, 13 Aug 2018 16:48:13 +0200},
        biburl        = {https://dblp.org/rec/journals/corr/LinMBHPRDZ14.bib},
        bibsource     = {dblp computer science bibliography, https://dblp.org}
        }
    """

    @check_cocodataset
    def __init__(self, dataset_dir, annotation_file, task="Detection", num_samples=None, num_parallel_workers=None,
                 shuffle=None, decode=False, sampler=None, num_shards=None, shard_id=None, cache=None,
                 extra_metadata=False):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.annotation_file = annotation_file
        self.task = replace_none(task, "Detection")
        self.decode = replace_none(decode, False)
        self.extra_metadata = extra_metadata

    def parse(self, children=None):
        return cde.CocoNode(self.dataset_dir, self.annotation_file, self.task, self.decode, self.sampler,
                            self.extra_metadata)

    def get_class_indexing(self):
        """
        Get the class index.

        Returns:
            dict, a str-to-list<int> mapping from label name to index.

        Examples:
            >>> coco_dataset_dir = "/path/to/coco_dataset_directory/images"
            >>> coco_annotation_file = "/path/to/coco_dataset_directory/annotation_file"
            >>>
            >>> # Read COCO data for Detection task
            >>> dataset = ds.CocoDataset(dataset_dir=coco_dataset_dir,
            ...                          annotation_file=coco_annotation_file,
            ...                          task='Detection')
            >>>
            >>> class_indexing = dataset.get_class_indexing()
        """
        if self.task not in {"Detection", "Panoptic"}:
            raise NotImplementedError("Only 'Detection' and 'Panoptic' support get_class_indexing.")
        if self._class_indexing is None:
            runtime_getter = self._init_tree_getters()
            self._class_indexing = dict(runtime_getter[0].GetClassIndexing())
        return self._class_indexing


class CoNLL2000Dataset(SourceDataset):
    """
    A source dataset that reads and parses CoNLL2000 dataset.

    The generated dataset has three columns: :py:obj:`[word, pos_tag, chunk_tag]`.
    The tensor of column :py:obj:`word` is of the string type.
    The tensor of column :py:obj:`pos_tag` is of the string type.
    The tensor of column :py:obj:`chunk_tag` is of the string type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be `train`, `test`,  or `all`. `train` will read from
            8936 train samples, `test` will read from 2,012 test samples,
            `all` will read from all 1,0948 samples (default=None, all samples).
        num_samples (int, optional): Number of samples (rows) to read (default=None, reads the full dataset).
        shuffle (Union[bool, Shuffle level], optional): Perform reshuffling of the data every epoch
            (default=Shuffle.GLOBAL).
            If shuffle is False, no shuffling will be performed;
            If shuffle is True, the behavior is the same as setting shuffle to be Shuffle.GLOBAL
            Otherwise, there are two levels of shuffling:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, `num_samples` reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_dir does not contain data files.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.

    Examples:
        >>> conll2000_dataset_dir = "/path/to/conll2000_dataset_dir"
        >>> dataset = ds.CoNLL2000Dataset(dataset_files=conll2000_dataset_dir, usage='all')
    """

    @check_conll2000_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, shuffle=Shuffle.GLOBAL, num_shards=None,
                 shard_id=None, num_parallel_workers=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, 'all')

    def parse(self, children=None):
        return cde.CoNLL2000Node(self.dataset_dir, self.usage, self.num_samples, self.shuffle_flag, self.num_shards,
                                 self.shard_id)


class CelebADataset(MappableDataset):
    """
    A source dataset for reading and parsing CelebA dataset.
    Only support to read `list_attr_celeba.txt` currently, which is the attribute annotations of the dataset.

    The generated dataset has two columns: :py:obj:`[image, attr]`.
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`attr` is of the uint32 type and one hot encoded.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        num_parallel_workers (int, optional): Number of workers to read the data (default=None, will use value set in
            the config).
        shuffle (bool, optional): Whether to perform shuffle on the dataset (default=None).
        usage (str, optional): Specify the `train`, `valid`, `test` part or `all` parts of dataset
            (default= `all`, will read all samples).
        sampler (Sampler, optional): Object used to choose samples from the dataset (default=None).
        decode (bool, optional): decode the images after reading (default=False).
        extensions (list[str], optional): List of file extensions to be included in the dataset (default=None).
        num_samples (int, optional): The number of images to be included in the dataset
            (default=None, will include all images).
        num_shards (int, optional): Number of shards that the dataset will be divided
            into (default=None). When this argument is specified, `num_samples` reflects
            the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` (default=None). This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_dir does not contain data files.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Note:
        - This dataset can take in a `sampler`. `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> celeba_dataset_dir = "/path/to/celeba_dataset_directory"
        >>>
        >>> # Read 5 samples from CelebA dataset
        >>> dataset = ds.CelebADataset(dataset_dir=celeba_dataset_dir, usage='train', num_samples=5)
        >>>
        >>> # Note: In celeba dataset, each data dictionary owns keys "image" and "attr"

    About CelebA dataset:

    CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset
    with more than 200K celebrity images, each with 40 attribute annotations.

    The images in this dataset cover large pose variations and background clutter.
    CelebA has large diversities, large quantities, and rich annotations, including

    * 10,177 number of identities,
    * 202,599 number of face images,
    * 5 landmark locations, 40 binary attributes annotations per image.

    The dataset can be employed as the training and test sets for the following computer
    vision tasks: face attribute recognition, face detection, landmark (or facial part)
    localization, and face editing & synthesis.

    Original CelebA dataset structure:

    .. code-block::

        .
        └── CelebA
             ├── README.md
             ├── Img
             │    ├── img_celeba.7z
             │    ├── img_align_celeba_png.7z
             │    └── img_align_celeba.zip
             ├── Eval
             │    └── list_eval_partition.txt
             └── Anno
                  ├── list_landmarks_celeba.txt
                  ├── list_landmarks_align_celeba.txt
                  ├── list_bbox_celeba.txt
                  ├── list_attr_celeba.txt
                  └── identity_CelebA.txt

    You can unzip the dataset files into the following structure and read by MindSpore's API.

    .. code-block::

        .
        └── celeba_dataset_directory
            ├── list_attr_celeba.txt
            ├── 000001.jpg
            ├── 000002.jpg
            ├── 000003.jpg
            ├── ...

    Citation:

    .. code-block::

        @article{DBLP:journals/corr/LiuLWT14,
        author        = {Ziwei Liu and Ping Luo and Xiaogang Wang and Xiaoou Tang},
        title         = {Deep Learning Face Attributes in the Wild},
        journal       = {CoRR},
        volume        = {abs/1411.7766},
        year          = {2014},
        url           = {http://arxiv.org/abs/1411.7766},
        archivePrefix = {arXiv},
        eprint        = {1411.7766},
        timestamp     = {Tue, 10 Dec 2019 15:37:26 +0100},
        biburl        = {https://dblp.org/rec/journals/corr/LiuLWT14.bib},
        bibsource     = {dblp computer science bibliography, https://dblp.org},
        howpublished  = {http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html}
        }
    """

    @check_celebadataset
    def __init__(self, dataset_dir, num_parallel_workers=None, shuffle=None, usage='all', sampler=None, decode=False,
                 extensions=None, num_samples=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.decode = replace_none(decode, False)
        self.extensions = replace_none(extensions, [])
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        if self.usage != "all":
            dataset_dir = os.path.realpath(self.dataset_dir)
            partition_file = os.path.join(dataset_dir, "list_eval_partition.txt")
            if os.path.exists(partition_file) is False:
                raise RuntimeError("Partition file can not be found when usage is not 'all'.")
        return cde.CelebANode(self.dataset_dir, self.usage, self.sampler, self.decode, self.extensions)


class CLUEDataset(SourceDataset, TextBaseDataset):
    """
    A source dataset that reads and parses CLUE datasets.
    Supported CLUE classification tasks: `AFQMC`, `TNEWS`, `IFLYTEK`, `CMNLI`, `WSC` and `CSL`.

    The generated dataset with different task setting has different output columns:

    - task = :py:obj:`AFQMC`
        - usage = :py:obj:`train`, output columns: :py:obj:`[sentence1, dtype=string]`, \
            :py:obj:`[sentence2, dtype=string]`, :py:obj:`[label, dtype=string]`.
        - usage = :py:obj:`test`, output columns: :py:obj:`[id, dtype=uint8]`, \
            :py:obj:`[sentence1, dtype=string]`, :py:obj:`[sentence2, dtype=string]`.
        - usage = :py:obj:`eval`, output columns: :py:obj:`[sentence1, dtype=string]`, \
            :py:obj:`[sentence2, dtype=string]`, :py:obj:`[label, dtype=string]`.

    - task = :py:obj:`TNEWS`
        - usage = :py:obj:`train`, output columns: :py:obj:`[label, dtype=string]`, \
            :py:obj:`[label_des, dtype=string]`, :py:obj:`[sentence, dtype=string]`, :py:obj:`[keywords, dtype=string]`.
        - usage = :py:obj:`test`, output columns: :py:obj:`[label, dtype=string]`, \
            :py:obj:`[label_des, dtype=string]`, :py:obj:`[sentence, dtype=string]`, :py:obj:`[keywords, dtype=string]`.
        - usage = :py:obj:`eval`, output columns: :py:obj:`[label, dtype=string]`, \
            :py:obj:`[label_des, dtype=string]`, :py:obj:`[sentence, dtype=string]`, :py:obj:`[keywords, dtype=string]`.

    - task = :py:obj:`IFLYTEK`
        - usage = :py:obj:`train`, output columns: :py:obj:`[label, dtype=string]`, \
            :py:obj:`[label_des, dtype=string]`, :py:obj:`[sentence, dtype=string]`.
        - usage = :py:obj:`test`, output columns: :py:obj:`[id, dtype=string]`, \
            :py:obj:`[sentence, dtype=string]`.
        - usage = :py:obj:`eval`, output columns: :py:obj:`[label, dtype=string]`, \
            :py:obj:`[label_des, dtype=string]`, :py:obj:`[sentence, dtype=string]`.

    - task = :py:obj:`CMNLI`
        - usage = :py:obj:`train`, output columns: :py:obj:`[sentence1, dtype=string]`, \
            :py:obj:`[sentence2, dtype=string]`, :py:obj:`[label, dtype=string]`.
        - usage = :py:obj:`test`, output columns: :py:obj:`[id, dtype=uint8]`, \
            :py:obj:`[sentence1, dtype=string]`, :py:obj:`[sentence2, dtype=string]`.
        - usage = :py:obj:`eval`, output columns: :py:obj:`[sentence1, dtype=string]`, \
            :py:obj:`[sentence2, dtype=string]`, :py:obj:`[label, dtype=string]`.

    - task = :py:obj:`WSC`
        - usage = :py:obj:`train`, output columns: :py:obj:`[span1_index, dtype=uint8]`, \
            :py:obj:`[span2_index, dtype=uint8]`, :py:obj:`[span1_text, dtype=string]`, \
            :py:obj:`[span2_text, dtype=string]`, :py:obj:`[idx, dtype=uint8]`, \
            :py:obj:`[text, dtype=string]`, :py:obj:`[label, dtype=string]`.
        - usage = :py:obj:`test`, output columns: :py:obj:`[span1_index, dtype=uint8]`, \
            :py:obj:`[span2_index, dtype=uint8]`, :py:obj:`[span1_text, dtype=string]`, \
            :py:obj:`[span2_text, dtype=string]`, :py:obj:`[idx, dtype=uint8]`, :py:obj:`[text, dtype=string]`.
        - usage = :py:obj:`eval`, output columns: :py:obj:`[span1_index, dtype=uint8]`, \
            :py:obj:`[span2_index, dtype=uint8]`, :py:obj:`[span1_text, dtype=string]`, \
            :py:obj:`[span2_text, dtype=string]`, :py:obj:`[idx, dtype=uint8]`, \
            :py:obj:`[text, dtype=string]`, :py:obj:`[label, dtype=string]`.

    - task = :py:obj:`CSL`
        - usage = :py:obj:`train`, output columns: :py:obj:`[id, dtype=uint8]`, \
            :py:obj:`[abst, dtype=string]`, :py:obj:`[keyword, dtype=string]`, :py:obj:`[label, dtype=string]`.
        - usage = :py:obj:`test`, output columns: :py:obj:`[id, dtype=uint8]`, \
            :py:obj:`[abst, dtype=string]`, :py:obj:`[keyword, dtype=string]`.
        - usage = :py:obj:`eval`, output columns: :py:obj:`[id, dtype=uint8]`, \
            :py:obj:`[abst, dtype=string]`, :py:obj:`[keyword, dtype=string]`, :py:obj:`[label, dtype=string]`.

    Args:
        dataset_files (Union[str, list[str]]): String or list of files to be read or glob strings to search for
            a pattern of files. The list will be sorted in a lexicographical order.
        task (str, optional): The kind of task, one of `AFQMC`, `TNEWS`, `IFLYTEK`, `CMNLI`, `WSC` and `CSL`.
            (default=AFQMC).
        usage (str, optional): Specify the `train`, `test` or `eval` part of dataset (default="train").
        num_samples (int, optional): The number of samples to be included in the dataset
            (default=None, will include all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        shuffle (Union[bool, Shuffle level], optional): Perform reshuffling of the data every epoch
            (default=Shuffle.GLOBAL).
            If shuffle is False, no shuffling will be performed;
            If shuffle is True, the behavior is the same as setting shuffle to be Shuffle.GLOBAL
            Otherwise, there are two levels of shuffling:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, `num_samples` reflects the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_files are not valid or do not exist.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.

    Examples:
        >>> clue_dataset_dir = ["/path/to/clue_dataset_file"] # contains 1 or multiple clue files
        >>> dataset = ds.CLUEDataset(dataset_files=clue_dataset_dir, task='AFQMC', usage='train')

    About CLUE dataset:

    CLUE, a Chinese Language Understanding Evaluation benchmark. It contains multiple
    tasks, including single-sentence classification, sentence pair classification, and machine
    reading comprehension.

    You can unzip the dataset files into the following structure and read by MindSpore's API,
    such as afqmc dataset:

    .. code-block::

        .
        └── afqmc_public
             ├── train.json
             ├── test.json
             └── dev.json

    Citation:

    .. code-block::

        @article{CLUEbenchmark,
        title   = {CLUE: A Chinese Language Understanding Evaluation Benchmark},
        author  = {Liang Xu, Xuanwei Zhang, Lu Li, Hai Hu, Chenjie Cao, Weitang Liu, Junyi Li, Yudong Li,
                Kai Sun, Yechen Xu, Yiming Cui, Cong Yu, Qianqian Dong, Yin Tian, Dian Yu, Bo Shi, Jun Zeng,
                Rongzhao Wang, Weijian Xie, Yanting Li, Yina Patterson, Zuoyu Tian, Yiwen Zhang, He Zhou,
                Shaoweihua Liu, Qipeng Zhao, Cong Yue, Xinrui Zhang, Zhengliang Yang, Zhenzhong Lan},
        journal = {arXiv preprint arXiv:2004.05986},
        year    = {2020},
        howpublished = {https://github.com/CLUEbenchmark/CLUE}
        }
    """

    @check_cluedataset
    def __init__(self, dataset_files, task='AFQMC', usage='train', num_samples=None, num_parallel_workers=None,
                 shuffle=Shuffle.GLOBAL, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_files = self._find_files(dataset_files)
        self.usage = replace_none(usage, 'train')
        self.task = replace_none(task, 'AFQMC')

    def parse(self, children=None):
        return cde.CLUENode(self.dataset_files, self.task, self.usage, self.num_samples, self.shuffle_flag,
                            self.num_shards, self.shard_id)


class CSVDataset(SourceDataset, TextBaseDataset):
    """
    A source dataset that reads and parses comma-separated values (CSV) datasets.
    The columns of generated dataset depend on the source CSV files.

    Args:
        dataset_files (Union[str, list[str]]): String or list of files to be read or glob strings to search
            for a pattern of files. The list will be sorted in a lexicographical order.
        field_delim (str, optional): A string that indicates the char delimiter to separate fields (default=',').
        column_defaults (list, optional): List of default values for the CSV field (default=None). Each item
            in the list is either a valid type (float, int, or string). If this is not provided, treats all
            columns as string type.
        column_names (list[str], optional): List of column names of the dataset (default=None). If this
            is not provided, infers the column_names from the first row of CSV file.
        num_samples (int, optional): The number of samples to be included in the dataset
            (default=None, will include all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        shuffle (Union[bool, Shuffle level], optional): Perform reshuffling of the data every epoch
            (default=Shuffle.GLOBAL).
            If shuffle is False, no shuffling will be performed;
            If shuffle is True, the behavior is the same as setting shuffle to be Shuffle.GLOBAL
            Otherwise, there are two levels of shuffling:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, `num_samples` reflects the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_files are not valid or do not exist.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.

    Examples:
        >>> csv_dataset_dir = ["/path/to/csv_dataset_file"] # contains 1 or multiple csv files
        >>> dataset = ds.CSVDataset(dataset_files=csv_dataset_dir, column_names=['col1', 'col2', 'col3', 'col4'])
    """

    @check_csvdataset
    def __init__(self, dataset_files, field_delim=',', column_defaults=None, column_names=None, num_samples=None,
                 num_parallel_workers=None, shuffle=Shuffle.GLOBAL, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_files = self._find_files(dataset_files)
        self.dataset_files.sort()
        self.field_delim = replace_none(field_delim, ',')
        self.column_defaults = replace_none(column_defaults, [])
        self.column_names = replace_none(column_names, [])

    def parse(self, children=None):
        return cde.CSVNode(self.dataset_files, self.field_delim, self.column_defaults, self.column_names,
                           self.num_samples, self.shuffle_flag, self.num_shards, self.shard_id)


class SBUDataset(MappableDataset):
    """
    A source dataset for reading and parsing the SBU dataset.

    The generated dataset has two columns :py:obj:`[image, caption]`.
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`caption` is of the string type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        decode (bool, optional): Decode the images after reading (default=False).
        num_samples (int, optional): The number of images to be included in the dataset
            (default=None, will read all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, will use value set in the config).
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset
            (default=None, expected order behavior shown in the table).
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, `num_samples` reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` (default=None). This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_dir does not contain data files.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Note:
        - This dataset can take in a sampler. 'sampler' and 'shuffle' are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using 'sampler' and 'shuffle'
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter 'sampler'
         - Parameter 'shuffle'
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> sbu_dataset_dir = "/path/to/sbu_dataset_directory"
        >>> # Read 3 samples from SBU dataset
        >>> dataset = ds.SBUDataset(dataset_dir=sbu_dataset_dir, num_samples=3)

    About SBU dataset:

    SBU dataset is a large captioned photo collection.
    It contains one million images with associated visually relevant captions.

    You should manually download the images using official download.m by replacing 'urls{i}(24, end)' with
    'urls{i}(24:1:end)' and keep the directory as below.

    .. code-block::

        .
        └─ dataset_dir
           ├── SBU_captioned_photo_dataset_captions.txt
           ├── SBU_captioned_photo_dataset_urls.txt
           └── sbu_images
               ├── m_3326_3596303505_3ce4c20529.jpg
               ├── ......
               └── m_2522_4182181099_c3c23ab1cc.jpg

    Citation:

    .. code-block::

        @inproceedings{Ordonez:2011:im2text,
          Author    = {Vicente Ordonez and Girish Kulkarni and Tamara L. Berg},
          Title     = {Im2Text: Describing Images Using 1 Million Captioned Photographs},
          Booktitle = {Neural Information Processing Systems ({NIPS})},
          Year      = {2011},
        }
    """

    @check_sbu_dataset
    def __init__(self, dataset_dir, num_samples=None, num_parallel_workers=None, shuffle=None, decode=False,
                 sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.decode = replace_none(decode, False)

    def parse(self, children=None):
        return cde.SBUNode(self.dataset_dir, self.decode, self.sampler)


class SogouNewsDataset(SourceDataset):
    """
    A source dataset that reads and parses Sogou News dataset.

    The generated dataset has three columns: :py:obj:`[index, title, content]`.
    The tensor of column :py:obj:`index` is of the string type.
    The tensor of column :py:obj:`title` is of the string type.
    The tensor of column :py:obj:`content` is of the string type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be `train`, `test` or `all` .
            `train` will read from 450,000 train samples, `test` will read from 60,000 test samples,
            `all` will read from all 510,000 samples (default=None, all samples).
        num_samples (int, optional): Number of samples (rows) to read (default=None, read all samples).
        shuffle (Union[bool, Shuffle level], optional): Perform reshuffling of the data every epoch
            (default=Shuffle.GLOBAL).
            If shuffle is False, no shuffling will be performed;
            If shuffle is True, the behavior is the same as setting shuffle to be Shuffle.GLOBAL
            Otherwise, there are two levels of shuffling:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.
        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, `num_samples` reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_dir does not contain data files.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.

    Examples:
        >>> sogou_news_dataset_dir = "/path/to/sogou_news_dataset_dir"
        >>> dataset = ds.SogouNewsDataset(dataset_files=sogou_news_dataset_dir, usage='all')

    About SogouNews Dataset:

    SogouNews dataset includes 3 columns, corresponding to class index (1 to 5), title and content. The title and
    content are escaped using double quotes ("), and any internal double quote is escaped by 2 double quotes ("").
    New lines are escaped by a backslash followed with an "n" character, that is "\n".

    You can unzip the dataset files into the following structure and read by MindSpore's API:

    .. code-block::

        .
        └── sogou_news_dir
             ├── classes.txt
             ├── readme.txt
             ├── test.csv
             └── train.csv

    Citation:

    .. code-block::

        @misc{zhang2015characterlevel,
            title={Character-level Convolutional Networks for Text Classification},
            author={Xiang Zhang and Junbo Zhao and Yann LeCun},
            year={2015},
            eprint={1509.01626},
            archivePrefix={arXiv},
            primaryClass={cs.LG}
        }
    """

    @check_sogou_news_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, shuffle=Shuffle.GLOBAL, num_shards=None,
                 shard_id=None, num_parallel_workers=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, 'all')

    def parse(self, children=None):
        return cde.SogouNewsNode(self.dataset_dir, self.usage, self.num_samples, self.shuffle_flag,
                                 self.num_shards, self.shard_id)


class _Flowers102Dataset:
    """
    Mainly for loading Flowers102 Dataset, and return one row each time.
    """

    def __init__(self, dataset_dir, task, usage, decode):
        self.dataset_dir = os.path.realpath(dataset_dir)
        self.task = task
        self.usage = usage
        self.decode = decode

        if self.task == "Classification":
            self.column_names = ["image", "label"]
        else:
            self.column_names = ["image", "segmentation", "label"]

        labels_path = os.path.join(self.dataset_dir, "imagelabels.mat")
        setid_path = os.path.join(self.dataset_dir, "setid.mat")
        # minus one to transform 1~102 to 0 ~ 101
        self.labels = (loadmat(labels_path)["labels"][0] - 1).astype(np.uint32)
        self.setid = loadmat(setid_path)

        if self.usage == 'train':
            self.indices = self.setid["trnid"][0].tolist()
        elif self.usage == 'test':
            self.indices = self.setid["tstid"][0].tolist()
        elif self.usage == 'valid':
            self.indices = self.setid["valid"][0].tolist()
        elif self.usage == 'all':
            self.indices = self.setid["trnid"][0].tolist()
            self.indices += self.setid["tstid"][0].tolist()
            self.indices += self.setid["valid"][0].tolist()
        else:
            raise ValueError("Input usage is not within the valid set of ['train', 'valid', 'test', 'all'].")

    def __getitem__(self, index):
        # range: 1 ~ 8189
        image_path = os.path.join(self.dataset_dir, "jpg", "image_" + str(self.indices[index]).zfill(5) + ".jpg")
        if not os.path.exists(image_path):
            raise RuntimeError("Can not find image file: " + image_path)

        if self.decode is True:
            image = np.asarray(Image.open(image_path).convert("RGB"))
        else:
            image = np.fromfile(image_path, dtype=np.uint8)

        label = self.labels[self.indices[index] - 1]

        if self.task == "Segmentation":
            segmentation_path = \
                os.path.join(self.dataset_dir, "segmim", "segmim_" + str(self.indices[index]).zfill(5) + ".jpg")
            if not os.path.exists(segmentation_path):
                raise RuntimeError("Can not find segmentation file: " + segmentation_path)
            if self.decode is True:
                segmentation = np.asarray(Image.open(segmentation_path).convert("RGB"))
            else:
                segmentation = np.fromfile(segmentation_path, dtype=np.uint8)
            return image, segmentation, label

        return image, label

    def __len__(self):
        return len(self.indices)


class Flowers102Dataset(GeneratorDataset):
    """
    A source dataset for reading and parsing Flowers102 dataset.

    The generated dataset has two columns :py:obj:`[image, label]` or three :py:obj:`[image, segmentation, label]`.
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`segmentation` is of the uint8 type.
    The tensor of column :py:obj:`label` is a scalar or a tensor of the uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        task (str): Specify the 'Classification' or 'Segmentation' task (default='Classification').
        usage (str): Specify the 'train', 'valid', 'test' part or 'all' parts of dataset
            (default='all', will read all samples).
        num_samples (int, optional): The number of samples to be included in the dataset (default=None, all images).
        num_parallel_workers (int, optional): Number of subprocesses used to fetch the dataset in parallel (default=1).
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Random accessible input is required.
            (default=None, expected order behavior shown in the table).
        decode (bool, optional): Whether or not to decode the images and segmentations after reading (default=False).
        sampler (Union[Sampler, Iterable], optional): Object used to choose samples from the dataset. Random accessible
            input is required (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            Random accessible input is required. When this argument is specified, 'num_samples' reflects the max
            sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This argument must be specified only
            when num_shards is also specified. Random accessible input is required.

    Raises:
        RuntimeError: If dataset_dir does not contain data files.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Note:
        - This dataset can take in a sampler. 'sampler' and 'shuffle' are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using 'sampler' and 'shuffle'
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter 'sampler'
         - Parameter 'shuffle'
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> flowers102_dataset_dir = "/path/to/flowers102_dataset_directory"
        >>> dataset = ds.Flowers102Dataset(dataset_dir=flowers102_dataset_dir,
        ...                                task="Classification",
        ...                                usage="all",
        ...                                decode=True)

    About Flowers102 dataset:

    Flowers102 dataset consists of 102 flower categories.
    The flowers commonly occur in the United Kingdom.
    Each class consists of between 40 and 258 images.

    Here is the original Flowers102 dataset structure.
    You can unzip the dataset files into this directory structure and read by MindSpore's API.

    .. code-block::
        .
        └── flowes102_dataset_dir
             ├── imagelabels.mat
             ├── setid.mat
             ├── jpg
                  ├── image_00001.jpg
                  ├── image_00002.jpg
                  ├── ...
             ├── segmim
                  ├── segmim_00001.jpg
                  ├── segmim_00002.jpg
                  ├── ...

    Citation:

    .. code-block::

        @InProceedings{Nilsback08,
          author       = "Maria-Elena Nilsback and Andrew Zisserman",
          title        = "Automated Flower Classification over a Large Number of Classes",
          booktitle    = "Indian Conference on Computer Vision, Graphics and Image Processing",
          month        = "Dec",
          year         = "2008",
        }
    """

    @check_flowers102dataset
    def __init__(self, dataset_dir, task="Classification", usage="all", num_samples=None, num_parallel_workers=1,
                 shuffle=None, decode=False, sampler=None, num_shards=None, shard_id=None):
        self.dataset_dir = os.path.realpath(dataset_dir)
        self.task = replace_none(task, "Classification")
        self.usage = replace_none(usage, "all")
        self.decode = replace_none(decode, False)
        dataset = _Flowers102Dataset(self.dataset_dir, self.task, self.usage, self.decode)
        super().__init__(dataset, column_names=dataset.column_names, num_samples=num_samples,
                         num_parallel_workers=num_parallel_workers, shuffle=shuffle, sampler=sampler,
                         num_shards=num_shards, shard_id=shard_id)

    def get_class_indexing(self):
        """
        Get the class index.

        Returns:
            dict, a str-to-int mapping from label name to index.
        """
        class_names = [
            "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
            "sweet pea", "english marigold", "tiger lily", "moon orchid",
            "bird of paradise", "monkshood", "globe thistle", "snapdragon",
            "colt's foot", "king protea", "spear thistle", "yellow iris",
            "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
            "giant white arum lily", "fire lily", "pincushion flower", "fritillary",
            "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers",
            "stemless gentian", "artichoke", "sweet william", "carnation",
            "garden phlox", "love in the mist", "mexican aster", "alpine sea holly",
            "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip",
            "lenten rose", "barbeton daisy", "daffodil", "sword lily", "poinsettia",
            "bolero deep blue", "wallflower", "marigold", "buttercup", "oxeye daisy",
            "common dandelion", "petunia", "wild pansy", "primula", "sunflower",
            "pelargonium", "bishop of llandaff", "gaura", "geranium", "orange dahlia",
            "pink-yellow dahlia?", "cautleya spicata", "japanese anemone",
            "black-eyed susan", "silverbush", "californian poppy", "osteospermum",
            "spring crocus", "bearded iris", "windflower", "tree poppy", "gazania",
            "azalea", "water lily", "rose", "thorn apple", "morning glory",
            "passion flower", "lotus", "toad lily", "anthurium", "frangipani",
            "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow",
            "magnolia", "cyclamen", "watercress", "canna lily", "hippeastrum",
            "bee balm", "ball moss", "foxglove", "bougainvillea", "camellia", "mallow",
            "mexican petunia", "bromelia", "blanket flower", "trumpet creeper",
            "blackberry lily"
        ]

        class_dict = {}
        for i, class_name in enumerate(class_names):
            class_dict[class_name] = i

        return class_dict


class LJSpeechDataset(MappableDataset):
    """
    A source dataset for reading and parsing LJSpeech dataset.

    The generated dataset has four columns :py:obj:`[waveform, sample_rate, transcription, normalized_transcript]`.
    The tensor of column :py:obj:`waveform` is a tensor of the float32 type.
    The tensor of column :py:obj:`sample_rate` is a scalar of the int32 type.
    The tensor of column :py:obj:`transcription` is a scalar of the string type.
    The tensor of column :py:obj:`normalized_transcript` is a scalar of the string type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        num_samples (int, optional): The number of audios to be included in the dataset
            (default=None, all audios).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        shuffle (bool, optional): Whether to perform shuffle on the dataset (default=None, expected
            order behavior shown in the table).
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided
            into (default=None). When this argument is specified, `num_samples` reflects
            the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_dir does not contain data files.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Note:
        - This dataset can take in a `sampler`. `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> lj_speech_dataset_dir = "/path/to/lj_speech_dataset_directory"
        >>>
        >>> # 1) Get all samples from LJSPEECH dataset in sequence
        >>> dataset = ds.LJSpeechDataset(dataset_dir=lj_speech_dataset_dir, shuffle=False)
        >>>
        >>> # 2) Randomly select 350 samples from LJSPEECH dataset
        >>> dataset = ds.LJSpeechDataset(dataset_dir=lj_speech_dataset_dir, num_samples=350, shuffle=True)
        >>>
        >>> # 3) Get samples from LJSPEECH dataset for shard 0 in a 2-way distributed training
        >>> dataset = ds.LJSpeechDataset(dataset_dir=lj_speech_dataset_dir, num_shards=2, shard_id=0)
        >>>
        >>> # In LJSPEECH dataset, each dictionary has keys "waveform", "sample_rate", "transcription"
        >>> # and "normalized_transcript"

    About LJSPEECH dataset:

    This is a public domain speech dataset consisting of 13,100 short audio clips of a single speaker
    reading passages from 7 non-fiction books. A transcription is provided for each clip.
    Clips vary in length from 1 to 10 seconds and have a total length of approximately 24 hours.

    The texts were published between 1884 and 1964, and are in the public domain.
    The audio was recorded in 2016-17 by the LibriVox project and is also in the public domain.

    Here is the original LJSPEECH dataset structure.
    You can unzip the dataset files into the following directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── LJSpeech-1.1
            ├── README
            ├── metadata.csv
            └── wavs
                ├── LJ001-0001.wav
                ├── LJ001-0002.wav
                ├── LJ001-0003.wav
                ├── LJ001-0004.wav
                ├── LJ001-0005.wav
                ├── LJ001-0006.wav
                ├── LJ001-0007.wav
                ├── LJ001-0008.wav
                ...
                ├── LJ050-0277.wav
                └── LJ050-0278.wav

    Citation:

    .. code-block::

        @misc{lj_speech17,
        author       = {Keith Ito and Linda Johnson},
        title        = {The LJ Speech Dataset},
        howpublished = {url{https://keithito.com/LJ-Speech-Dataset}},
        year         = 2017
        }
    """

    @check_lj_speech_dataset
    def __init__(self, dataset_dir, num_samples=None, num_parallel_workers=None, shuffle=None,
                 sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir

    def parse(self, children=None):
        return cde.LJSpeechNode(self.dataset_dir, self.sampler)


class TextFileDataset(SourceDataset, TextBaseDataset):
    """
    A source dataset that reads and parses datasets stored on disk in text format.
    The generated dataset has one column :py:obj:`[text]` with type string.

    Args:
        dataset_files (Union[str, list[str]]): String or list of files to be read or glob strings to search for a
            pattern of files. The list will be sorted in a lexicographical order.
        num_samples (int, optional): The number of samples to be included in the dataset
            (default=None, will include all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        shuffle (Union[bool, Shuffle level], optional): Perform reshuffling of the data every epoch
            (default=Shuffle.GLOBAL).
            If shuffle is False, no shuffling will be performed;
            If shuffle is True, the behavior is the same as setting shuffle to be Shuffle.GLOBAL
            Otherwise, there are two levels of shuffling:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, `num_samples` reflects the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_files are not valid or do not exist.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.

    Examples:
        >>> text_file_dataset_dir = ["/path/to/text_file_dataset_file"] # contains 1 or multiple text files
        >>> dataset = ds.TextFileDataset(dataset_files=text_file_dataset_dir)
    """

    @check_textfiledataset
    def __init__(self, dataset_files, num_samples=None, num_parallel_workers=None, shuffle=Shuffle.GLOBAL,
                 num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_files = self._find_files(dataset_files)
        self.dataset_files.sort()

    def parse(self, children=None):
        return cde.TextFileNode(self.dataset_files, self.num_samples, self.shuffle_flag, self.num_shards,
                                self.shard_id)


class _NumpySlicesDataset:
    """
    Mainly for dealing with several kinds of formats of Python data, and return one row each time.
    """

    def __init__(self, data, column_list=None):
        self.column_list = None
        # Convert dict data into tuple
        if isinstance(data, dict):
            data = self.process_dict(data)

        if isinstance(data, tuple):
            self.data = ()
            data_len = len(data)
            for i in range(data_len):
                self.data = self.data + (np.array(data[i]),)
        else:
            self.data = (np.array(data),)

        # check whether the data length in each column is equal
        data_len = [len(data_item) for data_item in self.data]
        if data_len[1:] != data_len[:-1]:
            raise ValueError("Data length in each column is not equal.")

        # Init column_name
        if column_list is not None:
            self.column_list = column_list
        elif self.column_list is None:
            self.column_list = []
            column_num = len(self.data)
            for i in range(column_num):
                self.column_list.append("column_" + str(i))

    def __getitem__(self, index):
        data_row = [d[index, ...] for d in self.data]
        data_res = tuple(data_row)
        return data_res

    def __len__(self):
        return len(self.data[0])

    def process_dict(self, input_data):
        """
        Convert the dict like data into tuple format, when input is a tuple of dicts then compose it into a dict first.
        """
        # Convert pandas like dict(has "values" column) into General dict
        data_keys = list(input_data.keys())
        data_col = input_data[data_keys[0]]
        if hasattr(data_col, "values"):
            new_dict = {}
            for key in data_keys:
                item1 = input_data.pop(key)
                new_dict[key] = item1.values
            input_data = new_dict

        # Convert the data in dict into tuple
        data = ()
        keys = list(input_data.keys())
        self.column_list = keys
        for key in keys:
            value = input_data[key]
            data = data + (list(value),)

        return data


class NumpySlicesDataset(GeneratorDataset):
    """
    Creates a dataset with given data slices, mainly for loading Python data into dataset.

    The column names and column types of generated dataset depend on Python data defined by users.

    Args:
        data (Union[list, tuple, dict]) Input of given data. Supported data types include: list, tuple, dict and other
            NumPy formats. Input data will be sliced along the first dimension and generate additional rows, if input is
            list, there will be one column in each row, otherwise there tends to be multi columns. Large data is not
            recommended to be loaded in this way as data is loading into memory.
        column_names (list[str], optional): List of column names of the dataset (default=None). If column_names is not
            provided, the output column names will be named as the keys of dict when the input data is a dict,
            otherwise they will be named like column_0, column_1 ...
        num_samples (int, optional): The number of samples to be included in the dataset (default=None, all samples).
        num_parallel_workers (int, optional): Number of subprocesses used to fetch the dataset in parallel (default=1).
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Random accessible input is required.
            (default=None, expected order behavior shown in the table).
        sampler (Union[Sampler, Iterable], optional): Object used to choose samples from the dataset. Random accessible
            input is required (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            Random accessible input is required. When this argument is specified, `num_samples` reflects the max
            sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This argument must be specified only
            when num_shards is also specified. Random accessible input is required.

    Note:
        - This dataset can take in a `sampler`. `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Raises:
        RuntimeError: If len of column_names does not match output len of data.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Examples:
        >>> # 1) Input data can be a list
        >>> data = [1, 2, 3]
        >>> dataset = ds.NumpySlicesDataset(data=data, column_names=["column_1"])
        >>>
        >>> # 2) Input data can be a dictionary, and column_names will be its keys
        >>> data = {"a": [1, 2], "b": [3, 4]}
        >>> dataset = ds.NumpySlicesDataset(data=data)
        >>>
        >>> # 3) Input data can be a tuple of lists (or NumPy arrays), each tuple element refers to data in each column
        >>> data = ([1, 2], [3, 4], [5, 6])
        >>> dataset = ds.NumpySlicesDataset(data=data, column_names=["column_1", "column_2", "column_3"])
        >>>
        >>> # 4) Load data from CSV file
        >>> import pandas as pd
        >>> df = pd.read_csv(filepath_or_buffer=csv_dataset_dir[0])
        >>> dataset = ds.NumpySlicesDataset(data=dict(df), shuffle=False)
    """

    @check_numpyslicesdataset
    def __init__(self, data, column_names=None, num_samples=None, num_parallel_workers=1, shuffle=None, sampler=None,
                 num_shards=None, shard_id=None):
        dataset = _NumpySlicesDataset(data, column_names)
        super().__init__(dataset, column_names=dataset.column_list, num_samples=num_samples,
                         num_parallel_workers=num_parallel_workers, shuffle=shuffle, sampler=sampler,
                         num_shards=num_shards, shard_id=shard_id)


class _PaddedDataset:
    """
    Mainly for combining false samples provided by users into a dataset.

    Args:
        padded_samples (list(dict)): Data provided by user to be added to the initial Dataset.
    """

    def __init__(self, padded_samples):
        self.column_names = list(padded_samples[0].keys())
        self.padded_samples = padded_samples

    def __getitem__(self, item):
        return (self.padded_samples[item][key] for key in self.column_names)

    def __len__(self):
        return len(self.padded_samples)


class PaddedDataset(GeneratorDataset):
    """
    Creates a dataset with filler data provided by user. Mainly used to add to the original data set
    and assign it to the corresponding shard.

    Args:
        padded_samples (list(dict)): Samples provided by user.

    Raises:
        TypeError: If padded_samples is not an instance of list.
        TypeError: If the element of padded_samples is not an instance of dict.
        ValueError: If the padded_samples is empty.

    Examples:
        >>> import numpy as np
        >>> data = [{'image': np.zeros(1, np.uint8)}, {'image': np.zeros(2, np.uint8)}]
        >>> dataset = ds.PaddedDataset(padded_samples=data)
    """

    @check_paddeddataset
    def __init__(self, padded_samples):
        dataset = _PaddedDataset(padded_samples)
        super().__init__(dataset, column_names=dataset.column_names, num_shards=None, shard_id=None, shuffle=False)
        self._dataset_size = len(dataset.padded_samples)
        self.padded_samples = padded_samples


class EMnistDataset(MappableDataset):
    """
    A source dataset for reading and parsing the EMNIST dataset.

    The generated dataset has two columns :py:obj:`[image, label]`.
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`label` is a scalar of the uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        name (str): Name of splits for this dataset, can be "byclass", "bymerge", "balanced", "letters", "digits"
            or "mnist".
        usage (str, optional): Usage of this dataset, can be "train", "test" or "all".
            (default=None, will read all samples).
        num_samples (int, optional): The number of images to be included in the dataset
            (default=None, will read all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, will use value set in the config).
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset
            (default=None, expected order behavior shown in the table).
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, `num_samples` reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` (default=None). This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Note:
        - This dataset can take in a `sampler`. `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> emnist_dataset_dir = "/path/to/emnist_dataset_directory"
        >>>
        >>> # Read 3 samples from EMNIST dataset
        >>> dataset = ds.EMnistDataset(dataset_dir=emnist_dataset_dir, name="mnist", num_samples=3)
        >>>
        >>> # Note: In emnist_dataset dataset, each dictionary has keys "image" and "label"

    About EMNIST dataset:

    The EMNIST dataset is a set of handwritten character digits derived from the NIST Special
    Database 19 and converted to a 28x28 pixel image format and dataset structure that directly
    matches the MNIST dataset. Further information on the dataset contents and conversion process
    can be found in the paper available at https://arxiv.org/abs/1702.05373v1.

    The numbers of characters and classes of each split of EMNIST are as follows:

    By Class: 814,255 characters and 62 unbalanced classes.
    By Merge: 814,255 characters and 47 unbalanced classes.
    Balanced: 131,600 characters and 47 balanced classes.
    Letters: 145,600 characters and 26 balanced classes.
    Digits: 280,000 characters and 10 balanced classes.
    MNIST: 70,000 characters and 10 balanced classes.

    Here is the original EMNIST dataset structure.
    You can unzip the dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── mnist_dataset_dir
             ├── emnist-mnist-train-images-idx3-ubyte
             ├── emnist-mnist-train-labels-idx1-ubyte
             ├── emnist-mnist-test-images-idx3-ubyte
             ├── emnist-mnist-test-labels-idx1-ubyte
             ├── ...

    Citation:

    .. code-block::

        @article{cohen_afshar_tapson_schaik_2017,
        title        = {EMNIST: Extending MNIST to handwritten letters},
        DOI          = {10.1109/ijcnn.2017.7966217},
        journal      = {2017 International Joint Conference on Neural Networks (IJCNN)},
        author       = {Cohen, Gregory and Afshar, Saeed and Tapson, Jonathan and Schaik, Andre Van},
        year         = {2017},
        howpublished = {https://www.westernsydney.edu.au/icns/reproducible_research/
                        publication_support_materials/emnist}
        }
    """

    @check_emnist_dataset
    def __init__(self, dataset_dir, name, usage=None, num_samples=None, num_parallel_workers=None,
                 shuffle=None, sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.name = name
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.EMnistNode(self.dataset_dir, self.name, self.usage, self.sampler)


class FakeImageDataset(MappableDataset):
    """
    A source dataset for generating fake images.

    The generated dataset has two columns :py:obj:`[image, label]`.
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`label` is a scalar of the uint32 type.

    Args:
        num_images (int, optional): Number of images to generate in the dataset (default=1000).
        image_size (tuple, optional):  Size of the fake image (default=(224, 224, 3)).
        num_classes (int, optional): Number of classes in the dataset (default=10).
        base_seed (int, optional): Offsets the index-based random seed used to generate each image (default=0).
        num_samples (int, optional): The number of images to be included in the dataset
            (default=None, will read all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, will use value set in the config).
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset
            (default=None, expected order behavior shown in the table).
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, `num_samples` reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` (default=None). This
            argument can only be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Note:
        - This dataset can take in a sampler. 'sampler' and 'shuffle' are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using 'sampler' and 'shuffle'
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter 'sampler'
         - Parameter 'shuffle'
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> # Read 3 samples from FakeImage dataset
        >>> dataset = ds.FakeImageDataset(num_images=1000, image_size=(224,224,3),
        ...                               num_classes=10, base_seed=0, num_samples=3)
        >>>
        >>> # Note: In FakeImage dataset, each dictionary has keys "image" and "label"
    """

    @check_fake_image_dataset
    def __init__(self, num_images=1000, image_size=(224, 224, 3), num_classes=10, base_seed=0, num_samples=None,
                 num_parallel_workers=None, shuffle=None, sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.num_images = num_images
        self.image_size = image_size
        self.num_classes = num_classes
        self.base_seed = base_seed

    def parse(self, children=None):
        return cde.FakeImageNode(self.num_images, self.image_size, self.num_classes, self.base_seed, self.sampler)


class FlickrDataset(MappableDataset):
    """
    A source dataset for reading and parsing Flickr8k and Flickr30k dataset.

    The generated dataset has two columns :py:obj:`[image, annotation]`.
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`annotation` is a tensor which contains 5 annotations string,
    such as ["a", "b", "c", "d", "e"].

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        annotation_file (str): Path to the root directory that contains the annotation.
        num_samples (int, optional): The number of images to be included in the dataset.
            (default=None, all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        shuffle (bool, optional): Whether to perform shuffle on the dataset (default=None, expected
            order behavior shown in the table).
        decode (bool, optional): Decode the images after reading (default=False).
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided
            into (default=None). When this argument is specified, `num_samples` reflects
            the max sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_dir is not valid or does not contain data files.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If dataset_dir is not exist.
        ValueError: If annotation_file is not exist.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Note:
        - This dataset can take in a `sampler`. `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> flickr_dataset_dir = "/path/to/flickr_dataset_directory"
        >>> annotation_file = "/path/to/flickr_annotation_file"
        >>>
        >>> # 1) Get all samples from FLICKR dataset in sequence
        >>> dataset = ds.FlickrDataset(dataset_dir=flickr_dataset_dir,
        ...                            annotation_file=annotation_file,
        ...                            shuffle=False)
        >>>
        >>> # 2) Randomly select 350 samples from FLICKR dataset
        >>> dataset = ds.FlickrDataset(dataset_dir=flickr_dataset_dir,
        ...                            annotation_file=annotation_file,
        ...                            num_samples=350,
        ...                            shuffle=True)
        >>>
        >>> # 3) Get samples from FLICKR dataset for shard 0 in a 2-way distributed training
        >>> dataset = ds.FlickrDataset(dataset_dir=flickr_dataset_dir,
        ...                            annotation_file=annotation_file,
        ...                            num_shards=2,
        ...                            shard_id=0)
        >>>
        >>> # In FLICKR dataset, each dictionary has keys "image" and "annotation"

    About Flickr8k dataset:

    The Flickr8k dataset consists of 8092 colour images. There are 40460 annotations in the Flickr8k.token.txt,
    each image has 5 annotations.

    You can unzip the dataset files into the following directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── Flickr8k
             ├── Flickr8k_Dataset
             │    ├── 1000268201_693b08cb0e.jpg
             │    ├── 1001773457_577c3a7d70.jpg
             │    ├── ...
             └── Flickr8k.token.txt

    Citation:

    .. code-block::

        @article{DBLP:journals/jair/HodoshYH13,
        author    = {Micah Hodosh and Peter Young and Julia Hockenmaier},
        title     = {Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics},
        journal   = {J. Artif. Intell. Res.},
        volume    = {47},
        pages     = {853--899},
        year      = {2013},
        url       = {https://doi.org/10.1613/jair.3994},
        doi       = {10.1613/jair.3994},
        timestamp = {Mon, 21 Jan 2019 15:01:17 +0100},
        biburl    = {https://dblp.org/rec/journals/jair/HodoshYH13.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org}
        }

    About Flickr30k dataset:

    The Flickr30k dataset consists of 31783 colour images. There are 158915 annotations in
    the results_20130124.token, each image has 5 annotations.

    You can unzip the dataset files into the following directory structure and read by MindSpore's API.

    Citation:

    .. code-block::

        .
        └── Flickr30k
             ├── flickr30k-images
             │    ├── 1000092795.jpg
             │    ├── 10002456.jpg
             │    ├── ...
             └── results_20130124.token

    .. code-block::

        @article{DBLP:journals/tacl/YoungLHH14,
        author    = {Peter Young and Alice Lai and Micah Hodosh and Julia Hockenmaier},
        title     = {From image descriptions to visual denotations: New similarity metrics
                     for semantic inference over event descriptions},
        journal   = {Trans. Assoc. Comput. Linguistics},
        volume    = {2},
        pages     = {67--78},
        year      = {2014},
        url       = {https://tacl2013.cs.columbia.edu/ojs/index.php/tacl/article/view/229},
        timestamp = {Wed, 17 Feb 2021 21:55:25 +0100},
        biburl    = {https://dblp.org/rec/journals/tacl/YoungLHH14.bib},
        bibsource = {dblp computer science bibliography, https://dblp.org}
        }
    """

    @check_flickr_dataset
    def __init__(self, dataset_dir, annotation_file, num_samples=None, num_parallel_workers=None, shuffle=None,
                 decode=None, sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.annotation_file = annotation_file
        self.decode = replace_none(decode, False)

    def parse(self, children=None):
        return cde.FlickrNode(self.dataset_dir, self.annotation_file, self.decode, self.sampler)


class SBDataset(GeneratorDataset):
    """
    A source dataset for reading and parsing Semantic Boundaries Dataset.

    The generated dataset has two columns: :py:obj:`[image, task]`.
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`task` contains 20 images of the uint8 type if `task` is `Boundaries` otherwise
    contains 1 image of the uint8 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        task (str, optional): Acceptable tasks include `Boundaries` or `Segmentation` (default= `Boundaries`).
        usage (str, optional): Acceptable usages include `train`, `val`, `train_noval` and `all` (default= `all`).
        num_samples (int, optional): The number of images to be included in the dataset.
            (default=None, all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        shuffle (bool, optional): Whether to perform shuffle on the dataset (default=None, expected
            order behavior shown in the table).
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided
            into (default=None). When this argument is specified, `num_samples` reflects
            the max sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.

    Raises:
        RuntimeError: If dataset_dir is not valid or does not contain data files.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If dataset_dir is not exist.
        ValueError: If task is not in [`Boundaries`, `Segmentation`].
        ValueError: If usage is not in [`train`, `val`, `train_noval`, `all`].
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Note:
        - This dataset can take in a sampler. `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> sb_dataset_dir = "/path/to/sb_dataset_directory"
        >>>
        >>> # 1) Get all samples from Semantic Boundaries Dataset in sequence
        >>> dataset = ds.SBDataset(dataset_dir=sb_dataset_dir, shuffle=False)
        >>>
        >>> # 2) Randomly select 350 samples from Semantic Boundaries Dataset
        >>> dataset = ds.SBDataset(dataset_dir=sb_dataset_dir, num_samples=350, shuffle=True)
        >>>
        >>> # 3) Get samples from Semantic Boundaries Dataset for shard 0 in a 2-way distributed training
        >>> dataset = ds.SBDataset(dataset_dir=sb_dataset_dir, num_shards=2, shard_id=0)
        >>>
        >>> # In Semantic Boundaries Dataset, each dictionary has keys "image" and "task"

    About Semantic Boundaries Dataset:

    The Semantic Boundaries Dataset consists of 11355 colour images. There are 8498 images' name in the train.txt,
    2857 images' name in the val.txt and 5623 images' name in the train_noval.txt. The category cls/
    contains the Segmentation and Boundaries results of category-level, the category inst/ catains the
    Segmentation and Boundaries results of instance-level.

    You can unzip the dataset files into the following structure and read by MindSpore's API:

    .. code-block::

         .
         └── benchmark_RELEASE
              ├── dataset
              ├── img
              │    ├── 2008_000002.jpg
              │    ├── 2008_000003.jpg
              │    ├── ...
              ├── cls
              │    ├── 2008_000002.mat
              │    ├── 2008_000003.mat
              │    ├── ...
              ├── inst
              │    ├── 2008_000002.mat
              │    ├── 2008_000003.mat
              │    ├── ...
              ├── train.txt
              └── val.txt

    .. code-block::

        @InProceedings{BharathICCV2011,
            author       = "Bharath Hariharan and Pablo Arbelaez and Lubomir Bourdev and
                            Subhransu Maji and Jitendra Malik",
            title        = "Semantic Contours from Inverse Detectors",
            booktitle    = "International Conference on Computer Vision (ICCV)",
            year         = "2011",
    """

    @check_sb_dataset
    def __init__(self, dataset_dir, task='Boundaries', usage='all', num_samples=None, num_parallel_workers=1,
                 shuffle=None, decode=None, sampler=None, num_shards=None, shard_id=None):
        dataset = _SBDataset(dataset_dir, task, usage, decode)
        super().__init__(dataset, column_names=dataset.column_list, num_samples=num_samples,
                         num_parallel_workers=num_parallel_workers, shuffle=shuffle, sampler=sampler,
                         num_shards=num_shards, shard_id=shard_id)


class _SBDataset:
    """
    Dealing with the data file with .mat extension, and return one row in tuple (image, task) each time.
    """

    def __init__(self, dataset_dir, task, usage, decode):
        self.column_list = ['image', 'task']
        self.task = task
        self.images_path = os.path.join(dataset_dir, 'img')
        self.cls_path = os.path.join(dataset_dir, 'cls')
        self._loadmat = loadmat
        self.categories = 20
        self.decode = replace_none(decode, False)

        if usage == "all":
            image_names = []
            for item in ["train", "val"]:
                usage_path = os.path.join(dataset_dir, item + '.txt')
                if not os.path.exists(usage_path):
                    raise FileNotFoundError("SBDataset: {0} not found".format(usage_path))
                with open(usage_path, 'r') as f:
                    image_names += [x.strip() for x in f.readlines()]
        else:
            usage_path = os.path.join(dataset_dir, usage + '.txt')
            if not os.path.exists(usage_path):
                raise FileNotFoundError("SBDataset: {0} not found".format(usage_path))
            with open(usage_path, 'r') as f:
                image_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(self.images_path, i + ".jpg") for i in image_names]
        self.clss = [os.path.join(self.cls_path, i + ".mat") for i in image_names]

        if len(self.images) != len(self.clss):
            raise ValueError("SBDataset: images count not equal to cls count")

        self._get_data = self._get_boundaries_data if self.task == "Boundaries" else self._get_segmentation_data
        self._get_item = self._get_decode_item if self.decode else self._get_undecode_item

    def _get_boundaries_data(self, mat_path):
        mat_data = self._loadmat(mat_path)
        return np.concatenate([np.expand_dims(mat_data['GTcls'][0][self.task][0][i][0].toarray(), axis=0)
                               for i in range(self.categories)], axis=0)

    def _get_segmentation_data(self, mat_path):
        mat_data = self._loadmat(mat_path)
        return Image.fromarray(mat_data['GTcls'][0][self.task][0])

    def _get_decode_item(self, idx):
        return Image.open(self.images[idx]).convert('RGB'), self._get_data(self.clss[idx])

    def _get_undecode_item(self, idx):
        return np.fromfile(self.images[idx], dtype=np.uint8), self._get_data(self.clss[idx])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self._get_item(idx)


class SpeechCommandsDataset(MappableDataset):
    """
    A source dataset for reading and parsing the SpeechCommands dataset.

    The generated dataset has five columns :py:obj:`[waveform, sample_rate, label, speaker_id, utterance_number]`.
    The tensor of column :py:obj:`waveform` is a vector of the float32 type.
    The tensor of column :py:obj:`sample_rate` is a scalar of the int32 type.
    The tensor of column :py:obj:`label` is a scalar of the string type.
    The tensor of column :py:obj:`speaker_id` is a scalar of the string type.
    The tensor of column :py:obj:`utterance_number` is a scalar of the int32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be `train`, `test`, `valid` or `all`. `train`
            will read from 84,843 samples, `test` will read from 11,005 samples, `valid` will read from 9,981
            test samples and `all` will read from all 105,829 samples (default=None, will read all samples).
        num_samples (int, optional): The number of samples to be included in the dataset
            (default=None, will read all samples).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, will use value set in the config).
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset
            (default=None, expected order behavior shown in the table).
        sampler (Sampler, optional): Object used to choose samples from the dataset
            (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, `num_samples` reflects the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` (default=None). This argument can only be specified
            when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_dir does not contain data files.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Note:
        - This dataset can take in a `sampler`. `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> speech_commands_dataset_dir = "/path/to/speech_commands_dataset_directory"
        >>>
        >>> # Read 3 samples from SpeechCommands dataset
        >>> dataset = ds.SpeechCommandsDataset(dataset_dir=speech_commands_dataset_dir, num_samples=3)
        >>>
        >>> # Note: In SpeechCommands dataset, each dictionary has keys "waveform", "sample_rate", "label",
        >>> # "speaker_id" and "utterance_number".

    About SpeechCommands dataset:

    The SpeechCommands is database for limited_vocabulary speech recognition, containing 105,829 audio samples of
    '.wav' format.

    Here is the original SpeechCommands dataset structure.
    You can unzip the dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── speech_commands_dataset_dir
             ├── cat
                  ├── b433eff_nohash_0.wav
                  ├── 5a33edf_nohash_1.wav
                  └──....
             ├── dog
                  ├── b433w2w_nohash_0.wav
                  └──....
             ├── four
             └── ....

    Citation:

    .. code-block::
        @article{2018Speech,
        title={Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition},
        author={Warden, P.},
        year={2018}
        }
    """

    @check_speech_commands_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None,
                 sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.SpeechCommandsNode(self.dataset_dir, self.usage, self.sampler)


class DeserializedDataset(Dataset):
    def __init__(self, input_obj):
        super().__init__()
        self.input_obj = input_obj

    def parse(self, children=None):
        if isinstance(self.input_obj, dict):
            json_str = json.dumps(self.input_obj)
            return cde.Dataset.from_json_string(json_str)
        return cde.Dataset.from_json_file(self.input_obj)


class CityscapesDataset(MappableDataset):
    """
    A source dataset for reading and parsing Cityscapes dataset.

    The generated dataset has two columns :py:obj:`[image, task]`.
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`task` is of the uint8 type if task is not 'polygon' otherwise task is
    a string tensor with serialize json.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str): Acceptable usages include `train`, `test`, `val` or `all` if quality_mode is `fine`
            otherwise `train`, `train_extra`, `val` or `all` (default= `train`).
        quality_mode (str): Acceptable quality_modes include `fine` or `coarse` (default= `fine`).
        task (str): Acceptable tasks include `instance`, `semantic`, `polygon` or `color` (default= `instance`).
        num_samples (int, optional): The number of images to be included in the dataset.
            (default=None, all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        shuffle (bool, optional): Whether to perform shuffle on the dataset (default=None, expected
            order behavior shown in the table).
        decode (bool, optional): Decode the images after reading (default=False).
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided
            into (default=None). When this argument is specified, `num_samples` reflects
            the max sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_dir is invalid or does not contain data files.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If dataset_dir is not exist.
        ValueError: If task is invalid.
        ValueError: If quality_mode is invalid.
        ValueError: If usage is invalid.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Note:
        - This dataset can take in a `sampler`. `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> cityscapes_dataset_dir = "/path/to/cityscapes_dataset_directory"
        >>>
        >>> # 1) Get all samples from Cityscapes dataset in sequence
        >>> dataset = ds.CityscapesDataset(dataset_dir=cityscapes_dataset_dir, task="instance", quality_mode="fine",
        ...                                usage="train", shuffle=False, num_parallel_workers=1)
        >>>
        >>> # 2) Randomly select 350 samples from Cityscapes dataset
        >>> dataset = ds.CityscapesDataset(dataset_dir=cityscapes_dataset_dir, num_samples=350, shuffle=True,
        ...                                num_parallel_workers=1)
        >>>
        >>> # 3) Get samples from Cityscapes dataset for shard 0 in a 2-way distributed training
        >>> dataset = ds.CityscapesDataset(dataset_dir=cityscapes_dataset_dir, num_shards=2, shard_id=0,
        ...                                num_parallel_workers=1)
        >>>
        >>> # In Cityscapes dataset, each dictionary has keys "image" and "task"

    About Cityscapes dataset:

    The Cityscapes dataset consists of 5000 colour images with high quality dense pixel annotations and
    19998 colour images with coarser polygonal annotations in 50 cities. There are 30 classes in this
    dataset and the polygonal annotations include dense semantic segmentation and instance segmentation
    for vehicle and people.

    You can unzip the dataset files into the following directory structure and read by MindSpore's API.

    Taking the quality_mode of `fine` as an example.

    .. code-block::

        .
        └── Cityscapes
             ├── leftImg8bit
             |    ├── train
             |    |    ├── aachen
             |    |    |    ├── aachen_000000_000019_leftImg8bit.png
             |    |    |    ├── aachen_000001_000019_leftImg8bit.png
             |    |    |    ├── ...
             |    |    ├── bochum
             |    |    |    ├── ...
             |    |    ├── ...
             |    ├── test
             |    |    ├── ...
             |    ├── val
             |    |    ├── ...
             └── gtFine
                  ├── train
                  |    ├── aachen
                  |    |    ├── aachen_000000_000019_gtFine_color.png
                  |    |    ├── aachen_000000_000019_gtFine_instanceIds.png
                  |    |    ├── aachen_000000_000019_gtFine_labelIds.png
                  |    |    ├── aachen_000000_000019_gtFine_polygons.json
                  |    |    ├── aachen_000001_000019_gtFine_color.png
                  |    |    ├── aachen_000001_000019_gtFine_instanceIds.png
                  |    |    ├── aachen_000001_000019_gtFine_labelIds.png
                  |    |    ├── aachen_000001_000019_gtFine_polygons.json
                  |    |    ├── ...
                  |    ├── bochum
                  |    |    ├── ...
                  |    ├── ...
                  ├── test
                  |    ├── ...
                  └── val
                       ├── ...

    Citation:

    .. code-block::

        @inproceedings{Cordts2016Cityscapes,
        title       = {The Cityscapes Dataset for Semantic Urban Scene Understanding},
        author      = {Cordts, Marius and Omran, Mohamed and Ramos, Sebastian and Rehfeld, Timo and Enzweiler,
                        Markus and Benenson, Rodrigo and Franke, Uwe and Roth, Stefan and Schiele, Bernt},
        booktitle   = {Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year        = {2016}
        }
    """

    @check_cityscapes_dataset
    def __init__(self, dataset_dir, usage="train", quality_mode="fine", task="instance", num_samples=None,
                 num_parallel_workers=None, shuffle=None, decode=None, sampler=None, num_shards=None,
                 shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.task = task
        self.quality_mode = quality_mode
        self.usage = usage
        self.decode = replace_none(decode, False)

    def parse(self, children=None):
        return cde.CityscapesNode(self.dataset_dir, self.usage, self.quality_mode, self.task, self.decode, self.sampler)


class DBpediaDataset(SourceDataset, TextBaseDataset):
    """
    A source dataset that reads and parses the DBpedia dataset.

    The generated dataset has three columns :py:obj:`[class, title, content]`.
    The tensor of column :py:obj:`class` is of the string type.
    The tensor of column :py:obj:`title` is of the string type.
    The tensor of column :py:obj:`content` is of the string type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be `train`, `test` or `all`.
            `train` will read from 560,000 train samples,
            `test` will read from 70,000 test samples,
            `all` will read from all 630,000 samples (default=None, all samples).
        num_samples (int, optional): The number of samples to be included in the dataset
            (default=None, will include all text).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        shuffle (Union[bool, Shuffle level], optional): Perform reshuffling of the data every epoch
            (default=Shuffle.GLOBAL).
            If shuffle is False, no shuffling will be performed;
            If shuffle is True, the behavior is the same as setting shuffle to be Shuffle.GLOBAL;
            Otherwise, there are two levels of shuffling:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, `num_samples` reflects the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_dir does not contain data files.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Examples:
        >>> dbpedia_dataset_dir = "/path/to/dbpedia_dataset_directory"
        >>>
        >>> # 1) Read 3 samples from DBpedia dataset
        >>> dataset = ds.DBpediaDataset(dataset_dir=dbpedia_dataset_dir, num_samples=3)
        >>>
        >>> # 2) Read train samples from DBpedia dataset
        >>> dataset = ds.DBpediaDataset(dataset_dir=dbpedia_dataset_dir, usage="train")

    About DBpedia dataset:

    The DBpedia dataset consists of 630,000 text samples in 14 classes, there are 560,000 samples in the train.csv
    and 70,000 samples in the test.csv.
    The 14 different classes represent Company, EducationaInstitution, Artist, Athlete, OfficeHolder,
    MeanOfTransportation, Building, NaturalPlace, Village, Animal, Plant, Album, Film, WrittenWork.

    Here is the original DBpedia dataset structure.
    You can unzip the dataset files into this directory structure and read by Mindspore's API.

    .. code-block::

        .
        └── dbpedia_dataset_dir
            ├── train.csv
            ├── test.csv
            ├── classes.txt
            └── readme.txt

    .. code-block::

        @article{DBpedia,
        title   = {DBPedia Ontology Classification Dataset},
        author  = {Jens Lehmann, Robert Isele, Max Jakob, Anja Jentzsch, Dimitris Kontokostas,
                Pablo N. Mendes, Sebastian Hellmann, Mohamed Morsey, Patrick van Kleef,
                    Sören Auer, Christian Bizer},
        year    = {2015},
        howpublished = {http://dbpedia.org}
        }
    """

    @check_dbpedia_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=Shuffle.GLOBAL,
                 num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.DBpediaNode(self.dataset_dir, self.usage, self.num_samples, self.shuffle_flag, self.num_shards,
                               self.shard_id)


class DIV2KDataset(MappableDataset):
    """
    A source dataset for reading and parsing DIV2KDataset dataset.

    The generated dataset has two columns :py:obj:`[hr_image, lr_image]`.
    The tensor of column :py:obj:`hr_image` is of the uint8 type.
    The tensor of column :py:obj:`lr_image` is of the uint8 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str): Acceptable usages include `train`, `valid` or `all` (default= `train`).
        downgrade (str): Acceptable downgrades include `bicubic`, `unknown`, `mild`, `difficult` or
            `wild` (default= `bicubic`).
        scale (int): Acceptable scales include 2, 3, 4 or 8 (default=2).
            When `downgrade` is `bicubic`, scale can be 2, 3, 4, 8.
            When `downgrade` is `unknown`, scale can only be 2, 3, 4.
            When `downgrade` is `mild`, `difficult` or `wild`, scale can only be 4.
        num_samples (int, optional): The number of images to be included in the dataset.
            (default=None, all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        shuffle (bool, optional): Whether to perform shuffle on the dataset (default=None, expected
            order behavior shown in the table).
        decode (bool, optional): Decode the images after reading (default=False).
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided
            into (default=None). When this argument is specified, `num_samples` reflects
            the max sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_dir is invalid or does not contain data files.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If dataset_dir is not exist.
        ValueError: If usage is invalid.
        ValueError: If downgrade is invalid.
        ValueError: If scale is invalid.
        ValueError: If scale equal to 8 and downgrade not equal to `bicubic`.
        ValueError: If downgrade in [`mild`, `difficult`, `wild`] and scale not equal to 4.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Note:
        - This dataset can take in a `sampler`. `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> div2k_dataset_dir = "/path/to/div2k_dataset_directory"
        >>>
        >>> # 1) Get all samples from DIV2K dataset in sequence
        >>> dataset = ds.DIV2KDataset(dataset_dir=div2k_dataset_dir, usage="train", scale=2, downgrade="bicubic",
        ...                           shuffle=False)
        >>>
        >>> # 2) Randomly select 350 samples from DIV2K dataset
        >>> dataset = ds.DIV2KDataset(dataset_dir=div2k_dataset_dir, usage="train", scale=2, downgrade="bicubic",
        ...                           num_samples=350, shuffle=True)
        >>>
        >>> # 3) Get samples from DIV2K dataset for shard 0 in a 2-way distributed training
        >>> dataset = ds.DIV2KDataset(dataset_dir=div2k_dataset_dir, usage="train", scale=2, downgrade="bicubic",
        ...                           num_shards=2, shard_id=0)
        >>>
        >>> # In DIV2K dataset, each dictionary has keys "hr_image" and "lr_image"

    About DIV2K dataset:

    The DIV2K dataset consists of 1000 2K resolution images, among which 800 images are for training, 100 images
    are for validation and 100 images are for testing. NTIRE 2017 and NTIRE 2018 include only training dataset
    and validation dataset.

    You can unzip the dataset files into the following directory structure and read by MindSpore's API.

    Take the training set as an example.

    .. code-block::

        .
        └── DIV2K
             ├── DIV2K_train_HR
             |    ├── 0001.png
             |    ├── 0002.png
             |    ├── ...
             ├── DIV2K_train_LR_bicubic
             |    ├── X2
             |    |    ├── 0001x2.png
             |    |    ├── 0002x2.png
             |    |    ├── ...
             |    ├── X3
             |    |    ├── 0001x3.png
             |    |    ├── 0002x3.png
             |    |    ├── ...
             |    └── X4
             |         ├── 0001x4.png
             |         ├── 0002x4.png
             |         ├── ...
             ├── DIV2K_train_LR_unknown
             |    ├── X2
             |    |    ├── 0001x2.png
             |    |    ├── 0002x2.png
             |    |    ├── ...
             |    ├── X3
             |    |    ├── 0001x3.png
             |    |    ├── 0002x3.png
             |    |    ├── ...
             |    └── X4
             |         ├── 0001x4.png
             |         ├── 0002x4.png
             |         ├── ...
             ├── DIV2K_train_LR_mild
             |    ├── 0001x4m.png
             |    ├── 0002x4m.png
             |    ├── ...
             ├── DIV2K_train_LR_difficult
             |    ├── 0001x4d.png
             |    ├── 0002x4d.png
             |    ├── ...
             ├── DIV2K_train_LR_wild
             |    ├── 0001x4w.png
             |    ├── 0002x4w.png
             |    ├── ...
             └── DIV2K_train_LR_x8
                  ├── 0001x8.png
                  ├── 0002x8.png
                  ├── ...
    Citation:

    .. code-block::

        @InProceedings{Agustsson_2017_CVPR_Workshops,
        author    = {Agustsson, Eirikur and Timofte, Radu},
        title     = {NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study},
        booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
        url       = "http://www.vision.ee.ethz.ch/~timofter/publications/Agustsson-CVPRW-2017.pdf",
        month     = {July},
        year      = {2017}
        }
    """

    @check_div2k_dataset
    def __init__(self, dataset_dir, usage="train", downgrade="bicubic", scale=2, num_samples=None,
                 num_parallel_workers=None, shuffle=None, decode=None, sampler=None, num_shards=None,
                 shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.usage = usage
        self.scale = scale
        self.downgrade = downgrade
        self.decode = replace_none(decode, False)

    def parse(self, children=None):
        return cde.DIV2KNode(self.dataset_dir, self.usage, self.downgrade, self.scale, self.decode, self.sampler)


class WIDERFaceDataset(MappableDataset):
    """
    A source dataset for reading and parsing WIDERFace dataset.

    When usage is "train", "valid" or "all", the generated dataset has eight columns ["image", "bbox", "blur",
    "expression", "illumination", "occlusion", "pose", "invalid"]. When usage is "test", it only has one column
    ["image"].
    The tensor of column :py:obj:`image` is a vector of the uint8 type.
    The tensor of column :py:obj:`bbox` is a scalar of the uint32 type.
    The tensor of column :py:obj:`blur` is a scalar of the uint32 type.
    The tensor of column :py:obj:`expression` is a scalar of the uint32 type.
    The tensor of column :py:obj:`illumination` is a scalar of the uint32 type.
    The tensor of column :py:obj:`occlusion` is a scalar of the uint32 type.
    The tensor of column :py:obj:`pose` is a scalar of the uint32 type.
    The tensor of column :py:obj:`invalid` is a scalar of the uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be `train`, `test`, `valid` or `all`. `train` will read
            from 12,880 samples, `test` will read from 16,097 samples, `valid` will read from 3,226 test samples
            and `all` will read all `train` and `valid` samples (default=None, will be set to `all`).
        num_samples (int, optional): The number of images to be included in the dataset
            (default=None, will read all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, will use value set in the config).
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset
            (default=None, expected order behavior shown in the table).
        decode (bool, optional): Decode the images after reading (default=False).
        sampler (Sampler, optional): Object used to choose samples from the dataset
            (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, `num_samples` reflects the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` (default=None). This argument can only be specified
            when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_dir does not contain data files.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).
        ValueError: If usage is not in [`train`, `test`, `valid`, `all`].
        ValueError: If annotation_file is not exist.
        ValueError: If dataset_dir is not exist.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Note:
        - This dataset can take in a `sampler`. `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> wider_face_dir = "/path/to/wider_face_dataset"
        >>>
        >>> # Read 3 samples from WIDERFace dataset
        >>> dataset = ds.WIDERFaceDataset(dataset_dir=wider_face_dir, num_samples=3)

    About WIDERFace dataset:

    The WIDERFace database of people faces has a training set of 12,880 samples, a testing set of 16,097 examples
    and a validating set of 3,226 examples. It is a subset of a larger set available from WIDER. The digits have
    been size-normalized and centered in a fixed-size image.

    The following is the original WIDERFace dataset structure.
    You can unzip the dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── wider_face_dir
             ├── WIDER_test
             │    └── images
             │         ├── 0--Parade
             │         │     ├── 0_Parade_marchingband_1_9.jpg
             │         │     ├── ...
             │         ├──1--Handshaking
             │         ├──...
             ├── WIDER_train
             │    └── images
             │         ├── 0--Parade
             │         │     ├── 0_Parade_marchingband_1_11.jpg
             │         │     ├── ...
             │         ├──1--Handshaking
             │         ├──...
             ├── WIDER_val
             │    └── images
             │         ├── 0--Parade
             │         │     ├── 0_Parade_marchingband_1_102.jpg
             │         │     ├── ...
             │         ├──1--Handshaking
             │         ├──...
             └── wider_face_split
                  ├── wider_face_test_filelist.txt
                  ├── wider_face_train_bbx_gt.txt
                  └── wider_face_val_bbx_gt.txt

    Citation:

    .. code-block::

        @inproceedings{2016WIDER,
          title={WIDER FACE: A Face Detection Benchmark},
          author={Yang, S. and Luo, P. and Loy, C. C. and Tang, X.},
          booktitle={IEEE},
          pages={5525-5533},
          year={2016},
        }
    """

    @check_wider_face_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None,
                 decode=False, sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")
        self.decode = replace_none(decode, False)

    def parse(self, children=None):
        return cde.WIDERFaceNode(self.dataset_dir, self.usage, self.decode, self.sampler)


class YelpReviewDataset(SourceDataset, TextBaseDataset):
    """
    A source dataset that reads and parses Yelp Review Polarity and Yelp Review Full dataset.

    The generated dataset has two columns: :py:obj:`[label, text]`.
    The tensor of column :py:obj:`label` is of the string type.
    The tensor of column :py:obj:`text` is of the string type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be `train`, `test` or `all`.
            For Polarity, `train` will read from 560,000 train samples, `test` will read from 38,000 test samples,
            `all` will read from all 598,000 samples.
            For Full, `train` will read from 650,000 train samples, `test` will read from 50,000 test samples,
            `all` will read from all 700,000 samples (default=None, all samples).
        num_samples (int, optional): Number of samples (rows) to read (default=None, reads all samples).
        shuffle (Union[bool, Shuffle level], optional): Perform reshuffling of the data every epoch
            (default=Shuffle.GLOBAL).
            If shuffle is False, no shuffling will be performed;
            If shuffle is True, the behavior is the same as setting shuffle to be Shuffle.GLOBAL
            Otherwise, there are two levels of shuffling:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.
        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, `num_samples` reflects the max sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_dir does not contain data files.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.

    Examples:
        >>> yelp_review_dataset_dir = "/path/to/yelp_review_dataset_dir"
        >>> dataset = ds.YelpReviewDataset(dataset_dir=yelp_review_dataset_dir, usage='all')

    About YelpReview Dataset:

    The Yelp Review Full dataset consists of reviews from Yelp. It is extracted from the Yelp Dataset Challenge 2015
    data, and it is mainly used for text classification.

    The Yelp Review Polarity dataset is constructed from the above dataset, by considering stars 1 and 2 negative, and 3
    and 4 positive.

    The directory structures of these two datasets are the same.
    You can unzip the dataset files into the following structure and read by MindSpore's API:

    .. code-block::

        .
        └── yelp_review_dir
             ├── train.csv
             ├── test.csv
             └── readme.txt

    Citation:

    For Yelp Review Polarity:

    .. code-block::

        @article{zhangCharacterlevelConvolutionalNetworks2015,
          archivePrefix = {arXiv},
          eprinttype = {arxiv},
          eprint = {1509.01626},
          primaryClass = {cs},
          title = {Character-Level {{Convolutional Networks}} for {{Text Classification}}},
          abstract = {This article offers an empirical exploration on the use of character-level convolutional networks
                      (ConvNets) for text classification. We constructed several large-scale datasets to show that
                      character-level convolutional networks could achieve state-of-the-art or competitive results.
                      Comparisons are offered against traditional models such as bag of words, n-grams and their TFIDF
                      variants, and deep learning models such as word-based ConvNets and recurrent neural networks.},
          journal = {arXiv:1509.01626 [cs]},
          author = {Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
          month = sep,
          year = {2015},
        }

    Citation:

    For Yelp Review Full:

    .. code-block::

        @article{zhangCharacterlevelConvolutionalNetworks2015,
          archivePrefix = {arXiv},
          eprinttype = {arxiv},
          eprint = {1509.01626},
          primaryClass = {cs},
          title = {Character-Level {{Convolutional Networks}} for {{Text Classification}}},
          abstract = {This article offers an empirical exploration on the use of character-level convolutional networks
                      (ConvNets) for text classification. We constructed several large-scale datasets to show that
                      character-level convolutional networks could achieve state-of-the-art or competitive results.
                      Comparisons are offered against traditional models such as bag of words, n-grams and their TFIDF
                      variants, and deep learning models such as word-based ConvNets and recurrent neural networks.},
          journal = {arXiv:1509.01626 [cs]},
          author = {Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
          month = sep,
          year = {2015},
        }
    """

    @check_yelp_review_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, shuffle=Shuffle.GLOBAL, num_shards=None,
                 shard_id=None, num_parallel_workers=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, 'all')

    def parse(self, children=None):
        return cde.YelpReviewNode(self.dataset_dir, self.usage, self.num_samples, self.shuffle_flag,
                                  self.num_shards, self.shard_id)


class YesNoDataset(MappableDataset):
    """
    A source dataset for reading and parsing the YesNo dataset.

    The generated dataset has three columns :py:obj:`[waveform, sample_rate, labels]`.
    The tensor of column :py:obj:`waveform` is a vector of the float32 type.
    The tensor of column :py:obj:`sample_rate` is a scalar of the int32 type.
    The tensor of column :py:obj:`labels` is a scalar of the int32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        num_samples (int, optional): The number of images to be included in the dataset
            (default=None, will read all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, will use value set in the config).
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset
            (default=None, expected order behavior shown in the table).
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, `num_samples` reflects the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` (default=None). This argument can only
            be specified when `num_shards` is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_dir does not contain data files.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Note:
        - This dataset can take in a `sampler`. `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> yes_no_dataset_dir = "/path/to/yes_no_dataset_directory"
        >>>
        >>> # Read 3 samples from YesNo dataset
        >>> dataset = ds.YesNoDataset(dataset_dir=yes_no_dataset_dir, num_samples=3)
        >>>
        >>> # Note: In YesNo dataset, each dictionary has keys "waveform", "sample_rate", "label"

    About YesNo dataset:

    Yesno is an audio dataset consisting of 60 recordings of one individual saying yes or no in Hebrew; each
    recording is eight words long. It was created for the Kaldi audio project by an author who wishes to
    remain anonymous.

    Here is the original YesNo dataset structure.
    You can unzip the dataset files into this directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── yes_no_dataset_dir
             ├── 1_1_0_0_1_1_0_0.wav
             ├── 1_0_0_0_1_1_0_0.wav
             ├── 1_1_0_0_1_1_0_0.wav
             └──....

    Citation:

    .. code-block::

        @NetworkResource{Kaldi_audio_project,
        author    = {anonymous},
        url       = "http://wwww.openslr.org/1/"
        }
    """

    @check_yes_no_dataset
    def __init__(self, dataset_dir, num_samples=None, num_parallel_workers=None, shuffle=None,
                 sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir

    def parse(self, children=None):
        return cde.YesNoNode(self.dataset_dir, self.sampler)


class SemeionDataset(MappableDataset):
    """
    A source dataset for reading and parsing Semeion dataset.

    The generated dataset has two columns :py:obj:`[image, label]`.
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`label` is a scalar of the uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        num_samples (int, optional): The number of samples to be included in the dataset
            (default=None, will read all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        shuffle (bool, optional): Whether to perform shuffle on the dataset (default=None, expected
            order behavior shown in the table).
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided
            into (default=None). When this argument is specified, `num_samples` reflects
            the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Note:
        - This dataset can take in a `sampler`. `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> semeion_dataset_dir = "/path/to/semeion_dataset_directory"
        >>>
        >>> # 1) Get all samples from SEMEION dataset in sequence
        >>> dataset = ds.SemeionDataset(dataset_dir=semeion_dataset_dir, shuffle=False)
        >>>
        >>> # 2) Randomly select 10 samples from SEMEION dataset
        >>> dataset = ds.SemeionDataset(dataset_dir=semeion_dataset_dir, num_samples=10, shuffle=True)
        >>>
        >>> # 3) Get samples from SEMEION dataset for shard 0 in a 2-way distributed training
        >>> dataset = ds.SemeionDataset(dataset_dir=semeion_dataset_dir, num_shards=2, shard_id=0)
        >>>
        >>> # In SEMEION dataset, each dictionary has keys: image, label.

    About SEMEION dataset:

    The dataset was created by Tactile Srl, Brescia, Italy (http://www.tattile.it) and donated in 1994
    to Semeion Research Center of Sciences of Communication, Rome, Italy (http://www.semeion.it),
    for machine learning research.

    This dataset consists of 1593 records (rows) and 256 attributes (columns). Each record represents
    a handwritten digit, originally scanned with a resolution of 256 grey scale. Each pixel of the each
    original scanned image was first stretched, and after scaled between 0 and 1
    (setting to 0 every pixel whose value was under the value 127 of the grey scale (127 included)
    and setting to 1 each pixel whose original value in the grey scale was over 127). Finally, each binary image
    was scaled again into a 16x16 square box (the final 256 binary attributes).

    .. code-block::

        .
        └── semeion_dataset_dir
            └──semeion.data
            └──semeion.names

    Citation:

    .. code-block::

        @article{
          title={The Theory of Independent Judges, in Substance Use & Misuse 33(2)1998, pp 439-461},
          author={M Buscema, MetaNet},
        }
    """

    @check_semeion_dataset
    def __init__(self, dataset_dir, num_samples=None, num_parallel_workers=None, shuffle=None,
                 sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir

    def parse(self, children=None):
        return cde.SemeionNode(self.dataset_dir, self.sampler)


class TedliumDataset(MappableDataset):
    """
    A source dataset for reading and parsing Tedlium dataset.
    The columns of generated dataset depend on the source SPH files and the corresponding STM files.

    The generated dataset has six columns :py:obj:`[waveform, sample_rate, transcript, talk_id, speaker_id,
    identifier]`.

    The tensor of column :py:obj:`waveform` is of the float32 type.
    The tensor of column :py:obj:`sample_rate` is a scalar of the int32 type.
    The tensor of column :py:obj:`transcript` is a scalar of the string type.
    The tensor of column :py:obj:`talk_id` is a scalar of the string type.
    The tensor of column :py:obj:`speaker_id` is a scalar of the string type.
    The tensor of column :py:obj:`identifier` is a scalar of the string type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        release (str): Release of the dataset, can be "release1", "release2", "release3".
        usage (str, optional): Usage of this dataset.
            For release1 or release2, can be `train`, `test`, ` dev` or `all`.
            `train` will read from train samples,
            `test` will read from test samples,
            `dev` will read from dev samples,
            `all` will read from all samples.
            For release3, can only be "all", it will read from data samples (default=None, all samples).
        extensions (str): Extensions of the SPH files, only '.sph' is valid.
            (default=None, ".sph").
        num_samples (int, optional): The number of audio samples to be included in the dataset
            (default=None, all samples).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        shuffle (bool, optional): Whether to perform shuffle on the dataset (default=None, expected
            order behavior shown in the table).
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided
            into (default=None). When this argument is specified, `num_samples` reflects
            the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_dir does not contain stm files.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Note:
        - This dataset can take in a `sampler`. `sampler` and `shuffle` are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using `sampler` and `shuffle`
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter `sampler`
         - Parameter `shuffle`
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> # 1) Get all train samples from TEDLIUM_release1 dataset in sequence.
        >>> dataset = ds.TedliumDataset(dataset_dir="/path/to/tedlium1_dataset_directory",
        ...                             release="release1", shuffle=False)
        >>>
        >>> # 2) Randomly select 10 samples from TEDLIUM_release2 dataset.
        >>> dataset = ds.TedliumDataset(dataset_dir="/path/to/tedlium2_dataset_directory",
        ...                             release="release2", num_samples=10, shuffle=True)
        >>>
        >>> # 3) Get samples from TEDLIUM_release-3 dataset for shard 0 in a 2-way distributed training.
        >>> dataset = ds.TedliumDataset(dataset_dir="/path/to/tedlium3_dataset_directory",
        ...                             release="release3", num_shards=2, shard_id=0)
        >>>
        >>> # In TEDLIUM dataset, each dictionary has keys : waveform, sample_rate, transcript, talk_id,
        >>> # speaker_id and identifier.

    About TEDLIUM_release1 dataset:

    The TED-LIUM corpus is English-language TED talks, with transcriptions, sampled at 16kHz.
    It contains about 118 hours of speech.

    About TEDLIUM_release2 dataset:

    This is the TED-LIUM corpus release 2, licensed under Creative Commons BY-NC-ND 3.0. All talks and text are
    property of TED Conferences LLC. The TED-LIUM corpus was made from audio talks and their transcriptions available
    on the TED website. We have prepared and filtered these data in order to train acoustic models to participate to
    the International Workshop on Spoken Language Translation 2011 (the LIUM English/French SLT system reached the
    first rank in the SLT task).

    About TEDLIUM_release-3 dataset:

    This is the TED-LIUM corpus release 3, licensed under Creative Commons BY-NC-ND 3.0. All talks and text are
    property of TED Conferences LLC. This new TED-LIUM release was made through a collaboration between the Ubiqus
    company and the LIUM (University of Le Mans, France).

    You can unzip the dataset files into the following directory structure and read by MindSpore's API.

    The structure of TEDLIUM release2 is the same as TEDLIUM release1, only the data is different.

    .. code-block::

        .
        └──TEDLIUM_release1
            └── dev
                ├── sph
                    ├── AlGore_2009.sph
                    ├── BarrySchwartz_2005G.sph
                ├── stm
                    ├── AlGore_2009.stm
                    ├── BarrySchwartz_2005G.stm
            └── test
                ├── sph
                    ├── AimeeMullins_2009P.sph
                    ├── BillGates_2010.sph
                ├── stm
                    ├── AimeeMullins_2009P.stm
                    ├── BillGates_2010.stm
            └── train
                ├── sph
                    ├── AaronHuey_2010X.sph
                    ├── AdamGrosser_2007.sph
                ├── stm
                    ├── AaronHuey_2010X.stm
                    ├── AdamGrosser_2007.stm
            └── readme
            └── TEDLIUM.150k.dic

    .. code-block::

        .
        └──TEDLIUM_release-3
            └── data
                ├── ctl
                ├── sph
                    ├── 911Mothers_2010W.sph
                    ├── AalaElKhani.sph
                ├── stm
                    ├── 911Mothers_2010W.stm
                    ├── AalaElKhani.stm
            └── doc
            └── legacy
            └── LM
            └── speaker-adaptation
            └── readme
            └── TEDLIUM.150k.dic

    Citation:

    .. code-block::

        @article{
          title={TED-LIUM: an automatic speech recognition dedicated corpus},
          author={A. Rousseau, P. Deléglise, Y. Estève},
          journal={Proceedings of the Eighth International Conference on Language Resources and Evaluation (LREC'12)},
          year={May 2012},
          biburl={https://www.openslr.org/7/}
        }

        @article{
          title={Enhancing the TED-LIUM Corpus with Selected Data for Language Modeling and More TED Talks},
          author={A. Rousseau, P. Deléglise, and Y. Estève},
          journal={Proceedings of the Eighth International Conference on Language Resources and Evaluation (LREC'12)},
          year={May 2014},
          biburl={https://www.openslr.org/19/}
        }

        @article{
          title={TED-LIUM 3: twice as much data and corpus repartition for experiments on speaker adaptation},
          author={François Hernandez, Vincent Nguyen, Sahar Ghannay, Natalia Tomashenko, and Yannick Estève},
          journal={the 20th International Conference on Speech and Computer (SPECOM 2018)},
          year={September 2018},
          biburl={https://www.openslr.org/51/}
        }
    """

    @check_tedlium_dataset
    def __init__(self, dataset_dir, release, usage=None, extensions=None, num_samples=None,
                 num_parallel_workers=None, shuffle=None, sampler=None, num_shards=None,
                 shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.extensions = replace_none(extensions, ".sph")
        self.release = release
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.TedliumNode(self.dataset_dir, self.release, self.usage, self.extensions, self.sampler)


class _SVHNDataset:
    """
    Mainly for loading SVHN Dataset, and return two rows each time.
    """

    def __init__(self, dataset_dir, usage):
        self.dataset_dir = os.path.realpath(dataset_dir)
        self.usage = usage
        self.column_names = ["image", "label"]
        self.usage_all = ["train", "test", "extra"]
        self.data = np.array([], dtype=np.uint8)
        self.labels = np.array([], dtype=np.uint32)

        if self.usage == "all":
            for _usage in self.usage_all:
                data, label = self._load_mat(_usage)
                self.data = np.concatenate((self.data, data)) if self.data.size else data
                self.labels = np.concatenate((self.labels, label)) if self.labels.size else label
        else:
            self.data, self.labels = self._load_mat(self.usage)

    def _load_mat(self, mode):
        filename = mode + "_32x32.mat"
        mat_data = loadmat(os.path.join(self.dataset_dir, filename))
        data = np.transpose(mat_data['X'], [3, 0, 1, 2])
        label = mat_data['y'].astype(np.uint32).squeeze()
        np.place(label, label == 10, 0)
        return data, label

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


class SVHNDataset(GeneratorDataset):
    """
    A source dataset for reading and parsing SVHN dataset.

    The generated dataset has two columns: :py:obj:`[image, label]`.
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`label` is of a scalar of uint32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Specify the 'train', 'test', 'extra' or 'all' parts of dataset
            (default=None, will read all samples).
        num_samples (int, optional): The number of samples to be included in the dataset (default=None, all images).
        num_parallel_workers (int, optional): Number of subprocesses used to fetch the dataset in parallel (default=1).
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset. Random accessible input is required.
            (default=None, expected order behavior shown in the table).
        sampler (Union[Sampler, Iterable], optional): Object used to choose samples from the dataset. Random accessible
            input is required (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            Random accessible input is required. When this argument is specified, 'num_samples' reflects the max
            sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This argument must be specified only
            when num_shards is also specified. Random accessible input is required.

    Raises:
        RuntimeError: If dataset_dir is not valid or does not exist or does not contain data files.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If usage is invalid.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Note:
        - This dataset can take in a sampler. 'sampler' and 'shuffle' are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using 'sampler' and 'shuffle'
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter 'sampler'
         - Parameter 'shuffle'
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> svhn_dataset_dir = "/path/to/svhn_dataset_directory"
        >>> dataset = ds.SVHNDataset(dataset_dir=svhn_dataset_dir, usage="train")

    About SVHN dataset:

    SVHN dataset consists of 10 digit classes.
    SVHN is obtained from house numbers in Google Street View images.
    73257 digits for training, 26032 digits for testing, and 531131 additional extra training data.

    Here is the original SVHN dataset structure.
    You can unzip the dataset files into this directory structure and read by MindSpore's API.

    .. code-block::
        .
        └── svhn_dataset_dir
             ├── train_32x32.mat
             ├── test_32x32.mat
             └── extra_32x32.mat

    Citation:

    .. code-block::

        @article{
          title={Reading Digits in Natural Images with Unsupervised Feature Learning},
          author={Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, Andrew Y. Ng},
          conference={NIPS Workshop on Deep Learning and Unsupervised Feature Learning 2011.},
          year={2011},
          publisher={NIPS}
          url={http://ufldl.stanford.edu/housenumbers}
        }

    """

    @check_svhn_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=1, shuffle=None,
                 sampler=None, num_shards=None, shard_id=None):
        self.dataset_dir = os.path.realpath(dataset_dir)
        self.usage = replace_none(usage, "all")
        dataset = _SVHNDataset(self.dataset_dir, self.usage)

        super().__init__(dataset, column_names=dataset.column_names, num_samples=num_samples,
                         num_parallel_workers=num_parallel_workers, shuffle=shuffle, sampler=sampler,
                         num_shards=num_shards, shard_id=shard_id)


class STL10Dataset(MappableDataset):
    """
    A source dataset for reading and parsing STL10 dataset.

    The generated dataset has two columns: :py:obj:`[image, label]`.
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`label` is of a scalar of int32 type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be "train", "test",
            "unlabeled", "train+unlabeled" or "all" . "train" will read from 5,000
            train samples, "test" will read from 8,000 test samples,
            "unlabeled" will read from all 100,000 samples, and "train+unlabeled"
            will read from 105000 samples, "all" will read all the samples
            (default=None, all samples).
        num_samples (int, optional): The number of images to be included in the dataset.
            (default=None, all images).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        shuffle (bool, optional): Whether to perform shuffle on the dataset (default=None, expected
            order behavior shown in the table).
        sampler (Sampler, optional): Object used to choose samples from the
            dataset (default=None, expected order behavior shown in the table).
        num_shards (int, optional): Number of shards that the dataset will be divided
            into (default=None). When this argument is specified, 'num_samples' reflects
            the max sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_dir is not valid or does not exist or does not contain data files.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If sampler and shuffle are specified at the same time.
        RuntimeError: If sampler and sharding are specified at the same time.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If usage is invalid.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Note:
        - This dataset can take in a sampler. 'sampler' and 'shuffle' are mutually exclusive.
          The table below shows what input arguments are allowed and their expected behavior.

    .. list-table:: Expected Order Behavior of Using 'sampler' and 'shuffle'
       :widths: 25 25 50
       :header-rows: 1

       * - Parameter 'sampler'
         - Parameter 'shuffle'
         - Expected Order Behavior
       * - None
         - None
         - random order
       * - None
         - True
         - random order
       * - None
         - False
         - sequential order
       * - Sampler object
         - None
         - order defined by sampler
       * - Sampler object
         - True
         - not allowed
       * - Sampler object
         - False
         - not allowed

    Examples:
        >>> stl10_dataset_dir = "/path/to/stl10_dataset_directory"
        >>>
        >>> # 1) Get all samples from STL10 dataset in sequence
        >>> dataset = ds.STL10Dataset(dataset_dir=stl10_dataset_dir, shuffle=False)
        >>>
        >>> # 2) Randomly select 350 samples from STL10 dataset
        >>> dataset = ds.STL10Dataset(dataset_dir=stl10_dataset_dir, num_samples=350, shuffle=True)
        >>>
        >>> # 3) Get samples from STL10 dataset for shard 0 in a 2-way distributed training
        >>> dataset = ds.STL10Dataset(dataset_dir=stl10_dataset_dir, num_shards=2, shard_id=0)

    About STL10 dataset:

    STL10 dataset consists of 10 classes: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck.
    STL10 is is inspired by the CIFAR-10 dataset.
    Images are 96x96 pixels, color.
    500 training images, 800 test images per class and 100000 unlabeled images.
    Labels are 0-indexed, and unlabeled images have -1 as their labels.

    Here is the original STL10 dataset structure.
    You can unzip the dataset files into this directory structure and read by MindSpore's API.

    .. code-block::
        .
        └── stl10_dataset_dir
             ├── train_X.bin
             ├── train_y.bin
             ├── test_X.bin
             ├── test_y.bin
             └── unlabeled_X.bin

    Citation of STL10 dataset.

    .. code-block::

        @techreport{Coates10,
        author       = {Adam Coates},
        title        = {Learning multiple layers of features from tiny images},
        year         = {20010},
        howpublished = {https://cs.stanford.edu/~acoates/stl10/},
        description  = {The STL-10 dataset consists of 96x96 RGB images in 10 classes,
                        with 500 training images and 800 testing images per class.
                        There are 5000 training images and 8000 test images.
                        It also has 100000 unlabeled images for unsupervised learning.
                        These examples are extracted from a similar but broader distribution of images.
                        }
        }
    """

    @check_stl10_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None,
                 sampler=None, num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, sampler=sampler, num_samples=num_samples,
                         shuffle=shuffle, num_shards=num_shards, shard_id=shard_id, cache=cache)

        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.STL10Node(self.dataset_dir, self.usage, self.sampler)


class EnWik9Dataset(SourceDataset):
    """
    A source dataset that reads and parses EnWik9 dataset.

    The generated dataset has one column :py:obj:`[text]` with type string.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        num_samples (int, optional): The number of samples to be included in the dataset
            (default=None, will include all samples).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        shuffle (Union[bool, Shuffle level], optional): Perform reshuffling of the data every epoch
            (default=True).
            If shuffle is False, no shuffling will be performed;
            If shuffle is True, the behavior is the same as setting shuffle to be Shuffle.GLOBAL
            Otherwise, there are two levels of shuffling:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, `num_samples` reflects the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing
            (default=None, which means no cache is used).

    Examples:
        >>> en_wik9_dataset_dir = "/path/to/en_wik9_dataset"
        >>> dataset2 = ds.EnWik9Dataset(dataset_dir=en_wik9_dataset_dir, num_samples=2,
        ...                             shuffle=True)

    About EnWik9 dataset:

    The data of EnWik9 is UTF-8 encoded XML consisting primarily of English text. It contains 243,426 article titles,
    of which 85,560 are #REDIRECT to fix broken links, and the rest are regular articles.

    The data is UTF-8 clean. All characters are in the range U'0000 to U'10FFFF with valid encodings of 1 to
    4 bytes. The byte values 0xC0, 0xC1, and 0xF5-0xFF never occur. Also, in the Wikipedia dumps,
    there are no control characters in the range 0x00-0x1F except for 0x09 (tab) and 0x0A (linefeed).
    Linebreaks occur only on paragraph boundaries, so they always have a semantic purpose.

    You can unzip the dataset files into the following directory structure and read by MindSpore's API.

    .. code-block::

        .
        └── EnWik9
             ├── enwik9

    Citation:

    .. code-block::

        @NetworkResource{Hutter_prize,
        author    = {English Wikipedia},
        url       = "https://cs.fit.edu/~mmahoney/compression/textdata.html",
        month     = {March},
        year      = {2006}
        }
    """

    @check_en_wik9_dataset
    def __init__(self, dataset_dir, num_samples=None, num_parallel_workers=None, shuffle=True,
                 num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir

    def parse(self, children=None):
        return cde.EnWik9Node(self.dataset_dir, self.num_samples, self.shuffle_flag, self.num_shards,
                              self.shard_id)


class YahooAnswersDataset(SourceDataset):
    """
    A source dataset that reads and parses the YahooAnswers dataset.

    The generated dataset has three columns :py:obj:`[class, title, content, answer]`.
    The tensor of column :py:obj:`class` is of the string type.
    The tensor of column :py:obj:`title` is of the string type.
    The tensor of column :py:obj:`content` is of the string type.
    The tensor of column :py:obj:`answer` is of the string type.

    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        usage (str, optional): Usage of this dataset, can be `train`, `test` or `all`. `train` will read
            from 1,400,000 train samples, `test` will read from 60,000 test samples, `all` will read from
            all 1,460,000 samples (default=None, all samples).
        num_samples (int, optional): The number of samples to be included in the dataset
            (default=None, will include all text).
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        shuffle (Union[bool, Shuffle level], optional): Perform reshuffling of the data every epoch
            (default=Shuffle.GLOBAL).
            If shuffle is False, no shuffling will be performed;
            If shuffle is True, the behavior is the same as setting shuffle to be Shuffle.GLOBAL
            Otherwise, there are two levels of shuffling:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided into (default=None).
            When this argument is specified, `num_samples` reflects the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_dir does not contain data files.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Examples:
        >>> yahoo_answers_dataset_dir = "/path/to/yahoo_answers_dataset_directory"
        >>>
        >>> # 1) Read 3 samples from YahooAnswers dataset
        >>> dataset = ds.YahooAnswersDataset(dataset_dir=yahoo_answers_dataset_dir, num_samples=3)
        >>>
        >>> # 2) Read train samples from YahooAnswers dataset
        >>> dataset = ds.YahooAnswersDataset(dataset_dir=yahoo_answers_dataset_dir, usage="train")

    About YahooAnswers dataset:

    The YahooAnswers dataset consists of 630,000 text samples in 14 classes,
    There are 560,000 samples in the train.csv and 70,000 samples in the test.csv.
    The 10 different classes represent Society & Culture, Science & Mathematics, Health, Education & Reference,
    Computers & Internet, Sports, Business & Finance, Entertainment & Music, Family & Relationships,
    Politics & Government.

    Here is the original YahooAnswers dataset structure.
    You can unzip the dataset files into this directory structure and read by Mindspore's API.

    .. code-block::

        .
        └── yahoo_answers_dataset_dir
            ├── train.csv
            ├── test.csv
            ├── classes.txt
            └── readme.txt

    .. code-block::

        @article{YahooAnswers,
        title   = {Yahoo! Answers Topic Classification Dataset},
        author  = {Xiang Zhang},
        year    = {2015},
        howpublished = {}
        }
    """

    @check_yahoo_answers_dataset
    def __init__(self, dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=Shuffle.GLOBAL,
                 num_shards=None, shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.dataset_dir = dataset_dir
        self.usage = replace_none(usage, "all")

    def parse(self, children=None):
        return cde.YahooAnswersNode(self.dataset_dir, self.usage, self.num_samples, self.shuffle_flag,
                                    self.num_shards, self.shard_id)
