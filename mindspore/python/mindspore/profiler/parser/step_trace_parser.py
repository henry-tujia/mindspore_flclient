# Copyright 2020 Huawei Technologies Co., Ltd
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
# ============================================================================
"""The parser for step trace data."""
import csv
import json
import os
import stat
import struct
from collections import namedtuple
from decimal import Decimal
from abc import abstractmethod

from mindspore.profiler.common.exceptions.exceptions import ProfilerPathErrorException, \
    ProfilerIOException, ProfilerRawFileException
from mindspore import log
from mindspore.profiler.common.util import get_summary_for_step_trace
from mindspore.profiler.common.validator.validate_path import \
    validate_and_normalize_path

ProfilingHeadStruct = namedtuple(
    'ProfilingHeadStruct', ['mode', 'rptType', 'bufSize']
)

StepTraceStruct = namedtuple(
    'StepTraceStruct', ['timeStamp', 'index_id', 'model_id', 'stream_id', 'task_id', 'tag_id']
)


class BaseStepTraceParser:
    """
    The parser for step trace data.

    Args:
        input_dir (str): The directory that contains original step trace data.
        output_file_path (str): The output file path.
        job_id (int): The job id used to define the start of new step. Default: 0.
        skip_first_step (bool): Whether skip the first step or not.
        is_training_mode (bool): Whether in training mode or not.
        is_gpu_kernel_async_launch (bool): Whether is gpu kernel async launch or not.
    """

    def __init__(self, input_dir, output_file_path, job_id=0, skip_first_step=False,
                 is_training_mode=True, is_gpu_kernel_async_launch=False):
        self._input_dir = input_dir
        self._output_path = output_file_path
        self._job_id = job_id
        self._skip_first_step = skip_first_step
        self._result = []
        self._header = []
        self._step_num = 0
        self._tag_map = {}
        self._is_training_mode = is_training_mode
        self._step_end_tag_id = 4
        self._is_gpu_kernel_async_launch = is_gpu_kernel_async_launch
        self._model_start_tag_id = 0
        self._model_end_tag_id = 1
        self._fp_tag_id = 2
        self._bp_tag_id = 3
        self._reduce_min_tag_id = 10000
        self._reduce_max_tag_id = 20000
        self._profiling_head_len = 4
        self._profiling_head_pad_len = 4
        self._st_data_len = 8 + 8 + 8 + 2 + 2 + 2

    @property
    def output_file(self):
        """The property of step trace header."""
        file_name = self._output_path.rsplit('/', 2)
        return file_name[-1] if len(file_name) == 3 else ''

    def show(self):
        """The property of step trace info."""
        summary_info = {}
        if self._result:
            summary_info = get_summary_for_step_trace(self._result[-1], self._header, self._is_training_mode)
            summary_info['total_steps'] = len(self._result) - 1
        print('\nStep trace summary info (unit: syscnt):')
        print(summary_info)
        print('\nThe step trace parse result saves under ${summary_dir}/profiler/%s'
              % self.output_file)

    def parse_and_save(self):
        """Parse step trace files and save the result."""
        try:
            source_files = self._get_step_trace_files()
            if self._is_gpu_kernel_async_launch:
                self._parse_async_launch(source_files)
            else:
                self._parse(source_files)
            self._save()
        except IOError as err:
            log.warning(err)
            raise ProfilerIOException()
        else:
            log.info("Finish to save intermediate result for step trace file.")

    def record_point_info(self, point_info, output_path):
        """
        Record point info into json.

        Args:
            point_info (dict): The point info about tag id and relative op name.
            output_path (str): The output path for saving point info.

        Returns:
            dict, parsed point info.
        """

    def update_tag_op_type_map(self, point_info):
        """
        update the map from tag id to op type.

        Args:
            point_info (dict): The point info about tag id and relative op name.
        """
        self._get_step_trace_files()
        tag_map = {}
        for tag, op_name in point_info.items():
            op_type = self._get_op_type(tag, op_name)
            tag_map[tag] = op_type
        log.info("Get tag types for step trace analysis: %s", tag_map)
        self._tag_map = tag_map

    def _get_op_type(self, tag, name):
        """
        Get op type from tag and name.

        Args:
            tag (int): The tag id.
            name (str): The op name.

        Returns:
            str, the op type or communication op name.
        """
        tag_map = {self._fp_tag: 'fp', self._bp_tag: 'bp', self._step_end_tag_id: 'end'}
        # get solid tag type
        op_type = tag_map.get(tag, '')
        if op_type:
            return op_type
        # check if the tag is step tag.
        if tag == 0:
            return 'start'
        # analyze the reduce tag
        op_name = name.rsplit('/', 1)[-1]
        if not op_name:
            log.warning("Unexpected op name:%s", name)

        return op_name

    def _get_step_trace_files(self):
        """Get step trace files."""
        return self._input_dir

    @staticmethod
    def _search_file(input_dir):
        """Search step trace file under specific input directory."""
        # validate input_dir
        if not os.path.isdir(input_dir):
            raise ProfilerPathErrorException(
                '{} does not exist or is not a dir'.format(input_dir)
            )
        # get step trace files
        files = os.listdir(input_dir)
        step_trace_files = list(
            filter(
                lambda file: file.startswith('ts_track.data') and not file.endswith('.done'),
                files
            )
        )
        # validate result
        if len(step_trace_files) > 1:
            # the format of file name is like
            # `training_trace.46.dev.profiler_default_tag.$id.slice_$number`
            # use the $number as the sorted key
            try:
                step_trace_files.sort(key=lambda path: int(path.rsplit('_', 1)[-1]))
            except ValueError as err:
                log.warning("Unable to parse file names: %s. %s", step_trace_files, err)
                step_trace_files = []
        else:
            training_trace_files = list(
                filter(
                    lambda file: file.startswith('training_trace') and not file.endswith('.done'),
                    files
                )
            )
            if len(training_trace_files) >= 1:
                log.warning("The training_trace file structure is changed, please upgrade "
                            "mindspore and regenerate profiling data")

        file_paths = [os.path.join(input_dir, file) for file in step_trace_files]
        log.info("Find %d step trace files.", len(file_paths))
        return file_paths

    @abstractmethod
    def _parse(self, source_files):
        """Parse source step trace files."""

    def _get_next_step_trace(self, content, event_info):
        """
        Get next step trace info.

        Args:
            content (bytes): The input step trace info.
            event_info (dict): The event info.

        Returns:
            Generator, return the step trace one by one.
        """
        start_time = event_info.get('end', '-')
        event_info['start'] = start_time
        if 'reduce' not in event_info.keys():
            event_info['reduce'] = {}

        i = 0
        while i < len(content):
            profiling_head_data = content[i:i + self._profiling_head_len]
            parsed_head = struct.unpack('BBH', profiling_head_data)
            profiling_head = ProfilingHeadStruct(*parsed_head)
            if profiling_head.rptType == 10:
                st_data = content[i + self._profiling_head_len + self._profiling_head_pad_len:
                                  i + self._profiling_head_len + self._profiling_head_pad_len + self._st_data_len]
                parsed_data = struct.unpack('QQQHHH', st_data)
                next_event = StepTraceStruct(*parsed_data)
                self._construct_event_info(next_event, event_info)

                if event_info.get('end'):
                    yield event_info
                    start_time = event_info.get('end', '-')
                    event_info.clear()
                    event_info['start'] = start_time
                    event_info['reduce'] = {}
            i = i + profiling_head.bufSize

    def _construct_event_info(self, next_event, event_info):
        """Construct event info according to next_event."""
        end_flag: bool = lambda tag: tag == self._step_end_tag_id
        fp_flag: bool = lambda tag: tag == self._fp_tag_id
        bp_flag: bool = lambda tag: tag == self._bp_tag_id
        reduce_flag: bool = lambda tag: self._reduce_min_tag_id <= tag < self._reduce_max_tag_id

        def _on_reduce_event(reduce_tag_id):
            """Handle reduce event."""
            stream_id = next_event.stream_id
            if event_info['reduce'].get(stream_id):
                event_info['reduce'][stream_id].append((reduce_tag_id, time_stamp))
            else:
                event_info['reduce'][stream_id] = [(reduce_tag_id, time_stamp)]

        tag_id = next_event.tag_id
        time_stamp = next_event.timeStamp
        if end_flag(tag_id):
            event_info['end'] = time_stamp
        elif fp_flag(tag_id):
            event_info['fp'] = time_stamp
        elif bp_flag(tag_id):
            event_info['bp'] = time_stamp
        elif reduce_flag(tag_id):
            _on_reduce_event(tag_id)

    def _record_trace_event(self, step_trace):
        """Record trace event."""
        self._step_num += 1
        start_time = step_trace.get('start')
        end_time = step_trace.get('end')
        fp_time = step_trace.get('fp')
        bp_time = step_trace.get('bp')
        if not (start_time and end_time and fp_time and bp_time):
            log.warning("The step %d lacks basic time.", self._step_num)
            return
        if start_time == '-':
            start_time = fp_time
        row_data = {
            'step_num': self._step_num,
            'start_point': start_time,
            'end_point': end_time,
            'total': end_time - start_time,
            'fp_point': fp_time,
            'bp_point': bp_time,
            'iteration_interval': fp_time - start_time,
            'fp_and_bp': bp_time - fp_time,
            'tail': end_time - bp_time
        }
        # update reduce info
        self._update_reduce_info(step_trace, row_data)
        # save the row data
        if not self._header:
            self._header = list(row_data.keys())
        row_data_list = [row_data.get(header_name, 0) for header_name in self._header]
        self._result.append(row_data_list)

    def _update_reduce_info(self, step_trace, row_data):
        """Extract reduce info."""
        reduce_time = step_trace.get('reduce', {})
        for stream_id, time_points in reduce_time.items():
            time_point_num = len(time_points)
            if time_point_num % 2:
                log.warning("Stream %d has %d reduce time points.", stream_id, time_point_num)
                continue
            for index, point_id in enumerate(range(0, time_point_num, 2)):
                field_name = f'stream_{stream_id}_{index}'
                reduce_info = self._get_single_reduce_event_info(
                    field_name, time_points[point_id], time_points[point_id + 1])
                row_data.update(reduce_info)

    def _get_single_reduce_event_info(self, field_name, start_point, end_point):
        """
        Get single reduce info.

        Args:
            field_name (str): The field name.
            start_point (Tuple[int, int]): Start point time info, including (tag_id, sys_count).
            end_point (Tuple[int, int]): End point time info, including (tag_id, sys_count).

        Returns:
            dict, reduce info.
        """
        ret_dict = {}
        return ret_dict

    def _record_average_info(self):
        """Calculate average info."""
        result_size = len(self._result)
        # calculate average data for each column in result data
        average_data = [0] * len(self._header)
        if result_size >= 2:
            for row_info in self._result[1:]:
                average_data = [
                    Decimal(i) + Decimal(j) for i, j in zip(row_info, average_data)
                ]
            average_data = [
                round((item / (result_size - 1))) for item in average_data
            ]
            # change step num info in average_data to None
            step_num_index = self._header.index('step_num')
            average_data[step_num_index] = '-'
        self._result.append(average_data)
        log.info("Finish add average info for step trace.")

    def _save(self):
        """save step trace file."""
        bp_point, tail, fp_duration = 5, -1, -2
        log.info("Start to save step trace file.")
        if not self._header:
            return
        try:
            with open(self._output_path, 'w') as file_handle:
                csv_writer = csv.writer(file_handle)
                if not self._is_training_mode:
                    self._header[fp_duration] = 'fp'
                    self._header = self._header[:bp_point] + self._header[bp_point + 1:tail]
                csv_writer.writerow(self._header)
                for row_data in self._result:
                    if not self._is_training_mode:
                        row_data[fp_duration] += row_data[tail]
                        row_data = row_data[:bp_point] + row_data[bp_point + 1:tail]
                    csv_writer.writerow(row_data)
            os.chmod(self._output_path, stat.S_IREAD | stat.S_IWRITE)
        except (IOError, OSError) as err:
            log.warning('Failed to save step trace raw info. %s', err)
            raise ProfilerIOException


class GpuStepTraceParser(BaseStepTraceParser):
    """The parser for gpu step trace data."""

    def get_fp_bp(self, f_obj, all_step_fp, all_step_bp):
        """Parser the fp and bp."""
        fp_start, bp_end = 0, 1
        if self._is_gpu_kernel_async_launch:
            for line in f_obj:
                line = line.strip().split()
                all_step_fp.append(line[1].split(',')[0])
                all_step_bp.append(line[2].split(',')[0])
        else:
            lines = f_obj.readlines()
            all_step_fp.append(lines[fp_start].split()[0])
            all_step_bp.append(lines[bp_end].split()[0])

    def record_point_info(self, source_file, output_path):
        """
        Record point info into json.

        Args:
            source_file (str): The file path of step trace original data.
            output_path (str): The output path for saving point info.

        Returns:
            dict, parsed point info.
        """
        all_step_points = []
        all_step_fp = []
        all_step_bp = []
        try:
            with open(source_file, 'r') as f_obj:
                self.get_fp_bp(f_obj, all_step_fp, all_step_bp)
        except (IOError, OSError) as err:
            log.warning(f'Failed to read {source_file}', err)
            raise ProfilerIOException

        for fp_name, bp_name in zip(all_step_fp, all_step_bp):
            if self._is_training_mode:
                points = {
                    'fp_start': fp_name,
                    'bp_end': bp_name
                }
            else:
                points = {
                    'fp_start': fp_name,
                }
            all_step_points.append(points)

        try:
            with open(output_path, 'w') as json_file:
                if self._is_gpu_kernel_async_launch:
                    json.dump(all_step_points, json_file)
                else:
                    json.dump(all_step_points[0], json_file)
            os.chmod(output_path, stat.S_IREAD | stat.S_IWRITE)
        except (IOError, OSError) as err:
            log.warning('Failed to save point info. %s', err)
            raise ProfilerIOException

        return all_step_points[0]

    def _get_step_trace_files(self):
        """Get step trace files."""
        return self._input_dir

    def _parse(self, source_file):
        """Parse source step trace files."""
        log.info("Start to parse step trace file.")
        fp_start, bp_end, iter_end, iter_start = 0, 1, 2, 3
        reduce_start = 4
        start_time, end_time = 0, 1
        step_trace_point_count = 3

        source_file = validate_and_normalize_path(source_file)
        try:
            with open(source_file, 'r') as f:
                lines = f.readlines()
                if len(lines) < step_trace_point_count:
                    raise ProfilerRawFileException(
                        f"Failed to parse {source_file} file. The FP_POINT/BP_POINT/ITER_END_POINT "
                        f"do not recognized correctly. Try to set the environment variable'PROFILING_FP_START' "
                        f"and 'PROFILING_BP_END' to solve this problem. For example, "
                        f"'export PROFILING_FP_START=Default/xxx/Conv2d-op1' ")
                step_trace_info_all = [line.strip().split()[1:] for line in lines]
                num_of_step = len(step_trace_info_all[0])
                for step_trace_point in step_trace_info_all:
                    if len(step_trace_point) != num_of_step:
                        raise ProfilerRawFileException(
                            f"Failed to parse {source_file} file. Due to the profiled "
                            f"step_num of FP/BP/ITER_END Point are not equal")
                iter_start_info = [step_trace_info_all[fp_start][0]] + \
                                  step_trace_info_all[iter_end][:num_of_step]
                step_trace_info_all.insert(iter_start, iter_start_info)
        except (IOError, OSError) as err:
            log.warning(f'Failed to read {source_file}', err)
            raise ProfilerIOException

        for step_num in range(num_of_step):
            step_trace = {
                'start': int(step_trace_info_all[iter_start][step_num].split(',')[start_time]),
                'fp': int(step_trace_info_all[fp_start][step_num].split(',')[start_time]),
                'bp': int(step_trace_info_all[bp_end][step_num].split(',')[end_time]),
                'end': int(step_trace_info_all[iter_end][step_num].split(',')[end_time]),
                'reduce': {}
            }
            num_of_step_point = len(step_trace_info_all)
            if num_of_step_point > reduce_start:
                reduce_info = {}
                reduce_time_info = []
                for reduce_idx in range(reduce_start, num_of_step_point):
                    cur_reduce_time = step_trace_info_all[reduce_idx][step_num]
                    reduce_time_info += cur_reduce_time.split(',')
                reduce_info['ops'] = reduce_time_info
                step_trace['reduce'] = reduce_info
            self._record_trace_event(step_trace)
        self._record_average_info()
        log.info("Finish to parse step trace file.")

    def _parse_one_step(self, line):
        """
        Parse step text line to dict obj.

        Args:
            line (str): The step trace line text, it contains five parts, each part is separated by a space.
                part 1: start_op_name,start_op_time
                part 2: fp_op_name,fp_time
                part 3: bp_op_name,bp_time
                part 4: end_op_name,end_time
                part 5: [reduce_op_name,reduce1_start],it contains multiple reduce, each reduce is separated by a space.
        """

        line = line.strip().split()
        start_time = int(line[0].split(',')[1][:-1])
        fp_time = int(line[1].split(',')[1][:-1])
        bp_time = int(line[2].split(',')[1][:-1])
        end_time = int(line[3].split(',')[1][:-1])
        reduce_info = {}
        reduce_time_info = []

        for reduce_item in line[4:]:
            # add communication op start and end time, time unit from ns to 10ns.
            reduce_time_info.append(reduce_item.split(',')[1][:-1])
            reduce_time_info.append(reduce_item.split(',')[2][:-1])
        step_trace = {
            'start': start_time,
            'fp': fp_time,
            'bp': bp_time,
            'end': end_time
        }
        if reduce_time_info:
            reduce_info['ops'] = reduce_time_info
        step_trace['reduce'] = reduce_info
        self._record_trace_event(step_trace)

    def _parse_async_launch(self, source_file):
        """Parse source step trace files generated from async launch kernel."""
        log.info("Start to parse step trace file.")
        source_file = validate_and_normalize_path(source_file)

        try:
            with open(source_file, 'r') as f_obj:
                for line in f_obj:
                    self._parse_one_step(line)

        except (IOError, OSError) as err:
            log.warning(f'Failed to read {source_file}', err)
            raise ProfilerIOException

        self._record_average_info()
        log.info("Finish to parse step trace file.")

    def _get_single_reduce_event_info(self, field_name, start_point, end_point):
        """
        Get single reduce info.

        Args:
            field_name (str): The field name.
            start_point (str): Start point time.
            end_point (str): End point time.

        Returns:
            dict, reduce info.
        """
        reduce_info = {}

        op_type = 'AllReduce'
        # append field name with op type.
        field_name += '_' + op_type
        reduce_info[field_name] = int(end_point) - int(start_point)
        reduce_info[field_name + '_start_point'] = start_point
        reduce_info[field_name + '_end_point'] = end_point

        return reduce_info


class AscendStepTraceParser(BaseStepTraceParser):
    """The parser for ascend step trace data."""
    _event_size = 20
    _fp_tag = 2
    _bp_tag = 3
    _step_trace_files = []

    def record_point_info(self, point_info, output_path):
        """
        Record point info into json.

        Args:
            point_info (dict): The point info about tag id and relative op name.
            output_path (str): The output path for saving point info.

        Returns:
            dict, parsed point info.
        """
        if self._is_training_mode:
            points = {
                'fp_start': point_info.get(self._fp_tag, ''),
                'bp_end': point_info.get(self._bp_tag, '')
            }
        else:
            points = {
                'fp_start': point_info.get(self._fp_tag, ''),
            }
        if os.path.exists(output_path):
            return points
        try:
            with open(output_path, 'w') as json_file:
                json.dump(points, json_file)
            os.chmod(output_path, stat.S_IREAD | stat.S_IWRITE)
        except (IOError, OSError) as err:
            log.warning('Failed to save point info. %s', err)
            raise ProfilerIOException
        return points

    def _get_step_trace_files(self):
        """Get step trace files."""
        # step trace files may under $profiler_dir or $profiler_dir/data
        if self._step_trace_files:
            return self._step_trace_files

        profiler_dir = self._input_dir
        step_trace_files = self._search_file(profiler_dir)
        if not step_trace_files:
            # try to find step trace files under $profiler_dir/data
            profiler_dir = os.path.join(profiler_dir, 'data')
            step_trace_files = self._search_file(profiler_dir)
        if not step_trace_files:
            raise ProfilerPathErrorException('Training trace file does not exist.')
        self._step_trace_files = step_trace_files

        return step_trace_files

    def _parse(self, source_files):
        """Parse source step trace files."""
        log.info("Start to parse step trace file.")
        event_info = {}

        for source_file in source_files:
            source_file = validate_and_normalize_path(source_file)
            try:
                with open(source_file, 'rb') as handler:
                    content = handler.read()
                    for step_trace in self._get_next_step_trace(content, event_info):
                        if self._skip_first_step:
                            self._skip_first_step = False
                            continue
                        self._record_trace_event(step_trace)
            except (IOError, OSError) as err:
                log.warning(f'Failed to read {source_file}', err)
                raise ProfilerIOException

        self._record_average_info()
        log.info("Finish to parse step trace file.")

    def _get_single_reduce_event_info(self, field_name, start_point, end_point):
        """
        Get single reduce info.

        Args:
            field_name (str): The field name.
            start_point (Tuple[int, int]): Start point time info, including (tag_id, sys_count).
            end_point (Tuple[int, int]): End point time info, including (tag_id, sys_count).

        Returns:
            dict, reduce info.
        """
        reduce_info = {}
        if end_point[0] - start_point[0] != 1 or start_point[0] % 2:
            log.warning("Unmatched reduce event <%s, %s>.", start_point, end_point)
            return reduce_info
        op_type = self._tag_map.get(start_point[0])
        # append field name with op type.
        if not op_type:
            log.warning("Can't recognize the inner type for point tag: %d.", start_point[0])
            field_name += '_parallel'
        else:
            field_name += '_' + op_type
        reduce_info[field_name] = end_point[1] - start_point[1]
        reduce_info[field_name + '_start_point'] = start_point[1]
        reduce_info[field_name + '_end_point'] = end_point[1]

        return reduce_info
