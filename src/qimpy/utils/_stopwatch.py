import time
import torch
import qimpy as qp
import numpy as np
from ._runconfig import RunConfig
from typing import ClassVar, Dict, List, Tuple, Union


class StopWatch:
    """Simple profiling utility for blocks of code / functions.
    To use, create a StopWatch associated with a name at the start of
    the code block or function being profiled, and call stop() at the
    end of that block. The class collects together statistics of
    execution time by name, which can be logged using print_stats()."""

    __slots__ = ['name', 'use_cuda', 't_start']
    name: str  #: name of code block
    use_cuda: bool  #: whether profiling is for a GPU device
    t_start: Union[float, torch.cuda.Event]  #: start time of current event

    #: timing statistics: list of durations by name
    _stats: ClassVar[Dict[str, List[float]]] = {}

    #: CUDA events for asynchronous timing on GPUs
    _cuda_events: ClassVar[List[Tuple[str, torch.cuda.Event,
                                      torch.cuda.Event]]] = []

    def __init__(self, name: str, rc: RunConfig):
        """Start profiling a block of code named `name`."""
        self.use_cuda = rc.use_cuda
        self.name = name
        if self.use_cuda:
            self.t_start = torch.cuda.Event(enable_timing=True)
            self.t_start.record()
        else:
            self.t_start = time.time()

    def stop(self):
        "Stop this watch and collect statistics on it."
        if self.t_start:
            if self.use_cuda:
                t_stop = torch.cuda.Event(enable_timing=True)
                t_stop.record()
                self._cuda_events.append([self.name, self.t_start, t_stop])
                if len(self._cuda_events) > 100:
                    self._process_cuda_events()
            else:
                duration = time.time() - self.t_start
                self._add_stat(self.name, duration)
            self.t_start = None  # prevents repeated stopping

    @classmethod
    def _add_stat(cls, name: str, duration: float):
        "Add a single entry to the timing statistics"
        stats_entry = cls._stats.get(name)
        if stats_entry:
            stats_entry.append(duration)
        else:
            cls._stats[name] = [duration]

    @classmethod
    def _process_cuda_events(cls):
        "Process pending cuda events"
        if cls._cuda_events:
            torch.cuda.synchronize()
            for name, t_start, t_stop in cls._cuda_events:
                cls._add_stat(name, t_start.elapsed_time(t_stop)*1e-3)
            cls._cuda_events = []

    @classmethod
    def print_stats(cls):
        """Print statistics of all timings measured using class StopWatch."""
        cls._process_cuda_events()
        qp.log.info('')
        for name, times in cls._stats.items():
            t = np.array(times)
            qp.log.info(
                f'StopWatch: {name:30s}  {t.mean():10.6f} +/- {t.std():10.6f}'
                f' s, {len(t):4d} calls, {t.sum():10.6f} s total')
