import time
import torch
import qimpy as qp
import numpy as np
import functools
from typing import ClassVar, Dict, List, Tuple, Union


class StopWatch:
    """Simple profiling utility for blocks of code / functions.
    To use, create a StopWatch associated with a name at the start of the code
    block or function being profiled, and call stop() at the end of that block.
    For timing entire functions, use decorator `stopwatch` instead for convenience.
    Use `print_stats` at the end of the run to log statistics of execution times
    of each named code block or function."""

    __slots__ = ["name", "t_start"]
    name: str  #: name of code block
    t_start: Union[float, torch.cuda.Event]  #: start time of current event

    #: timing statistics: list of durations by name
    _stats: ClassVar[Dict[str, List[float]]] = {}

    #: CUDA events for asynchronous timing on GPUs
    _cuda_events: ClassVar[List[Tuple[str, torch.cuda.Event, torch.cuda.Event]]] = []

    def __init__(self, name: str):
        """Start profiling a block of code named `name`."""
        self.name = name
        if qp.rc.use_cuda:
            torch.cuda.nvtx.range_push(name)
            self.t_start = torch.cuda.Event(enable_timing=True)
            self.t_start.record()
        else:
            self.t_start = time.time()

    def stop(self):
        """Stop this watch and collect statistics on it."""
        if self.t_start:
            if qp.rc.use_cuda:
                t_stop = torch.cuda.Event(enable_timing=True)
                t_stop.record()
                torch.cuda.nvtx.range_pop()
                self._cuda_events.append([self.name, self.t_start, t_stop])
                if len(self._cuda_events) > 100:
                    self._process_cuda_events()
            else:
                duration = time.time() - self.t_start
                self._add_stat(self.name, duration)
            self.t_start = None  # prevents repeated stopping

    @classmethod
    def _add_stat(cls, name: str, duration: float):
        """Add a single entry to the timing statistics"""
        stats_entry = cls._stats.get(name)
        if stats_entry:
            stats_entry.append(duration)
        else:
            cls._stats[name] = [duration]

    @classmethod
    def _process_cuda_events(cls):
        """Process pending cuda events"""
        if cls._cuda_events:
            torch.cuda.synchronize()
            for name, t_start, t_stop in cls._cuda_events:
                cls._add_stat(name, t_start.elapsed_time(t_stop) * 1e-3)
            cls._cuda_events = []

    @classmethod
    def print_stats(cls):
        """Print statistics of all timings measured using class StopWatch."""
        cls._process_cuda_events()
        qp.log.info("")
        t_total = 0.0
        n_calls_total = 0
        for name, times in sorted(cls._stats.items()):
            t = np.array(times)
            t_total += t.sum()
            n_calls_total += len(t)
            t_mid = np.median(t)
            t_mad = np.median(abs(t - t_mid))  # mean absolute deviation
            qp.log.info(
                f"StopWatch: {name:30s}  {t_mid:10.6f} +/- {t_mad:10.6f}"
                f" s, {len(t):4d} calls, {t.sum():10.6f} s total"
            )
        qp.log.info(
            f'StopWatch: {"Total":30s}    {"-"*25} {n_calls_total:5d}'
            f" calls, {t_total:10.6f} s total"
        )


def stopwatch(_func=None, *, name=None):
    """Decorator to wrap `StopWatch` around call to function.
    Without arguments, decorator `@stopwatch` uses `__qualname__` of the function
    for `StopWatch.name`. To override this, use the decorator with a keyword-argument
    name i.e. `@stopwatch(name=function_name)`."""

    def stopwatch_named(func):
        @functools.wraps(func)
        def stopwatch_wrapper(*args, **kwargs):
            watch = StopWatch(func.__qualname__ if (name is None) else name)
            result = func(*args, **kwargs)
            watch.stop()
            return result

        return stopwatch_wrapper

    if _func is None:
        return stopwatch_named
    else:
        return stopwatch_named(_func)
