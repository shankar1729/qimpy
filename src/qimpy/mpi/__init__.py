"""MPI communication utilities."""

# List exported symbols for doc generation
__all__ = (
    "BufferView",
    "ProcessGrid",
    "get_comm",
    "TaskDivision",
    "TaskDivisionCustom",
    "get_block_slices",
    "Waitable",
    "Waitless",
    "globalreduce",
)

from ._bufferview import BufferView
from ._process_grid import ProcessGrid, get_comm
from ._taskdivision import TaskDivision, TaskDivisionCustom, get_block_slices
from ._waitable import Waitable, Waitless
from . import globalreduce
