"""MPI communication utilities."""
# List exported symbols for doc generation
__all__ = (
    "BufferView",
    "ProcessGrid",
    "TaskDivision",
    "TaskDivisionCustom",
    "get_block_slices",
    "Waitable",
    "Waitless",
    "Iallreduce_in_place",
    "globalreduce",
)

from ._bufferview import BufferView
from ._process_grid import ProcessGrid
from ._taskdivision import TaskDivision, TaskDivisionCustom, get_block_slices
from ._async_reduce import Iallreduce_in_place
from ._waitable import Waitable, Waitless
from . import globalreduce
