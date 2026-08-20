"""MPI communication utilities."""

# List exported symbols for doc generation
__all__ = (
    "ProcessGrid",
    "get_comm",
    "TaskDivision",
    "TaskDivisionCustom",
    "get_block_slices",
    "Waitable",
    "Waitless",
    "all_gather_padded",
    "globalreduce",
)

from ._process_grid import ProcessGrid, get_comm
from ._taskdivision import TaskDivision, TaskDivisionCustom, get_block_slices
from ._waitable import Waitable, Waitless
from ._wrappers import all_gather_padded
from . import globalreduce
