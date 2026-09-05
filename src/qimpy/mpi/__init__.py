"""MPI communication utilities."""

# List exported symbols for doc generation
__all__ = (
    "ProcessGrid",
    "all_gather_padded",
    "all_gather_scalars",
    "all_reduce_scalars",
    "TaskDivision",
    "TaskDivisionCustom",
    "get_block_slices",
    "Waitable",
    "Waitless",
    "globalreduce",
)

from ._process_grid import ProcessGrid
from ._wrappers import all_gather_padded, all_gather_scalars, all_reduce_scalars
from ._taskdivision import TaskDivision, TaskDivisionCustom, get_block_slices
from ._waitable import Waitable, Waitless
from . import globalreduce
