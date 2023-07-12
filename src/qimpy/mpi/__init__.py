"""Shared utility functions and classes"""
# List exported symbols for doc generation
__all__ = (
    "globalreduce",
    "ProcessGrid",
    "TaskDivision",
    "TaskDivisionCustom",
    "get_block_slices",
    "BufferView",
    "Iallreduce_in_place",
)

# Temporary alias before actual split of utils into mpi
from ..utils import (
    globalreduce,
    ProcessGrid,
    TaskDivision,
    TaskDivisionCustom,
    get_block_slices,
    BufferView,
    Iallreduce_in_place,
)
