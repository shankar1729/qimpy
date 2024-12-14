from typing import Type, Sequence
from functools import cache

from qimpy import rc
from qimpy.io import Unit
from qimpy.lattice import Lattice
from qimpy.symmetries import Symmetries
from . import Grid
from ._field import FieldType


@cache
def get_sequential_grid(shape: Sequence[int]) -> Grid:
    lattice, symmetries = get_grid_inputs()
    return Grid(lattice=lattice, symmetries=symmetries, shape=shape, comm=None)


@cache
def get_parallel_grid(shape: Sequence[int]) -> Grid:
    lattice, symmetries = get_grid_inputs()
    return Grid(lattice=lattice, symmetries=symmetries, shape=shape, comm=rc.comm)


@cache
def get_reference_field(
    cls: Type[FieldType], grid: Grid, shape_batch: tuple[int, ...] = (2, 3)
) -> FieldType:
    """MPI-reproducible field of specified type on given `grid`."""
    result = cls(grid, shape_batch=shape_batch)  # all zeroes
    result.randomize(seed=0)
    return result


@cache
def get_grid_inputs() -> tuple[Lattice, Symmetries]:
    """Get dummy lattice etc. needed to create grid."""
    lattice = Lattice(
        system=dict(
            name="triclinic",
            a=2.1,
            b=2.2,
            c=2.3,
            alpha=75 * Unit.MAP["deg"],
            beta=80 * Unit.MAP["deg"],
            gamma=85 * Unit.MAP["deg"],
        )
    )  # pick one with no symmetries
    return lattice, Symmetries(lattice=lattice)
