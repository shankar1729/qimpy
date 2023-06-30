import qimpy as qp
import numpy as np
import functools
import torch
from ._field import FieldType
from typing import Type, Sequence


@functools.cache
def get_sequential_grid(shape: Sequence[int]) -> qp.grid.Grid:
    lattice, symmetries = get_grid_inputs()
    return qp.grid.Grid(lattice=lattice, symmetries=symmetries, shape=shape, comm=None)


@functools.cache
def get_parallel_grid(shape: Sequence[int]) -> qp.grid.Grid:
    lattice, symmetries = get_grid_inputs()
    return qp.grid.Grid(
        lattice=lattice, symmetries=symmetries, shape=shape, comm=qp.rc.comm
    )


@functools.cache
def get_reference_field(
    cls: Type[FieldType], grid: qp.grid.Grid, shape_batch: tuple[int, ...] = (2, 3)
) -> FieldType:
    """MPI-reproducible field of specified type on given `grid`."""
    result = cls(grid, shape_batch=shape_batch)  # all zeroes
    shape_mine = result.data.shape
    shape_full = shape_batch + grid.shape  # without split or H symmetry
    offsets = [0] * len(shape_full)
    if cls is qp.grid.FieldH:
        offsets[-1] = grid.split2H.i_start
    elif cls is qp.grid.FieldG:
        offsets[-1] = grid.split2.i_start
    else:
        offsets[-3] = grid.split0.i_start
    # Make each entry unique:
    for i_dim, offset in enumerate(offsets):
        cur_offset = torch.arange(shape_mine[i_dim], device=qp.rc.device) + offset
        stride = np.prod(shape_full[i_dim + 1 :])
        bcast_shape = [1] * len(shape_full)
        bcast_shape[i_dim] = -1
        result.data += stride * cur_offset.view(bcast_shape)
    # Introduce imaginary parts where not zero in general:
    if result.dtype().is_complex:
        result.data *= 1 + 0.1j
        if cls is qp.grid.FieldH:
            i2_real = torch.where(grid.weight2H == 1.0)[0]
            for i2 in i2_real.tolist():
                result.data[..., i2] = result.data[..., i2].real
    return result


@functools.cache
def get_grid_inputs() -> tuple[qp.lattice.Lattice, qp.symmetries.Symmetries]:
    """Get dummy lattice etc. needed to create grid."""
    lattice = qp.lattice.Lattice(
        system="triclinic", a=2.1, b=2.2, c=2.3, alpha=75, beta=80, gamma=85
    )  # pick one with no symmetries
    ions = qp.ions.Ions(lattice=lattice)
    symmetries = qp.symmetries.Symmetries(lattice=lattice, ions=ions)
    return lattice, symmetries
