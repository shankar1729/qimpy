import pytest
import torch
import qimpy as qp
import numpy as np
from ._field import Field, FieldType
from .test_common import get_sequential_grid, get_parallel_grid, get_reference_field
from typing import Type, Sequence, Any


def get_test_field(cls: Type[FieldType], grid: qp.grid.Grid) -> Field[Any]:
    """A highly oscillatory and non-trivial function to test resampling."""
    if cls is qp.grid.FieldC:
        field = get_test_field(qp.grid.FieldR, grid)
        return qp.grid.FieldC(grid, data=field.data * (0.6 + 0.4j))
    assert cls is qp.grid.FieldR
    x = grid.get_mesh("R") / torch.tensor(grid.shape, device=qp.rc.device)
    k1 = (2 * np.pi) * torch.tensor([2, -1, 3], device=qp.rc.device)
    k2 = (2 * np.pi) * torch.tensor([1, 0, -2], device=qp.rc.device)
    return qp.grid.FieldR(grid, data=torch.exp(torch.cos(x @ k1) + torch.sin(x @ k2)))


def get_test_shapes() -> Sequence[Sequence[int]]:
    return (36, 40, 48), (40, 48, 64)


def get_shape_field_combinations() -> Sequence[tuple[Sequence[int], Type]]:
    shapes = get_test_shapes()
    field_types = (qp.grid.FieldR, qp.grid.FieldC, qp.grid.FieldH, qp.grid.FieldG)
    return [(shape, field_type) for shape in shapes for field_type in field_types]


@pytest.mark.mpi
@pytest.mark.parametrize("shape, cls", get_shape_field_combinations())
def test_scatter_gather(shape: Sequence[int], cls: Type[FieldType]) -> None:
    """Check scatter and gather for grids of same shape."""
    # Create sequential and parallel grids of same shape:
    grid_s = get_sequential_grid(shape)
    grid_p = get_parallel_grid(shape)
    # Create fields that are supposed to be identical on both grids:
    field_s = get_reference_field(cls, grid_s)
    field_p = get_reference_field(cls, grid_p)
    # Check that scatter/gather match:
    assert (field_s - field_p.to(grid_s)).norm().max().item() < 1e-8
    assert (field_p - field_s.to(grid_p)).norm().max().item() < 1e-8


@pytest.mark.mpi
@pytest.mark.parametrize("cls", (qp.grid.FieldH, qp.grid.FieldG))
def test_resample_mpi(cls: Type[FieldType]) -> None:
    """Check up/down sampling consistency between MPI and serial versions."""
    # Create sequential and parallel grids of two shapes:
    shapes = get_test_shapes()[:2]
    grids_s = [get_sequential_grid(shape) for shape in shapes]
    grids_p = [get_parallel_grid(shape) for shape in shapes]
    # Create corresponding fields:
    fields_s = [get_reference_field(cls, grid_s) for grid_s in grids_s]
    fields_p = [get_reference_field(cls, grid_p) for grid_p in grids_p]
    # Compare sequential and parallel resampling (in both directions):
    for field1s, field1p, grid2s, grid2p in zip(
        fields_s, fields_p, grids_s[::-1], grids_p[::-1]
    ):
        field2s = field1s.to(grid2s)  # serial reference
        field2p = field1p.to(grid2p)  # parallel-to-parallel sampling
        assert (field2s - field1p.to(grid2s)).norm().max().item() < 1e-8
        assert (field2s.to(grid2p) - field2p).norm().max().item() < 1e-8
        assert (field1s.to(grid2p) - field2p).norm().max().item() < 1e-8


@pytest.mark.mpi_skip
@pytest.mark.parametrize("cls", (qp.grid.FieldR, qp.grid.FieldC))
def test_resample(cls: Type[FieldType]) -> None:
    shapes = get_test_shapes()[:2]
    grid1, grid2 = [get_sequential_grid(shape) for shape in shapes]
    v1 = get_test_field(cls, grid1)
    v2 = get_test_field(cls, grid2)
    tol = 1e-5
    assert (v2 - ~((~v1).to(grid2))).data.abs().max().item() < tol
    assert (v1 - ~((~v2).to(grid1))).data.abs().max().item() < tol


def get_plot_slice(v: Field) -> tuple[np.ndarray, np.ndarray]:
    assert v.grid.comm is None
    Lz = v.grid.lattice.Rbasis[:, 2].norm().item()
    Nz = v.grid.shape[2]
    z = np.arange(Nz) * (Lz / Nz)
    return z, v.data[0, 0].to(qp.rc.cpu).numpy()


def main() -> None:
    qp.utils.log_config()
    qp.rc.init()
    if qp.rc.is_head:
        # Visually inspect resampling sequentially (MPI equivalence captured in tests):
        import matplotlib.pyplot as plt

        grid1, grid2 = [get_sequential_grid(shape) for shape in get_test_shapes()[:2]]
        v1 = get_test_field(qp.grid.FieldR, grid1)
        v2 = get_test_field(qp.grid.FieldR, grid2)
        v12 = ~((~v1).to(grid2))
        v21 = ~((~v2).to(grid1))
        plt.plot(*get_plot_slice(v1), "r", label="Created on 1")
        plt.plot(*get_plot_slice(v21), "r+", label=r"Sampled 2$\to$1")
        plt.plot(*get_plot_slice(v2), "b", label="Created on 2")
        plt.plot(*get_plot_slice(v12), "b+", label=r"Sampled 1$\to$2")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
