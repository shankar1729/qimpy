from typing import Sequence, Type

import pytest
import torch

from . import FieldType
from .test_common import get_parallel_grid, get_reference_field
from .test_fft import get_shape_batch_field_combinations


@pytest.mark.parametrize(
    "shape, n_batch, cls", get_shape_batch_field_combinations(include_tilde=False)
)
def test_parseval(
    shape: Sequence[int], n_batch: Sequence[int], cls: Type[FieldType], n_repeat=0
) -> None:
    grid = get_parallel_grid(shape)
    field = get_reference_field(cls, grid, n_batch)
    field_tilde = ~field
    result = field ^ field
    result_tilde = field_tilde ^ field_tilde
    assert (result - result_tilde).norm() < 1e-8 * result.norm()


@pytest.mark.parametrize(
    "shape, n_batch, cls", get_shape_batch_field_combinations(include_tilde=False)
)
def test_integral(
    shape: Sequence[int], n_batch: Sequence[int], cls: Type[FieldType], n_repeat=0
) -> None:
    grid = get_parallel_grid(shape)
    field = get_reference_field(cls, grid, n_batch)
    # Test integral against dot product with 1s
    ones = cls(grid, data=torch.ones_like(field.data))
    integral_ref = ones ^ field
    tol = 1e-8 * integral_ref.norm()
    assert (field.integral() - integral_ref).norm() < tol
    assert ((~field).integral() - integral_ref).norm() < tol


@pytest.mark.mpi_skip
def test_gradient_operator_follows_lattice() -> None:
    """The gradient operator must use the current lattice (which changes in
    variable-cell relaxations), not the one at its first use."""
    from qimpy.lattice import Lattice
    from qimpy.symmetries import Symmetries
    from . import Grid

    lattice = Lattice(system=dict(name="cubic", a=5.0))
    grid = Grid(
        lattice=lattice,
        symmetries=Symmetries(lattice=lattice),
        shape=(8, 8, 8),
        group=None,
    )
    iG = grid.get_mesh("H").permute(3, 0, 1, 2).to(torch.double)
    grid.get_gradient_operator("H", False)  # first use (cached mesh)
    lattice.update(1.1 * lattice.Rbasis, report_change=False)
    expected = 1j * torch.tensordot(lattice.Gbasis, iG, dims=1)
    assert torch.allclose(grid.get_gradient_operator("H", False), expected)
