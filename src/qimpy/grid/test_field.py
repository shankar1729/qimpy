import pytest
import torch
from typing import Sequence, Type
from ._field import FieldType
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
