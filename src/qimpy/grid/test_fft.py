from typing import Sequence, Type

import pytest

from qimpy import rc, log
from qimpy.io import log_config
from qimpy.utils import StopWatch
from . import FieldType, FieldR, FieldH, FieldC, FieldG
from .test_common import get_sequential_grid, get_parallel_grid, get_reference_field


def get_shape_batch_field_combinations(
    include_tilde: bool,
) -> Sequence[tuple[Sequence[int], Sequence[int], Type]]:
    shapes = ((48, 64, 96), (64, 72, 128))
    n_batches = ((2, 3), tuple[int, ...]())
    field_types = [FieldR, FieldC]
    if include_tilde:
        field_types += [FieldH, FieldG]
    return [
        (shape, n_batch, field_type)
        for shape, n_batch in zip(shapes, n_batches)
        for field_type in field_types
    ]


@pytest.mark.mpi
@pytest.mark.parametrize(
    "shape, n_batch, cls", get_shape_batch_field_combinations(include_tilde=True)
)
def test_fft(
    shape: Sequence[int], n_batch: Sequence[int], cls: Type[FieldType], n_repeat=0
) -> None:
    """Check parallel FFT against serial version."""
    # Create sequential and parallel grids of same shape:
    grid_s = get_sequential_grid(shape)
    grid_p = get_parallel_grid(shape)
    # Create fields that are supposed to be identical on both grids:
    field_s = get_reference_field(cls, grid_s, n_batch)
    field_p = get_reference_field(cls, grid_p, n_batch)
    # Check that serial and parallel versions match:
    field_s_tilde = ~field_s
    tol = 1e-8 * field_s_tilde.norm().max()
    assert (field_s_tilde.to(grid_p) - (~field_p)).norm().max().item() < tol
    # Time repetitions if needed:
    if n_repeat:
        for field, name in ((field_s, "seq"), (field_p, "par")):
            for i_repeat in range(n_repeat):
                watch = StopWatch(f"{cls.__name__}.fft({name})")
                field_tilde = ~field
                watch.stop()
                log.info(f"Rep: {i_repeat}  norm: {field_tilde.norm().max().item()}")


def main():
    log_config()
    rc.init()
    for shape, n_batch, field_type in get_shape_batch_field_combinations(
        include_tilde=True
    ):
        test_fft(shape, n_batch, field_type, n_repeat=10)
    StopWatch.print_stats()


if __name__ == "__main__":
    main()
