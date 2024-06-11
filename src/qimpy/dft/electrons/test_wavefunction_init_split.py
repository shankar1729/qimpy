from typing import Optional
import functools

import torch
import pytest

from qimpy import rc, dft
from qimpy.io import Unit
from qimpy.mpi import TaskDivision
from . import Wavefunction


@functools.cache
def make_system(real: bool, spinorial: bool, polarized: bool) -> dft.System:
    """Make a system to test with."""
    if real:
        # Gamma-point only, non-spinorial (make a bit larger since no k)
        assert not spinorial
        lattice = {
            "system": {
                "name": "monoclinic",
                "a": 15.0,
                "b": 20.0,
                "c": 18.0,
                "beta": 105.0 * Unit.MAP["deg"],
            }
        }
        kmesh = {}
    else:
        lattice = {"system": {"name": "hexagonal", "a": 9.0, "c": 12.0}}
        kmesh = {"size": [4, 4, 3]}
    return dft.System(
        lattice=lattice,
        electrons={"k-mesh": kmesh, "basis": {"real-wavefunctions": real}},
        process_grid_shape=(1, 1, -1),  # ensure basis-split
    )


@functools.cache
def get_Cg(
    system: dft.System, n_bands: int, b_start: int = 0, b_stop: Optional[int] = None
) -> Wavefunction:
    """Create basis-split wavefunction, selectively randomized."""
    Cg = Wavefunction(system.electrons.basis, n_bands=n_bands)
    Cg.randomize(b_start=b_start, b_stop=b_stop)
    return Cg


@functools.cache
def get_Cb(
    system: dft.System, n_bands: int, b_start: int = 0, b_stop: Optional[int] = None
) -> Wavefunction:
    """Create band-split wavefunction, selectively randomized."""
    div = system.electrons.basis.division
    band_division = TaskDivision(n_tot=n_bands, n_procs=div.n_procs, i_proc=div.i_proc)
    Cb = Wavefunction(system.electrons.basis, band_division=band_division)
    Cb.randomize(b_start=b_start, b_stop=b_stop)
    return Cb


# Combinations of system, n_bands, b_start, b_stop
parameter_combinations = [
    (make_system(real=True, spinorial=False, polarized=True), 317, 8, 51),
    (make_system(real=False, spinorial=False, polarized=False), 128, 16, 96),
    (make_system(real=False, spinorial=True, polarized=False), 64, 0, 64),
]

# Combinations with system, n_bands alone:
system_band_combinations = [combination[:2] for combination in parameter_combinations]


@pytest.mark.mpi
@pytest.mark.parametrize("system, n_bands, b_start, b_stop", parameter_combinations)
def test_split_bands_inverse(
    system: dft.System, n_bands: int, b_start: int, b_stop: int
) -> None:
    """Check that inv(split_bands) = split_basis."""
    Cg = get_Cg(system, n_bands, b_start, b_stop)
    Cgb = Cg.split_bands().wait()
    Cgbg = Cgb.split_basis().wait()
    assert (Cg - Cgbg).norm() < 1e-8


@pytest.mark.mpi
@pytest.mark.parametrize("system, n_bands, b_start, b_stop", parameter_combinations)
def test_split_basis_inverse(
    system: dft.System, n_bands: int, b_start: int, b_stop: int
) -> None:
    """Check that inv(split_basis) = split_bands."""
    Cb = get_Cb(system, n_bands, b_start, b_stop)
    Cbg = Cb.split_basis().wait()
    Cbgb = Cbg.split_bands().wait()
    assert (Cb - Cbgb).norm() < 1e-8


@pytest.mark.mpi
@pytest.mark.parametrize("system, n_bands, b_start, b_stop", parameter_combinations)
def test_random_equiv(
    system: dft.System, n_bands: int, b_start: int, b_stop: int
) -> None:
    """Check that randomization is consistent between different splits."""
    Cg = get_Cg(system, n_bands, b_start, b_stop)
    Cb = get_Cb(system, n_bands, b_start, b_stop)
    assert (Cb - Cg.split_bands().wait()).norm() < 1e-8
    assert (Cg - Cb.split_basis().wait()).norm() < 1e-8


@pytest.mark.parametrize("system, n_bands", system_band_combinations)
def test_orthonormalize(system: dft.System, n_bands: int) -> None:
    C = get_Cg(system, n_bands).orthonormalize()
    C_OC = C.dot_O(C).wait()
    expected = torch.eye(n_bands, device=rc.device)[None, None]
    assert (C_OC - expected).abs().max().item() < 1e-8
