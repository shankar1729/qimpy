"""Lattice gradient (stress) against finite differences of the energy, for silicon with
a partial-core pseudopotential: GBRV, found via QIMPY_PSEUDO_DIR (e.g. the JDFTx
library), else skipped. Covers the volume term of the partial-core density, which is
about 2 Eh of the isotropic lattice gradient here."""

import os

import pytest

from qimpy.dft import System

PSEUDOPOTENTIAL = "GBRV/$ID_pbe.uspp"  # includes a partial core
DELTA = 5e-4  # isotropic strain for the central difference


def find_pseudopotential() -> bool:
    """Whether the silicon pseudopotential can be located."""
    prefixes = [""] + os.environ.get("QIMPY_PSEUDO_DIR", "").split(":")
    filename = PSEUDOPOTENTIAL.replace("$ID", "si")
    return any(os.path.isfile(os.path.join(prefix, filename)) for prefix in prefixes)


def get_system() -> System:
    a = 5.175323929535814  # half the cubic lattice constant (bohr)
    return System(
        lattice=dict(
            Rbasis=[[0.0, a, a], [a, 0.0, a], [a, a, 0.0]], compute_stress=True
        ),
        ions=dict(
            pseudopotentials=PSEUDOPOTENTIAL,
            coordinates=[["Si", 0.0, 0.0, 0.0], ["Si", 0.25, 0.25, 0.25]],
        ),
        electrons=dict(
            basis=dict(ke_cutoff=10.0),
            k_mesh=dict(size=[2, 2, 2]),
            scf=dict(n_iterations=100, energy_threshold=1e-11),
        ),
        grid=dict(ke_cutoff=100.0),
        geometry="fixed",
        checkpoint=None,
        checkpoint_out=None,
    )


def scale_lattice(system: System, scale: float) -> float:
    """Scale the lattice in place, as the geometry stepper does, and return the
    self-consistent energy. The wavefunction basis keeps its G-vectors, so the
    energy difference contains no Pulay term, like the analytic lattice gradient."""
    lattice = system.lattice
    lattice.update(lattice.Rbasis * scale, report_change=False)
    system.coulomb.update_lattice_dependent()
    system.energy.clear()
    system.ions.update(system)
    system.electrons.run(system)
    return float(system.energy)


@pytest.mark.mpi_skip
@pytest.mark.skipif(not find_pseudopotential(), reason="needs GBRV pseudopotential")
def test_stress_partial_core() -> None:
    """For the isotropic strain Rbasis -> (1 + delta) Rbasis, dE/d(delta) is the trace
    of the lattice gradient (dE/d(strain), i.e. stress times volume)."""
    system = get_system()
    system.run()
    assert system.lattice.grad is not None
    dE_analytic = float(system.lattice.grad.trace())
    E_plus = scale_lattice(system, 1.0 + DELTA)
    E_minus = scale_lattice(system, (1.0 - DELTA) / (1.0 + DELTA))
    dE_numerical = (E_plus - E_minus) / (2.0 * DELTA)
    print(f"dE/d(delta): analytic {dE_analytic:.8f}, numerical {dE_numerical:.8f} Eh")
    assert abs(dE_analytic - dE_numerical) < 1e-3
