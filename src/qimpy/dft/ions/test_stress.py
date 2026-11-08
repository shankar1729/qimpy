import pytest
import torch

from qimpy.dft import System


@pytest.mark.parametrize("system_name", ["system_Si", "system_Si_perturbed"])
def test_stress(system_name: str, request) -> None:
    """For the isotropic strain Rbasis -> (1 + delta) Rbasis, dE/d(delta) is the trace
    of the lattice gradient (dE/d(strain), i.e. stress times volume)."""
    system: System = request.getfixturevalue(system_name)
    lattice = system.lattice
    lattice.compute_stress = True
    system.run()
    assert lattice.grad is not None

    # Generate a commensurate perturbation:
    torch.manual_seed(0)
    strain = system.symmetries.symmetrize_matrix(torch.rand_like(lattice.grad))
    dRbasis = strain @ lattice.Rbasis
    print(lattice.grad)

    def step(scale: float) -> float:
        lattice.update(lattice.Rbasis + scale * dRbasis, report_change=False)
        system.coulomb.update_lattice_dependent()
        system.energy.clear()
        system.ions.update(system)
        system.electrons.C = system.electrons.C.orthonormalize()
        system.electrons.update(system)
        return float(system.energy)

    dE_analytic = float((lattice.grad * strain).sum())
    DELTA = 5.0e-4
    E_plus = step(DELTA)
    E_minus = step(-2 * DELTA)
    dE_numerical = (E_plus - E_minus) / (2 * DELTA)
    dE_ratio = dE_analytic / dE_numerical
    print(f"{system_name}: dE_ratio: {dE_ratio:.8f}")
    assert abs(dE_ratio - 1.0) < 1e-3
