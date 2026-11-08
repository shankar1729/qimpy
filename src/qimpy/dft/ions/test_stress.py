from qimpy.dft import System

DELTA = 5e-4  # isotropic strain for the central difference


def scale_lattice(system: System, scale: float) -> float:
    """Scale the lattice in place, as the geometry stepper does, and return the
    self-consistent energy. The wavefunction basis keeps its G-vectors, so the
    energy difference contains no Pulay term, like the analytic lattice gradient."""
    lattice = system.lattice
    lattice.update(lattice.Rbasis * scale, report_change=False)
    system.coulomb.update_lattice_dependent()
    system.energy.clear()
    system.ions.update(system)
    system.electrons.C = system.electrons.C.orthonormalize()
    system.electrons.update(system)
    # system.electrons.run(system)
    return float(system.energy)


def test_stress_partial_core(system_Si) -> None:
    """For the isotropic strain Rbasis -> (1 + delta) Rbasis, dE/d(delta) is the trace
    of the lattice gradient (dE/d(strain), i.e. stress times volume)."""
    system = system_Si
    system.lattice.compute_stress = True
    system.run()
    assert system.lattice.grad is not None
    dE_analytic = float(system.lattice.grad.trace())
    E_plus = scale_lattice(system, 1.0 + DELTA)
    E_minus = scale_lattice(system, (1.0 - DELTA) / (1.0 + DELTA))
    dE_numerical = (E_plus - E_minus) / (2.0 * DELTA)
    print(f"dE/d(delta): analytic {dE_analytic:.8f}, numerical {dE_numerical:.8f} Eh")
    assert abs(dE_analytic - dE_numerical) < 1e-3
