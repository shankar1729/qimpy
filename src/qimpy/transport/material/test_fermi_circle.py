from __future__ import annotations

import numpy as np
import torch
import pytest

from qimpy import log, rc
from qimpy.io import log_config
from qimpy.mpi import get_process_grid
from qimpy.transport.material import FermiCircle

pytestmark = pytest.mark.mpi_skip  # All tests in this module are serial


def get_fermi_circle(**kwargs) -> FermiCircle:
    """Wrap FermiCircle construction, handling process grid and adding more defaults."""
    kwargs.setdefault("n_theta", 32)
    kwargs.setdefault("kF", 1.5)
    kwargs.setdefault("vF", 0.8)
    kwargs.setdefault("dk_r", 0.1)
    kwargs.setdefault("T", kwargs["vF"] * kwargs["dk_r"])
    return FermiCircle(**kwargs, process_grid=get_process_grid("rk", (1, 1)))


def test_contact_n_r_equivalence() -> None:
    """Verify that contact distribution at large n_r > 1 matches n_r = 1."""
    t = 0.0
    dmu = 1.0e-6
    n_hat = torch.tensor([[1.0, 0.0]], device=rc.device)
    fc0 = get_fermi_circle(n_r=1)
    n_ref = fc0.measure_observables(fc0.get_contactor(n_hat, dmu=dmu)(t), t)
    for n_r in [2, 4, 8, 16, 32]:
        fc = get_fermi_circle(n_r=n_r)
        n_check = fc.measure_observables(fc.get_contactor(n_hat, dmu=dmu)(t), t)[:, 0]
        relative_error = (n_check / n_ref - 1.0).item()
        log.info(f"{n_r=}: {relative_error=:.3e}")
        assert abs(relative_error) < 2 * np.exp(-0.5 * n_ref / dmu)


@pytest.mark.parametrize("n_r", [1, 4])
@pytest.mark.parametrize(
    "tau_p, tau_ee, r_c, conserve_p",
    [
        (np.inf, np.inf, np.inf, True),
        (np.inf, 0.1, np.inf, True),
        (2.0, 0.1, np.inf, False),
        (np.inf, np.inf, 0.5, False),
    ],
)
def test_scattering_conservation(
    n_r: int, tau_p: float, tau_ee: float, r_c: float, conserve_p: bool
) -> None:
    """Test conservation properties of scattering kernel (and magentic field)"""
    fc = get_fermi_circle(n_r=n_r, tau_p=tau_p, tau_ee=tau_ee, r_c=r_c)
    torch.manual_seed(0)
    rho = torch.rand(100, fc.nk_mine) * 2 - 1  # Random on [-1, 1]
    t = 0.0
    rho_dot = fc.rho_dot(rho, t, 0)

    # Charge (and energy for n_r > 1):
    obs_dot = fc.measure_observables(rho_dot, t)
    check_conservation("Charge", obs_dot[:, 0], 1.0e-11)
    if n_r > 1:
        check_conservation("Energy", obs_dot[:, 1], 1.0e-11)

    # Momentum:
    if conserve_p:
        p_dot = (rho_dot[:, None] * fc.transport_velocity.T) @ fc.wk
        check_conservation("Momentum", p_dot[:, 1], 1.0e-11)


def check_conservation(name: str, value_dot: torch.Tensor, tol: float) -> None:
    err_mean = value_dot.abs().mean().item()
    err_max = value_dot.abs().max().item()
    log.info(f"{name} conservation error: mean={err_mean:.3e} max={err_max:.3e}")
    if err_max > tol:
        raise AssertionError(f"{name} conservation error {err_max:.3e} > {tol:.3e}")


# TODO: port the following test ideas from test_fermi_surface
# + scattering conservation of charge, momentum and energy, as appropriate
# + reflection propertes at all angles: charge, energy, tang. mom. conservation


def main():
    log_config()
    pytest.main([__file__, "-vs"])


if __name__ == "__main__":
    main()
