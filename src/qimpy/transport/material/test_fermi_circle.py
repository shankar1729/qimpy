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
    log.info("\n")
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
        assert abs(relative_error) < 2 * np.exp(-0.5 * n_ref.item() / dmu)


@pytest.mark.parametrize("n_r", [1, 4])
@pytest.mark.parametrize(
    "tau_p, tau_ee, r_c, conserve_p, conserve_E",
    [
        (np.inf, np.inf, np.inf, True, True),
        (np.inf, 0.1, np.inf, True, True),
        (2.0, 0.1, np.inf, False, False),
        (np.inf, np.inf, 0.5, False, True),
    ],
)
def test_scattering_conservation(
    n_r: int,
    tau_p: float,
    tau_ee: float,
    r_c: float,
    conserve_p: bool,
    conserve_E: bool,
) -> None:
    """Test conservation properties of scattering kernel (and magnetic field)"""
    log.info("\n")
    fc = get_fermi_circle(n_r=n_r, tau_p=tau_p, tau_ee=tau_ee, r_c=r_c)
    torch.manual_seed(0)
    rho = torch.rand(100, fc.nk_mine, device=rc.device) * 2 - 1  # Random on [-1, 1]
    t = 0.0
    rho_dot = fc.rho_dot(rho, t)

    # Charge (and energy for n_r > 1):
    obs_dot = fc.measure_observables(rho_dot, t)
    check_conservation("Charge", obs_dot[:, 0], 1.0e-11)
    if (n_r > 1) and conserve_E:
        check_conservation("Energy", obs_dot[:, 1], 1.0e-11)

    # Momentum:
    if conserve_p:
        p_dot = (rho_dot[:, None] * fc.transport_velocity.T) @ fc.wk
        check_conservation("Momentum", p_dot[:, 1], 1.0e-11)


@pytest.mark.parametrize("n_r", [1, 4])
@pytest.mark.parametrize("specularity", [0.37, 1.0])
def test_reflector_conservation(n_r: int, specularity: float) -> None:
    log.info("\n")
    fc = get_fermi_circle(n_r=n_r, n_theta=256, specularity=specularity)
    torch.manual_seed(0)
    theta = torch.rand(100, device=rc.device) * (2 * np.pi)  # Random wall orientations
    n_hat = torch.stack((theta.cos(), theta.sin()), dim=1)
    t_hat = torch.stack((-n_hat[:, 1], n_hat[:, 0]), dim=1)  # tangential directions
    v = fc.transport_velocity
    v_n = n_hat @ v.T
    v_t = t_hat @ v.T
    rho_incident = torch.rand(len(theta), fc.nk_mine, device=rc.device) * 2 - 1
    rho_incident[v_t < 0.0] *= -1  # Introduce net tangential bias (else flux ~ 0)
    rho_reflected = fc.get_reflector(n_hat)(rho_incident)

    # Compare incident and reflected normal fluxes:
    v_n_i = v_n.clamp(min=0.0)
    v_n_r = (-v_n).clamp(min=0.0)
    flux_incident = fc.measure_observables(rho_incident * v_n_i, 0.0)
    flux_reflected = fc.measure_observables(rho_reflected * v_n_r, 0.0)
    flux_rel_err = flux_reflected / flux_incident - 1.0
    check_conservation("Relative charge flux", flux_rel_err[:, 0], 1.0e-11)
    if n_r > 1:
        check_conservation("Relative energy flux", flux_rel_err[:, 1], 1.0e-11)

    # Compare incident and reflected tangential fluxes (not exactly conserved):
    flux_t_incident = fc.measure_observables(rho_incident * v_t * v_n_i, 0.0)
    flux_t_reflected = fc.measure_observables(rho_reflected * v_t * v_n_r, 0.0)
    flux_t_rel_err = flux_t_reflected / flux_t_incident - specularity
    check_conservation("Tangential charge flux", flux_t_rel_err[:, 0], 0.1, False)
    if n_r > 1:
        check_conservation("Tangential energy flux", flux_t_rel_err[:, 1], 0.1, False)


def check_conservation(
    name: str, value_dot: torch.Tensor, tol: float, check_max: bool = True
) -> None:
    err_mean = value_dot.abs().mean().item()
    err_max = value_dot.abs().max().item()
    log.info(f"{name} conservation error: mean={err_mean:.3e} max={err_max:.3e}")
    err_check = err_max if check_max else err_mean
    if err_check > tol:
        raise AssertionError(f"{name} conservation error {err_check:.3e} > {tol:.3e}")


def main():
    log_config()
    pytest.main([__file__, "-vs"])


if __name__ == "__main__":
    main()
