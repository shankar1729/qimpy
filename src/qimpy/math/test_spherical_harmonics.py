import numpy as np
import torch
import pytest

from qimpy import rc, log
from qimpy.io import log_config
from . import spherical_harmonics as sh
from ._spherical_harmonics_generate import get_harmonics_ref


def get_r_ylm() -> tuple[torch.Tensor, torch.Tensor]:
    """Get test data for all the harmonics tests."""
    torch.manual_seed(0)
    r = torch.randn(10, 10000, 3, device=rc.device)
    ylm = sh.get_harmonics(sh.L_MAX, r)
    return r, ylm


@pytest.fixture(scope="module")
def r_ylm() -> tuple[torch.Tensor, torch.Tensor]:
    return get_r_ylm()


@pytest.mark.mpi_skip
def test_ylm(r_ylm: tuple[torch.Tensor, torch.Tensor]) -> None:
    log.info("Testing spherical harmonics:")
    r, ylm = r_ylm
    rel_err_all = []
    ylm_ref = torch.from_numpy(get_harmonics_ref(sh.L_MAX, r.to(rc.cpu).numpy())).to(
        rc.device
    )
    for l in range(sh.L_MAX + 1):
        l_slice = slice(l**2, (l + 1) ** 2)
        err = (ylm[l_slice] - ylm_ref[l_slice]).norm().item()
        rel_err = err / (ylm_ref[l_slice]).norm().item()
        rel_err_all.append(rel_err)
        log.info(f"  l: {l} Err: {rel_err:9.3e}")
    rel_err_overall = np.sqrt((np.array(rel_err_all) ** 2).mean())
    log.info(f"  Overall Err: {rel_err_overall:9.3e}\n")
    assert rel_err_overall < 1e-14


@pytest.mark.mpi_skip
def test_ylm_prime(r_ylm: tuple[torch.Tensor, torch.Tensor]) -> None:
    log.info("Testing derivatives:")
    # Analytical calculation:
    r, ylm = r_ylm
    _, ylm_prime = sh.get_harmonics_and_prime(sh.L_MAX, r)
    err = torch.zeros((ylm.shape[0], 3), device=r.device, dtype=r.dtype)
    err_den_sq = torch.zeros(ylm.shape[0], device=r.device, dtype=r.dtype)
    # Compare to numerical derivative:
    dr_mag = 1e-4
    dr_shape = (1,) * (len(r.shape) - 1) + (3,)
    for i_dir in range(3):
        dr = torch.zeros(dr_shape, device=r.device, dtype=r.dtype)
        dr[..., i_dir] = dr_mag
        ylm_prime_num = (0.5 / dr_mag) * (
            sh.get_harmonics(sh.L_MAX, r + dr) - sh.get_harmonics(sh.L_MAX, r - dr)
        )
        err[:, i_dir] = (ylm_prime_num - ylm_prime[i_dir]).flatten(1).norm(dim=-1)
        err_den_sq += ylm_prime_num.flatten(1).norm(dim=1).square()
    err[1:] *= 1.0 / err_den_sq.sqrt()[1:, None]  # to relative error
    # Report:
    err_np = err.to(rc.cpu).numpy()
    for l in range(sh.L_MAX + 1):
        for m in range(-l, l + 1):
            err_x, err_y, err_z = err_np[l * (l + 1) + m]
            log.info(f"  l: {l} m: {m:+d}  Err: {err_x:.3e} {err_y:.3e} {err_z:.3e}")
    # Overall error:
    err_avg = err_np.mean(axis=0)
    err_x, err_y, err_z = err_avg
    log.info(f"      Overall Err: {err_x:.3e} {err_y:.3e} {err_z:.3e}")
    assert np.all(err_avg < 1e-8)


@pytest.mark.mpi_skip
def test_ylm_product() -> None:
    log.info("Testing product coefficients:")
    l_max = sh.L_MAX_HLF
    coeff = sh.get_product_coefficients(l_max)
    torch.manual_seed(1)
    r_hat = torch.nn.functional.normalize(torch.randn(1000, 3, device=rc.device), dim=1)
    Y = sh.get_harmonics(2 * l_max, r_hat)
    n_lm = (l_max + 1) ** 2
    product = Y[:n_lm, None] * Y[None, :n_lm]
    assert torch.allclose(torch.einsum("abL, Lr -> abr", coeff, Y), product, atol=1e-13)
    assert torch.equal(coeff, coeff.transpose(0, 1))
    Y00 = 1.0 / np.sqrt(4 * np.pi)  # Y_a Y_b integrates to delta_ab = c_ab^00 / Y00
    assert torch.allclose(coeff[:, :, 0], Y00 * torch.eye(n_lm, device=rc.device))


def main():
    """Run test with verbose log."""
    log_config()
    rc.init()
    assert rc.n_procs == 1
    r, ylm = get_r_ylm()
    test_ylm((r, ylm))
    test_ylm_prime((r, ylm))
    test_ylm_product()


if __name__ == "__main__":
    main()
