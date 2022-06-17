import qimpy as qp
import numpy as np
from qimpy.ions import spherical_harmonics as sh
from qimpy.ions.spherical_harmonics_generate import get_harmonics_ref
from typing import Tuple
import torch
import pytest


def get_r_ylm() -> Tuple[torch.Tensor, torch.Tensor]:
    """Get test data for all the harmonics tests."""
    torch.manual_seed(0)
    r = torch.randn(10, 10000, 3, device=qp.rc.device)
    ylm = sh.get_harmonics(sh.L_MAX, r)
    return r, ylm


@pytest.fixture(scope="module")
def r_ylm() -> Tuple[torch.Tensor, torch.Tensor]:
    return get_r_ylm()


@pytest.mark.mpi_skip
def test_ylm(r_ylm: Tuple[torch.Tensor, torch.Tensor]) -> None:
    qp.log.info("Testing spherical harmonics:")
    r, ylm = r_ylm
    rel_err_all = []
    ylm_ref = torch.from_numpy(get_harmonics_ref(sh.L_MAX, r.to(qp.rc.cpu).numpy())).to(
        qp.rc.device
    )
    for l in range(sh.L_MAX + 1):
        l_slice = slice(l ** 2, (l + 1) ** 2)
        err = (ylm[l_slice] - ylm_ref[l_slice]).norm().item()
        rel_err = err / (ylm_ref[l_slice]).norm().item()
        rel_err_all.append(rel_err)
        qp.log.info(f"  l: {l} Err: {rel_err:9.3e}")
    rel_err_overall = np.sqrt((np.array(rel_err_all) ** 2).mean())
    qp.log.info(f"  Overall Err: {rel_err_overall:9.3e}\n")
    assert rel_err_overall < 1e-14


@pytest.mark.mpi_skip
def test_ylm_prod(r_ylm: Tuple[torch.Tensor, torch.Tensor]) -> None:
    qp.log.info("Testing product coefficients:")
    r, ylm = r_ylm
    r_sq = (r ** 2).sum(dim=-1)
    if not sh._YLM_PROD:
        sh._initialize_device(qp.rc.device)
    rel_err_all = []
    dl_shape = (-1,) + (1,) * len(r_sq.shape)  # bcast with r_sq[None]
    for l1 in range(sh.L_MAX_HLF + 1):
        l1_slice = slice(l1 ** 2, (l1 + 1) ** 2)
        for l2 in range(l1 + 1):
            l2_slice = slice(l2 ** 2, (l2 + 1) ** 2)
            product_ref = ylm[l1_slice, None, :] * ylm[None, l2_slice, :]
            product = torch.zeros_like(product_ref)
            for m1 in range(-l1, l1 + 1):
                ilm1 = l1 * (l1 + 1) + m1
                for m2 in range(-l2, l2 + 1):
                    ilm2 = l2 * (l2 + 1) + m2
                    index, coeffs = sh._YLM_PROD[(max(ilm1, ilm2), min(ilm1, ilm2))]
                    l_net = index.sqrt().floor().to(torch.int)
                    dl_by_2 = (l1 + l2 - l_net).div(2, rounding_mode="floor")
                    prod_terms = ylm[index] * (
                        r_sq[None, ...] ** dl_by_2.view(dl_shape)
                    )
                    product[l1 + m1, l2 + m2] = torch.tensordot(
                        coeffs, prod_terms, dims=1
                    )
            err = (product - product_ref).norm().item()
            rel_err = err / product_ref.norm().item()
            rel_err_all.append(rel_err)
            qp.log.info(f"  l: {l1} {l2} Err: {rel_err:9.3e}")
    rel_err_overall = np.sqrt((np.array(rel_err_all) ** 2).mean())
    qp.log.info(f"  Overall Err: {rel_err_overall:9.3e}")
    assert rel_err_overall < 1e-14


@pytest.mark.mpi_skip
def test_ylm_prime(r_ylm: Tuple[torch.Tensor, torch.Tensor]) -> None:
    qp.log.info("Testing derivatives:")
    # Analytical calculation:
    r, ylm = r_ylm
    ylm_prime = sh.get_harmonics_prime(sh.L_MAX, r)
    err = torch.zeros((ylm.shape[0], 3), device=r.device, dtype=r.dtype)
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
    # Report:
    err_np = err.to(qp.rc.cpu).numpy()
    for l in range(sh.L_MAX + 1):
        for m in range(-l, l + 1):
            err_x, err_y, err_z = err_np[l * (l + 1) + m]
            qp.log.info(f"  l: {l} m: {m:+d}  Err: {err_x:.3e} {err_y:.3e} {err_z:.3e}")


def main():
    """Run test with verbose log."""
    qp.utils.log_config()
    qp.rc.init()
    assert qp.rc.n_procs == 1
    r, ylm = get_r_ylm()
    test_ylm((r, ylm))
    test_ylm_prod((r, ylm))
    test_ylm_prime((r, ylm))


if __name__ == "__main__":
    main()
