import qimpy as qp
import torch
import pytest
import functools
from qimpy.dft.electrons._fillings import _smearing_funcs, SmearingFunc


@functools.cache
def get_smear_test_inputs() -> tuple[torch.Tensor, float, float, float]:
    """Get input set for smearing tests."""
    sigma = 0.005
    mu = -0.157
    deig = 0.01 * sigma
    delta_eig = 100 * sigma
    eig = torch.arange(mu - delta_eig, mu + delta_eig, deig, device=qp.rc.device)
    return eig, mu, sigma, deig


@pytest.mark.mpi_skip
@pytest.mark.parametrize("smearing_func", _smearing_funcs.values())
def test_f_eig(smearing_func: SmearingFunc) -> None:
    """Check df/deig consistency with f."""
    eig, mu, sigma, deig = get_smear_test_inputs()
    f, f_eig, S = smearing_func(eig, mu, sigma)
    f_eig_num = (f[2:] - f[:-2]) / (2 * deig)  # central difference derivative
    err_max = (deig**2) / (sigma**3)  # because above is second-order correct
    assert torch.allclose(f_eig_num, f_eig[1:-1], atol=err_max, rtol=0.0)


@pytest.mark.mpi_skip
@pytest.mark.parametrize("smearing_func", _smearing_funcs.values())
def test_S(smearing_func: SmearingFunc) -> None:
    """Check S consistency with f: dS/df = 2(eig - mu)/sigma."""
    eig, mu, sigma, deig = get_smear_test_inputs()
    f, f_eig, S = smearing_func(eig, mu, sigma)
    df = f[2:] - f[:-2]
    dS = S[2:] - S[:-2]
    dS_expected = df * 2 * (eig[1:-1] - mu) / sigma
    tol = (deig / sigma) ** 2
    assert torch.allclose(dS, dS_expected, rtol=tol, atol=tol)
