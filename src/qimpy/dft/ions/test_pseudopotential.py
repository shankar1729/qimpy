import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torch.distributed as dist

from qimpy import rc
from qimpy.io import log_config
from qimpy.math import RadialFunction
from . import Pseudopotential

# Get list of filenames to test load:
ps_path = os.getenv("QIMPY_PSEUDOPOTENTIAL_TEST_PATH", "")
ps_names = os.getenv("QIMPY_PSEUDOPOTENTIAL_TEST_NAMES", "H").split() if ps_path else []
ps_filenames = [os.path.join(ps_path, ps_name) for ps_name in ps_names]


@pytest.mark.mpi_skip
@pytest.mark.parametrize("filename", ps_filenames)
def test_pseudopotential(filename: str) -> None:
    plot_pseudopotential(filename)


@pytest.mark.mpi_skip
@pytest.mark.parametrize("filename", ps_filenames)
def test_augmentation_data(filename: str) -> None:
    """Check selection rules and sum rules in ultrasoft data."""
    ps = Pseudopotential(filename)
    if not ps.is_ultrasoft:
        assert (ps.Q is None) and (ps.q is None) and (ps.Q_pair is None)
        return
    assert (ps.Q is not None) and (ps.q is not None) and (ps.Q_pair is not None)
    l_beta = ps.beta.l
    i, j = ps.Q_pair.T
    L = ps.Q.l
    # Gaunt selection rules for the angular part Y_l_i Y_l_j -> Y_L:
    assert (i <= j).all()
    assert ((l_beta[i] - l_beta[j]).abs() <= L).all() and (
        L <= l_beta[i] + l_beta[j]
    ).all()
    assert ((l_beta[i] + l_beta[j] + L) % 2 == 0).all()
    # Overlap coefficients: symmetric, zero between different l, and matching Q^0(G=0):
    assert torch.equal(ps.q, ps.q.T)
    assert (ps.q[l_beta[:, None] != l_beta[None, :]] == 0.0).all()
    assert dist.group.WORLD is not None
    RadialFunction.transform([ps.Q], 1.0, dist.group.WORLD)
    sel = torch.where(L == 0)[0]
    q_from_Q = ps.Q.f_tilde[sel, 0] / (4 * np.pi)
    assert torch.allclose(ps.q[i[sel], j[sel]], q_from_Q, rtol=1e-13, atol=1e-15)


def plot_pseudopotential(filename: str) -> None:
    ps = Pseudopotential(filename)
    for plot_func in (plot_local, plot_projectors, plot_orbitals):
        plt.figure()
        plot_func(ps)


def plot_local(ps: Pseudopotential) -> None:
    """Plot local potential and densities."""
    r = ps.r.to(rc.cpu)
    plt.title(f"{ps.element} density/potential")
    plt.plot(r, ps.rho_atom.f.to(rc.cpu)[0], label=r"$\rho_{\mathrm{atom}}(r)$")
    if hasattr(ps, "nCore"):
        plt.plot(r, ps.n_core.f.to(rc.cpu)[0], label=r"$n_{\mathrm{core}}(r)$")
    plt.plot(r, r * ps.Vloc.f.to(rc.cpu)[0], label=r"$r V_{\mathrm{loc}}(r)$")
    plt.xlabel(r"$r$")
    plt.xlim(0, 10.0)
    plt.legend()


def plot_projectors(ps: Pseudopotential) -> None:
    """Plot projectors."""
    r = ps.r.to(rc.cpu)
    plt.title(f"{ps.element} projectors")
    for i, beta_i in enumerate(ps.beta.f.to(rc.cpu)):
        l_i = int(ps.beta.l[i].item())
        plt.plot(r, beta_i, label=r"$\beta_" + "spdf"[l_i] + f"(r)/r^{l_i}$")
    plt.xlabel(r"$r$")
    plt.xlim(0, 10.0)
    plt.legend()


def plot_orbitals(ps: Pseudopotential) -> None:
    """Plot projectors."""
    r = ps.r.to(rc.cpu)
    plt.title(f"{ps.element} orbitals")
    for i, psi_i in enumerate(ps.psi.f.to(rc.cpu)):
        l_i = int(ps.psi.l[i].item())
        plt.plot(r, psi_i, label=r"$\psi_" + "spdf"[l_i] + f"(r)/r^{l_i}$")
    plt.xlabel(r"$r$")
    plt.xlim(0, 10.0)
    plt.legend()


def main():
    log_config()
    rc.init()
    if not ps_filenames:
        print("""
            Specify environment variables QIMPY_PSEUDOPOTENTIAL_TEST_PATH
            and QIMPY_PSEUDOPOTENTIAL_TEST_NAMES to select pseudos to plot.
            """)
    if rc.is_head:
        for ps_filename in ps_filenames:
            plot_pseudopotential(ps_filename)
        plt.show()


if __name__ == "__main__":
    main()
