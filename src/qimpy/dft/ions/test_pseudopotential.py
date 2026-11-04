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
    """Ultrasoft data read from file obey the selection rules and sum rules.
    Norm-conserving files must load without any augmentation.
    For example, test the GBRV set with QIMPY_PSEUDOPOTENTIAL_TEST_NAMES set to
    "si_pbe.uspp p_pbe.uspp nb_pbe.uspp" (in the GBRV directory).
    (Note that the atomic density need not integrate to Z: GBRV generates some
    elements in ionized configurations, e.g. 15 electrons for Fe with Z = 16.)"""
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
    # Overlap coefficients: symmetric, zero between different l, and equal to
    # the G = 0 limit of the transformed Q^0 (which makes the charge conserved):
    assert torch.equal(ps.q, ps.q.T)
    assert (ps.q[l_beta[:, None] != l_beta[None, :]] == 0.0).all()
    assert dist.group.WORLD is not None
    RadialFunction.transform([ps.Q], 1.0, dist.group.WORLD)
    sel = torch.where(L == 0)[0]
    q_from_Q = ps.Q.f_tilde[sel, 0] / (4 * np.pi)
    assert torch.allclose(ps.q[i[sel], j[sel]], q_from_Q, rtol=1e-13, atol=1e-15)


@pytest.mark.mpi_skip
def test_set_augmentation() -> None:
    """Processing in Pseudopotential.set_augmentation on synthetic data with known
    results: per-L functions and q, inner pseudization, pruning, and a norm-conserving
    pseudopotential in ultrasoft format (all Q = 0)."""
    r_np = np.linspace(1e-4, 10.0, 4001)
    r = torch.tensor(r_np, device=rc.device)
    dr = torch.full_like(r, r_np[1] - r_np[0])
    ps = Pseudopotential.__new__(Pseudopotential)  # without reading a file
    ps.r, ps.dr = r, dr
    ps.beta = RadialFunction(r, dr, torch.zeros((3, len(r))), np.array([0, 1, 1]))

    # Per-L input r^2 Q^L_ij = a_ij r^(L+2) exp(-r^2), with a zero pair (1, 2):
    a = np.array([[1.0, 0.5, 0.25], [0.5, 2.0, 0.0], [0.25, 0.0, 3.0]])
    L_max = 2
    r_L = r_np[None, :] ** np.arange(L_max + 1)[:, None]  # (L, r)
    r2_Q = a[:, :, None, None] * (r_L * (r_np**2) * np.exp(-(r_np**2)))
    ps.set_augmentation(r2_Q, np.zeros((3, 3)))
    assert ps.is_ultrasoft and (ps.Q is not None) and (ps.q is not None)
    expected_pairs_L = [(0, 0, 0), (0, 1, 1), (0, 2, 1), (1, 1, 0), (1, 1, 2)]
    expected_pairs_L += [(2, 2, 0), (2, 2, 2)]  # pair (1, 2) pruned
    assert ps.Q_pair is not None
    got = [(*pair, L) for pair, L in zip(ps.Q_pair.tolist(), ps.Q.l.tolist())]
    assert got == expected_pairs_L
    for (i, j, L), f in zip(got, ps.Q.f):
        assert torch.allclose(f, a[i, j] * torch.exp(-(r**2)))  # stored as Q^L / r^L
    q_expected = a * (np.sqrt(np.pi) / 4) * np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1]])
    np.testing.assert_allclose(ps.q.to(rc.cpu).numpy(), q_expected, atol=1e-10)

    # L-independent input with inner Taylor series sum_k c_k r^(L + 2k):
    r_inner = np.array([0.5, 0.6, 0.7])
    Q_taylor = np.zeros((3, 3, 3, 2))
    Q_taylor[..., 0] = 7.0
    Q_taylor[..., 1] = -3.0
    ps.set_augmentation(r2_Q[:, :, 0], np.zeros((3, 3)), r_inner, Q_taylor)
    assert ps.Q_pair is not None and ps.Q is not None
    for (i, j), L, f in zip(ps.Q_pair.tolist(), ps.Q.l.tolist(), ps.Q.f):
        inner = r < r_inner[L]
        outer = torch.logical_not(inner)
        assert torch.allclose(f[inner], 7.0 - 3.0 * (r[inner] ** 2))
        Q_outer = a[i, j] * torch.exp(-(r[outer] ** 2))
        assert torch.allclose(f[outer], Q_outer / (r[outer] ** L))

    # Norm-conserving pseudopotential written in ultrasoft format:
    ps.set_augmentation(np.zeros_like(r2_Q), np.zeros((3, 3)))
    assert not ps.is_ultrasoft


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
        print(
            """
            Specify environment variables QIMPY_PSEUDOPOTENTIAL_TEST_PATH
            and QIMPY_PSEUDOPOTENTIAL_TEST_NAMES to select pseudos to plot.
            """
        )
    if rc.is_head:
        for ps_filename in ps_filenames:
            plot_pseudopotential(ps_filename)
        plt.show()


if __name__ == "__main__":
    main()
