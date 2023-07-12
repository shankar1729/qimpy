import os

import matplotlib.pyplot as plt
import pytest

from qimpy import rc
from qimpy.io import log_config
from . import Pseudopotential


# Get list of filenames to test load:
ps_path = os.getenv("QIMPY_PSEUDOPOTENTIAL_TEST_PATH", "")
ps_names = os.getenv("QIMPY_PSEUDOPOTENTIAL_TEST_NAMES", "H").split() if ps_path else []
ps_filenames = [os.path.join(ps_path, ps_name) for ps_name in ps_names]


@pytest.mark.mpi_skip
@pytest.mark.parametrize("filename", ps_filenames)
def test_pseudopotential(filename: str) -> None:
    plot_pseudopotential(filename)


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
