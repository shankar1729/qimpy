import torch
import numpy as np

from qimpy import rc, log
from qimpy.io import log_config
from qimpy.mpi import ProcessGrid
from . import SingleBand


def test_scatter():
    process_grid = ProcessGrid(rc.comm, "rk", (-1, 1))
    vF = 0.375  # Graphene Fermi velocity
    material = SingleBand(
        process_grid=process_grid,
        lattice=dict(
            periodic=[True, True, False], system=dict(name="hexagonal", a=4.651, c=15)
        ),
        kmesh=[1200, 1200, 1],
        mu=0.01,  # ~ 0.3 eV
        T=0.0002,  # ~ 60 K
        v=vF,
        scatter=dict(dE=0.00005, epsilon_bg=1.0, lambda_D=10.0),
    )

    # Create single-carrier perturbations at each k:
    f0 = material.rho0.flatten()  # equilibrium distribution
    f = f0.tile((len(f0), 1))
    df = torch.where(f0 < 0.5, +0.01, -0.01)
    f += torch.diag(df)

    # Compute rate of change of occupations and various moments
    f_dot = material.rho_dot(f, 0.0, 0)
    tau_inv_ee = -torch.diag(f_dot) / df
    N_dot = f_dot.sum(dim=1).abs().sum().item()
    E_dot = (f_dot @ material.E.flatten()).abs().sum().item()
    k_dot = (f_dot @ material.k).norm(dim=-1).sum().item()

    # Report accuracy of conservation rules:
    f_dot_sum = torch.diag(f_dot).abs().sum().item()  # typical scale for rates
    N_dot_ref = f0.sum().item() * f_dot_sum  # characteristic number rate scale
    E_dot_ref = N_dot_ref * material.T  # corresponding energy scale
    k_dot_ref = N_dot_ref * (material.mu / vF)  # corresponding momentum scale
    log.info(f"Number conservation accuracy: {N_dot / N_dot_ref:.3e}")
    log.info(f"Energy conservation accuracy: {E_dot / E_dot_ref:.3e}")
    log.info(f"Momentum conservation accuracy: {k_dot / k_dot_ref:.3e}")

    # Plot carrier scattering rate vs energy:
    from matplotlib import pyplot as plt

    Ediff = material.E.to(rc.cpu).numpy().flatten() - material.mu
    tau_inv_ee = tau_inv_ee.to(rc.cpu).numpy().flatten()
    quadratic_model = Ediff**2 + (np.pi * material.T) ** 2
    D_ee = np.linalg.lstsq(quadratic_model[:, None], tau_inv_ee, rcond=None)[0][0]
    log.info(f"Lifetime fit parameter, {D_ee = :.3e}")
    plt.scatter(Ediff, tau_inv_ee, marker="+", label="single_band.Scatter")
    E_plot = np.linspace(Ediff.min(), Ediff.max(), 200)
    plt.plot(
        E_plot,
        D_ee * (E_plot**2 + (np.pi * material.T) ** 2),
        label=r"Single-parameter fit, $D_{ee}((E-\mu)^2 + (\pi T)^2)$",
    )
    plt.xlabel(r"$E - \mu$")
    plt.ylabel(r"$\tau_{ee}^{-1}$")
    plt.ylim(0, None)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    log_config()
    rc.init()
    test_scatter()
