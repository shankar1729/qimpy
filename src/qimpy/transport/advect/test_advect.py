import torch
import numpy as np

from qimpy import log, rc
from qimpy.io import log_config
from qimpy.profiler import StopWatch
from ._advect import Advect


def movie(Nxy, N_theta, diag=True):
    import matplotlib.pyplot as plt

    log_config()
    rc.init()
    assert rc.n_procs == 1  # MPI not yet supported

    v_F = 200.0
    Lx = 1.0
    Ly = 1.0
    sim = Advect(
        reflect_boundaries=False,
        contact_width=0.0,
        v_F=v_F,
        Lx=Lx,
        Ly=Ly,
        Nx=Nxy,
        Ny=Nxy,
        N_theta=N_theta,
        init_angle=np.pi / 4 if diag else 0.0,
    )
    sigma = 0.05
    sim.rho[:, :, 0] = torch.exp(
        -((sim.q[:, :, 0] - sim.Lx / 2) ** 2 + (sim.q[:, :, 1] - sim.Ly / 2) ** 2)
        / sigma**2
    ).detach()
    density_init = torch.clone(sim.density)

    t_final = (Lx**2 + Ly**2) ** 0.5 / v_F if diag else Lx / v_F
    time_steps = round(t_final / sim.dt)
    log.info(f"Running for {time_steps} steps.")

    for time_step in range(time_steps):
        log.info(f"{time_step = }")
        plt.clf()
        sim.plot_streamlines(plt, dict(levels=100), dict(linewidth=1.0))
        plt.gca().set_aspect("equal")
        plt.savefig(
            f"advect_animation/blob_advect_{time_step:04d}.png",
            bbox_inches="tight",
            dpi=200,
        )
        sim.time_step()

    # Plot only at end (for easier performance benchmarking of time steps):
    log.info("Plotting density and streamlines")

    StopWatch.print_stats()
    diff = float(((density_init - sim.density) ** 2).sum() / (sim.Nx * sim.Ny))
    return diff


def convergence(Nxy, N_theta, diag=True):
    # import matplotlib.pyplot as plt

    log_config()
    rc.init()
    assert rc.n_procs == 1  # MPI not yet supported

    v_F = 200.0
    Lx = 1.0
    Ly = 1.0
    sim = Advect(
        reflect_boundaries=False,
        contact_width=0.0,
        v_F=v_F,
        Lx=Lx,
        Ly=Ly,
        Nx=Nxy,
        Ny=Nxy,
        N_theta=N_theta,
        init_angle=np.pi / 4 - np.pi / N_theta if diag else 0.0,
    )
    sigma = 0.05
    sim.rho[:, :, 0] = torch.exp(
        -((sim.q[:, :, 0] - sim.Lx / 2) ** 2 + (sim.q[:, :, 1] - sim.Ly / 2) ** 2)
        / sigma**2
    ).detach()
    density_init = torch.clone(sim.density)

    t_final = (Lx**2 + Ly**2) ** 0.5 / v_F if diag else Lx / v_F
    time_steps = round(t_final / sim.dt)
    print(f"Running for {time_steps} steps.")

    for time_step in range(time_steps):
        log.info(f"{time_step = }")
        # plt.clf()
        # sim.plot_streamlines(plt, dict(levels=100), dict(linewidth=1.0))
        # plt.gca().set_aspect("equal")
        # plt.savefig(
        #    f"advect_animation/blob_advect_{time_step:04d}.png",
        #    bbox_inches="tight",
        #    dpi=200,
        # )
        sim.time_step()

    # Plot only at end (for easier performance benchmarking of time steps):
    log.info("Plotting density and streamlines")

    StopWatch.print_stats()
    diff = float(((density_init - sim.density) ** 2).sum() / (sim.Nx * sim.Ny))
    return diff


def main():
    movie(256, 256, diag=True)
    # errors = dict()
    # Nthetas = [2, 4, 256]
    # Ns = [64, 128, 256, 512]
    # for i in Nthetas:
    #    for j in Ns:
    #        if not (i == 256 and j == 1024):
    #            errors[(i, j)] = convergence(j, i, diag=(True if i > 2 else False))
    #            print(i, j, errors[(i, j)])
    # print(errors)


if __name__ == "__main__":
    main()
