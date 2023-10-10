import torch
import numpy as np

from qimpy import log, rc
from qimpy.io import log_config
from qimpy.profiler import StopWatch
from qimpy.transport.advect._advect import Advect, to_numpy


def gaussian_blob(
    q: torch.Tensor, q0: torch.Tensor, sigma: float = 0.05
) -> torch.Tensor:
    return torch.exp(-(q - q0).square().sum(axis=-1) / sigma**2).detach()


def run(*, Nxy, N_theta, diag, plot_metric=False):
    import matplotlib.pyplot as plt

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

    # Set the time for slightly more than one period
    L_period = np.hypot(Lx, Ly) if diag else Lx
    t_period = L_period / v_F
    time_steps = int(np.ceil(1.25 * t_period / sim.dt))
    t_final = time_steps * sim.dt
    log.info(f"Running for {time_steps} steps.")

    # Initialize initial and expected final density
    q0 = torch.tensor([0.25 * Lx, 0.25 * Ly], device=rc.device)
    sim.rho[:, :, 0] = gaussian_blob(sim.q, q0)
    q_final = q0 + sim.v[0, 0, 0] * (t_final - t_period)
    density_final = (
        gaussian_blob(sim.q, q_final)[sim.non_ghost, sim.non_ghost] * sim.dtheta
    )

    if plot_metric:
        # Plot metric
        plt.figure()
        plt.imshow(np.squeeze(to_numpy(sim.g)))
        plt.gca().set_aspect("equal")
        plt.colorbar()
        plt.savefig("transform_metric.png", dpi=300)

        # Plot Jacobian
        for index, jac_comp in enumerate((sim.dX_dx, sim.dX_dy, sim.dY_dx, sim.dY_dy)):
            plt.figure()
            plt.imshow(np.squeeze(to_numpy(jac_comp)))
            plt.gca().set_aspect("equal")
            plt.colorbar()
            plt.savefig(f"jacobian_{index}.png", dpi=300)

        # Plot velocity
        for index, v_comp in enumerate((sim.v_X, sim.v_Y)):
            plt.figure()
            plt.imshow(np.squeeze(to_numpy(v_comp[..., 0])))
            plt.gca().set_aspect("equal")
            plt.colorbar()
            plt.savefig(f"v_{'XY'[index]}.png", dpi=300)
        exit()

    plot_interval = round(0.01 * time_steps)
    plot_frame = 0
    for time_step in range(time_steps):
        log.info(f"{time_step = }")
        if time_step % plot_interval == 0:
            log.info("Plotting density and streamlines")
            plt.clf()
            sim.plot_streamlines(plt, dict(levels=100), dict(linewidth=1.0))
            plt.gca().set_aspect("equal")
            plt.savefig(
                f"animation/blob_advect_{plot_frame:04d}.png",
                bbox_inches="tight",
                dpi=200,
            )
            plot_frame += 1
        sim.time_step()

    # Return RMS error in density:
    return (density_final - sim.density).square().mean().sqrt().item()


def main():
    log_config()
    rc.init()
    assert rc.n_procs == 1  # MPI not yet supported

    RMSE = run(Nxy=512, N_theta=1, diag=True)
    log.info(f"{RMSE = }")
    # errors = dict()
    # Nthetas = [64]
    # Ns = [64, 128, 256, 512, 1024, 2048]
    # for i in Nthetas:
    #   for j in Ns:
    #       if not (i == 256 and j == 1024):
    #           errors[(i, j)] = convergence(j, i, diag=(True if i > 2 else False))
    #           print(i, j, errors[(i, j)])
    #           print(errors)
    # print(errors)

    StopWatch.print_stats()


if __name__ == "__main__":
    main()
