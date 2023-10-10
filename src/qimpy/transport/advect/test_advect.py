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


def run(*, Nxy, N_theta, diag, plot_frames=False):
    import matplotlib.pyplot as plt

    sim = Advect(
        reflect_boundaries=False,
        contact_width=0.0,
        v_F=200.0,
        L=(1.0, 1.0),
        N=(Nxy, Nxy),
        N_theta=N_theta,
        init_angle=np.pi / 4 if diag else 0.0,
    )

    # Set the time for slightly more than one period
    L_period = np.hypot(*sim.L) if diag else sim.L[0]
    t_period = L_period / sim.v_F
    time_steps = int(np.ceil(1.25 * t_period / sim.dt))
    t_final = time_steps * sim.dt
    log.info(f"Running for {time_steps} steps.")

    # Initialize initial and expected final density
    q0 = 0.25 * torch.tensor(sim.L, device=rc.device)
    sim.rho[:, :, 0] = gaussian_blob(sim.q, q0)
    q_final = q0 + sim.v[0, 0, 0] * (t_final - t_period)
    density_final = (
        gaussian_blob(sim.q, q_final)[sim.non_ghost, sim.non_ghost] * sim.dtheta
    )

    plot_interval = round(0.01 * time_steps)
    plot_frame = 0
    for time_step in range(time_steps):
        log.info(f"{time_step = }")
        if plot_frames and (time_step % plot_interval == 0):
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

    # Plot final density error:
    log.info("Plotting error in final density")
    plt.clf()
    density_err = density_final - sim.density
    q = to_numpy(sim.q[sim.non_ghost, sim.non_ghost])
    plt.contourf(q[..., 0], q[..., 1], to_numpy(density_err), levels=100, cmap="bwr")
    plt.gca().set_aspect("equal")
    plt.colorbar()
    plt.savefig(f"density_err_{Nxy}.png", bbox_inches="tight", dpi=200)

    # Return RMS error in density:
    RMSE = density_err.square().mean().sqrt().item()
    RMS_value = density_final.square().mean().sqrt().item()
    RelativeRMSE = RMSE / RMS_value
    log.info(f"{RelativeRMSE = }")
    return RelativeRMSE


def main():
    import argparse

    log_config()
    rc.init()
    assert rc.n_procs == 1  # MPI not yet supported

    parser = argparse.ArgumentParser()
    parser.add_argument("--Nxy", help="spatial resolution", type=int, required=True)
    parser.add_argument("--Ntheta", help="angular resolution", type=int, required=True)
    parser.add_argument("--diag", help="move diagonally", action="store_true")
    parser.add_argument("--Nxy_min", help="start resolution for convergence", type=int)
    args = parser.parse_args()

    if args.Nxy_min is None:
        run(Nxy=args.Nxy, N_theta=args.Ntheta, diag=args.diag, plot_frames=True)
    else:
        # Convergence test:
        assert isinstance(args.Nxy_min, int)
        assert args.Nxy_min < args.Nxy
        Ns = []
        RMSEs = []
        Nxy = args.Nxy_min
        while Nxy <= args.Nxy:
            RMSE = run(Nxy=Nxy, N_theta=args.Ntheta, diag=args.diag, plot_frames=False)
            Ns.append(Nxy)
            RMSEs.append(RMSE)
            Nxy *= 2

        # Print
        log.info("\n#Nxy RMSE(relative)")
        for Nxy, RMSE in zip(Ns, RMSEs):
            log.info(f"{Nxy:4d} {RMSE:.6f}")

        # Plot
        import matplotlib.pyplot as plt

        plt.scatter(Ns, RMSEs, marker="+")
        plt.plot([Ns[0], Ns[-1]], [RMSEs[0], RMSEs[0] * Ns[0] / Ns[-1]], "k--")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(r"$N_{xy}$")
        plt.ylabel("Density RMSE (relative)")
        plt.savefig("convergence.pdf", bbox_inches="tight")

    rc.report_end()
    StopWatch.print_stats()


if __name__ == "__main__":
    main()
