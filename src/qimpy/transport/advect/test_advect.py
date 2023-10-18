import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from qimpy import log, rc
from qimpy.io import log_config
from qimpy.profiler import StopWatch
from qimpy.transport.advect._advect import Advect, to_numpy


def gaussian_blob(
    q: torch.Tensor, q0: torch.Tensor, L: torch.Tensor, sigma: float
) -> torch.Tensor:
    dq = q - q0
    dq -= torch.floor(0.5 + dq / L) * L  # Minimum-image convention
    return torch.exp(-dq.square().sum(axis=-1) / sigma**2).detach()


def gaussian_blob_error(
    g: torch.Tensor,
    rho: torch.Tensor,
    q: torch.Tensor,
    q0: torch.Tensor,
    L: torch.Tensor,
    sigma: float,
) -> tuple[torch.Tensor, float]:
    """Compute error profile and MAE of density `rho`."""
    rho_err = rho - gaussian_blob(q, q0, L, sigma)
    rho_norm = (g[..., 0] * rho).sum()
    rho_mae = (g[..., 0] * rho_err.abs()).sum() / rho_norm
    return rho_err, rho_mae.item()


def run(*, Nxy, N_theta, diag, plot_frames=False) -> float:
    """Run simulation and report error in final density."""
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
    L = torch.tensor(sim.L, device=rc.device)
    L_period = np.hypot(*sim.L) if diag else sim.L[0]
    t_period = L_period / sim.v_F
    time_steps = round(1.25 * t_period / sim.dt)
    t_final = time_steps * sim.dt
    log.info(f"\nRunning for {time_steps} steps at {Nxy = }:")

    # Initialize initial and expected final density
    sigma = 0.05
    q = sim.q
    g = sim.g
    q0 = 0.25 * L
    sim.rho[sim.non_ghost, sim.non_ghost, 0] = gaussian_blob(q, q0, L, sigma)

    plot_interval = round(0.01 * time_steps)
    plot_frame = 0
    for time_step in tqdm(range(time_steps)):
        if plot_frames and (time_step % plot_interval == 0):
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
    plt.clf()
    q_final = q0 + sim.v[0] * t_final
    rho = sim.rho[sim.non_ghost, sim.non_ghost, 0]
    rho_err, rho_mae = gaussian_blob_error(g, rho, q, q_final, L, sigma)
    q_np = to_numpy(q)
    plt.contourf(q_np[..., 0], q_np[..., 1], to_numpy(rho_err), levels=100, cmap="bwr")
    plt.gca().set_aspect("equal")
    plt.colorbar()
    plt.savefig(f"density_err_{Nxy}.png", bbox_inches="tight", dpi=200)

    # Return RMS error in density:
    log.info(f"{rho_mae = } for {Nxy = }")
    return rho_mae


def main():
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
        errs = []
        Nxy = args.Nxy_min
        while Nxy <= args.Nxy:
            err = run(Nxy=Nxy, N_theta=args.Ntheta, diag=args.diag, plot_frames=False)
            Ns.append(Nxy)
            errs.append(err)
            Nxy *= 2

        # Print
        log.info("\n#Nxy MAE")
        for Nxy, err in zip(Ns, errs):
            log.info(f"{Nxy:4d} {err:.6f}")

        # Plot
        plt.figure()
        plt.scatter(Ns, errs, marker="+", label="Observed Errors")
        # Add scaling guides:
        x_scale = np.array([0.5 * Ns[0], 2 * Ns[-1]])
        for exponent, ls in zip((1, 2), ("dashed", "dotted")):
            plt.plot(
                x_scale,
                errs[0] * (Ns[0] / x_scale) ** exponent,
                color="k",
                ls=ls,
                lw=1,
                label=r"$N^{-" + str(exponent) + r"}$ scaling",
            )
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(r"$N_{xy}$")
        plt.ylabel(r"MAE($\rho$)")
        plt.xlim(*x_scale)
        plt.ylim(0.5 * min(errs), 2 * max(errs))
        plt.legend()
        plt.savefig("convergence.pdf", bbox_inches="tight")

    rc.report_end()
    StopWatch.print_stats()


if __name__ == "__main__":
    main()
