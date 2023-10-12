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


def gaussian_blob_errors(
    g: torch.Tensor,
    rho: torch.Tensor,
    q: torch.Tensor,
    q0: torch.Tensor,
    L: torch.Tensor,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Compute errors in position, shape and MAE of shape of density `rho`."""
    dq = q - q0
    dq -= torch.floor(0.5 + dq / L) * L  # Minimum-image convention
    rho_norm = (g[..., 0] * rho).sum()
    dq_average = (g * rho[..., None] * dq).sum(axis=(0, 1)) / rho_norm
    # Accounting for position error, dq_average, determine shape error:
    rho_err = rho - gaussian_blob(q, q0 + dq_average, L, sigma)
    rho_mae = (g[..., 0] * rho_err.abs()).sum() / rho_norm
    return dq_average, rho_err, rho_mae.item()


def run(*, Nxy, N_theta, diag, plot_frames=False) -> tuple[float, float]:
    """Run simulation and report errors in position and shape."""
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
    q = sim.q[sim.non_ghost, sim.non_ghost]
    g = sim.g[sim.non_ghost, sim.non_ghost]
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
    dq_average, rho_err, rho_mae = gaussian_blob_errors(g, rho, q, q_final, L, sigma)
    q_np = to_numpy(q)
    plt.contourf(q_np[..., 0], q_np[..., 1], to_numpy(rho_err), levels=100, cmap="bwr")
    plt.gca().set_aspect("equal")
    plt.colorbar()
    plt.savefig(f"density_err_{Nxy}.png", bbox_inches="tight", dpi=200)

    # Return RMS error in density:
    pos_err = dq_average.norm().item()
    log.info(f"{pos_err = } and {rho_mae = } for {Nxy = }")
    return pos_err, rho_mae


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
        pos_errs = []
        shape_errs = []
        Nxy = args.Nxy_min
        while Nxy <= args.Nxy:
            pos_err, shape_err = run(
                Nxy=Nxy, N_theta=args.Ntheta, diag=args.diag, plot_frames=False
            )
            Ns.append(Nxy)
            pos_errs.append(pos_err)
            shape_errs.append(shape_err)
            Nxy *= 2

        # Print
        log.info("\n#Nxy PosErr ShapeErr")
        for Nxy, pos_err, shape_err in zip(Ns, pos_errs, shape_errs):
            log.info(f"{Nxy:4d} {pos_err:.6f} {shape_err:.6f}")

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        plt.subplots_adjust(wspace=0.3)
        for ax, errs, name in zip(axes, (pos_errs, shape_errs), ("Position", "Shape")):
            plt.sca(ax)
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
            plt.ylabel(f"{name} error")
            plt.xlim(*x_scale)
            plt.ylim(0.5 * min(errs), 2 * max(errs))
            plt.legend()
        plt.savefig("convergence.pdf", bbox_inches="tight")

    rc.report_end()
    StopWatch.print_stats()


if __name__ == "__main__":
    main()
