import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

from qimpy import log, rc
from qimpy.io import log_config, Checkpoint, CheckpointPath, CheckpointContext
from qimpy.profiler import StopWatch
from .. import Transport


def gaussian_blob(
    q: torch.Tensor, q0: torch.Tensor, Rbasis: torch.Tensor, sigma: float
) -> torch.Tensor:
    dq = q - q0
    dq -= torch.floor(0.5 + dq @ torch.linalg.inv(Rbasis.T)) @ Rbasis.T
    return torch.exp(-dq.square().sum(dim=-1) / sigma**2).detach()


def gaussian_blob_error(
    g: torch.Tensor,
    rho: torch.Tensor,
    q: torch.Tensor,
    q0: torch.Tensor,
    Rbasis: torch.Tensor,
    sigma: float,
) -> tuple[float, float]:
    """Compute sum and error-sum of density `rho`."""
    rho_err = rho - gaussian_blob(q, q0, Rbasis, sigma)
    rho_sum = (g[..., 0] * rho).sum().item()
    rho_err_sum = (g[..., 0] * rho_err.abs()).sum().item()
    return rho_sum, rho_err_sum


def run(*, grid_spacing, N_theta, q0, v0, svg_file, save_frames=False) -> float:
    """Run simulation and report error in final density."""

    # Initialize transport system:
    vF = v0.norm().item()
    init_angle = torch.atan2(v0[1], v0[0]).item()
    transport = Transport(
        fermi_circle=dict(kF=1.0, vF=vF, N_theta=N_theta, theta0=init_angle),
        geometry=dict(svg_file=svg_file, grid_spacing=grid_spacing),
    )
    geometry = transport.geometry

    # Detect periodicity:
    tol = 1e-3
    displacements = (
        torch.from_numpy(geometry.quad_set.displacements).flatten(0, 1).to(rc.device)
    )
    displacements = displacements[torch.where(displacements.norm(dim=-1) > tol)[0]]
    if not len(displacements):
        log.error("No non-zero displacements to find lattice vectors for periodicity.")
        exit(1)
    is_equal = torch.logical_or(
        (displacements[:, None] - displacements[None]).norm(dim=-1) < tol,
        (displacements[:, None] + displacements[None]).norm(dim=-1) < tol,
    )
    Rbasis = displacements[torch.unique(torch.where(is_equal, 1, 0).argmax(dim=0))].T
    if Rbasis.shape != (2, 2):
        log.error("Could not detect two unique lattice vectors for periodicity.")
        exit(1)

    # Set the time for slightly more than one period
    distance = torch.linalg.det(Rbasis).abs().sqrt().item()  # move ~ one cell
    time_steps = round(distance / (vF * geometry.dt))
    t_final = time_steps * geometry.dt
    log.info(
        f"\nStarting {time_steps}-step run for {grid_spacing = :g}"
        f" at t[s]: {rc.clock():.2f}"
    )

    # Initialize initial and expected final density
    sigma = 5.0
    for patch in geometry.patches:
        patch.rho[..., 0] = gaussian_blob(patch.q, q0, Rbasis, sigma)

    plot_interval = round(0.02 * time_steps)
    save_frame = 0
    for time_step in range(time_steps):
        log.info(f"Step {time_step} at t[s]: {rc.clock():.2f}")
        if save_frames and (time_step % plot_interval == 0):
            watch = StopWatch("save_frame")
            with Checkpoint(
                f"animation/advect_{save_frame:04d}.h5", writable=True, rotate=False
            ) as cp:
                transport.save_checkpoint(CheckpointPath(cp), CheckpointContext(""))
            watch.stop()
            save_frame += 1
        geometry.time_step()

    # Compute final density error:
    q_final = q0 + v0 * t_final
    rho_sum_tot = 0.0
    rho_err_tot = 0.0
    for patch in geometry.patches:
        rho_sum, rho_err_sum = gaussian_blob_error(
            patch.g, patch.rho[..., 0], patch.q, q_final, Rbasis, sigma
        )
        rho_sum_tot += rho_sum
        rho_err_tot += rho_err_sum
    rho_sum_tot = rc.comm.allreduce(rho_sum_tot)
    rho_err_tot = rc.comm.allreduce(rho_err_tot)
    rho_mae = rho_err_tot / rho_sum_tot

    # Return RMS error in density:
    log.info(
        f"Done with {rho_mae = :.6f} for {grid_spacing = :g}"
        f" at t[s]: {rc.clock():.2f}\n"
    )
    return rho_mae


def main():
    log_config()
    rc.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("--h", help="grid spacing", type=float, required=True)
    parser.add_argument("--Ntheta", help="angular resolution", type=int, required=True)
    parser.add_argument("--q0", help="origin", nargs=2, type=float, required=True)
    parser.add_argument("--v0", help="velocity", nargs=2, type=float, required=True)
    parser.add_argument("--h_max", help="max spacing for convergence test", type=float)
    parser.add_argument("--svg", help="SVG geometry file", type=str, required=True)
    args = parser.parse_args()

    if args.h_max is None:
        run(
            grid_spacing=args.h,
            N_theta=args.Ntheta,
            q0=torch.tensor(args.q0, device=rc.device),
            v0=torch.tensor(args.v0, device=rc.device),
            svg_file=args.svg,
            save_frames=True,
        )
    else:
        # Convergence test:
        assert isinstance(args.h_max, float)
        assert args.h_max > args.h
        h_list = []
        err_list = []
        h = args.h
        while h <= args.h_max:
            err = run(
                grid_spacing=h,
                N_theta=args.Ntheta,
                q0=torch.tensor(args.q0, device=rc.device),
                v0=torch.tensor(args.v0, device=rc.device),
                svg_file=args.svg,
                save_frames=False,
            )
            h_list.append(h)
            err_list.append(err)
            h *= 2

        # Print
        log.info("\n# h  MAE")
        for h, err in zip(h_list, err_list):
            log.info(f"{h:4g} {err:.6f}")

        if rc.is_head:
            # Plot
            plt.figure()
            plt.scatter(h_list, err_list, marker="+", label="Observed Errors")
            # Add scaling guides:
            x_scale = np.array([0.5 * h_list[0], 2 * h_list[-1]])
            for exponent, ls in zip((1, 2), ("dashed", "dotted")):
                plt.plot(
                    x_scale,
                    err_list[-1] * (x_scale / h_list[-1]) ** exponent,
                    color="k",
                    ls=ls,
                    lw=1,
                    label=r"$h^{" + str(exponent) + r"}$ scaling",
                )
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel(r"Grid spacing, $h$")
            plt.ylabel(r"MAE($\rho$)")
            plt.xlim(*x_scale)
            plt.ylim(0.5 * min(err_list), 2 * max(err_list))
            plt.legend()
            plt.savefig("convergence.pdf", bbox_inches="tight")

    rc.report_end()
    StopWatch.print_stats()


if __name__ == "__main__":
    main()
