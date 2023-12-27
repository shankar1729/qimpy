from __future__ import annotations
from typing import Optional
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

from qimpy import log, rc
from qimpy.io import log_config
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


def run(
    *,
    grid_spacing: float,
    N_theta: int,
    sigma: float,
    q0: torch.Tensor,
    v0: torch.Tensor,
    t_max: float,
    svg_file: str,
    save_frames: bool = False,
) -> float:
    """Run simulation and report error in final density."""

    # Initialize transport system:
    vF = v0.norm().item()
    init_angle = torch.atan2(v0[1], v0[0]).item()
    dt_save = (0.01 if save_frames else 2) * t_max
    transport = Transport(
        fermi_circle=dict(
            kF=1.0,
            vF=vF,
            N_theta=N_theta,
            theta0=init_angle,
            tau_p=np.inf,
            tau_ee=np.inf,
        ),
        geometry=dict(svg_file=svg_file, grid_spacing=grid_spacing, contacts={}),
        time_evolution=dict(t_max=t_max, dt_save=dt_save, n_collate=10),
        checkpoint_out="animation/advect_{:04d}.h5",
    )
    geometry = transport.geometry

    # Detect periodicity:
    displacements = torch.from_numpy(geometry.quad_set.displacements).flatten(0, 1)
    Rbasis = detect_lattice_vectors(displacements.to(rc.device))
    if Rbasis is None:
        log.info("Lattice vectors set to bounding box: IGNORE reported rho_mae.")
        vertices = geometry.quad_set.vertices
        bbox_size = np.max(vertices, axis=0) - np.min(vertices, axis=0)
        Rbasis = torch.diag(torch.from_numpy(bbox_size)).to(rc.device)

    # Initialize density
    for patch in geometry.patches:
        patch.rho[..., 0] = gaussian_blob(patch.q, q0, Rbasis, sigma)

    transport.run()

    # Compute final density error:
    q_final = q0 + v0 * transport.time_evolution.t
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


def detect_lattice_vectors(
    displacements: torch.Tensor, tol=1e-3
) -> Optional[torch.Tensor]:
    """Detect periodicity from edge-equivalence displacements."""
    displacements = displacements[torch.where(displacements.norm(dim=-1) > tol)[0]]
    if not len(displacements):
        log.info("No displacements to find lattice vectors for periodicity.")
        return None
    is_equal = torch.logical_or(
        (displacements[:, None] - displacements[None]).norm(dim=-1) < tol,
        (displacements[:, None] + displacements[None]).norm(dim=-1) < tol,
    )
    Rbasis = displacements[torch.unique(torch.where(is_equal, 1, 0).argmax(dim=0))].T
    if Rbasis.shape != (2, 2):
        log.info("Could not detect two unique lattice vectors for periodicity.")
        return None
    return Rbasis


def main():
    log_config()
    rc.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("--h", help="grid spacing", type=float, required=True)
    parser.add_argument("--Ntheta", help="angular resolution", type=int, required=True)
    parser.add_argument("--sigma", help="gaussian width", type=float, required=True)
    parser.add_argument("--q0", help="origin", nargs=2, type=float, required=True)
    parser.add_argument("--v0", help="velocity", nargs=2, type=float, required=True)
    parser.add_argument("--t_max", help="stopping time", type=float, required=True)
    parser.add_argument("--h_max", help="max spacing for convergence test", type=float)
    parser.add_argument("--svg", help="SVG geometry file", type=str, required=True)
    args = parser.parse_args()

    run_args = dict(
        N_theta=args.Ntheta,
        sigma=args.sigma,
        q0=torch.tensor(args.q0, device=rc.device),
        v0=torch.tensor(args.v0, device=rc.device),
        t_max=args.t_max,
        svg_file=args.svg,
    )
    if args.h_max is None:
        run(grid_spacing=args.h, save_frames=True, **run_args)
    else:
        # Convergence test:
        assert isinstance(args.h_max, float)
        assert args.h_max > args.h
        h_list = []
        err_list = []
        h = args.h
        while h <= args.h_max:
            err = run(grid_spacing=h, save_frames=False, **run_args)
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
