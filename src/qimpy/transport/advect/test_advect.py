import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from qimpy import log, rc
from qimpy.io import log_config
from qimpy.profiler import StopWatch
from qimpy.transport.advect._advect import Advect, to_numpy
from qimpy.transport.test_spline import get_splines


class BicubicPatch:
    """Transformation based on cubic spline edges."""

    control_points: torch.Tensor  #: Control point coordinates (4 x 4 x 2)

    def __init__(self, boundary: torch.Tensor):
        """Initialize from 12 x 2 coordinates of control points on perimeter."""
        control_points = torch.empty((4, 4, 2), device=rc.device)
        # Set boundary control points:
        control_points[:, 0] = boundary[:4]
        control_points[-1, 1:] = boundary[4:7]
        control_points[:-1, -1] = boundary[7:10].flipud()
        control_points[0, 1:-1] = boundary[10:12].flipud()

        # Set internal control points based on parallelogram completion:
        def complete_parallelogram(
            v: torch.Tensor, i0: int, j0: int, i1: int, j1: int
        ) -> None:
            v[i1, j1] = v[i0, j1] + v[i1, j0] - v[i0, j0]

        complete_parallelogram(control_points, 0, 0, 1, 1)
        complete_parallelogram(control_points, 3, 0, 2, 1)
        complete_parallelogram(control_points, 0, 3, 1, 2)
        complete_parallelogram(control_points, 3, 3, 2, 2)
        self.control_points = control_points

    def __call__(self, Qfrac: torch.Tensor) -> torch.Tensor:
        """Define mapping from fractional mesh to Cartesian coordinates."""
        return torch.einsum(
            "uvi, u..., v... -> ...i",
            self.control_points,
            cubic_bernstein(Qfrac[..., 0]),
            cubic_bernstein(Qfrac[..., 1]),
        )


def cubic_bernstein(t: torch.Tensor) -> torch.Tensor:
    """Return basis of cubic Bernstein polynomials."""
    t_bar = 1.0 - t
    return torch.stack(
        (t_bar**3, 3.0 * (t_bar**2) * t, 3.0 * t_bar * (t**2), t**3)
    )


def gaussian_blob(
    q: torch.Tensor, q0: torch.Tensor, Rbasis: torch.Tensor, sigma: float
) -> torch.Tensor:
    dq = q - q0
    dq -= torch.floor(0.5 + dq @ torch.linalg.inv(Rbasis.T)) @ Rbasis.T
    return torch.exp(-dq.square().sum(axis=-1) / sigma**2).detach()


def gaussian_blob_error(
    g: torch.Tensor,
    rho: torch.Tensor,
    q: torch.Tensor,
    q0: torch.Tensor,
    Rbasis: torch.Tensor,
    sigma: float,
) -> tuple[torch.Tensor, float]:
    """Compute error profile and MAE of density `rho`."""
    rho_err = rho - gaussian_blob(q, q0, Rbasis, sigma)
    rho_norm = (g[..., 0] * rho).sum()
    rho_mae = (g[..., 0] * rho_err.abs()).sum() / rho_norm
    return rho_err, rho_mae.item()


def run(*, Nxy, N_theta, diag, transformation, plot_frames=False) -> float:
    """Run simulation and report error in final density."""

    origin = transformation(torch.zeros((1, 2), device=rc.device))
    Rbasis = (transformation(torch.eye(2, device=rc.device)) - origin).T
    delta_Qfrac = torch.tensor([1.0, 1.0] if diag else [1.0, 0.0], device=rc.device)
    delta_q = delta_Qfrac @ Rbasis.T

    # Initialize velocities (eventually should be in Material):
    v_F = 200.0
    init_angle = torch.atan2(delta_q[1], delta_q[0]).item()
    dtheta = 2 * np.pi / N_theta
    theta = torch.arange(N_theta, device=rc.device) * dtheta + init_angle
    v = v_F * torch.stack([theta.cos(), theta.sin()], dim=-1)

    sim = Advect(transformation=transformation, v=v, N=(Nxy, Nxy))

    # Set the time for slightly more than one period
    t_period = delta_q.norm().item() / v_F
    time_steps = round(1.25 * t_period / sim.dt)
    t_final = time_steps * sim.dt
    log.info(f"\nRunning for {time_steps} steps at {Nxy = }:")

    # Initialize initial and expected final density
    sigma = 0.05
    q = sim.q
    g = sim.g
    q0 = origin + torch.tensor([0.5, 0.5], device=rc.device) @ Rbasis.T
    sim.rho[..., 0] = gaussian_blob(q, q0, Rbasis, sigma)

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
    rho = sim.rho[..., 0]
    rho_err, rho_mae = gaussian_blob_error(g, rho, q, q_final, Rbasis, sigma)
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
    parser.add_argument(
        "--input_svg", help="Input patch transformation (SVG file)", type=str
    )
    args = parser.parse_args()

    if args.input_svg is not None:
        patch_coords = get_splines(args.input_svg)
        boundary = torch.cat([spline[:-1] for spline in patch_coords])
        boundary /= 100.0
        print(boundary)
    else:
        boundary = torch.tensor(
            [
                [0.0, 0.0],
                [0.4, 0.0],
                [0.6, 0.2],
                [1.0, 0.2],
                [1.0, 0.5],
                [1.1, 0.7],
                [1.1, 1.0],
                [0.7, 1.0],
                [0.5, 0.8],
                [0.1, 0.8],
                [0.1, 0.5],
                [0.0, 0.3],
            ],
            device=rc.device,
        )

    # Initialize geometric transformation:
    transformation = BicubicPatch(boundary=boundary)

    if args.Nxy_min is None:
        run(
            Nxy=args.Nxy,
            N_theta=args.Ntheta,
            diag=args.diag,
            transformation=transformation,
            plot_frames=True,
        )
    else:
        # Convergence test:
        assert isinstance(args.Nxy_min, int)
        assert args.Nxy_min < args.Nxy
        Ns = []
        errs = []
        Nxy = args.Nxy_min
        while Nxy <= args.Nxy:
            err = run(
                Nxy=Nxy,
                N_theta=args.Ntheta,
                diag=args.diag,
                transformation=transformation,
                plot_frames=False,
            )
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
