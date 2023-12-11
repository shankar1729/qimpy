import argparse

import torch
import matplotlib.pyplot as plt
from qimpy import rc

from . import Geometry


def gaussian_blob(
    q: torch.Tensor, q0: torch.Tensor, Rbasis: torch.Tensor, sigma: float
) -> torch.Tensor:
    dq = q - q0
    dq -= torch.floor(0.5 + dq @ torch.linalg.inv(Rbasis.T)) @ Rbasis.T
    return torch.exp(-dq.square().sum(axis=-1) / sigma**2).detach()


def test_geometry(input_svg):
    v_F = 200.0
    N_theta = 1
    N = (64, 64)
    diag = True
    time_steps = 100

    geometry = Geometry(svg_file=input_svg, v_F=v_F, N=N, N_theta=N_theta, diag=diag)

    # Initialize density
    sigma = 5 
    q = geometry.patches[0].q
    q0 = (
        geometry.patches[0].origin
        + torch.tensor([0.5, 0.5], device=rc.device) @ geometry.patches[0].Rbasis.T
    )
    geometry.patches[0].rho[..., 0] = gaussian_blob(
        q, q0, geometry.patches[0].Rbasis, sigma
    )

    for time_step in range(time_steps):
        for i, patch in enumerate(geometry.patches):
            plt.clf()
            patch.plot_streamlines(plt, dict(levels=100), dict(linewidth=1.0))
            plt.gca().set_aspect("equal")
            plt.savefig(
                f"animation/advect_{time_step:04d}_patch_{i}.png",
                bbox_inches="tight",
                dpi=200,
            )
        geometry.time_step()
        print(f"{time_step=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_svg", help="Input patch (SVG file)", type=str)
    args = parser.parse_args()
    test_geometry(args.input_svg)
