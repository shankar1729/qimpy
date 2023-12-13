import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    time_steps = 400
    steps_per_plot = 4

    geometry = Geometry(svg_file=input_svg, v_F=v_F, N=N, N_theta=N_theta, diag=diag)

    # Initialize density
    sigma = 5
    patch = geometry.patches[0]  # select a patch to put the blob within
    q = patch.q
    q0 = patch.origin + torch.tensor([0.5, 0.5], device=rc.device) @ patch.Rbasis.T
    patch.rho[..., 0] = gaussian_blob(q, q0, patch.Rbasis, sigma)

    # Run time steps
    for plot_step in tqdm(range(time_steps // steps_per_plot)):
        # Plot all patches on single MPL plot
        plt.clf()
        plt.gca().set_aspect("equal")
        rho_max = max(patch.density.max().item() for patch in geometry.patches)
        contour_opts = dict(levels=np.linspace(0.0, rho_max, 100))
        for i, patch in enumerate(geometry.patches):
            patch.plot_streamlines(plt, contour_opts, dict(linewidth=1.0))
        plt.savefig(
            f"animation/advect_{plot_step:04d}.png",
            bbox_inches="tight",
            dpi=200,
        )
        for time_step in range(steps_per_plot):
            geometry.time_step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_svg", help="Input patch (SVG file)", type=str)
    args = parser.parse_args()
    test_geometry(args.input_svg)
