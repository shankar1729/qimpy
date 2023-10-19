from __future__ import annotations

import torch
import numpy as np

from qimpy import rc


def get_splines(svg_file: str) -> list[torch.Tensor]:
    """Read a list of splines from an SVG file.
    Each entry in the result must be 4 x 2 (control points x dim).
    """
    # TODO
    # Temporary test splines:
    return [
        torch.tensor(coords, device=rc.device)
        for coords in [
            [[0.0, 0.0], [0.4, 0.0], [0.6, 0.2], [1.0, 0.2]],
            [[1.0, 0.2], [1.0, 0.5], [1.1, 0.7], [1.1, 1.0]],
            [[1.1, 1.0], [0.7, 1.0], [0.5, 0.8], [0.1, 0.8]],
            [[0.1, 0.8], [0.1, 0.5], [0.0, 0.3], [0.0, 0.0]],
        ]
    ]


def plot_spline(ax, spline: torch.Tensor, n_points: int = 64) -> None:
    t = np.linspace(0.0, 1.0, n_points + 1)[:, None]
    t_bar = 1.0 - t
    # Evaluate cubic spline by De Casteljau's algorithm:
    control_points = spline.to(rc.cpu).numpy()
    result = control_points[:, None, :]
    for iter in range(len(spline) - 1):
        result = result[:-1] * t_bar + result[1:] * t
    points = result[0]
    # Plot
    ax.plot(points[:, 0], points[:, 1], color="k")
    if len(spline) == 4:
        ax.plot(control_points[:2, 0], control_points[:2, 1], color="r")
        ax.plot(control_points[2:, 0], control_points[2:, 1], color="r")
        ax.scatter(control_points[1:3, 0], control_points[1:3, 1], color="r")


def main():
    import matplotlib.pyplot as plt

    splines = get_splines("test.svg")
    plt.figure()
    ax = plt.gca()
    ax.set_aspect("equal")
    for spline in splines:
        plot_spline(ax, spline)
    plt.show()


if __name__ == "__main__":
    main()
