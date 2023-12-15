from __future__ import annotations

import numpy as np
import torch

from qimpy import rc


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


def plot_spline(ax, spline: np.ndarray, n_points: int = 64) -> None:
    assert len(spline) == 4
    t = np.linspace(0.0, 1.0, n_points + 1)[:, None]
    t_bar = 1.0 - t
    # Evaluate cubic spline by De Casteljau's algorithm:
    result = spline[:, None, :]
    for i_iter in range(len(spline) - 1):
        result = result[:-1] * t_bar + result[1:] * t
    points = result[0]
    # Plot
    ax.plot(points[:, 0], points[:, 1], color="k")
    ax.plot(spline[:2, 0], spline[:2, 1], color="r")
    ax.plot(spline[2:, 0], spline[2:, 1], color="r")
    ax.scatter(spline[1:3, 0], spline[1:3, 1], color="r")


def spline_length(spline: np.ndarray, n_points: int = 64) -> np.ndarray:
    """Compute cubic spline lengths, batched over any preceding dimensions.
    The shape of spline should end in (4, d), where d is the dimension of space.
    """
    basis = cubic_bernstein(torch.linspace(0.0, 1.0, n_points + 1)).numpy()
    points = np.einsum("ct, ...ci -> ...ti", basis, spline)
    return np.linalg.norm(np.diff(points, axis=-2), axis=-1).sum(axis=-1)
