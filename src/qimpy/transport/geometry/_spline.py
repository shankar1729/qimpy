from __future__ import annotations
from typing import Optional

import numpy as np
import torch

from qimpy import rc


class CubicSpline:
    spline_params: torch.Tensor  # 4 x 2 tensor
    neighbor_edge: Optional[CubicSpline]
    n_points: int

    def __init__(self, spline_params, neighbor_edge=None, n_points=64):
        self.spline_params = spline_params
        self.n_points = n_points
        self.neighbor_edge = None

    def __repr__(self):
        numpy_spline = self.spline_params.numpy()
        has_edge = self.neighbor_edge is not None
        return f"{numpy_spline[0, :]} -> {numpy_spline[-1, :]} (neighbor: {has_edge})"

    def points(self):
        assert len(self.spline_params) == 4
        t = np.linspace(0.0, 1.0, self.n_points + 1)[:, None]
        t_bar = 1.0 - t
        # Evaluate cubic spline by De Casteljau's algorithm:
        control_points = self.spline_params.to(rc.cpu).numpy()
        result = control_points[:, None, :]
        for i_iter in range(len(self.spline_params) - 1):
            result = result[:-1] * t_bar + result[1:] * t
        return result[0]


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


def plot_spline(ax, spline: torch.Tensor, n_points: int = 64) -> None:
    assert len(spline) == 4
    t = np.linspace(0.0, 1.0, n_points + 1)[:, None]
    t_bar = 1.0 - t
    # Evaluate cubic spline by De Casteljau's algorithm:
    control_points = spline.to(rc.cpu).numpy()
    result = control_points[:, None, :]
    for i_iter in range(len(spline) - 1):
        result = result[:-1] * t_bar + result[1:] * t
    points = result[0]
    # Plot
    ax.plot(points[:, 0], points[:, 1], color="k")
    ax.plot(control_points[:2, 0], control_points[:2, 1], color="r")
    ax.plot(control_points[2:, 0], control_points[2:, 1], color="r")
    ax.scatter(control_points[1:3, 0], control_points[1:3, 1], color="r")
