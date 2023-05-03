from __future__ import annotations
from .. import TreeNode, rc
from ..utils import CpPath
from typing import Sequence, Union
import numpy as np
import torch


class Geometry(TreeNode):
    """Geometry specification."""

    vertices: torch.Tensor  #: Cartesian coordinates of vertices (n_vertices x 2)
    edges: torch.Tensor  #: 0-based vertex indices and edge resolution (n_edges x 4)
    edge_splines: list[QuadraticSpline]

    def __init__(
        self,
        *,
        vertices: Union[Sequence[Sequence[float]], np.ndarray, torch.Tensor],
        edges: Union[Sequence[Sequence[float]], np.ndarray, torch.Tensor],
        checkpoint_in: CpPath = CpPath(),
    ):
        """
        Initialize geometry parameters.

        Parameters
        ----------
        vertices
            :yaml:`Caretsian vertex coordinates (n_vertices x 2).`
        edges
            :yaml:`Indices of vertices in edges (n_edges x 4.`
            Each edge should have three 0-based indices into the vertices array,
            for the starting, end and mid point. The midpoint index can be set
            to -1 to make the edge linear (autocompute midpoint from extremes).
            The final entry is the number of subdivisions, or -1 for automatic
        """
        super().__init__()

        self.vertices = torch.tensor(vertices, device=rc.device)
        assert len(self.vertices.shape) == 2
        assert self.vertices.shape[1] == 2

        self.edges = torch.tensor(edges, device=rc.device, dtype=torch.int)
        assert len(self.edges.shape) == 2
        assert self.edges.shape[1] == 4

        self.edge_splines = []
        for i0, i1, i_mid, n_points in self.edges.to(rc.cpu):
            v0 = self.vertices[i0]
            v1 = self.vertices[i1]
            v_mid = (0.5 * (v0 + v1)) if (i_mid < 0) else self.vertices[i_mid]
            self.edge_splines.append(QuadraticSpline(v0, v1, v_mid, n_points))


class QuadraticSpline:

    n_points: int  #: Number of basis points along edge
    coeff: torch.Tensor  #: Quadratic coefficients

    def __init__(
        self, v0: torch.Tensor, v1: torch.Tensor, v_mid: torch.Tensor, n_points: int
    ):
        """
        Create quadratic spline from `v0` to `v1` via `v_mid` with `n_points` points.
        """
        self.n_points = n_points

        # Solve for coefficients:
        Lhs = (
            (n_points * torch.tensor([0, 0.5, 1]))[:, None] ** torch.arange(3)[None, :]
        ).to(rc.device)
        rhs = torch.stack((v0, v_mid, v1))
        self.coeff = torch.linalg.solve(Lhs, rhs)
        print(self.coeff)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Get spline coordinate values for a sequence of x."""
        return (x[:, None] ** torch.arange(3, device=rc.device)[None, :]) @ self.coeff

    @property
    def points(self) -> torch.Tensor:
        """Get spline points at original resolution, including end points."""
        return self.value(
            torch.arange(self.n_points + 1, dtype=torch.double, device=rc.device)
        )
