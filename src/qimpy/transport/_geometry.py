from __future__ import annotations
from typing import Sequence, Union, Any, Optional

import numpy as np
import torch

from qimpy import TreeNode, rc
from qimpy.io import CheckpointPath


class Geometry(TreeNode):
    """Geometry specification."""

    vertices: torch.Tensor  #: Cartesian coordinates of vertices (n_vertices x 2)
    edges: torch.Tensor  #: 0-based vertex indices and edge resolution (n_edges x 4)
    quads: torch.Tensor  #: 0-based edge indices in each quad (n_quads x 4)
    edge_splines: list[QuadraticSpline]

    def __init__(
        self,
        *,
        vertices: Union[Sequence[Sequence[float]], np.ndarray, torch.Tensor],
        edges: Union[Sequence[Sequence[int]], np.ndarray, torch.Tensor],
        quads: Union[Sequence[Sequence[int]], np.ndarray, torch.Tensor],
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ):
        """
        Initialize geometry parameters.

        Parameters
        ----------
        vertices
            :yaml:`Caretsian vertex coordinates (n_vertices x 2).`
        edges
            :yaml:`Indices of vertices in edges (n_edges x 4).`
            Each edge should have three 0-based indices into the vertices array,
            for the starting, end and mid point. The midpoint index can be set
            to -1 to make the edge linear (autocompute midpoint from extremes).
            The final entry is the number of subdivisions, or -1 for automatic
        quads
            :yaml:`Indices of edges in each quad (n_quads x 4).`
        """
        super().__init__()

        self.vertices = _make_check_tensor(vertices, (-1, 2))
        self.edges = _make_check_tensor(edges, (-1, 4), dtype=torch.int)
        self.quads = _make_check_tensor(quads, (-1, 4), dtype=torch.int)

        self.edge_splines = []
        for i0, i1, i_mid, n_points in self.edges.to(rc.cpu):
            v0 = self.vertices[i0]
            v1 = self.vertices[i1]
            v_mid = (0.5 * (v0 + v1)) if (i_mid < 0) else self.vertices[i_mid]
            self.edge_splines.append(QuadraticSpline(v0, v1, v_mid, n_points))

        # Construct equivalence classes of opposite edges in quads:
        # edge_class = np.arange(self.edges.shape[0])
        edge_pairs = self.quads.reshape(-1, 2, 2).transpose(-1, -2).flatten(0, 1)
        print(equivalence_classes(edge_pairs.to(rc.cpu).numpy()))


class QuadraticSpline:
    v0: torch.Tensor  #: Starting point
    v1: torch.Tensor  #: End point
    v_mid: torch.Tensor  #: Mid point (can be used to curve the segment)
    n_points: int  #: Number of basis points along edge
    coeff: torch.Tensor  #: Quadratic coefficients

    def __init__(
        self, v0: torch.Tensor, v1: torch.Tensor, v_mid: torch.Tensor, n_points: int
    ):
        """
        Create quadratic spline from `v0` to `v1` via `v_mid` with `n_points` points.
        """
        self.v0 = v0
        self.v1 = v1
        self.v_mid = v_mid
        self.n_points = n_points

        # Solve for coefficients:
        Lhs = (torch.tensor([0, 0.5, 1])[:, None] ** torch.arange(3)[None, :]).to(
            rc.device
        )
        rhs = torch.stack((v0, v_mid, v1))
        self.coeff = torch.linalg.solve(Lhs, rhs)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Get spline coordinate values for a sequence of fractional coordinates x."""
        return (x[:, None] ** torch.arange(3, device=rc.device)[None, :]) @ self.coeff

    @property
    def points(self) -> torch.Tensor:
        """Get spline points at original resolution, including end points."""
        assert self.n_points > 0
        return self.value(torch.linspace(0.0, 1.0, self.n_points + 1, device=rc.device))

    def length(self, n_segments: int = 20) -> float:
        """Compute length of edge, using n_segments if n_points is not specified."""
        N = self.n_points if (self.n_points > 0) else n_segments
        points = self.value(torch.linspace(0.0, 1.0, N + 1, device=rc.device))
        return torch.diff(points, dim=0).norm(dim=1).sum().item()


def _make_check_tensor(
    data: Union[Sequence[Sequence[Any]], np.ndarray, torch.Tensor],
    dims: Sequence[int],
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    result = torch.tensor(data, device=rc.device, dtype=dtype)
    assert len(result.shape) == len(dims)
    for result_shape_i, dim_i in zip(result.shape, dims):
        if dim_i >= 0:
            assert result_shape_i == dim_i
    return result


def equivalence_classes(pairs: torch.Tensor) -> torch.Tensor:
    """Given Npair x 2 array of index pairs that are equivalent,
    compute equivalence class numbers for each original index."""
    # Construct adjacency matrix:
    N = pairs.max() + 1
    i_pair, j_pair = pairs.T
    adjacency_matrix = torch.eye(N, device=rc.device)
    adjacency_matrix[i_pair, j_pair] = 1.0
    adjacency_matrix[j_pair, i_pair] = 1.0

    # Expand to indirect neighbors by repeated multiplication:
    n_non_zero_prev = torch.count_nonzero(adjacency_matrix)
    for i_mult in range(N):
        adjacency_matrix = adjacency_matrix @ adjacency_matrix
        n_non_zero = torch.count_nonzero(adjacency_matrix)
        if n_non_zero == n_non_zero_prev:
            break  # highest-degree connection reached
        n_non_zero_prev = n_non_zero

    # Find first non-zero entry of above (i.e. first equivalent index):
    is_first = torch.logical_and(
        adjacency_matrix.cumsum(dim=1) == adjacency_matrix, adjacency_matrix != 0.0
    )
    first_index = torch.nonzero(is_first)[:, 1]
    assert len(first_index) == N
    return torch.unique(first_index, return_inverse=True)[1]  # minimal class indices
