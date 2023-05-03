from __future__ import annotations
from .. import TreeNode, rc
from ..utils import CpPath
from typing import Sequence, Union, Any, Optional
import numpy as np
import torch


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
        checkpoint_in: CpPath = CpPath(),
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

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Get spline coordinate values for a sequence of x."""
        return (x[:, None] ** torch.arange(3, device=rc.device)[None, :]) @ self.coeff

    @property
    def points(self) -> torch.Tensor:
        """Get spline points at original resolution, including end points."""
        return self.value(
            torch.arange(self.n_points + 1, dtype=torch.double, device=rc.device)
        )


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

def affine(X, Y, x_y_corners):
    #    a = [a0, a1, a2, a3, a4, a5, a6, a7]

    """
    Inputs :
        - Grid in X and Y. X, Y need to be outputs of np.meshgrid()
        - coordinates of 4 points on the original grid (x, y)
    Output : Transformed grid
    """

    x_y_bottom_left, x_y_bottom_right, x_y_top_right, x_y_top_left = x_y_corners

    N1 = X.shape[0]
    N2 = X.shape[1]
    X_Y_bottom_left = [0, 0]
    X_Y_bottom_right = [N1 - 1, 0]
    X_Y_top_right = [N1 - 1, N2 - 1]
    X_Y_top_left = [0, N2 - 1]

    x0, y0 = x_y_bottom_left
    X0, Y0 = X_Y_bottom_left
    x1, y1 = x_y_bottom_right
    X1, Y1 = X_Y_bottom_right
    x2, y2 = x_y_top_right
    X2, Y2 = X_Y_top_right
    x3, y3 = x_y_top_left
    X3, Y3 = X_Y_top_left

    # x = a0 + a1*X + a2*Y + a3*X*Y
    # y = b0 + b1*X + b2*Y + b3*X*Y

    # x0 = a0 + a1*X0 + a2*Y0 + a3*X0*Y0
    # x1 = a0 + a1*X1 + a2*Y1 + a3*X1*Y1
    # x2 = a0 + a1*X2 + a2*Y2 + a3*X2*Y2
    # x3 = a0 + a1*X3 + a2*Y3 + a3*X3*Y3

    # A x = b
    A = np.array(
        [
            [1, X0, Y0, X0 * Y0],
            [1, X1, Y1, X1 * Y1],
            [1, X2, Y2, X2 * Y2],
            [1, X3, Y3, X3 * Y3],
        ]
    )
    b = np.array([[x0], [x1], [x2], [x3]])

    a0, a1, a2, a3 = np.linalg.solve(A, b)

    a0 = a0[0]
    a1 = a1[0]
    a2 = a2[0]
    a3 = a3[0]

    # y0 = b0 + b1*X0 + b2*Y0 + b3*X0*Y0
    # y1 = b0 + b1*X1 + b2*Y1 + b3*X1*Y1
    # y2 = b0 + b1*X2 + b2*Y2 + b3*X2*Y2
    # y3 = b0 + b1*X3 + b2*Y3 + b3*X3*Y3

    b = np.array([[y0], [y1], [y2], [y3]])

    b0, b1, b2, b3 = np.linalg.solve(A, b)

    b0 = b0[0]
    b1 = b1[0]
    b2 = b2[0]
    b3 = b3[0]

    x = a0 + a1 * X + a2 * Y + a3 * X * Y
    y = b0 + b1 * X + b2 * Y + b3 * X * Y

    # Calculate and return analytical jacobian
    dx_dX = a1 + a3 * Y
    dx_dY = a2 + a3 * X

    dy_dX = b1 + b3 * Y
    dy_dY = b2 + b3 * X

    jacobian = [[dx_dX, dx_dY], [dy_dX, dy_dY]]

    return (x, y, jacobian)


def jacobian(X, Y, transform, x_y_corners):

    x, y, jacobian_ = transform(X, Y, x_y_corners)

    return jacobian_


def jacobian_inv(X, Y, transform, x_y_corners):

    """Returns dX/dx in a 2x2 array

    Usage :
    [[dX_dx, dX_dy], [dY_dx, dY_dy]] = jacobian_inv(X, Y)

    """

    A = jacobian(X, Y, transform, x_y_corners)

    a = A[0][0]
    b = A[0][1]
    c = A[1][0]
    d = A[1][1]

    det_A = a * d - b * c

    inv_A = [[0, 0], [0, 0]]

    inv_A[0][0] = d / det_A
    inv_A[0][1] = -b / det_A
    inv_A[1][0] = -c / det_A
    inv_A[1][1] = a / det_A

    return inv_A


def sqrt_det_g(X, Y, transform, x_y_corners):

    jac = jacobian(X, Y, transform, x_y_corners)

    dx_dX = jac[0][0]
    dx_dY = jac[0][1]
    dy_dX = jac[1][0]
    dy_dY = jac[1][1]

    g_11 = (dx_dX) ** 2.0 + (dy_dX) ** 2.0

    g_12 = dx_dX * dx_dY + dy_dX * dy_dY

    g_21 = g_12

    g_22 = (dx_dY) ** 2.0 + (dy_dY) ** 2.0

    det_g = g_11 * g_22 - g_12 * g_21

    return np.sqrt(det_g)
