from __future__ import annotations
from typing import Sequence, Union, Any, Optional

import numpy as np
import torch
from xml.dom import minidom
from svg.path import parse_path, CubicBezier, Line, Close

from qimpy import TreeNode, rc
from qimpy.io import CheckpointPath


class CubicSpline:
    spline_params: torch.Tensor  # 4 x 2 tensor
    neighbor_edge: CubicSpline
    n_points: int

    def __init__(self, spline_params, neighbor_edge=None, n_points=64):
        self.spline_params = spline_params
        self.n_points = n_points
        self.neighbor_edge = None

    def __repr__(self):
        numpy_spline = self.spline_params.numpy()
        return f"{numpy_spline[0, :]} -> {numpy_spline[-1, :]} (neighbor: {self.neighbor_edge is not None})"

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

    def __init__(self, edges: Sequence[CubicSpline]):
        """Initialize from 12 x 2 coordinates of control points on perimeter."""
        self.edges = edges
        boundary = torch.cat([spline.spline_params[:-1] for spline in self.edges])
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


def weld_points(coords: torch.Tensor, tol: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Weld `coords` within tolerance `tol`, returning indices and unique coordinates.
    Here, coords has dimensions (..., d), where d is the dimension of space.
    The first output is a flat list of unique welded vertices of shape (N_uniq, d).
    The second output contains indices into this unique list of dimensions (...)."""
    coords_flat = coords.flatten(end_dim=-2)
    distances = (coords_flat[:, None] - coords_flat[None]).norm(dim=-1)
    equiv_index = torch.where(distances < tol, 1, 0).argmax(dim=1)
    _, inverse, counts = torch.unique(
        equiv_index, return_inverse=True, return_counts=True
    )
    # Compute the centroid of each set of equivalent vertices:
    coords_uniq = torch.zeros((len(counts), coords_flat.shape[-1]), device=rc.device)
    coords_uniq.index_add_(0, inverse, coords_flat)  # sum equivalent coordinates
    coords_uniq *= (1.0 / counts)[:, None]  # convert to mean
    return coords_uniq, inverse.view(coords.shape[:-1])


PATCH_SIDES: int = 4  #: Support only quad-patches (implicitly required throughout)


def get_splines(svg_file: str) -> torch.Tensor:
    doc = minidom.parse(svg_file)
    svg_elements = []
    svg_paths = doc.getElementsByTagName("path")

    # Concatenate segments from all paths in SVG file
    for path in svg_paths:
        svg_elements.extend(parse_path(path.getAttribute("d")))

    def segment_to_tensor(segment):
        if isinstance(segment, CubicBezier):
            control1, control2 = segment.control1, segment.control2
        # Both Line and Close can produce linear segments
        elif isinstance(segment, (Line, Close)):
            # Generate a spline from a linear segment
            disp_third = (segment.end - segment.start) / 3.0
            control1 = segment.start + disp_third
            control2 = segment.start + 2 * disp_third
        else:
            raise ValueError("All segments must be cubic splines or lines")
        return torch.view_as_real(
            torch.tensor(
                [segment.start, control1, control2, segment.end],
                device=rc.device,
            )
        )

    # Ignore all elements that are not lines or cubic splines (essentially ignore moves)
    # In the future we may want to throw an error for unsupported segments
    # (e.g. quadratic splines)
    return torch.stack(
        [
            segment_to_tensor(segment)
            for segment in svg_elements
            if isinstance(segment, (Line, Close, CubicBezier))
        ]
    )


def edge_sequence(cycle):
    return list(zip(cycle[:-1], cycle[1:])) + [(cycle[-1], cycle[0])]


class SVGParser:
    def __init__(self, svg_file, epsilon=0.005):
        self.splines = get_splines(svg_file)
        self.vertices, self.edges = weld_points(self.splines[:, (0, -1)], tol=epsilon)
        self.edges_lookup = {
            (edge[0], edge[1]): ind for ind, edge in enumerate(self.edges.tolist())
        }

        self.cycles = []
        self.find_cycles()

        self.patches = []

        patch_edges = {}

        # Now build the patches, ensuring each spline goes along
        # the direction of the cycle
        for cycle in self.cycles:
            patch_splines = []
            for edge in edge_sequence(cycle):
                # Edges lookup reflects the original ordering of the edges
                # if an edge's order doesn't appear in here, it needs to be flipped
                if edge not in self.edges_lookup:
                    new_spline = CubicSpline(
                        torch.flip(
                            self.splines[self.edges_lookup[edge[::-1]]], dims=[0]
                        )
                    )
                else:
                    new_spline = CubicSpline(self.splines[self.edges_lookup[edge]])
                patch_edges[edge] = new_spline
                patch_splines.append(new_spline)
            self.patches.append(BicubicPatch(patch_splines))

        for edge, spline in patch_edges.items():
            if edge[::-1] in patch_edges:
                patch_edges[edge[::-1]].neighbor_edge = spline

    # Determine whether a cycle goes counter-clockwise or clockwise
    # (Return 1 or -1 respectively)
    def cycle_handedness(self, cycle):
        cycle_vertices = [self.vertices[j] for j in cycle]
        edges = edge_sequence(cycle_vertices)
        handed_sum = 0.0
        for v1, v2 in edges:
            handed_sum += (v2[0] - v1[0]) / (v2[1] + v1[1])
        # NOTE: SVG uses a left-handed coordinate system
        return np.sign(handed_sum)

    def add_cycle(self, cycle):
        # Add a cycle if it is unique

        def unique(path):
            return path not in self.cycles

        def normalize_cycle_order(cycle):
            min_index = cycle.index(min(cycle))
            return cycle[min_index:] + cycle[:min_index]

        new_cycle = normalize_cycle_order(cycle)
        # Check both directions
        if unique(new_cycle) and unique(normalize_cycle_order(new_cycle[::-1])):
            self.cycles.append(new_cycle)

    def find_cycles(self):
        # Graph traversal using recursion
        def cycle_search(cycle, depth=1):
            # Don't look for cycles that exceed a single patch (limit recursion depth)
            if depth > PATCH_SIDES:
                return
            start_vertex = cycle[-1]
            for edge in self.edges:
                if start_vertex in edge:
                    next_vertex = int(edge[1] if edge[0] == start_vertex else edge[0])
                    if next_vertex not in cycle:
                        cycle_search(cycle + [next_vertex], depth=depth + 1)
                    elif len(cycle) > 2 and next_vertex == cycle[0]:
                        self.add_cycle(cycle)

        # Search for cycles from each starting vertex
        for first_vertex in range(len(self.vertices)):
            cycle_search([first_vertex])

        # Make sure each cycle goes counter-clockwise
        self.cycles = [
            cycle if self.cycle_handedness(cycle) > 0 else cycle[::-1]
            for cycle in self.cycles
        ]


class Geometry(TreeNode):
    """Geometry specification."""

    edges: list[BicubicPatch]  # Patch objects

    def __init__(
        self,
        *,
        svg_file: str,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ):
        """
        Initialize geometry parameters.

        Parameters
        ----------
        svg_file
            :yaml:`Path to an SVG file containing the input geometry.
        """
        super().__init__()

        svg_parser = SVGParser(svg_file)
        self.patches = svg_parser.patches


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
