from __future__ import annotations
from typing import Iterable
from collections import defaultdict
from dataclasses import dataclass
import os

import numpy as np
from svg.path import parse_path, CubicBezier, Line, Close
from xml.dom import minidom

from qimpy import rc
from qimpy.io import InvalidInputException
from . import spline_length, evaluate_spline, within_circles_np


QUAD_N_SIDES: int = 4  #: Support only quads (implicitly required throughout)


@dataclass
class QuadSet:
    """Set of quads, defining simulation geometry."""

    vertices: np.ndarray  #: Nverts x 2 vertex coordinates (including control points)
    quads: np.ndarray  #: Nquads x 4 x 4 vertex indices for each quad and edge within
    adjacency: np.ndarray  #: Nquads x 4 x 2: neighbor indices for each (quad, edge)
    displacements: np.ndarray  #: Nquads x 4 x 2 edge displacements for each adjacency
    grid_size: np.ndarray  #: Nquads x 2 grid dimensions for each quad
    contacts: np.ndarray  #: Ncontacts x 3: center x, y  and radius of each circle
    apertures: np.ndarray  #: Napertures x 3: center x, y  and radius of each circle
    aperture_names: list[str]  #: Napertures labels for each aperture in SVG
    has_apertures: np.ndarray  #: Nquads x 4: whether each edge has any apertures on it

    def get_boundary(self, i_quad: int) -> np.ndarray:
        """Get sequence of boundary points (12 x 2) defining a specified quad.
        Suitable for initializing a `BicubicPatch`."""
        indices = self.quads[i_quad, :, :-1]  # drop last redundant point in each edge
        return self.vertices[indices.flatten()]


def parse_svg(
    svg_file: str,
    svg_unit: float,
    grid_spacing: float,
    contact_names: Iterable[str],
    tol: float = 1e-3,
) -> QuadSet:
    """Parse SVG file into QuadSet, sampled with `grid_spacing`,
    and with vertex equivalence determined with tolerance `tol`.
    All distances are scaled by `svg_unit` in the output.
    Note that `grid_spacing` is in the output units since it sets the
    simulation resolution, while `tol` applies in the SVG units before
    scaling since it deals with design tolerance in the SVG editor."""
    svg_xml = minidom.parse(svg_file)
    splines, colors, dashed = get_splines(svg_xml)
    circles, circle_names = get_circles(svg_xml)

    # Transition SVG quantities to real units:
    splines *= svg_unit
    circles *= svg_unit
    tol *= svg_unit  # At input, `tol` is in SVG units

    # Check contact specification:
    contact_indices = []
    for contact_name in contact_names:
        if contact_name.startswith("aperture"):
            raise InvalidInputException(
                "Contact names cannot start with 'aperture', which is reserved for"
                " circles in the SVG that control pass-through in internal edges."
            )
        try:
            contact_indices.append(circle_names.index(contact_name))
        except ValueError:
            raise InvalidInputException(
                f"Contact '{contact_name}' not found in {svg_file}."
                " (Each contact name must match the id of a circle in the svg.)"
            )
    contacts = circles[contact_indices] if contact_indices else np.zeros((0, 3))

    # Add apertures:
    aperture_indices = [
        i_circle
        for i_circle, circle_name in enumerate(circle_names)
        if circle_name.startswith("aperture")
    ]
    apertures = circles[aperture_indices] if aperture_indices else np.zeros((0, 3))
    aperture_names = [circle_names[index] for index in aperture_indices]

    # Check non-dashed splines for apertures (then they are partially pass-through):
    pass_throughs = []
    for spline, is_dashed in zip(splines, dashed):
        if is_dashed:
            pass_throughs.append(True)  # fully-pass through if dashed
        else:
            Npoints = 2 * int(np.ceil(spline_length(spline) / grid_spacing))
            t_spline = (np.arange(Npoints)[:, None] + 0.5) / Npoints
            points = evaluate_spline(spline, t_spline)
            pass_throughs.append(bool(np.any(within_circles_np(apertures, points))))

    # Process mesh geometry:
    vertices, edges = weld_points(splines, tol=tol)
    cycles = find_cycles(edges, vertices)

    # Randomly permute cycles to test transfer in general case:
    if "QIMPY_CYCLE_PERMUTE" in os.environ:
        random = rc.comm.bcast(np.random.default_rng())
        for cycle in cycles:
            roll = random.integers(4)
            cycle[:] = cycle[roll:] + cycle[:roll]

    # Look-up table for edges by end-point vertex indices:
    # Second index is +1 for edges in forward direction and -1 for reverse
    edges_lookup = {
        tuple(edge[::direction]): (i_spline, direction)
        for i_spline, edge in enumerate(edges[:, [0, -1]])
        for direction in (+1, -1)
    }

    # Build quads, ensuring each spline goes along the direction of the cycle
    quads = np.empty((len(cycles), 4, 4), dtype=int)
    has_apertures = np.empty(quads.shape[:2], dtype=bool)
    quad_edges = {}  # map edge (vertex index pair) to quad, edge-within indices
    color_edges = defaultdict(list)  # list of edges (vert index pair) by color
    edge_colors = {}  # map edges to colors (only for non-black edges)
    pass_through_edges = set()  # only edges that need adjacency (dashed / apertures)
    for i_quad, cycle in enumerate(cycles):
        cycle_next = cycle[1:] + [cycle[0]]  # next entry for each in cycle
        for i_edge, edge in enumerate(zip(cycle, cycle_next)):
            i_spline, direction = edges_lookup[edge]
            color = colors[i_spline]
            pass_through = pass_throughs[i_spline]
            # Add edge in appropriate direction to new list:
            quads[i_quad, i_edge] = edges[i_spline][::direction]
            has_apertures[i_quad, i_edge] = pass_through and (not dashed[i_spline])
            # Update edge look-ups:
            quad_edges[edge] = (i_quad, i_edge)
            if color != "#000000":  # only use non-black colors for adjacency
                color_edges[color].append(edge)
                edge_colors[edge] = color
            if pass_through:
                pass_through_edges.add(edge)

    # Compute adjacency:
    adjacency = np.full((len(quads), QUAD_N_SIDES, 2), -1)
    for edge, i_quad_edge in quad_edges.items():
        if edge not in pass_through_edges:
            continue  # only dashed edges are allowed to have any adjacency

        # Handle inner adjacency
        if edge[::-1] in quad_edges:
            adjacency[i_quad_edge] = quad_edges[edge[::-1]]
            assert edge not in edge_colors
            continue

        # Handle color adjacency
        if edge in edge_colors:
            similar_edges = color_edges[edge_colors[edge]]
            assert len(similar_edges) == 2
            for other_edge in similar_edges:
                if other_edge != edge:
                    adjacency[i_quad_edge] = quad_edges[other_edge]

    # Determine edge displacements and equivalence:
    displacements, ij_quad, ij_edge = get_displacements(vertices, quads, adjacency, tol)
    ij_quad_edge = 2 * ij_quad + (ij_edge % 2)  # flattened, only 2 indep. edges/quad
    edge_classes = equivalence_classes(ij_quad_edge, 2 * len(quads))  # flattened index

    # Determine sample counts based on maximum edge length in each class:
    lengths = spline_length(vertices[quads])
    lengths = lengths.reshape(-1, 2, 2).max(axis=1)  # max over equiv edges in each quad
    lengths = lengths.flatten()  # now corresponds to flattened index used above
    n_points = np.empty(len(lengths), dtype=int)
    for edge_class in range(edge_classes.max() + 1):
        sel = np.where(edge_class == edge_classes)[0]
        max_length = lengths[sel].max()
        n_points[sel] = int(np.ceil(max_length / grid_spacing))
    grid_size = n_points.reshape(-1, 2)  # now n_quads x 2 grid dimensions

    return QuadSet(
        vertices,
        quads,
        adjacency,
        displacements,
        grid_size,
        contacts,
        apertures,
        aperture_names,
        has_apertures,
    )


def parse_style(style_str: str):
    return {
        prop: value for prop, value in [cmd.split(":") for cmd in style_str.split(";")]
    }


def get_splines(svg_xml: minidom.Document) -> tuple[np.ndarray, list[str], list[bool]]:
    """Get spline geometries and colors from SVG file."""
    svg_paths = svg_xml.getElementsByTagName("path")

    # Concatenate segments from all paths in SVG file, and parse associated styles
    segments = []
    styles = []
    for path in svg_paths:
        paths = parse_path(path.getAttribute("d"))
        segments.extend(paths)
        styles.extend(len(paths) * [parse_style(path.getAttribute("style"))])

    # Ignore all elements that are not lines or cubic splines (essentially ignore moves)
    # In the future we may want to throw an error for unsupported segments
    # (e.g. quadratic splines)
    splines_complex = []  # coordinates as complex numbers from svg.path library
    colors = []
    dashed = []
    for segment, style in zip(segments, styles):
        if isinstance(segment, (Line, Close, CubicBezier)):
            splines_complex.append(ensure_cubic_spline(segment))
            colors.append(style["stroke"])
            dashed.append(style.get("stroke-dasharray", "none") != "none")
    splines = np.array(splines_complex)
    splines = np.stack((splines.real, splines.imag), axis=-1)  # to real array
    return splines, colors, dashed


def get_circles(svg_xml: minidom.Document) -> tuple[np.ndarray, list]:
    """Get all circle parameters from SVG file."""
    svg_circles = svg_xml.getElementsByTagName("circle")

    # Gather parameters from all circles in SVG file
    params = []
    ids = []
    for circle in svg_circles:
        cx = float(circle.getAttribute("cx"))
        cy = float(circle.getAttribute("cy"))
        r = float(circle.getAttribute("r"))
        circle_id = str(circle.getAttribute("id"))
        params.append([cx, cy, r])
        ids.append(circle_id)

    return np.array(params), ids


def ensure_cubic_spline(segment) -> list[complex]:
    """Convert supported segments to cubic splines."""
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
    return [segment.start, control1, control2, segment.end]


def find_cycles(edges: np.ndarray, vertices: np.ndarray) -> list[list[int]]:
    """Find length-4 cycles within `edges`.
    Only the initial and final vertex indices of edges are used.
    Any intermediate control points within edges are ignored.
    This uses `vertices` to ensure counter-clockwise traversal direction.
    """
    cycles = []

    def cycle_search(cycle: list[int], depth: int = 1) -> None:
        """Find and add 4-cycles by graph traversal using recursion."""
        start_vertex = cycle[-1]
        for edge in edges:
            if start_vertex == edge[0]:
                next_vertex = edge[-1]
            elif start_vertex == edge[-1]:
                next_vertex = edge[0]
            else:
                continue
            if (depth < QUAD_N_SIDES) and (next_vertex not in cycle):
                cycle_search(cycle + [next_vertex], depth=depth + 1)
            elif (depth == QUAD_N_SIDES) and (next_vertex == cycle[0]):
                add_cycle(cycle)

    def add_cycle(cycle: list[int]) -> None:
        """Add a cycle with normalized vertex order and handedness, if unique."""
        # Ensure cycle is counter-clockwise:
        dv = vertices[cycle[1:]] - vertices[cycle[0]]
        area = np.cross(dv[0], dv[1]) + np.cross(dv[1], dv[2])
        if area < 0.0:
            cycle = cycle[::-1]

        # Normalize vertex order in cycle:
        min_index = int(np.argmin(cycle))
        cycle = cycle[min_index:] + cycle[:min_index]

        if cycle not in cycles:
            cycles.append(cycle)

    # Search for cycles from each starting vertex
    for first_vertex in range(len(vertices)):
        cycle_search([first_vertex])
    return cycles


def weld_points(coords: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Weld `coords` within tolerance `tol`, returning indices and unique coordinates.
    Here, coords has dimensions (..., d), where d is the dimension of space.
    The first output is a flat list of unique welded vertices of shape (N_uniq, d).
    The second output contains indices into this unique list of dimensions (...)."""
    coords_flat = coords.reshape(-1, 2)
    distances = np.linalg.norm(coords_flat[:, None] - coords_flat[None], axis=-1)
    equiv_index = np.where(distances < tol, 1, 0).argmax(axis=1)
    _, inverse, counts = np.unique(equiv_index, return_inverse=True, return_counts=True)
    # Compute the centroid of each set of equivalent vertices:
    coords_uniq = np.zeros((len(counts), coords_flat.shape[-1]))
    np.add.at(coords_uniq, inverse, coords_flat)  # sum equivalent coordinates
    coords_uniq *= (1.0 / counts)[:, None]  # convert to mean
    return coords_uniq, inverse.reshape(coords.shape[:-1])


def get_displacements(
    vertices: np.ndarray, quads: np.ndarray, adjacency: np.ndarray, tol: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Check consistency and collect displacements between adjacent edges.
    Also return ij_quad and ij_edge pairs for eaqch displacement (each Npairs x 2)."""
    i_quad, i_edge = np.where(adjacency[..., 0] >= 0)
    j_quad, j_edge = adjacency[i_quad, i_edge].T
    verts_i = vertices[quads[i_quad, i_edge]]
    verts_j = vertices[quads[j_quad, j_edge]][:, ::-1]
    deltas = verts_i - verts_j
    assert np.all(deltas.std(axis=1) < tol).item()
    displacements = np.zeros(adjacency.shape)
    displacements[i_quad, i_edge] = deltas.mean(axis=1)
    return displacements, np.stack((i_quad, j_quad)).T, np.stack((i_edge, j_edge)).T


def equivalence_classes(pairs: np.ndarray, N: int) -> np.ndarray:
    """Given Npair x 2 array of index pairs (into a sequence of length N) that
    are equivalent, compute equivalence class numbers for each original index."""
    # Construct adjacency matrix:
    i_pair, j_pair = pairs.T
    adjacency_matrix = np.eye(N)
    adjacency_matrix[i_pair, j_pair] = 1.0
    adjacency_matrix[j_pair, i_pair] = 1.0

    # Expand to indirect neighbors by repeated multiplication:
    n_non_zero_prev = np.count_nonzero(adjacency_matrix)
    for i_mult in range(N):
        adjacency_matrix = adjacency_matrix @ adjacency_matrix
        n_non_zero = np.count_nonzero(adjacency_matrix)
        if n_non_zero == n_non_zero_prev:
            break  # highest-degree connection reached
        n_non_zero_prev = n_non_zero

    # Find first non-zero entry of above (i.e. first equivalent index):
    is_first = np.logical_and(
        adjacency_matrix.cumsum(axis=1) == adjacency_matrix, adjacency_matrix != 0.0
    )
    first_index = np.where(is_first)[1]
    assert len(first_index) == N
    return np.unique(first_index, return_inverse=True)[1]  # minimal class indices
