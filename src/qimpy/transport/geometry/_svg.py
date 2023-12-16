from __future__ import annotations
from typing import NamedTuple
from collections import defaultdict
import os

import numpy as np
from svg.path import parse_path, CubicBezier, Line, Close
from xml.dom import minidom


class PatchSet(NamedTuple):
    vertices: np.ndarray  #: Nverts x 2 vertex coordinates (including control points)
    edges: np.ndarray  #: Nedges x 4 vertex indices for each c-spline edge
    quads: np.ndarray  #: Nquads x 4 edge indices within each quad
    adjacency: np.ndarray  #: Nquads x 4 x 2: neighbor indices for each (quad, edge)


def parse_svg(svg_file: str, tol: float = 1e-3) -> PatchSet:
    """Parse SVG file into PatchSet, with vertices identified with tolerance `tol`."""
    return SVGParser(svg_file, tol).patch_set


PATCH_SIDES: int = 4  #: Support only quad-patches (implicitly required throughout)


def parse_style(style_str: str):
    return {
        prop: value for prop, value in [cmd.split(":") for cmd in style_str.split(";")]
    }


def get_splines(svg_file: str) -> tuple[np.ndarray, list]:
    """Get spline geometries and colors from SVG file."""
    svg_paths = minidom.parse(svg_file).getElementsByTagName("path")

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
    splines = []
    colors = []
    for segment, style in zip(segments, styles):
        if isinstance(segment, (Line, Close, CubicBezier)):
            splines.append(ensure_cubic_spline(segment))
            colors.append(style["stroke"])
    splines = np.array(splines)
    splines = np.stack((splines.real, splines.imag), axis=-1)  # to real array
    return splines, colors


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


class SVGParser:
    def __init__(self, svg_file, tol=0.001):
        splines, colors = get_splines(svg_file)
        vertices, edges = weld_points(splines, tol=tol)
        cycles = find_cycles(edges, vertices)

        # Randomly permute cycles to test transfer in general case:
        if "QIMPY_CYCLE_PERMUTE" in os.environ:
            random = np.random.default_rng()
            for cycle in cycles:
                roll = random.integers(4)
                cycle[:] = cycle[roll:] + cycle[:roll]

        # Look-up table for edges by end-point vertex indices:
        # Second index is +1 for edges in forward direction and -1 for reverse
        edges_lookup = {(edge[0], edge[-1]): (ind, 1) for ind, edge in enumerate(edges)}
        edges_lookup.update(
            {(edge[-1], edge[0]): (ind, -1) for ind, edge in enumerate(edges)}
        )  # include reversed direction

        quads = []
        quad_edges = {}  # map global edge index to quad, edge-within indices
        color_adj = {}

        color_pairs = defaultdict(list)
        for i, color in enumerate(colors):
            # Ignore black edges
            if color != "#000000":
                color_pairs[color].append(i)

        # Only include pairs, exclude all others
        color_pairs = {key: val for key, val in color_pairs.items() if len(val) == 2}

        # Build quads, ensuring each spline goes along the direction of the cycle
        edges_new = []  # New edges with corrected directions of traversal
        for cycle in cycles:
            cur_quad = []
            cycle_next = cycle[1:] + [cycle[0]]  # next entry for each in cycle
            for edge in zip(cycle, cycle_next):
                i_spline, direction = edges_lookup[edge]
                color = colors[i_spline]
                # Add edge in appropriate direction to new list:
                i_edge = len(edges_new)
                edges_new.append(edges[i_spline][::direction])
                cur_quad.append(i_edge)
                quad_edges[edge] = (len(quads), len(cur_quad) - 1)
                if color in color_pairs:
                    color_adj[edge] = color
            quads.append(cur_quad)
        edges = np.stack(edges_new)

        # Compute adjacency:
        adjacency = np.full((len(quads), PATCH_SIDES, 2), -1)
        for edge, adj in quad_edges.items():
            quad, edge_ind = adj
            # Handle inner adjacency
            if edge[::-1] in quad_edges:
                adjacency[quad, edge_ind, :] = quad_edges[edge[::-1]]

            # Handle color adjacency
            if edge in color_adj:
                color = color_adj[edge]
                # N^2 lookup, fine for now
                for other_edge, other_color in color_adj.items():
                    if other_color == color and edge != other_edge:
                        adjacency[quad, edge_ind, :] = quad_edges[other_edge]

        self.patch_set = PatchSet(vertices, np.array(edges), np.array(quads), adjacency)


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
            if (depth < PATCH_SIDES) and (next_vertex not in cycle):
                cycle_search(cycle + [next_vertex], depth=depth + 1)
            elif (depth == PATCH_SIDES) and (next_vertex == cycle[0]):
                add_cycle(cycle)

    def add_cycle(cycle: list[int]) -> None:
        """Add a cycle with normalized vertex order and handedness, if unique."""
        # Ensure cycle is counter-clockwise:
        dv = vertices[cycle[1:]] - vertices[cycle[0]]
        area = np.cross(dv[0], dv[1]) + np.cross(dv[1], dv[2])
        if area < 0.0:
            cycle = cycle[::-1]

        # Normalize vertex order in cycle:
        min_index = np.argmin(cycle)
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
