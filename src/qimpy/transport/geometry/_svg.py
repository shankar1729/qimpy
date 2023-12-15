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


def parse_svg(svg_file: str) -> PatchSet:
    return SVGParser(svg_file).patch_set


PATCH_SIDES: int = 4  #: Support only quad-patches (implicitly required throughout)


def parse_style(style_str: str):
    return {
        prop: value for prop, value in [cmd.split(":") for cmd in style_str.split(";")]
    }


def get_splines(svg_file: str) -> tuple[np.ndarray, list]:
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


def edge_sequence(cycle):
    return list(zip(cycle[:-1], cycle[1:])) + [(cycle[-1], cycle[0])]


class SVGParser:
    def __init__(self, svg_file, tol=0.001):
        self.splines, self.colors = get_splines(svg_file)
        self.vertices, self.edges = weld_points(self.splines[:, (0, -1)], tol=tol)
        self.edges_lookup = {
            (edge[0], edge[1]): ind for ind, edge in enumerate(self.edges)
        }

        self.cycles = []
        self.find_cycles()

        # Randomly permute cycles to test transfer in general case:
        if "QIMPY_CYCLE_PERMUTE" in os.environ:
            random = np.random.default_rng()
            for cycle in self.cycles:
                roll = random.integers(4)
                cycle[:] = cycle[roll:] + cycle[:roll]

        verts = np.copy(self.vertices)
        edges = []
        quads = []

        control_pt_lookup = {}
        quad_edges = {}
        color_adj = {}

        color_pairs = defaultdict(list)
        for i, color in enumerate(self.colors):
            # Ignore black edges
            if color != "#000000":
                color_pairs[color].append(i)

        # Only include pairs, exclude all others
        color_pairs = {key: val for key, val in color_pairs.items() if len(val) == 2}

        # Now build the patches, ensuring each spline goes along
        # the direction of the cycle
        for cycle in self.cycles:
            cur_quad = []
            for edge in edge_sequence(cycle):
                # Edges lookup reflects the original ordering of the edges
                # if an edge's order doesn't appear in here, it needs to be flipped
                if edge not in self.edges_lookup:
                    spline = self.splines[self.edges_lookup[edge[::-1]]][::-1]
                    color = self.colors[self.edges_lookup[edge[::-1]]]
                else:
                    spline = self.splines[self.edges_lookup[edge]]
                    color = self.colors[self.edges_lookup[edge]]
                cp1 = tuple(spline[1].tolist())
                cp2 = tuple(spline[2].tolist())
                # Get control points from spline and add to vertices
                # Ensure that control points are unique by lookup dict
                if cp1 not in control_pt_lookup:
                    verts = np.concatenate((verts, spline[1:3]), 0)
                    control_pt_lookup[cp1] = verts.shape[0] - 2
                    control_pt_lookup[cp2] = verts.shape[0] - 1
                edges.append(
                    [edge[0], control_pt_lookup[cp1], control_pt_lookup[cp2], edge[1]]
                )
                cur_quad.append(len(edges) - 1)
                quad_edges[edge] = (len(quads), len(cur_quad) - 1)
                if color in color_pairs:
                    color_adj[edge] = color
            quads.append(cur_quad)

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

        self.patch_set = PatchSet(verts, np.array(edges), np.array(quads), adjacency)

    # Determine whether a cycle goes counter-clockwise or clockwise
    # (Return 1 or -1 respectively)
    def cycle_handedness(self, cycle):
        cycle_vertices = [self.vertices[j] for j in cycle]
        edges = edge_sequence(cycle_vertices)
        handed_sum = 0.0
        for v1, v2 in edges:
            handed_sum += ((v2[0] - v1[0]) / (v2[1] + v1[1])).item()
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
