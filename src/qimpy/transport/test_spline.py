from __future__ import annotations
import argparse

import torch
import numpy as np
from xml.dom import minidom
from svg.path import parse_path, CubicBezier, Line, Close

from qimpy import rc


class Patches:
    def __init__(self, splines, epsilon=0.005, patch_sides=4):
        self.patch_sides = patch_sides
        # Vertex list
        # (in complex form for ease of distance calculation)
        self.vertices = []
        self.edges = []
        self.cycles = []
        self.patches = []
        self.epsilon = epsilon

        # First pass: Identify all unique vertices (vertex welding)
        for spline in splines:
            # Add starting and ending vertex for each spline
            self.add_vertex(complex(spline[0][0] + spline[0][1] * 1j))
            self.add_vertex(complex(spline[-1][0] + spline[-1][1] * 1j))
        self.N = len(self.vertices)
        for spline in splines:
            # Identify each spline to its respective vertices, building graph
            i = self.match_vertex(complex(spline[0][0] + spline[0][1] * 1j))
            j = self.match_vertex(complex(spline[-1][0] + spline[-1][1] * 1j))
            self.edges.append((i, j))
        self.find_cycles()
        for cycle in self.cycles:
            self.patches.append([splines[i] for i in cycle])

    def add_vertex(self, new_vertex):
        if self.match_vertex(new_vertex) is None:
            self.vertices.append(new_vertex)

    def match_vertex(self, search_vertex):
        # Uniqueness check
        for i, vertex in enumerate(self.vertices):
            if np.abs(vertex - search_vertex) < self.epsilon:
                return i
        return None

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
        def cycle_search(self, cycle, depth=1):
            # Don't bother looking for cycles that exceed a single patch
            # (limit recursion depth)
            if depth > self.patch_sides:
                return
            start_vertex = cycle[-1]
            for edge in self.edges:
                if start_vertex in edge:
                    next_vertex = edge[1] if edge[0] == start_vertex else edge[0]
                    if next_vertex not in cycle:
                        cycle_search(self, cycle + [next_vertex], depth=depth + 1)
                    elif len(cycle) > 2 and next_vertex == cycle[0]:
                        self.add_cycle(cycle)

        # Search for cycles from each starting vertex
        for start_vertex in list(range(self.N)):
            cycle_search(self, [start_vertex])


def get_splines(svg_file: str) -> list[torch.Tensor]:
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
    return [
        segment_to_tensor(segment)
        for segment in svg_elements
        if isinstance(segment, (Line, Close, CubicBezier))
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

    parser = argparse.ArgumentParser()
    parser.add_argument("input_svg", help="Input patch (SVG file)", type=str)
    args = parser.parse_args()

    splines = get_splines(args.input_svg)

    # Find each patch and print its respective vertices
    patches = Patches(splines)

    def complex_to_list(z):
        return [z.real, z.imag]

    [print([complex_to_list(patches.vertices[j]) for j in i]) for i in patches.cycles]

    plt.figure()
    ax = plt.gca()
    ax.set_aspect("equal")
    for spline in splines:
        plot_spline(ax, spline)
    plt.show()


if __name__ == "__main__":
    main()
