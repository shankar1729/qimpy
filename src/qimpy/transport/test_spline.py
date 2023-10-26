from __future__ import annotations
import argparse
from collections import defaultdict

import torch
import numpy as np
from xml.dom import minidom
from svg.path import parse_path, CubicBezier, Line, Close

from qimpy import rc


class AdjacencyMatrix:
    def __init__(self, splines, epsilon=0.005):
        # Vertex list
        # (in complex form for ease of distance calculation)
        self.vertices = []
        self.epsilon = epsilon
        # Identify all unique vertices (vertex welding)
        for spline in splines:
            # Add starting and ending vertex for each spline
            self.add_vertex(complex(spline[0][0] + spline[0][1] * 1j))
            self.add_vertex(complex(spline[-1][0] + spline[-1][1] * 1j))
        self.N = len(self.vertices)
        self.data = torch.zeros((self.N, self.N))
        for spline in splines:
            # Add starting and ending vertex for each spline
            i = self.match_vertex(complex(spline[0][0] + spline[0][1] * 1j))
            j = self.match_vertex(complex(spline[-1][0] + spline[-1][1] * 1j))
            self.data[i, j] = 1
            self.data[j, i] = 1

    def add_vertex(self, new_vertex):
        if self.match_vertex(new_vertex) is None:
            self.vertices.append(new_vertex)

    def match_vertex(self, search_vertex):
        # Uniqueness check
        for i, vertex in enumerate(self.vertices):
            if np.abs(vertex - search_vertex) < self.epsilon:
                return i
        return None


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
    adjacency_matrix = AdjacencyMatrix(splines)
    print(adjacency_matrix.data)
    plt.figure()
    ax = plt.gca()
    ax.set_aspect("equal")
    for spline in splines:
        plot_spline(ax, spline)
    plt.show()


if __name__ == "__main__":
    main()
