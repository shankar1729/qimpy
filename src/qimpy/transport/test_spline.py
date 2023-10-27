from __future__ import annotations
import argparse
from xml.dom import minidom

import torch
import numpy as np
from svg.path import parse_path, CubicBezier, Line, Close

from qimpy import rc


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


class Patches:
    def __init__(self, splines, epsilon=0.005):
        self.vertices, self.edges = weld_points(splines[:, (0, -1)], tol=epsilon)

        self.cycles = []
        self.find_cycles()

        self.patches = []
        for cycle in self.cycles:
            self.patches.append([splines[i] for i in cycle])

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
                    next_vertex = edge[1] if edge[0] == start_vertex else edge[0]
                    if next_vertex not in cycle:
                        cycle_search(cycle + [next_vertex], depth=depth + 1)
                    elif len(cycle) > 2 and next_vertex == cycle[0]:
                        self.add_cycle(cycle)

        # Search for cycles from each starting vertex
        for first_vertex in range(len(self.vertices)):
            cycle_search([first_vertex])


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


def main():
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("input_svg", help="Input patch (SVG file)", type=str)
    args = parser.parse_args()

    splines = get_splines(args.input_svg)

    # Find each patch and print its respective vertices
    patches = Patches(splines)

    print(f"Found {len(patches.patches)} patches:")
    for cycle in patches.cycles:
        print([patches.vertices[j] for j in cycle])

    plt.figure()
    ax = plt.gca()
    ax.set_aspect("equal")
    for spline in splines:
        plot_spline(ax, spline)
    plt.show()


if __name__ == "__main__":
    main()
