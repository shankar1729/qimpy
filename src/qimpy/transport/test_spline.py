from __future__ import annotations
import argparse

import torch
import numpy as np
from xml.dom import minidom
from svg.path import parse_path, CubicBezier, Line, Close

from qimpy import rc


def get_splines(svg_file: str) -> list[torch.Tensor]:
    doc = minidom.parse(svg_file)
    svg_elements = []
    svg_paths = doc.getElementsByTagName("path")

    # Concatenate segments from all paths in SVG file
    for path in svg_paths:
        svg_elements.extend(parse_path(path.getAttribute("d")))

    def segment_to_tensor(segment):
        if isinstance(segment, CubicBezier):
            return torch.view_as_real(
                torch.tensor(
                    [segment.start, segment.control1, segment.control2, segment.end],
                    device=rc.device,
                )
            )
        # Both Line and Close can produce linear segments
        elif isinstance(segment, (Line, Close)):
            # Generate a spline from a linear segment
            disp_third = (segment.end - segment.start) / 3.0
            return torch.view_as_real(
                torch.tensor(
                    [
                        segment.start,
                        segment.start + disp_third,
                        segment.start + 2 * disp_third,
                        segment.end,
                    ],
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
    plt.figure()
    ax = plt.gca()
    ax.set_aspect("equal")
    for spline in splines:
        plot_spline(ax, spline)
    plt.show()


if __name__ == "__main__":
    main()
