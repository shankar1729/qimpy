from __future__ import annotations
import argparse

import torch
import numpy as np
from xml.dom import minidom
from svg.path import parse_path, CubicBezier

from qimpy import rc


def get_splines(svg_file: str) -> list[torch.Tensor]:
    doc = minidom.parse(svg_file)
    # We assume that there is only one path patch and that it is the first.
    # When we want to support more later, we can iterate through these elements.
    svg_path = doc.getElementsByTagName("path")[0].getAttribute("d")
    svg_xml = parse_path(svg_path)
    return [
        torch.view_as_real(
            torch.tensor(
                [segment.start, segment.control1, segment.control2, segment.end],
                device=rc.device,
            )
        )
        for segment in svg_xml
        if isinstance(segment, CubicBezier)
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