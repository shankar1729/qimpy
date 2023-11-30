import argparse

import torch

from . import Geometry


def test_geometry(input_svg):
    v_F = 200.0
    N_theta = 1
    N = (64, 64)
    diag = True

    geometry = Geometry(svg_file=input_svg, v_F=v_F, N=N, N_theta=N_theta, diag=diag)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_svg", help="Input patch (SVG file)", type=str)
    args = parser.parse_args()
    test_geometry(args.input_svg)
