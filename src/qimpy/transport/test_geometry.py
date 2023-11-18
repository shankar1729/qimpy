import argparse

from . import Geometry


def test_geometry(input_svg):
    geometry = Geometry(svg_file=input_svg)
    for i, patch in enumerate(geometry.patches):
        print(f"Patch {i}")
        for edge in patch.edges:
            print(edge)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_svg", help="Input patch (SVG file)", type=str)
    args = parser.parse_args()
    test_geometry(args.input_svg)
