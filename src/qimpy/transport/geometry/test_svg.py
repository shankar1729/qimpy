from __future__ import annotations
import argparse

from . import parse_svg, plot_spline


def main():
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("input_svg", help="Input patch (SVG file)", type=str)
    args = parser.parse_args()

    quad_set = parse_svg(args.input_svg, grid_spacing=1.0)

    print(f"Found {len(quad_set.quads)} quads:")
    for i_quad, (quad, adjacency, grid_size) in enumerate(
        zip(quad_set.quads, quad_set.adjacency, quad_set.grid_size)
    ):
        print(f"Quad {i_quad} sampled by {tuple(grid_size)} grid:")
        for i_edge, edge_index in enumerate(quad):
            i_verts = quad_set.edges[edge_index, [0, -1]]
            v_start, v_stop = quad_set.vertices[i_verts]
            neigh_patch, neigh_edge = adjacency[i_edge]
            if neigh_patch >= 0:
                neigh_str = f" (Neighbor: quad {neigh_patch} edge {neigh_edge})"
            else:
                neigh_str = ""
            print(f"  Edge {i_edge}: {v_start} -> {v_stop}{neigh_str}")

    plt.figure()
    ax = plt.gca()
    ax.set_aspect("equal")
    for edge in quad_set.edges:
        plot_spline(ax, quad_set.vertices[edge])
    plt.show()


if __name__ == "__main__":
    main()
