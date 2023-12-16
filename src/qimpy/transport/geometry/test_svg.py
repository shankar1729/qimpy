from __future__ import annotations
import argparse

import matplotlib.pyplot as plt

from qimpy import log, rc
from qimpy.io import log_config
from . import parse_svg, plot_spline
from ._subdivide import select_division


def main():
    log_config()
    rc.init()
    assert rc.n_procs == 1  # MPI does not make sense for this

    parser = argparse.ArgumentParser()
    parser.add_argument("input_svg", help="Input SVG filename", type=str)
    parser.add_argument("n_processes", help="# processes to test division", type=int)
    args = parser.parse_args()

    quad_set = parse_svg(args.input_svg, grid_spacing=1.0)

    log.info(f"Found {len(quad_set.quads)} quads:")
    for i_quad, (quad, adjacency, grid_size) in enumerate(
        zip(quad_set.quads, quad_set.adjacency, quad_set.grid_size)
    ):
        log.info(f"Quad {i_quad} sampled by {tuple(grid_size)} grid:")
        for i_edge, edge_index in enumerate(quad):
            i_verts = quad_set.edges[edge_index, [0, -1]]
            v_start, v_stop = quad_set.vertices[i_verts]
            neigh_patch, neigh_edge = adjacency[i_edge]
            if neigh_patch >= 0:
                neigh_str = f" (Neighbor: quad {neigh_patch} edge {neigh_edge})"
            else:
                neigh_str = ""
            log.info(f"  Edge {i_edge}: {v_start} -> {v_stop}{neigh_str}")

    plt.figure()
    ax = plt.gca()
    ax.set_aspect("equal")
    for edge in quad_set.edges:
        plot_spline(ax, quad_set.vertices[edge])

    # Test subdivision:
    select_division(quad_set, args.n_processes)

    rc.report_end()
    plt.show()


if __name__ == "__main__":
    main()
