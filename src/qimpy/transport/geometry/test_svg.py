from __future__ import annotations
import argparse

import matplotlib.pyplot as plt

from qimpy import log, rc
from qimpy.io import log_config
from . import parse_svg, plot_spline
from ._subdivide import select_division, subdivide


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
        for i_edge, i_verts in enumerate(quad):
            v_start = quad_set.vertices[i_verts[0]]
            v_stop = quad_set.vertices[i_verts[-1]]
            neigh_patch, neigh_edge = adjacency[i_edge]
            if neigh_patch >= 0:
                neigh_str = f" (Neighbor: quad {neigh_patch} edge {neigh_edge})"
            else:
                neigh_str = ""
            log.info(f"  Edge {i_edge}: {v_start} -> {v_stop}{neigh_str}")

    plt.figure()
    ax = plt.gca()
    ax.set_aspect("equal")
    for edge in quad_set.quads.reshape(-1, 4):
        plot_spline(ax, quad_set.vertices[edge])

    # Test subdivision:
    grid_size_max = select_division(quad_set, args.n_processes)
    sub_quad_set = subdivide(quad_set, grid_size_max)
    log.info(sub_quad_set)

    rc.report_end()
    plt.show()


if __name__ == "__main__":
    main()
