from __future__ import annotations
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from qimpy import log, rc
from qimpy.io import log_config
from . import parse_svg, plot_spline, BicubicPatch
from ._subdivide import select_division, subdivide


def main():
    log_config()
    rc.init()
    assert rc.n_procs == 1  # MPI does not make sense for this

    parser = argparse.ArgumentParser()
    parser.add_argument("input_svg", help="Input SVG filename", type=str)
    parser.add_argument("n_processes", help="# processes to test division", type=int)
    args = parser.parse_args()

    grid_spacing = 1.0
    quad_set = parse_svg(args.input_svg, grid_spacing, contact_names=[])

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
        plot_spline(ax, quad_set.vertices[edge], show_handles=True)

    # Test subdivision:
    grid_size_max = select_division(quad_set, args.n_processes)
    sub_quad_set = subdivide(quad_set, grid_size_max)
    patches = [
        BicubicPatch(
            boundary=torch.from_numpy(quad_set.get_boundary(i_quad)).to(rc.device)
        )
        for i_quad in range(len(quad_set.quads))
    ]

    # Show subdivided quads:
    plt.figure()
    ax = plt.gca()
    ax.set_aspect("equal")
    midpoints = []  # collect midpoints for indicating adjacency below
    for i_quad, grid_start, grid_stop in zip(
        sub_quad_set.quad_index, sub_quad_set.grid_start, sub_quad_set.grid_stop
    ):
        patch = patches[i_quad]
        N0, N1 = quad_set.grid_size[i_quad]
        t0 = torch.arange(grid_start[0] + 0.5, grid_stop[0], device=rc.device) / N0
        t1 = torch.arange(grid_start[1] + 0.5, grid_stop[1], device=rc.device) / N1
        for i_edge, Qfrac in enumerate(
            (
                torch.stack((t0, t1[:1].expand(len(t0))), dim=1),
                torch.stack((t0[-1:].expand(len(t1)), t1), dim=1),
                torch.stack((t0, t1[-1:].expand(len(t0))), dim=1),  # flipped direction
                torch.stack((t0[:1].expand(len(t1)), t1), dim=1),  # flipped direction
            )
        ):
            coords = patch(Qfrac).to(rc.cpu).numpy()
            plt.plot(*coords.T, lw=1, color="k")
            midpoints.append(coords[len(coords) // 2])

    # Annotate adjacency
    midpoints = np.stack(midpoints)  # (n_sub_quads * 4) x 2: flattened quad-edge
    adjacency = (sub_quad_set.adjacency @ np.array((4, 1))).flatten()  # in same index
    i_quad_edges = np.where(adjacency >= 0)[0]
    j_quad_edges = adjacency[i_quad_edges]
    if len(i_quad_edges):
        i_segments = np.stack((i_quad_edges, j_quad_edges), axis=1)
        segments = midpoints[i_segments].swapaxes(-2, -1)  # Cartesian axis to center
        for segment in segments:
            is_long = np.linalg.norm(np.diff(segment, axis=1)) > 2 * grid_spacing
            plt.plot(*segment, "r", ls=("dotted" if is_long else "solid"))

    # Annotate apertures
    for center_x, center_y, radius in quad_set.apertures:
        ax.add_patch(plt.Circle((center_x, center_y), radius, color="gray", alpha=0.5))

    rc.report_end()
    plt.show()


if __name__ == "__main__":
    main()
