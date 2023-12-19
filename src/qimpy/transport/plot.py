from __future__ import annotations
import argparse
import glob
import logging

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import ListedColormap
import numpy as np
import h5py

from qimpy import rc, log
from qimpy.profiler import stopwatch, StopWatch
from qimpy.io import log_config
from .geometry import BOUNDARY_SLICES, plot_spline


def main() -> None:
    log_config()
    rc.init()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoints", help="Filename pattern for checkpoints", type=str
    )
    parser.add_argument(
        "--streamlines", help="Whether to draw streamlines", type=bool, default=False
    )
    parser.add_argument(
        "--transparency", help="Transparency in streamlines", type=bool, default=False
    )
    args = parser.parse_args()

    # Distirbute tasks over MPI:
    file_list = rc.comm.bcast(sorted(glob.glob(args.checkpoints)))
    geom = PlotGeometry(file_list[0])
    orig_log_level = log.getEffectiveLevel()
    log.setLevel(logging.INFO)  # Capture output from all processes
    for checkpoint_file in file_list[rc.i_proc :: rc.n_procs]:
        plot_file = checkpoint_file.replace(".h5", ".png")

        # Load data from checkpoint:
        rho_list = []
        vx_list = []
        vy_list = []
        with h5py.File(checkpoint_file, "r") as cp:
            n_quads = cp["/geometry/quads"].shape[0]
            for i_quad in range(n_quads):
                prefix = f"/geometry/quad{i_quad}"
                rho_list.append(np.array(cp[f"{prefix}/rho"]))
                if args.streamlines:
                    v = np.array(cp[f"{prefix}/v"])
                    vx_list.append(v[..., 0])
                    vy_list.append(v[..., 1])

        # Density:
        plt.clf()
        rho_max_abs = max(np.max(np.abs(rho)) for rho in rho_list)
        img = plt.imshow(
            geom.interpolate(rho_list),
            extent=geom.extents,
            origin="lower",
            cmap="bwr",
            interpolation="bilinear",
            vmin=-rho_max_abs,
            vmax=+rho_max_abs,
        )

        # Optional streamlines:
        if args.streamlines:
            vx = geom.interpolate(vx_list)
            vy = geom.interpolate(vy_list)
            stream_kwargs = dict(color="k", linewidth=1, arrowsize=0.7)
            if args.transparency:
                v_mag = np.hypot(vx, vy).filled(0.0)
                v_rel = v_mag / v_mag.max()
                colors = np.zeros((256, 4))
                colors[:, -1] = np.linspace(v_rel.min(), 1.0, len(colors))
                alpha_cmap = ListedColormap(colors)
                stream_kwargs.update(dict(color=v_rel, cmap=alpha_cmap))
            plt.streamplot(geom.x_grid, geom.y_grid, vx, vy, **stream_kwargs)

        # Draw domain boundaries:
        for spline in geom.exterior_splines:
            plot_spline(plt.gca(), spline)

        plt.gca().set_aspect("equal")
        rho_max_str = r"|\rho|_{\mathrm{max}}"
        cbar = plt.colorbar(
            img,
            ticks=[-rho_max_abs, 0, +rho_max_abs],
            label=f"Density (${rho_max_str}$ = {rho_max_abs:6.2e})",
        )
        cbar.ax.set_yticklabels([rf"$-{rho_max_str}$", "0", rf"$+{rho_max_str}$"])
        spines = plt.gca().spines
        spines["top"].set_visible(False)
        spines["right"].set_visible(False)
        plt.savefig(plot_file, bbox_inches="tight", dpi=200)
        log.info(f"Saved {plot_file}")

    log.setLevel(orig_log_level)  # Switch log back to single process
    rc.comm.Barrier()
    rc.report_end()
    StopWatch.print_stats()


class PlotGeometry:
    """Load geometry and cache quantities to accelerate plotting from checkpoint."""

    @stopwatch(name="PlotGeometry.init")
    def __init__(self, checkpoint_file: str):
        # Load geometry:
        q_list = []
        triangles = []
        edge_indices = []
        with h5py.File(checkpoint_file, "r") as cp:
            grid_spacing = float(cp["/geometry"].attrs["grid_spacing"])
            vertices = np.array(cp["/geometry/vertices"])
            quads = np.array(cp["/geometry/quads"])
            adjacency = np.array(cp["/geometry/adjacency"])
            displacement_magnitudes = np.linalg.norm(
                np.array(cp["/geometry/displacements"]), axis=-1
            )
            n_quads = len(quads)
            n_points_prev = 0
            for i_quad in range(n_quads):
                prefix = f"/geometry/quad{i_quad}"
                q_list.append(np.array(cp[f"{prefix}/q"]))
                grid_size = q_list[-1].shape[:-1]
                n_points = np.prod(grid_size)
                indices = n_points_prev + np.arange(n_points).reshape(grid_size)
                triangles.append(
                    np.stack(
                        (
                            indices[:-1, :-1],
                            indices[1:, :-1],
                            indices[1:, 1:],
                            indices[:-1, :-1],
                            indices[1:, 1:],
                            indices[:-1, 1:],
                        ),
                        axis=-1,
                    ).reshape(-1, 3)
                )
                n_points_prev += n_points

                # Store edge indices for triangulating between patches:
                edge_indices.append([indices[boundary] for boundary in BOUNDARY_SLICES])

        # Add triangles between adjacent segments and collect exterior splines:
        tol = 1e-3
        self.exterior_splines = []
        for i_quad, adjacency_quad in enumerate(adjacency):
            for i_edge, (j_quad, j_edge) in enumerate(adjacency_quad):
                is_exterior = (j_quad < 0) or (
                    displacement_magnitudes[i_quad, i_edge] > tol
                )
                if is_exterior:
                    self.exterior_splines.append(vertices[quads[i_quad, i_edge]])
                else:
                    indices_i = edge_indices[i_quad][i_edge]
                    indices_j = edge_indices[j_quad][j_edge][::-1]
                    triangles.append(
                        np.stack(
                            (indices_i[:-1], indices_j[:-1], indices_j[1:]), axis=-1
                        ).reshape(-1, 3)
                    )

        # Comnstruct triangulation:
        triangles = np.concatenate(triangles, axis=0)
        q_all = np.concatenate([q.reshape(-1, 2) for q in q_list])
        self.triangulation = mtri.Triangulation(*q_all.T, triangles)

        # Construct target grid for interpolation
        x_min, y_min = q_all.min(axis=0)
        x_max, y_max = q_all.max(axis=0)
        self.extents = (x_min, x_max, y_min, y_max)
        Nx = 1 + int(np.round((x_max - x_min) / grid_spacing))
        Ny = 1 + int(np.round((y_max - y_min) / grid_spacing))
        x_grid_1d = np.linspace(x_min, x_max, Nx)
        y_grid_1d = np.linspace(y_min, y_max, Ny)
        self.x_grid, self.y_grid = np.meshgrid(x_grid_1d, y_grid_1d)

    @stopwatch
    def interpolate(self, values_list: list[np.ndarray]) -> np.ndarray:
        """Interpolate from geometry to bounding-box grid."""
        values_all = np.concatenate([values.flatten() for values in values_list])
        interpolator = mtri.LinearTriInterpolator(self.triangulation, values_all)
        return interpolator(self.x_grid, self.y_grid)


if __name__ == "__main__":
    main()
