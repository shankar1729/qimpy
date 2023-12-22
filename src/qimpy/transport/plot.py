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
        "checkpoints", help="Filename pattern (wildcard) for checkpoints", type=str
    )
    parser.add_argument(
        "output",
        help="Output filename pattern with integer format string (eg. {:04d})",
        type=str,
    )
    parser.add_argument(
        "--streamlines", help="Whether to draw streamlines", type=bool, default=False
    )
    parser.add_argument(
        "--transparency", help="Transparency in streamlines", type=bool, default=False
    )
    args = parser.parse_args()

    # Distribute tasks over MPI:
    file_list = rc.comm.bcast(sorted(glob.glob(args.checkpoints)))
    mine = slice(rc.i_proc, None, rc.n_procs)  # divide frames within each file
    geom = PlotGeometry(file_list[0])
    orig_log_level = log.getEffectiveLevel()
    log.setLevel(logging.INFO)  # Capture output from all processes
    for checkpoint_file in file_list:
        # Load data from checkpoint:
        rho_list = []
        vx_list = []
        vy_list = []
        with h5py.File(checkpoint_file, "r") as cp:
            cp_geom = cp["/geometry"]
            n_quads = cp_geom["quads"].shape[0]
            i_step_list = np.array(cp_geom.attrs["i_step"])[mine]
            t_list = np.array(cp_geom.attrs["i_step"])[mine]
            for i_quad in range(n_quads):
                cp_quad = cp_geom[f"quad{i_quad}"]
                rho_list.append(np.array(cp_quad["rho"][mine]))
                if args.streamlines:
                    v = np.array(cp_quad["v"][mine])
                    vx_list.append(v[..., 0])
                    vy_list.append(v[..., 1])

        # Plot each frame:
        for i_frame_mine, (i_step, t) in enumerate(zip(i_step_list, t_list)):
            plot_file = args.output.format(i_step)
            # Density:
            plt.clf()
            plt.title(f"$t$ = {t:.4g}")
            rho = geom.interpolate(rho_list, i_frame_mine)
            rho_max_abs = np.max(np.abs(rho))
            img = plt.imshow(
                rho,
                extent=geom.extents,
                origin="lower",
                cmap="bwr",
                interpolation="bilinear",
                vmin=-rho_max_abs,
                vmax=+rho_max_abs,
            )

            # Optional streamlines:
            if args.streamlines:
                vx = geom.interpolate(vx_list, i_frame_mine)
                vy = geom.interpolate(vy_list, i_frame_mine)
                stream_kwargs = dict(color="k", linewidth=1, arrowsize=0.7)
                if args.transparency:
                    v_mag = np.hypot(vx, vy).filled(0.0)
                    v_rel = np.sqrt(v_mag / v_mag.max())  # partially suppress low v
                    colors = np.zeros((256, 4))
                    colors[:, -1] = np.linspace(v_rel.min(), 1.0, len(colors))
                    alpha_cmap = ListedColormap(colors)
                    stream_kwargs.update(dict(color=v_rel, cmap=alpha_cmap))
                plt.streamplot(geom.x_grid, geom.y_grid, vx, vy, **stream_kwargs)

            # Draw domain boundaries:
            for spline in geom.exterior_splines:
                points = plot_spline(plt.gca(), spline)

                # Mark contacts if any:
                for i_contact, contact in enumerate(geom.contacts):
                    center = contact[:2]
                    radius = contact[2]
                    distances = np.linalg.norm(points - center, axis=1)
                    selection = np.where(distances <= radius)[0]
                    if len(selection):
                        contact_points = points[selection]
                        plt.plot(contact_points[:, 0], contact_points[:, 1], "w")
                        i_mid = len(selection) // 2
                        dq = np.diff(contact_points[i_mid : (i_mid + 2)], axis=0)[0]
                        angle = np.rad2deg(np.arctan2(dq[1], dq[0]))
                        plt.text(
                            *contact_points[i_mid],
                            geom.contact_names[i_contact],
                            rotation=angle,
                            ha="center",
                            va="top",
                            rotation_mode="anchor",
                        )

            ax = plt.gca()
            ax.set_aspect("equal")
            ax.margins(0.1)
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
            self.contact_names = str(cp["/geometry"].attrs["contact_names"]).split(",")
            self.contacts = np.array(cp["/geometry/contacts"])
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
    def interpolate(self, values_list: list[np.ndarray], i_frame: int) -> np.ndarray:
        """Interpolate from geometry to bounding-box grid."""
        values_all = np.concatenate(
            [values[i_frame].flatten() for values in values_list]
        )
        interpolator = mtri.LinearTriInterpolator(self.triangulation, values_all)
        return interpolator(self.x_grid, self.y_grid)


if __name__ == "__main__":
    main()
