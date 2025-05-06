from __future__ import annotations
from typing import Optional, Any, Union
import argparse
import glob
import logging

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import ListedColormap, SymLogNorm
from matplotlib.patches import FancyArrowPatch
import numpy as np

from qimpy import rc, log, io
from qimpy.profiler import stopwatch, StopWatch
from qimpy.io import log_config, Checkpoint, CheckpointPath
from .geometry import BOUNDARY_SLICES, evaluate_spline, within_circles_np


def main() -> None:
    log_config()
    rc.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="YAML input file", type=str)
    args = parser.parse_args()
    input_dict = io.dict.key_cleanup(io.yaml.load(args.input_file))
    run(**input_dict)

    rc.report_end()
    StopWatch.print_stats()


def run(
    *,
    checkpoints: str,
    output: str,
    density: Optional[dict] = None,
    streamlines: Optional[dict] = None,
    density_prefix: str = "",
    current_prefix: str = "",
    apertures: Optional[dict] = None,
    dpi: int = 200,
) -> None:
    if density is None:
        density = {}
    # Distribute tasks over MPI:
    file_list = rc.comm.bcast(sorted(glob.glob(checkpoints)))
    mine = slice(rc.i_proc, None, rc.n_procs)  # divide frames within each file
    pg = PlotGeometry(
        file_list[0], **density, plot_apertures=(apertures if apertures else {})
    )
    orig_log_level = log.getEffectiveLevel()
    log.setLevel(logging.INFO)  # Capture output from all processes
    streams = None
    frame_data_mine = []
    for checkpoint_file in file_list:
        # Load data from checkpoint:
        rho_list = []
        vx_list = []
        vy_list = []
        with Checkpoint(checkpoint_file) as cp:
            cp_geom = cp["/geometry"]
            n_quads = cp_geom["quads"].shape[0]
            i_step_list = np.array(cp_geom["i_step"])[mine]
            t_list = np.array(cp_geom["t"])[mine]
            for i_quad in range(n_quads):
                cp_quad = cp_geom[f"quad{i_quad}"]
                observables = np.array(cp_quad["observables"][mine])
                rho_list.append(observables[..., 0])  # TODO: handle ab_initio cases
                vx_list.append(observables[..., 1])
                vy_list.append(observables[..., 2])

        # Plot each frame:
        for i_frame_mine, (i_step, t) in enumerate(zip(i_step_list, t_list)):
            plot_file = output.format(i_step)
            # Density:
            pg.title.set_text(f"$t$ = {t:.4g}")
            rho, rho_flat = pg.interpolate(rho_list, i_frame_mine)
            rho_max_abs = np.max(np.abs(rho))
            pg.img.set_data(rho / rho_max_abs)
            rho_max_str = r"$\times|\rho|_{\mathrm{max}}$"
            pg.cbar.set_label(f"Density ({rho_max_str} = {rho_max_abs:6.2e})")

            # Optional streamlines:
            vx, vx_flat = pg.interpolate(vx_list, i_frame_mine)
            vy, vy_flat = pg.interpolate(vy_list, i_frame_mine)
            if streamlines is not None:
                streams = plot_streamlines(streams, pg, vx, vy, **streamlines)
            plt.savefig(plot_file, bbox_inches="tight", dpi=dpi)
            log.info(f"Saved {plot_file}")

            # Collect per-frame density and/or current:
            if density_prefix or current_prefix:
                frame_data = [t]  # time, followed by density, current for each location
                for index, normal in pg.current_props.values():
                    g_sel = pg.g_flat[index]  # integration weight for selected points
                    v_sel = np.stack((vx_flat[index], vy_flat[index]), axis=1)
                    density_avg = (rho_flat[index] @ g_sel) / g_sel.sum()
                    current_tot = (v_sel * normal).sum()
                    frame_data.extend((density_avg, current_tot))
                frame_data_mine.append(frame_data)

    log.setLevel(orig_log_level)  # Switch log back to single process
    rc.comm.Barrier()

    # Collect density/current data on head for plotting:
    if density_prefix or current_prefix:
        if not rc.is_head:
            # Send data to head:
            data_mine = np.array(frame_data_mine)
            rc.comm.send(len(data_mine), dest=0, tag=0)
            if len(data_mine):
                rc.comm.Send(data_mine, dest=0, tag=1)
        else:
            # Collect data from all:
            data_list = [np.array(frame_data_mine)]
            n_columns = data_list[0].shape[1]
            for i_proc in range(1, rc.n_procs):
                len_data_other = rc.comm.recv(None, source=i_proc, tag=0)
                if len_data_other:
                    buf = np.zeros((len_data_other, n_columns))
                    rc.comm.Recv(buf, source=i_proc, tag=1)
                    data_list.append(buf)
            data_all = np.concatenate(data_list, axis=0)

            # Sort by time and separate properties:
            data_all = data_all[data_all[:, 0].argsort()]
            t = data_all[:, 0]
            densities = data_all[:, 1::2]
            currents = data_all[:, 2::2]
            labels = pg.current_props.keys()

            for quantity_prefix, quantities, ylabel in (
                (density_prefix, densities, r"Density, $\bar\rho$"),
                (current_prefix, currents, r"Current, $I$"),
            ):
                if quantity_prefix:
                    plt.figure()
                    for quantity, label in zip(quantities.T, labels):
                        plt.plot(t, quantity, label=label)
                    plt.axhline(0, color="k", ls="dotted", lw=1)
                    plt.xlim(t[0], t[-1])
                    plt.xlabel(r"Time, $t$")
                    plt.ylabel(ylabel)
                    plt.legend()
                    plt.savefig(f"{quantity_prefix}.pdf", bbox_inches="tight")
                    np.savetxt(
                        f"{quantity_prefix}.dat",
                        np.hstack((t[:, None], quantities)),
                        header="t " + " ".join(labels),
                    )


def plot_streamlines(
    streams: Any,
    pg: PlotGeometry,
    vx: np.ndarray,
    vy: np.ndarray,
    *,
    transparency: bool = False,
    **stream_kwargs,
) -> Any:
    if streams is not None:
        # Remove previous streamlines:
        streams.lines.remove()
        for art in plt.gca().get_children():
            if isinstance(art, FancyArrowPatch):
                art.remove()
    kwargs = dict(color="k", linewidth=1, arrowsize=0.7)  # defaults
    kwargs.update(**stream_kwargs)
    if transparency:
        v_mag = np.hypot(vx, vy).filled(0.0)  # type: ignore
        v_rel = np.sqrt(v_mag / v_mag.max())  # partially suppress low v
        colors = np.zeros((256, 4))
        colors[:, -1] = np.linspace(v_rel.min(), 1.0, len(colors))
        alpha_cmap = ListedColormap(colors)
        kwargs.update(dict(color=v_rel, cmap=alpha_cmap))
    return plt.streamplot(pg.x_grid, pg.y_grid, vx, vy, **kwargs)


class PlotGeometry:
    """Load geometry and cache quantities to accelerate plotting from checkpoint."""

    @stopwatch(name="PlotGeometry.init")
    def __init__(
        self,
        checkpoint_file: str,
        *,
        cmap: str = "bwr",
        interpolation: str = "bilinear",
        linthresh: float = 0.1,
        plot_apertures: dict,
    ):
        # Load geometry:
        q_list = []
        triangles = []
        edge_indices = []
        with Checkpoint(checkpoint_file) as cp:
            cp_geom = CheckpointPath(cp, "/geometry")
            grid_spacing = float(cp_geom.attrs["grid_spacing"])
            contact_names = split_names(cp_geom.read_str("contact_names"))
            aperture_names = split_names(cp_geom.read_str("aperture_names"))
            contacts = np.array(cp["/geometry/contacts"])
            apertures = np.array(cp["/geometry/apertures"])
            vertices = np.array(cp["/geometry/vertices"])
            quads = np.array(cp["/geometry/quads"])
            adjacency = np.array(cp["/geometry/adjacency"])
            displacement_magnitudes = np.linalg.norm(
                np.array(cp["/geometry/displacements"]), axis=-1
            )
            n_quads = len(quads)
            n_points_prev = 0
            g_list_flat = []
            for i_quad in range(n_quads):
                prefix = f"/geometry/quad{i_quad}"
                g_list_flat.append(np.array(cp[f"{prefix}/g"]).flatten())
                q_list.append(np.array(cp[f"{prefix}/q"]))
                grid_size = q_list[-1].shape[:-1]
                n_points = int(np.prod(grid_size))
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
                edge_indices.append(
                    [indices[boundary] for boundary in BOUNDARY_SLICES]  # type: ignore
                )
            self.g_flat = np.concatenate(g_list_flat)

        # Add triangles between adjacent segments and collect exterior splines:
        tol = 1e-3
        current_props: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # index, normals
        for i_quad, adjacency_quad in enumerate(adjacency):
            for i_edge, (j_quad, j_edge) in enumerate(adjacency_quad):
                is_periodic_bc = displacement_magnitudes[i_quad, i_edge] > tol
                linestyle = "dashed" if is_periodic_bc else "solid"
                is_exterior = (j_quad < 0) or is_periodic_bc
                verts = vertices[quads[i_quad, i_edge]]

                # Plot exterior/interior boundaries with contacts/apertures
                indices_i = edge_indices[i_quad][i_edge]
                Npoints = len(indices_i)
                t = np.arange(2 * Npoints + 1)[:, None] / (2 * Npoints)
                points_all = evaluate_spline(verts, t)  # includes vertices & midpoints
                points = points_all[::2]  # vertices along spline only
                points_mid = points_all[1::2]  # midpoints of spline segments only
                circles = contacts if is_exterior else apertures
                circle_names = contact_names if is_exterior else aperture_names
                within_each = within_circles_np(circles, points_mid)
                within_any = np.any(within_each, axis=0)
                triangle_selection: Union[slice, np.ndarray] = slice(None)
                if np.count_nonzero(within_any):
                    # Draw partial boundary due to contacts or apertures:
                    sel_blocked = np.where(np.logical_not(within_any))[0]
                    if len(sel_blocked):  # otherwise no boundary left to draw
                        i_breaks = np.where(sel_blocked[:-1] + 1 != sel_blocked[1:])[0]
                        i_starts = np.concatenate(
                            (sel_blocked[:1], sel_blocked[i_breaks + 1])
                        )
                        i_stops = np.concatenate(
                            (sel_blocked[i_breaks] + 2, sel_blocked[-1:] + 2)
                        )
                        for i_start, i_stop in zip(i_starts, i_stops):
                            plt.plot(*points[i_start:i_stop].T, "k", ls=linestyle)

                    # For interior case, only keep triangles within apertures:
                    triangle_selection = np.where(
                        np.logical_and(within_any[:-1], within_any[1:])
                    )[0]

                    # Annotate contacts:
                    tangents_mid = np.diff(points, axis=0)
                    if is_exterior:
                        text_args = dict(ha="center", va="top", rotation_mode="anchor")
                        for within_contact, contact_name in zip(
                            within_each, contact_names
                        ):
                            selection = np.where(within_contact)[0]
                            if len(selection):
                                i_mid = selection[len(selection) // 2]
                                position = points_mid[i_mid]
                                dq = tangents_mid[i_mid]
                                angle = np.rad2deg(np.arctan2(dq[1], dq[0]))
                                plt.text(
                                    *position, contact_name, rotation=angle, **text_args
                                )

                    # Collect quantities to calculate current:
                    normals_mid = np.stack(
                        (tangents_mid[..., 1], -tangents_mid[..., 0]), axis=-1
                    )  # outward normals with length = boundary segment length
                    for i_circle, (within_circle, circle_name) in enumerate(
                        zip(within_each, circle_names)
                    ):
                        sel = np.where(within_circle)[0]
                        if len(sel):
                            index_sel = indices_i[sel]
                            normals_sel = normals_mid[sel]
                            if circle_name in current_props:
                                # Combine calculations from both sides of aperture:
                                index_old, normals_old = current_props[circle_name]
                                assert len(sel) == len(index_old)
                                index_net = np.concatenate((index_old, index_sel))
                                normals_net = (
                                    np.concatenate((-normals_old, normals_sel), axis=0)
                                    * 0.5
                                )  # still count to same total length
                                current_props[circle_name] = (index_net, normals_net)
                            else:
                                current_props[circle_name] = (index_sel, normals_sel)

                elif is_exterior:
                    # Draw uninterrupted spline
                    plt.plot(*points.T, "k", ls=linestyle)

                # Add connecting triangles for interior edges (within apertures):
                if not is_exterior:
                    indices_j = edge_indices[j_quad][j_edge][::-1]
                    new_triangles = np.stack(
                        (indices_i[:-1], indices_j[:-1], indices_j[1:]), axis=-1
                    ).reshape(-1, 3)
                    triangles.append(new_triangles[triangle_selection])

        # Construct triangulation:
        triangles = np.concatenate(triangles, axis=0)
        q_all = np.concatenate([q.reshape(-1, 2) for q in q_list])
        self.triangulation = mtri.Triangulation(*q_all.T, triangles)

        # Select current calculators (all contacts and apertures named in input):
        self.current_props = {
            contact_name: current_props[contact_name] for contact_name in contact_names
        }
        for aperture_label, aperture_props in plot_apertures.items():
            aperture_name: str = aperture_props["name"]
            aperture_outward = np.array(aperture_props["outward"])
            assert aperture_outward.shape == (2,)
            aperture_index, aperture_normals = current_props[aperture_name]
            if aperture_normals.mean(axis=0) @ aperture_outward < 0.0:
                aperture_normals *= -1  # flip to match specified "outward" direction
            self.current_props[aperture_label] = (aperture_index, aperture_normals)

        # Construct target grid for interpolation
        x_min, y_min = q_all.min(axis=0)
        x_max, y_max = q_all.max(axis=0)
        Nx = 1 + 2 * int(np.round((x_max - x_min) / grid_spacing))
        Ny = 1 + 2 * int(np.round((y_max - y_min) / grid_spacing))
        x_grid_1d = np.linspace(x_min, x_max, Nx)
        y_grid_1d = np.linspace(y_min, y_max, Ny)
        self.x_grid, self.y_grid = np.meshgrid(x_grid_1d, y_grid_1d)

        # Create all plot elements upfront (later only update data):
        self.title = plt.title("$t$ =")
        test_rho = (self.x_grid / np.max(self.x_grid)) * 2 - 1
        self.img = plt.imshow(
            test_rho,
            extent=(x_min, x_max, y_min, y_max),
            origin="lower",
            cmap=cmap,
            interpolation=interpolation,
            norm=SymLogNorm(linthresh=linthresh, vmin=-1, vmax=+1),
        )
        ax = plt.gca()
        ax.set_aspect("equal")
        ax.margins(0.1)
        cbar_orientation = "horizontal" if (Nx > 1.5 * Ny) else "vertical"
        self.cbar = plt.colorbar(self.img, label="Temp", orientation=cbar_orientation)
        spines = ax.spines
        spines["top"].set_visible(False)
        spines["right"].set_visible(False)

    @stopwatch
    def interpolate(
        self, values_list: list[np.ndarray], i_frame: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate from geometry to bounding-box grid at given frame.
        Also return the flattened data on original geometry for probe calculations."""
        values_all = np.concatenate(
            [values[i_frame].flatten() for values in values_list]
        )
        interpolator = mtri.LinearTriInterpolator(self.triangulation, values_all)
        return interpolator(self.x_grid, self.y_grid), values_all


def split_names(input: str) -> list[str]:
    return input.split(",") if input else []


if __name__ == "__main__":
    main()
