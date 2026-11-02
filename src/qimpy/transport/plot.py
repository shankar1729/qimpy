from __future__ import annotations
from typing import Optional
import argparse
import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.colors import SymLogNorm, Normalize
from matplotlib.collections import LineCollection

from qimpy import rc, log, io
from qimpy.profiler import StopWatch, stopwatch
from qimpy.io import log_config, Checkpoint, CheckpointPath, Unit
from qimpy.transport.geometry import Mesh


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


@stopwatch
def run(
    *,
    checkpoints: str,
    output: str,
    probes: str,
    density: Optional[dict] = None,
    streamlines: Optional[dict] = None,
    dpi: int = 200,
) -> None:
    """Plot cell-centered scalars as color map and optionally face-centered fluxes
    as streamlnies for all `checkpoints`. Additionally plot scalar and flux probes
    as a function of time, comparing defined boundary and interior regions."""

    # Construct geometry from first checkpoint:
    file_list = rc.comm.bcast(sorted(glob.glob(checkpoints)))
    with Checkpoint(file_list[0]) as checkpoint:
        mesh = Mesh(
            group=None, checkpoint_in=CheckpointPath(checkpoint, "/geometry/mesh")
        )
        cp_geometry = CheckpointPath(checkpoint, "/geometry")
        observable_names = cp_geometry.read_str_list("observable_names")
        flux_indices = cp_geometry.read_np("flux_indices")
        flux_names = [observable_names[index] for index in flux_indices]
    triangulation = Triangulation(*mesh.vertices.T, mesh.cells[:, :-1])
    cell_weights = mesh.get_cell_weights()

    if density is None:
        density = {}
    boundary = Boundary(mesh, params=density)
    cmap = density.get("cmap", "bwr")
    if (linthresh := float(density.get("linthresh", 1.0))) < 1.0:
        norm = SymLogNorm(linthresh=linthresh, vmin=-1.0, vmax=1.0, base=10)
    else:
        norm = Normalize(vmin=-1.0, vmax=1.0)
    observable_name = density.get("name", "n")
    i_observable = observable_names.index(observable_name)
    i_flux = 0
    if streamlines is not None:
        i_flux = flux_names.index(streamlines.get("name", "n"))

    # Prepare grid on bounding box (only used for streamlines)
    bbox_min = mesh.vertices.min(axis=0)
    bbox_max = mesh.vertices.max(axis=0)
    h_max = np.sqrt(0.1 * cell_weights.min())  # few grid points per triangle
    n_grid = np.maximum(500, np.ceil((bbox_max - bbox_min) / h_max).astype(int)) + 1
    grids_1d = [np.linspace(bbox_min[i], bbox_max[i], n_grid[i]) for i in range(2)]
    grid = np.stack(np.meshgrid(*grids_1d))
    grid_cells = triangulation.get_trifinder()(*grid)  # cell index per grid point

    # Distribute frames over MPI:
    t_probes = []
    probes_region = []
    probes_flux = []
    mine = slice(rc.i_proc, None, rc.n_procs)
    for filename in file_list:
        with Checkpoint(filename) as checkpoint:
            cp_geometry = CheckpointPath(checkpoint, "/geometry")
            i_step_list = cp_geometry.read_np("i_step")[mine]
            t_list = cp_geometry.read_np("t")[mine]
            observables = cp_geometry.read_np("observables")[mine, :, i_observable]
            if streamlines is not None:
                face_flux = cp_geometry.read_np("face_flux")[mine, :, i_flux]
            t_probes.append(cp_geometry.read_np("t_probe"))
            probes_region.append(cp_geometry.read_np("probes"))
            probes_flux.append(cp_geometry.read_np("probe_flux"))

        for i_frame, (i_step, t) in enumerate(zip(i_step_list, t_list)):
            fig, ax = plt.subplots(figsize=(6, 6), layout="compressed")
            boundary.plot(ax)

            # Plot observable scalar field as color map:
            values = observables[i_frame]
            deviations = values - np.nanmean(values)
            max_deviation = float(np.nanmax(np.abs(deviations)))
            if not np.isfinite(max_deviation) or max_deviation == 0.0:
                max_deviation = 1.0
            scaled_deviations = deviations / max_deviation
            color_plot = ax.tripcolor(
                triangulation,
                facecolors=scaled_deviations,
                cmap=cmap,
                norm=norm,
            )
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_xlim(bbox_min[0], bbox_max[0])
            ax.set_ylim(bbox_min[1], bbox_max[1])

            color_bar = fig.colorbar(
                color_plot, ax=ax, location="bottom", fraction=0.04, pad=0.07
            )
            color_bar.set_label(
                rf"${observable_name}-\langle {observable_name}\rangle$"
                rf" ($\times${max_deviation:.2e})"
                rf" at $t$ = {float(t) / Unit.MAP['ps']:.4g} ps"
            )

            if streamlines is not None:
                # Get flux per cell area on all faces per cell:
                cell_flux_density = np.zeros(mesh.faces.shape[:2])
                i_cell, i_face = mesh.face_map[:, 0].T  # interior and boundary
                cell_flux_density[i_cell, i_face] = face_flux[i_frame]
                interior = mesh.face_map[:, 1, 0] >= 0
                i_cell, i_face = mesh.face_map[interior, 1].T  # interior (opposite)
                cell_flux_density[i_cell, i_face] = -face_flux[i_frame, interior]
                cell_flux_density /= cell_weights[:, None]

                # Interpolate within each cell onto grid:
                grid_flux_density = np.zeros_like(grid)
                for i_face in range(3):
                    weight = 0.5 * cell_flux_density[grid_cells, i_face]
                    vertex = mesh.vertices[mesh.cells[grid_cells, i_face]]
                    grid_flux_density += weight * (grid - vertex.transpose(2, 0, 1))
                grid_flux_density[np.where(grid_cells[None] < 0)] = np.nan

                # Plot streamlines:
                ax.streamplot(
                    *grids_1d,
                    *grid_flux_density,
                    density=streamlines.get("density", 1.5),
                    linewidth=streamlines.get("linewidth", 0.9),
                    arrowsize=streamlines.get("arrowsize", 0.9),
                    color="k",
                )

            # Save:
            filename = output.format(i_step)
            fig.savefig(filename, bbox_inches="tight", dpi=dpi)
            plt.close(fig)
            log.info(f"Saved {filename}")

    # Plot probes:
    if probes and rc.is_head:
        t = np.concatenate(t_probes)
        probes_region = np.concatenate(probes_region, axis=0)
        probes_flux = np.concatenate(probes_flux, axis=0)
        order = t.argsort()
        t = t[order]
        probes_region = probes_region[order]

        # Scalars:
        for i_name, name in enumerate(observable_names):
            plt.figure()
            plt.xlabel("t")
            plt.ylabel(name)
            for i_region, region_name in enumerate(
                mesh.region_names + mesh.boundary_names
            ):
                plt.plot(t, probes_region[:, i_region, i_name], label=region_name)
            plt.legend()
            filename = probes.format(name)
            plt.savefig(filename, bbox_inches="tight")
            log.info(f"Saved {filename}")

        # Fluxes:
        for i_name, name in enumerate(flux_names):
            plt.figure()
            plt.xlabel("t")
            plt.ylabel(f"{name} flux")
            for i_boundary, boundary_name in enumerate(mesh.boundary_names):
                plt.plot(t, probes_flux[:, i_boundary, i_name], label=boundary_name)
            plt.legend()
            filename = probes.format(f"flux_{name}")
            plt.savefig(filename, bbox_inches="tight")
            log.info(f"Saved {filename}")
        plt.show()


class Boundary:
    def __init__(self, mesh: Mesh, params: dict) -> None:
        # Get all boundary edges and their region IDs:
        boundary_sel = mesh.face_map[:, 1, 0] == -1
        boundary_c, boundary_f = mesh.face_map[boundary_sel, 0].T
        boundary_region = mesh.face_map[boundary_sel, 1, 1]
        edges = mesh.faces[boundary_c, boundary_f]
        edge_vertices = mesh.vertices[edges]

        # Outline:
        lw = params.get("outline_lw", 2.0)
        segments = edge_vertices[boundary_region == -1]
        lines = [dict(segments=segments, colors="k", linewidths=lw, zorder=5)]

        # Contacts:
        lw = params.get("contact_lw", 5.0)
        color = params.get("contact_color", "orange")
        texts = []
        text_args = dict(ha="center", va="center", zorder=7, rotation_mode="anchor")
        text_args |= dict(size=params.get("contact_fontsize", 9), weight="bold", c="k")
        for i_region, name in enumerate(mesh.boundary_names):
            indices = np.where(boundary_region == i_region)
            if len(indices) == 0:
                continue
            segments = edge_vertices[indices]
            lines.append(dict(segments=segments, colors=color, linewidths=lw, zorder=6))

            # Labels:
            for component in split_continuous(edges[indices], mesh.vertices):
                # Locate and orient label to the central segment of each component:
                v1 = mesh.vertices[component[(len(component) - 1) // 2, 0]]
                v2 = mesh.vertices[component[len(component) // 2, 1]]
                x, y = 0.5 * (v1 + v2)
                tangent = v2 - v1
                angle = np.rad2deg(np.arctan2(tangent[1], tangent[0]))
                if abs(angle) > 90:
                    # Avoid upside down text
                    angle -= np.copysign(180, angle)
                    label = f"{name}\n\n"
                else:
                    label = f"\n\n{name}"
                texts.append(dict(x=x, y=y, s=label, rotation=angle, **text_args))

        # Store plot parameters for repeated use:
        self.lines = lines
        self.texts = texts

    def plot(self, ax: plt.Axes) -> None:
        for line in self.lines:
            ax.add_collection(LineCollection(**line, clip_on=False))
        for text in self.texts:
            ax.text(**text)


def split_continuous(
    edges: np.ndarray, vertices: np.ndarray, cos_cutoff: float = 0.9
) -> list[np.ndarray]:
    """Split index pairs `edges` into connected arcs (in order). Further split arcs
    if cos(angle) between adajcent segments falls below `cos_cutoff`."""
    vertices1, vertices2 = edges.T
    map12 = {v1: v2 for v1, v2 in zip(vertices1, vertices2)}
    v_starts = set(vertices1) - set(vertices2)
    components = []
    for v in v_starts:
        component = []
        tangent_prev = None
        while (v_next := map12.get(v, -1)) != -1:
            tangent = vertices[v_next] - vertices[v]
            tangent /= np.linalg.norm(tangent)
            if tangent_prev is not None and (tangent_prev @ tangent < cos_cutoff):
                # Break component due to large angle:
                components.append(np.array(component))
                component = []
                tangent = None
            component.append((v, v_next))
            v = v_next
            tangent_prev = tangent
        components.append(np.array(component))
    assert len(edges) == sum(map(len, components))
    return components


if __name__ == "__main__":
    main()
