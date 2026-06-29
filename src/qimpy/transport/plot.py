from __future__ import annotations
from typing import Optional
import argparse
import glob
import logging

import matplotlib.pyplot as plt
import numpy as np

from qimpy import rc, log, io
from qimpy.profiler import StopWatch
from qimpy.io import log_config, Checkpoint


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
    dpi: int = 200,
    **ignored,
) -> None:
    """Render finite-volume (FiniteVolume) transport checkpoints.

    ``**ignored`` absorbs legacy keys from older input files for compatibility.
    """
    if density is None:
        density = {}
    # Distribute frames over MPI:
    file_list = rc.comm.bcast(sorted(glob.glob(checkpoints)))
    mine = slice(rc.i_proc, None, rc.n_procs)
    with Checkpoint(file_list[0]) as cp:
        geom_type = cp["/geometry"].attrs.get("variant_name", b"")
        geom_type = (geom_type.decode() if isinstance(geom_type, bytes)
                     else str(geom_type))
    if geom_type != "finite_volume":
        raise ValueError(
            "qimpy.transport.plot renders the finite-volume 'finite_volume' geometry; "
            f"checkpoint has variant_name={geom_type!r}."
        )
    run_finite_volume(file_list, mine, output, density, streamlines, dpi)


def run_finite_volume(file_list, mine, output, density, streamlines, dpi) -> None:
    """Frame-parallel, mesh-native rendering of FiniteVolume (finite-volume) output.

    The finite-volume state is one average per triangle, so the density is drawn
    as a flat-shaded ``tripcolor`` (piecewise-constant, the honest FV picture)
    over the actual mesh, and current streamlines are traced from a linear
    interpolation of (jx, jy) off the cell centroids. Each rank renders its
    strided subset of frames, so post-processing scales like the solve."""
    import matplotlib.tri as mtri
    from scipy.interpolate import griddata
    cmap = density.get("cmap", "bwr")
    with Checkpoint(file_list[0]) as cp:
        g = cp["/geometry"]
        verts = np.array(g["mesh_vertices"])         # (Nv, 2)
        tris = np.array(g["mesh_triangles"])         # (K, 3)
        cen = np.array(g["cell_centroid"])           # (K, 2)
    triang = mtri.Triangulation(verts[:, 0], verts[:, 1], tris)
    if streamlines is not None:
        xs = np.linspace(verts[:, 0].min(), verts[:, 0].max(), 220)
        ys = np.linspace(verts[:, 1].min(), verts[:, 1].max(), 220)
        Xs, Ys = np.meshgrid(xs, ys)
    orig_level = log.getEffectiveLevel(); log.setLevel(logging.INFO)
    for checkpoint_file in file_list:
        with Checkpoint(checkpoint_file) as cp:
            g = cp["/geometry"]
            i_step_list = np.array(g["i_step"])[mine]
            t_list = np.array(g["t"])[mine]
            obs = np.array(g["fv_observables"][mine])   # (nframe, K, n_obs)
        for fr, (i_step, t) in enumerate(zip(i_step_list, t_list)):
            n_val = obs[fr, :, 0]                        # (K,) per-cell density
            vmax = float(np.nanmax(np.abs(n_val)))
            if not np.isfinite(vmax) or vmax == 0.0:
                vmax = 1.0
            fig, ax = plt.subplots(figsize=(6, 6))
            tpc = ax.tripcolor(triang, facecolors=n_val / vmax, cmap=cmap,
                               vmin=-1, vmax=1)         # flat shading = FV cell average
            ax.set_aspect("equal"); ax.set_title(f"$t$ = {t:.4g}"); ax.axis("off")
            cb = fig.colorbar(tpc, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(rf"Density ($\times|\rho|_{{\max}}$ = {vmax:.2e})")
            if streamlines is not None and obs.shape[-1] >= 3:
                U = np.nan_to_num(griddata(cen, obs[fr, :, 1], (Xs, Ys), method="linear"))
                V = np.nan_to_num(griddata(cen, obs[fr, :, 2], (Xs, Ys), method="linear"))
                ax.streamplot(xs, ys, U, V,
                              density=streamlines.get("density", 1.5),
                              linewidth=streamlines.get("linewidth", 0.6),
                              arrowsize=streamlines.get("arrowsize", 0.6), color="k")
            plot_file = output.format(i_step)
            fig.savefig(plot_file, bbox_inches="tight", dpi=dpi)
            plt.close(fig)
            log.info(f"Saved {plot_file}")
    log.setLevel(orig_level)
    rc.comm.Barrier()


def split_names(input: str) -> list[str]:
    return input.split(",") if input else []


if __name__ == "__main__":
    main()
