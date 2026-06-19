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
    boundary: Optional[dict] = None,
    contacts: Optional[dict] = None,
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
    if geom_type != "spatial_transport":
        raise ValueError(
            "qimpy.transport.plot renders the finite-volume 'spatial_transport'"
            f" geometry; checkpoint has variant_name={geom_type!r}."
        )
    run_finite_volume(file_list, mine, output, density, streamlines, boundary,
                      contacts, dpi)


def run_finite_volume(file_list, mine, output, density, streamlines, boundary,
                      contacts, dpi) -> None:
    """Frame-parallel, mesh-native rendering of FiniteVolume (finite-volume) output.

    The finite-volume state is one average per triangle, so the density is drawn
    as a flat-shaded ``tripcolor`` (piecewise-constant, the honest FV picture)
    over the actual mesh. Current streamlines are traced mesh-natively: the
    per-cell (jx, jy) is scattered to the vertices and sampled with matplotlib's
    ``LinearTriInterpolator``, which interpolates within the triangulation and
    returns values masked outside it -- so streamlines stop at the true device
    boundary with no convex-hull bleed (important for non-convex cross/Hall-bar
    domains). The device boundary itself is drawn from the mesh edges that border
    exactly one triangle. Each rank renders its strided subset of frames."""
    import matplotlib.tri as mtri
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch
    cmap = density.get("cmap", "bwr")
    bdraw = {} if boundary is None else boundary
    cdraw = {} if contacts is None else contacts
    with Checkpoint(file_list[0]) as cp:
        g = cp["/geometry"]
        verts = np.array(g["mesh_vertices"])         # (Nv, 2)
        tris = np.array(g["mesh_triangles"])         # (K, 3)
        mesh_path = g.attrs.get("mesh_file", "")
    mesh_path = mesh_path.decode() if isinstance(mesh_path, bytes) else str(mesh_path)
    triang = mtri.Triangulation(verts[:, 0], verts[:, 1], tris)
    nv = len(verts)
    # Device boundary = mesh edges that border exactly one triangle, i.e. have no
    # neighbour across them (Triangulation.neighbors == -1). neighbors[k, j] is
    # the triangle across the edge (tris[k, j] -> tris[k, (j+1)%3]).
    bk, bj = np.where(triang.neighbors < 0)
    bseg = np.stack([verts[tris[bk, bj]], verts[tris[bk, (bj + 1) % 3]]], axis=1)
    # Clip path for the streamlines = the EXACT device, built as the union of all
    # mesh triangles (each a closed subpath). The clip is a render-time
    # set-intersection of the drawn lines with this region, so no streamline ink
    # lands outside the device -- independent of the integration grid (the
    # velocity mask alone only bounds streamlines to ~one grid cell). Using the
    # triangle union (rather than an assembled boundary polygon) is robust for
    # star-shaped / pinched domains where a single outline can wind wrongly.
    tv = verts[tris]                                              # (K, 3, 2)
    clip_pts = np.concatenate([tv, tv[:, :1]], axis=1).reshape(-1, 2)  # v0,v1,v2,v0
    clip_codes = np.tile(
        np.array([MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO,
                  MplPath.CLOSEPOLY], np.uint8), len(tris))
    device_path = MplPath(clip_pts, clip_codes) if len(tris) else None
    # Contacts: the boundary markers live in the source mesh file (its path is
    # recorded in the checkpoint). Highlight the parametrized contacts (any marker
    # other than the unparametrized "wall") over the plain device boundary.
    contact_segs = {}
    if cdraw.get("draw", True) and mesh_path:
        try:
            md = np.load(mesh_path, allow_pickle=True)
            be = np.asarray(md["boundary_edges"])
            bm = np.asarray([str(x) for x in md["boundary_markers"]])
            for marker in sorted(set(bm)):
                if marker.lower() == "wall":
                    continue
                sel = bm == marker
                contact_segs[marker] = np.stack(
                    [verts[be[sel, 0]], verts[be[sel, 1]]], axis=1)
        except Exception:
            contact_segs = {}
    contact_palette = {"source": "#2ca02c", "drain": "#d62728"}
    _cyc = ["#1f77b4", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
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
                # Scatter per-triangle (jx, jy) to vertices (mean of incident
                # cells), then sample with the mesh-aware interpolator; values
                # off the triangulation come back masked -> NaN, so streamlines
                # stay inside the device.
                jxv = np.zeros(nv); jyv = np.zeros(nv); cnt = np.zeros(nv)
                np.add.at(jxv, tris.ravel(), np.repeat(obs[fr, :, 1], 3))
                np.add.at(jyv, tris.ravel(), np.repeat(obs[fr, :, 2], 3))
                np.add.at(cnt, tris.ravel(), 1.0)
                nz = cnt > 0; jxv[nz] /= cnt[nz]; jyv[nz] /= cnt[nz]
                U = mtri.LinearTriInterpolator(triang, jxv)(Xs, Ys)
                V = mtri.LinearTriInterpolator(triang, jyv)(Xs, Ys)
                sp = ax.streamplot(xs, ys, U.filled(np.nan), V.filled(np.nan),
                                   density=streamlines.get("density", 1.5),
                                   linewidth=streamlines.get("linewidth", 0.6),
                                   arrowsize=streamlines.get("arrowsize", 0.6),
                                   color="k")
                if device_path is not None and streamlines.get("clip", True):
                    # exact hard clip to the device (triangle union; no overshoot)
                    clip = PathPatch(device_path, transform=ax.transData,
                                     fc="none", ec="none")
                    ax.add_patch(clip)
                    sp.lines.set_clip_path(clip)
                    try:
                        sp.arrows.set_clip_path(clip)
                    except Exception:
                        for art in ax.patches:
                            if art is not clip:
                                art.set_clip_path(clip)
            if bdraw.get("draw", True) and len(bseg):
                ax.add_collection(LineCollection(
                    bseg, colors=bdraw.get("color", "0.2"),
                    linewidths=bdraw.get("linewidth", 1.3), zorder=3))
            if contact_segs:
                clw = cdraw.get("linewidth", 3.0)
                ccolors = cdraw.get("colors", {})
                handles = []
                for i_m, (marker, segs) in enumerate(contact_segs.items()):
                    col = ccolors.get(marker, contact_palette.get(
                        marker, _cyc[i_m % len(_cyc)]))
                    ax.add_collection(LineCollection(
                        segs, colors=col, linewidths=clw, zorder=4))
                    handles.append(Line2D([0], [0], color=col, lw=2.5, label=marker))
                ax.legend(handles=handles, loc="upper right", fontsize=8,
                          framealpha=0.85)
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
