from __future__ import annotations
import argparse
import glob
import logging

import matplotlib.pyplot as plt
import numpy as np
import h5py

from qimpy import rc, log
from qimpy.mpi import TaskDivision
from qimpy.io import log_config


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
    args = parser.parse_args()

    # Distirbute tasks over MPI:
    file_list = rc.comm.bcast(sorted(glob.glob(args.checkpoints)))
    division = TaskDivision(n_tot=len(file_list), n_procs=rc.n_procs, i_proc=rc.i_proc)

    orig_log_level = log.getEffectiveLevel()
    log.setLevel(logging.INFO)  # Capture output from all processes
    for checkpoint_file in file_list[division.i_start : division.i_stop]:
        plot_file = checkpoint_file.replace(".h5", ".png")

        # Load data from checkpoint:
        with h5py.File(checkpoint_file, "r") as cp:
            n_quads = cp["/geometry/quads"].shape[0]
            q_list = []
            rho_list = []
            v_list = []
            for i_quad in range(n_quads):
                prefix = f"/geometry/quad{i_quad}"
                q_list.append(np.array(cp[f"{prefix}/q"]))
                rho_list.append(np.array(cp[f"{prefix}/rho"]))
                v_list.append(np.array(cp[f"{prefix}/v"]))

        plt.clf()
        rho_max = max(np.max(rho) for rho in rho_list)
        contour_kwargs = dict(
            levels=np.linspace(-1e-3 * rho_max, rho_max, 20), cmap="bwr"
        )
        stream_kwargs = dict(density=2.0, linewidth=1.0, color="k", arrowsize=1.0)

        for q, rho, v in zip(q_list, rho_list, v_list):
            x = q[:, :, 0]
            y = q[:, :, 1]
            contour = plt.contourf(x, y, rho, **contour_kwargs)

            if args.streamlines:
                plt.streamplot(x, y, v[..., 0].T, v[..., 1].T, **stream_kwargs)

            # Label edges:
            NX, NY = x.shape
            midNX = slice(NX // 2, NX // 2 + 2)
            midNY = slice(NY // 2, NY // 2 + 2)
            text_kwargs = dict(ha="center", rotation_mode="anchor")
            for i_edge, q_mid in enumerate(
                (q[midNX, 0], q[-1, midNY], q[midNX, -1][::-1], q[0, midNY][::-1])
            ):
                dq = np.diff(q_mid, axis=0)[0]
                angle = np.rad2deg(np.arctan2(dq[1], dq[0]))
                plt.text(*q_mid[0], f"{i_edge}$\\to$", rotation=angle, **text_kwargs)

        plt.gca().set_aspect("equal")
        plt.colorbar(contour)
        plt.savefig(plot_file, bbox_inches="tight", dpi=200)
        log.info(f"Saved {plot_file}")

    log.setLevel(orig_log_level)  # Switch log back to single process
    rc.comm.Barrier()
    rc.report_end()


if __name__ == "__main__":
    main()
