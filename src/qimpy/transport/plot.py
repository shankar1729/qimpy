from __future__ import annotations
import argparse
import glob

import matplotlib.pyplot as plt
import numpy as np

from qimpy import rc, log
from qimpy.io import log_config, Checkpoint


def main() -> None:
    log_config()
    rc.init()
    assert rc.n_procs == 1  # Plot in serial mode only

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoints", help="Filename pattern for checkpoints", type=str
    )
    parser.add_argument(
        "--streamlines", help="Whether to draw streamlines", type=bool, default=False
    )
    args = parser.parse_args()

    for checkpoint_file in sorted(glob.glob(args.checkpoints)):
        plot_file = checkpoint_file.replace(".h5", ".png")

        # Load data from checkpoint:
        with Checkpoint(checkpoint_file) as cp:
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
        contour_kwargs = dict(levels=np.linspace(-1e-3, rho_max, 20), cmap="bwr")
        stream_kwargs = dict(density=2.0, linewidth=1.0, color="k", arrowsize=1.0)

        for q, rho, v in zip(q_list, rho_list, v_list):
            x = q[:, :, 0]
            y = q[:, :, 1]
            plt.contourf(x, y, rho, **contour_kwargs)

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
        plt.savefig(plot_file, bbox_inches="tight", dpi=200)
        log.info(f"Saved {plot_file}")


if __name__ == "__main__":
    main()
