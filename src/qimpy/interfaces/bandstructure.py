"""Plot band structure from HDF5 checkpoint file.

Command-line parameters (obtained using
:code:`python -m qimpy.interfaces.bandstructure -h`):

.. code-block:: bash

    python -m qimpy.interfaces.bandstructure [-h] -c FILE -o FILE

Options:

  -h, --help            show this help message and exit
  -c FILE, --checkpoint-file FILE
                        checkpoint file in HDF5 format
  -o FILE, --output-file FILE
                        output plot in matplotlib supported format (based on extension)
"""
import argparse
import ast

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot(checkpoint_file: str, output_file: str) -> None:
    """Plot band structure from HDF5 checkpoint name `checkpoint_file`.
    Save plot to `output_file` in a matplotlib supported format (based on extension).
    """
    # Read required quantities from checkpoint:
    with h5py.File(checkpoint_file, "r") as h5_file:
        eig = np.array(h5_file["/electrons/eig"])
        k_length = np.array(h5_file["/electrons/kpoints/k_length"])
        labels: dict[int, str] = ast.literal_eval(
            str(h5_file["/electrons/kpoints"].attrs["labels"])
        )
        mu = float(h5_file["/electrons/fillings"].attrs["mu"])
    n_spins, nk, n_bands = eig.shape

    # Check for semi-core gap (only among occupied bands):
    n_occupied_bands = max(np.where(eig.min(axis=(0, 1)) < mu)[0]) + 1
    gaps_start = eig[..., : n_occupied_bands - 1].max(axis=(0, 1))
    gaps_stop = eig[..., 1:n_occupied_bands].min(axis=(0, 1))
    gaps = gaps_stop - gaps_start
    gap_cut = 0.5  # threshold on semi-core gap at which to break y-axis
    i_gaps = np.where(gaps > gap_cut)[0] + 1
    split_axis = len(i_gaps) > 0
    i_band_edges = np.concatenate(
        (np.zeros(1, dtype=int), i_gaps, np.full(1, n_bands, dtype=int))
    )
    band_ranges = list(zip(i_band_edges[:-1], i_band_edges[1:]))

    # Prepare figure and panels:
    yformatter = ticker.ScalarFormatter(useOffset=False)
    ax_heights = [3] + [1] * (len(band_ranges) - 1)
    if split_axis:
        _, axes = plt.subplots(
            len(band_ranges),
            1,
            sharex="all",
            gridspec_kw={"height_ratios": ax_heights},
        )
        plt.subplots_adjust(hspace=0.03)
    else:
        plt.figure()
        axes = [plt.gca()]

    # Plot:
    tick_pos = [k_length[i] for i in labels.keys()]
    for i_ax, band_range in enumerate(band_ranges[::-1]):
        plt.sca(axes[i_ax])
        for i_spin in range(n_spins):
            plt.plot(
                k_length,
                eig[i_spin, :, slice(*band_range)],
                color="kr"[i_spin],
            )
        for pos in tick_pos[1:-1]:
            plt.axvline(pos, color="k", ls="dotted", lw=1)
        plt.xlim(0, k_length[-1])
        axes[i_ax].yaxis.set_major_formatter(yformatter)
        if (not np.isnan(mu)) and (not i_ax):
            plt.axhline(mu, color="k", ls="dotted", lw=1)

    # Axis settings for arbitrary number of splits:
    axes[0].set_ylabel(r"$E$ [$E_h$]")
    axes[0].set_ylim(None, eig[..., -1].min())
    for ax in axes[:-1]:
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", top=False, bottom=False)  # no x-ticks
    for ax in axes[1:]:
        ax.spines["top"].set_visible(False)
    axes[-1].xaxis.tick_bottom()
    axes[-1].set_xticks(tick_pos)
    axes[-1].set_xticklabels(labels.values())

    # Axis break annotations:
    for i_ax, ax in enumerate(axes):
        dx = 0.01
        dy = (dx * 0.5 * sum(ax_heights)) / ax_heights[i_ax]
        kwargs = dict(transform=ax.transAxes, color="k", clip_on=False, lw=1)
        for x0 in (0, 1):
            if i_ax + 1 < len(axes):
                ax.plot((x0 - dx, x0 + dx), (-dy, +dy), **kwargs)
            if i_ax:
                ax.plot((x0 - dx, x0 + dx), (1 - dy, 1 + dy), **kwargs)
    plt.savefig(output_file, bbox_inches="tight")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m qimpy.interfaces.bandstructure",
        description="Plot band structure from HDF5 checkpoint file",
    )
    parser.add_argument(
        "-c",
        "--checkpoint-file",
        metavar="FILE",
        help="checkpoint file in HDF5 format",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-file",
        metavar="FILE",
        help="output plot in matplotlib supported format (based on extension)",
        required=True,
    )
    args = parser.parse_args()
    plot(args.checkpoint_file, args.output_file)


if __name__ == "__main__":
    main()
