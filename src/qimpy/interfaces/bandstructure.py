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
from typing import Optional
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from qimpy.io import Unit, Checkpoint, CheckpointPath


def plot(
    checkpoint_files: list[str],
    output_file: str,
    plot_labels: Optional[list[str]],
    units: str,
) -> None:
    """Plot band structure from HDF5 checkpoint name `checkpoint_file`.
    Save plot to `output_file` in a matplotlib supported format (based on extension).
    """
    if plot_labels is not None:
        assert len(plot_labels) == len(checkpoint_files)
    eigs = []  # list of eigenvalues for each checkpoint
    mus = np.array([])  # list of chemical potentials for each checkpoint
    n_occupied_bands = np.array([], dtype=int)  #
    nbands_tot = 0
    plot_units = 1 if units == "Hartree" else Unit.convert(1, units).value
    for checkpoint_file in checkpoint_files:
        with Checkpoint(checkpoint_file) as checkpoint:
            eig = np.array(checkpoint["/electrons/eig"]) * plot_units
            eigs.append(eig)
            cp_k = CheckpointPath(checkpoint, "/electrons/kpoints")
            assert cp_k.attrs["variant_name"] == "k-path"
            k_length = np.array(cp_k["k_length"])
            label_names = cp_k.read_str("labels").split(",")
            label_indices = list(cp_k["label_indices"])  # type: ignore
            labels: dict[int, str] = dict(zip(label_indices, label_names))
            mu = float(checkpoint["/electrons/fillings"].attrs["mu"])
            mu *= plot_units
            mus = np.append(mus, mu)
            n_spins, nk, n_bands = eig.shape
            nbands_tot += n_bands
            # Check for semi-core gap (only among occupied bands):
            n_occupied_bands = np.append(
                n_occupied_bands, max(np.where(eig.min(axis=(0, 1)) < mu)[0]) + 1
            )

    occupied_eigs = np.concatenate(
        [
            eigs[i][..., 0 : n_occupied_bands[i]]
            for i in range(0, len(n_occupied_bands))
        ],
        axis=2,
    )
    occupied_eigs = np.sort(occupied_eigs, axis=2)
    gaps_start = occupied_eigs[..., :-1].max(axis=(0, 1))
    gaps_stop = occupied_eigs[..., 1:].min(axis=(0, 1))
    gaps = gaps_stop - gaps_start
    gap_cut = 0.5  # threshold on semi-core gap at which to break y-axis
    gap_cut *= plot_units
    i_gaps = np.where(gaps > gap_cut)[0] + 1
    split_axis = len(i_gaps) > 0
    i_band_edges = np.concatenate(
        (np.zeros(1, dtype=int), i_gaps, np.full(1, nbands_tot, dtype=int))
    )
    band_ranges = list(zip(i_band_edges[:-1], i_band_edges[1:]))
    ax_heights = [3] + [1] * (len(band_ranges) - 1)
    # Prepare figure and panels:
    yformatter = ticker.ScalarFormatter(useOffset=False)

    band_range_energies = [
        (
            occupied_eigs[..., r[0]].min() - 0.01 * plot_units,
            occupied_eigs[..., slice(r[1])].max() + 0.01 * plot_units,
        )
        for r in band_ranges
    ]
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

    for cf_num, (mu, eig) in enumerate(zip(mus, eigs)):
        plot_label = plot_labels[cf_num] if (plot_labels is not None) else None
        if cf_num == 0:
            # Plot:
            tick_pos = [k_length[i] for i in labels.keys()]
            for i_ax, brange_energy in enumerate(band_range_energies[::-1]):
                plt.sca(axes[i_ax])
                for i_spin in range(n_spins):
                    lines = plt.plot(
                        k_length,
                        eig[i_spin, ...],
                        color="kr"[i_spin],
                        linestyle="solid",
                    )
                    if (i_ax == 0) and (plot_label is not None):
                        first_legend = plt.gca().legend(
                            lines[:1], [plot_label], loc="upper right"
                        )
                        plt.gca().add_artist(first_legend)
                for pos in tick_pos[1:-1]:
                    plt.axvline(pos, color="k", ls="dotted", lw=1)
                plt.xlim(0, k_length[-1])
                axes[i_ax].yaxis.set_major_formatter(yformatter)
                axes[i_ax].set_ylim(brange_energy)
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

        else:
            axes = plt.gcf().axes
            for i_ax, ax in enumerate(axes):
                plt.sca(ax)
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                for i_spin in range(n_spins):
                    lines = plt.plot(
                        k_length,
                        eig[i_spin, :, :],
                        color="kr"[i_spin],
                        linestyle="dashed",
                        linewidth=2,
                    )
                    if (i_ax == 0) and (plot_label is not None):
                        additional_legend = plt.legend(
                            [*lines[:1]], [plot_label], loc="lower right"
                        )
                        plt.gca().add_artist(additional_legend)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                if (not np.isnan(mu)) and (not i_ax):
                    plt.axhline(mu, color="k", ls="dashed", lw=1)

    plt.savefig(output_file, bbox_inches="tight")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m qimpy.interfaces.bandstructure",
        description="Plot band structure from one or more HDF5 checkpoint file(s)",
    )
    parser.add_argument(
        "-c",
        "--checkpoint-file",
        metavar="FILE",
        nargs="+",
        help="checkpoint file(s) in HDF5 format",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-file",
        metavar="FILE",
        help="output plot in matplotlib supported format (based on extension)",
        required=True,
    )
    parser.add_argument(
        "-u",
        "--units",
        metavar="UNITS",
        help="energy units for band structure",
        choices=["Hartree", "eV"],
        default="Hartree",
    )
    parser.add_argument(
        "-l",
        "--labels",
        metavar="Labels",
        nargs="+",
        help="labels for band structure plots",
    )
    args = parser.parse_args()
    plot(args.checkpoint_file, args.output_file, args.labels, args.units)


if __name__ == "__main__":
    main()
