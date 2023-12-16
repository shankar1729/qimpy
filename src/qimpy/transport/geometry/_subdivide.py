from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from qimpy import log
from qimpy.math import ceildiv
from . import QuadSet


@dataclass
class SubQuadSet:
    """Specifies subdivision of a QuadSet (for flexibility in MPI parallelization)."""

    quad_index: np.ndarray  #: original quad index that each sub-quad corresponds to
    grid_start: np.ndarray  #: starting grid index of sub-quad (NsubQuads x 2)
    grid_stop: np.ndarray  #: stopping grid index of sub-quad (NsubQuads x 2)
    adjacency: np.ndarray  #: NsubQuads x 4 x 2, analogous to `QuadSet.adjacency`


def divided_count(quad_set: QuadSet, grid_size_max: int) -> tuple[int, float]:
    """Calculate number of quads if `quad_set` is divided such that
    the maximum sample count of any resulting quad is `grid_size`.
    Also return the percentage imbalance between subdivided quad sizes."""
    n_splits = ceildiv(quad_set.grid_size, grid_size_max)  # 2 splits for each quad
    n_divided_quads = n_splits.prod(axis=1).sum(axis=0)
    split_sizes = quad_set.grid_size / n_splits  # approximate: assuming equal splits
    split_counts = split_sizes.prod(axis=1)
    imbalance_percent = 100.0 * split_counts.std() / split_counts.mean()
    return n_divided_quads, imbalance_percent


def select_division(quad_set: QuadSet, n_processes: int) -> int:
    """Select `grid_size_max` suitable for division on `n_processes`.
    Reports all cases considered to guide selection of parallelization."""
    # Start with existing sizes, attempting to make domains squarer
    grid_size_max_list = np.unique(quad_set.grid_size)

    # Expand list with smaller entries if needed
    n_max = divided_count(quad_set, grid_size_max_list[0])[0]  # type:ignore
    needed_expansion = 4 * n_processes / n_max  # check quad counts till 4 n_processes
    if needed_expansion > 1.0:
        log_spacing = 0.2
        scale_factors = np.exp(
            -log_spacing - np.arange(0, 0.5 * np.log(needed_expansion), log_spacing)
        )
        additional_sizes = np.round(grid_size_max_list[0] * scale_factors)
        grid_size_max_list = np.unique(
            np.concatenate((grid_size_max_list, additional_sizes.astype(int)))
        )

    # Check and report candidates:
    log.info("\n--- Quad subdivision candidates ---")
    log.info("grid_size_max  n_quads  size_imbalance number_imbalance")
    best_grid_size_max = 0
    best_score = (np.inf,) * 3
    for grid_size_max in grid_size_max_list[::-1]:
        n_quads, size_imbalance = divided_count(quad_set, grid_size_max)
        n_quads_each = ceildiv(n_quads, n_processes)
        number_imbalance = 100.0 * (1.0 - n_quads / (n_quads_each * n_processes))
        score = (number_imbalance, size_imbalance, n_quads)
        if score < best_score:
            best_score = score
            best_grid_size_max = grid_size_max
        log.info(
            f"{grid_size_max:>9d} {n_quads:>9d}"
            f" {size_imbalance:>12.0f}% {number_imbalance:>12.0f}%"
        )
    log.info(
        f"Selecting grid_size_max = {best_grid_size_max} for {n_processes} processes"
    )
    return best_grid_size_max


def subdivide(quad_set: QuadSet, grid_size_max: int) -> SubQuadSet:
    """Subdividide `quad_set` till all grid dimensions are below `grid_size_max`."""
    # Divide lowest dimension (Nquads x 2 flattened) in each equivalence class first:
    dim_divisions: list[list[int]] = [[]] * len(quad_set.equivalent)
    for i_edge, equivalent_edge in enumerate(quad_set.equivalent):
        if i_edge == equivalent_edge:
            dim_divisions[i_edge] = []  # TODO
    return NotImplemented
