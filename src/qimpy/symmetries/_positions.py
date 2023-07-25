from __future__ import annotations
from typing import NamedTuple, Optional

import torch

from qimpy import symmetries
from qimpy.lattice import Lattice


class LabeledPositions(NamedTuple):
    """Fractional coordinates with labels for symmetry detection."""

    positions: torch.Tensor  #: fractional coordinates of points (N x 3)
    scalars: torch.Tensor  #: scalar labels (Ns x N)
    vectors: Optional[torch.Tensor] = None  #: vector labels (Nv x N x 3)
    pseudovectors: Optional[torch.Tensor] = None  #: pseudovector labels (Npv x N x 3)


def get_space_group(
    lattice_sym: torch.Tensor,
    lattice: Lattice,
    labeled_positions: LabeledPositions,
    tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find space group given point group `lattice_sym` and `labeled_positions`.
    Accuracy of symmetry detection is specified by relative threshold `tolerance`.

    Returns
    -------
    rot
        Rotations (n_sym x 3 x 3) in fractional coordinates.
    trans
        Translations (n_sym x 3) in lattice coordinates.
    position_map
        0-based index (n_sym x n_positions) of the position that each position
         maps to after each symmetry operation."""
    positions0, scalars, vectors0, pseudovectors0 = labeled_positions
    device = positions0.device
    n_ions = len(positions0)

    # Prepare masks and transformations:
    scalar_mask = (scalars[:, :, None] - scalars[:, None, :]).norm(dim=0) < tolerance
    if (vectors0 is not None) or (pseudovectors0 is not None):
        sym_vectors = (lattice.Rbasis @ lattice_sym) @ lattice.invRbasis
        sym_pseudovectors = sym_vectors * torch.linalg.det(lattice_sym).view(-1, 1, 1)

    rot_list = []
    trans_list = []
    position_map_list = []
    for i_sym in range(lattice_sym.shape[0]):
        rot_cur = lattice_sym[i_sym]

        # Compute all translations for each position that map it back to a position:
        # --- positions transform by rot, so transposed on right-multiply
        positions = positions0 @ rot_cur.T  # rotated positions
        offsets = positions0[None, ...] - positions[:, None, :]  # possible translations
        offsets -= torch.floor(0.5 + offsets)  # wrap to [-0.5,0.5)

        # Select those that map to position with compatible labels:
        mask = scalar_mask
        if vectors0 is not None:
            vectors = vectors0 @ sym_vectors[i_sym].T
            err = (vectors0[:, None] - vectors[:, :, None]).norm(dim=(0, 3))
            mask = torch.logical_and(err < tolerance, mask)
        if pseudovectors0 is not None:
            pseudovectors = vectors0 @ sym_pseudovectors[i_sym].T
            err = (pseudovectors0[:, None] - pseudovectors[:, :, None]).norm(dim=(0, 3))
            mask = torch.logical_and(err < tolerance, mask)

        # Find offsets that work for every position:
        common_offsets = None
        for i_ion in mask.count_nonzero(dim=1).argsort():
            # in ascending order of number of valid offsets
            offsets_cur = offsets[i_ion][mask[i_ion]]
            if common_offsets is None:
                common_offsets = offsets_cur
            else:
                # compute intersection of (common_offsets, offsets_cur)
                doffset = common_offsets[:, None, :] - offsets_cur[None, ...]
                doffset -= torch.floor(0.5 + doffset)  # wrap to [-0.5,0.5)
                is_common = (doffset.norm(dim=-1) < tolerance).any(dim=1)
                common_offsets = common_offsets[is_common]
        if common_offsets is None:
            continue

        # Determine position map for each offset and optimize it:
        index_offset = n_ions * torch.arange(n_ions, device=device)
        for offset in common_offsets:
            doffset = offsets - offset[None, None, :]
            doffset -= torch.floor(0.5 + doffset)  # wrap to [-0.5,0.5)
            position_map_cur = doffset.norm(dim=-1).argmin(dim=1)
            # Optimize offset by accounting for all atoms:
            doffset_best = doffset.view(-1, 3)[index_offset + position_map_cur]
            offset_opt = offset + doffset_best.mean(axis=0)
            # Add to space group:
            rot_list.append(rot_cur)
            trans_list.append(offset_opt)
            position_map_list.append(position_map_cur)

    rot = torch.stack(rot_list)
    trans = torch.stack(trans_list)
    position_map = torch.stack(position_map_list)
    return rot, trans, position_map


def symmetrize_positions(
    self: symmetries.Symmetries, positions: torch.Tensor
) -> torch.Tensor:
    """Symmetrize `positions` (n_ions x 3)"""
    pos_rot = positions @ self.rot.transpose(-2, -1) + self.trans[:, None]
    pos_mapped = positions[self.position_map, :]
    # Correction on rotated positions:
    dpos_rot = pos_mapped - pos_rot
    dpos_rot -= torch.floor(0.5 + dpos_rot)  # wrap to [-0.5,0.5)
    # Transform corrections back and average:
    dpos = dpos_rot @ torch.linalg.inv(self.rot.transpose(-2, -1))
    return positions + dpos.mean(dim=0)


def symmetrize_forces(
    self: symmetries.Symmetries, positions_grad: torch.Tensor
) -> torch.Tensor:
    """Symmetrize forces `positions_grad` (n_ions x 3) in lattice coordinates.
    Note that these are contravariant lattice coordinates, as in dE/dpositions.
    """
    return (positions_grad[self.position_map, :] @ self.rot).mean(dim=0)
