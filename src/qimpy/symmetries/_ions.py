from __future__ import annotations
import qimpy as qp
import torch
from typing import Tuple


def _get_space_group(
    lattice_sym: torch.Tensor,
    lattice: qp.lattice.Lattice,
    ions: qp.ions.Ions,
    tolerance: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find space group given point group `lattice_sym` and `ions`. Accuracy
    of symmetry detection is specified by relative threshold `tolerance`.

    Returns
    -------
    rot
        Rotations (n_sym x 3 x 3) in fractional coordinates.
    trans
        Translations (n_sym x 3) in lattice coordinates.
    ion_map
        0-based index (n_sym x n_ions) of the ion that each ion
         maps to after each symmetry operation."""

    # Special case of no ions:
    device = ions.rc.device
    n_ions = ions.n_ions
    if not n_ions:
        # space group = point group:
        rot = lattice_sym.clone().detach()
        trans = torch.zeros((lattice_sym.shape[0], 3), device=device)
        ion_map = torch.zeros((lattice_sym.shape[0], 0), dtype=torch.int, device=device)
        return rot, trans, ion_map

    # Prepare ion properties needed for space group determination:
    pos0 = ions.positions  # original ionic positions
    type_mask = ions.types[:, None] == ions.types[None, :]
    # --- magnetization:
    M0 = ions.M_initial
    if M0 is None:
        pass  # no handling needed
    elif len(M0.shape) == 1:
        # Scalar magnetization must be invariant; treat just like type:
        type_mask = torch.logical_and(
            type_mask, (M0[:, None] - M0[None, :]).abs() < tolerance
        )
        M0 = None  # no more handling of magnetization needed below
    else:
        # Vector magentization needs Cartesian rotation matrices:
        sym_M = (lattice.Rbasis @ lattice_sym) @ torch.linalg.inv(lattice.Rbasis)
        sym_M *= torch.linalg.det(lattice_sym).view(-1, 1, 1)  # pseudo-vector

    rot_list = []
    trans_list = []
    ion_map_list = []
    tol_sq = tolerance ** 2
    for i_sym in range(lattice_sym.shape[0]):
        rot_cur = lattice_sym[i_sym]

        # Compute all translations for each ion that map it back to an ion:
        # --- positions transform by rot, so transposed on right-multiply
        pos = pos0 @ rot_cur.T  # rotated positions of all ions
        offsets = pos0[None, ...] - pos[:, None, :]  # possible translations
        offsets -= torch.floor(0.5 + offsets)  # wrap to [-0.5,0.5)

        # Select those that map to ion with same type (and magnetization):
        if M0 is None:  # scalar or no magnetization
            mask = type_mask
        else:  # vector magnetization
            M = M0 @ sym_M[i_sym].T
            mask = torch.logical_and(
                (((M0[None, ...] - M[:, None, :]) ** 2).sum(dim=-1) < tol_sq), type_mask
            )

        # Find offsets that work for every ion:
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
                is_common = ((doffset ** 2).sum(dim=-1) < tol_sq).any(dim=1)
                common_offsets = common_offsets[is_common]
        if common_offsets is None:
            continue

        # Determine ion map for each offset and optimize it:
        index_offset = n_ions * torch.arange(n_ions, device=device)
        for offset in common_offsets:
            doffset = offsets - offset[None, None, :]
            doffset -= torch.floor(0.5 + doffset)  # wrap to [-0.5,0.5)
            ion_map_cur = (doffset ** 2).sum(dim=-1).argmin(dim=1)
            # Optimize offset by accounting for all atoms:
            doffset_best = doffset.view(-1, 3)[index_offset + ion_map_cur]
            offset_opt = offset + doffset_best.mean(axis=0)
            # Add to space group:
            rot_list.append(rot_cur)
            trans_list.append(offset_opt)
            ion_map_list.append(ion_map_cur)

    rot = torch.stack(rot_list)
    trans = torch.stack(trans_list)
    ion_map = torch.stack(ion_map_list)
    return rot, trans, ion_map


def _symmetrize_positions(
    self: qp.symmetries.Symmetries, positions: torch.Tensor
) -> torch.Tensor:
    """Symmetrize ionic `positions` (n_ions x 3)"""
    pos_rot = positions @ self.rot.transpose(-2, -1) + self.trans[:, None]
    pos_mapped = positions[self.ion_map, :]
    # Correction on rotated positions:
    dpos_rot = pos_mapped - pos_rot
    dpos_rot -= torch.floor(0.5 + dpos_rot)  # wrap to [-0.5,0.5)
    # Transform corrections back and average:
    dpos = dpos_rot @ torch.linalg.inv(self.rot.transpose(-2, -1))
    return positions + dpos.mean(dim=0)
