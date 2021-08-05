from __future__ import annotations
import qimpy as qp
import numpy as np
import torch
from ._lattice import _get_lattice_point_group, _symmetrize_lattice
from ._ions import _get_space_group, _symmetrize_positions
from ._grid import _check_grid_shape, _get_grid_shape
from typing import List, Dict


class Symmetries(qp.Constructable):
    """Space group symmetries.
    Detects space group from lattice and ions, and provides methods to
    symmetrize properties such as positions, forces and densities."""

    __slots__ = ('lattice', 'ions', 'tolerance', 'n_sym',
                 'rot', 'trans', 'ion_map', 'i_id', 'i_inv')
    lattice: qp.lattice.Lattice  #: Corresponding lattice vectors
    ions: qp.ions.Ions  #: Corresponding ionic geometry
    tolerance: float  #: Relative error threshold in detecting symmetries
    n_sym: int  #: Number of space group operations
    rot: torch.Tensor  #: Rotations in fractional coordinates (n_sym x 3 x 3)
    trans: torch.Tensor  #: Translations in fractional coordinates (n_sym x 3)
    ion_map: torch.Tensor  #: Ion index each ion maps to (n_sym x n_ions)
    i_id: int  #: Index of identity operation within space group
    i_inv: List[int]  #: Indices of any inversion operations in space group

    symmetrize_lattice = _symmetrize_lattice
    symmetrize_positions = _symmetrize_positions
    check_grid_shape = _check_grid_shape
    get_grid_shape = _get_grid_shape

    def __init__(self, *, co: qp.ConstructOptions,
                 lattice: qp.lattice.Lattice, ions: qp.ions.Ions,
                 axes: Dict[str, np.ndarray] = {},
                 tolerance: float = 1e-6) -> None:
        """Determine space group from `lattice` and `ions`.

        Parameters
        ----------
        tolerance
            Threshold for detecting symmetries. :yaml:
        """
        super().__init__(co=co)
        rc = self.rc
        self.lattice = lattice
        self.ions = ions
        self.tolerance = tolerance
        qp.log.info('\n--- Initializing Symmetries ---')

        # Lattice point group:
        lattice_sym = _get_lattice_point_group(lattice.Rbasis, tolerance)
        qp.log.info(f'Found {lattice_sym.shape[0]} point-group symmetries'
                    ' of the Bravais lattice')

        # Reduce symmetries by any global axes:
        sym_axis = ((lattice.Rbasis @ lattice_sym)
                    @ torch.linalg.inv(lattice.Rbasis))  # Cartesian axes
        for axis_name, axis_np in axes.items():
            axis = torch.from_numpy(axis_np).to(rc.device)
            sel = torch.where((sym_axis @ axis - axis).norm(dim=-1)
                              < tolerance)[0]
            lattice_sym = lattice_sym[sel]
            sym_axis = sym_axis[sel]
            qp.log.info(f'Reduced to {len(sel)} point-group symmetries'
                        f' with {axis_name}')

        # Space group:
        self.rot, self.trans, self.ion_map = _get_space_group(
            lattice_sym, lattice, ions, tolerance)
        self.n_sym = self.rot.shape[0]
        qp.log.info(f'Found {self.n_sym} space-group symmetries with basis:')
        for i_sym in range(self.n_sym):
            sym_str = '- ['
            for row in range(3):
                sym_str += rc.fmt(self.rot[i_sym, row].to(torch.int)) + ', '
            qp.log.info(sym_str + rc.fmt(self.trans[i_sym]) + ']')
        qp.log.debug('Ion map:\n' + rc.fmt(self.ion_map))

        # Enforce symmetries exactly on lattice:
        qp.log.info('Enforcing symmetries:')
        lattice.update(self.symmetrize_lattice(lattice.Rbasis))

        # Enforce symmetries exactly on ions:
        if ions.n_ions:
            positions_sym = self.symmetrize_positions(ions.positions)
            rms = ((positions_sym - ions.positions)**2).mean().sqrt()
            ions.positions = positions_sym
            qp.log.info(f'RMS change in fractional positions of ions: {rms:e}')

        # Identify location of special entries:
        # --- identity
        id = torch.eye(3, device=rc.device)  # identity matrix
        id_diff = (((self.rot - id)**2).sum(dim=(1, 2))
                   + (self.trans**2).sum(dim=1))
        self.i_id = int(id_diff.argmin().item())
        if id_diff[self.i_id] > tolerance**2:
            raise ValueError('Identity operation not found in space group.')
        # ---  inversion (if present: list of indices, else [])
        self.i_inv = torch.where(((self.rot + id)**2).sum(dim=(1, 2))
                                 < tolerance**2)[0].tolist()
