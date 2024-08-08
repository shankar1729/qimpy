from __future__ import annotations

import torch

from qimpy.lattice import Lattice
from qimpy.grid import Grid, FieldH, coulomb


class KernelWire:
    """Coulomb interactions between fields with 1D periodicity: Wigner-Seitz version."""

    grid: Grid
    i_dir: int  #: Periodic direction (wire axis)
    _kernel: torch.Tensor  # Coulomb kernel

    def __init__(self, coul: coulomb.Coulomb, i_dir: int) -> None:
        self.grid = coul.grid
        self.i_dir = i_dir
        raise NotImplementedError

    def __call__(self, rho: FieldH, correct_G0_width: bool = False) -> FieldH:
        assert self.grid is rho.grid
        raise NotImplementedError

    def stress(self, rho1: FieldH, rho2: FieldH) -> torch.Tensor:
        raise NotImplementedError


class KernelCylindrical:
    """Coulomb interactions between fields with 1D periodicity: analytic version."""

    grid: Grid
    i_dir: int  #: Periodic direction (cylinder axis)
    radius: float  #: Cylinder radius
    _kernel: torch.Tensor  #: Coulomb kernel

    def __init__(self, coul: coulomb.Coulomb, i_dir: int) -> None:
        self.grid = coul.grid
        self.i_dir = i_dir
        if coul.radius:
            self.radius = coul.radius
        else:
            raise NotImplementedError  # TODO: determine in-radius
        raise NotImplementedError

    def __call__(self, rho: FieldH, correct_G0_width: bool = False) -> FieldH:
        assert self.grid is rho.grid
        raise NotImplementedError

    def stress(self, rho1: FieldH, rho2: FieldH) -> torch.Tensor:
        raise NotImplementedError


class EwaldWire:
    """Coulomb interactions between point charges with 1D periodicity."""

    lattice: Lattice
    i_dir: int

    def __init__(self, lattice: Lattice, i_dir: int) -> None:
        self.lattice = lattice
        self.i_dir = i_dir
        raise NotImplementedError

    def __call__(self, positions: torch.Tensor, Z: torch.Tensor) -> float:
        raise NotImplementedError
