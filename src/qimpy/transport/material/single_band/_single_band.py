from __future__ import annotations
from typing import Union, Sequence, Callable, Optional
from functools import cache
from dataclasses import dataclass

import numpy as np
import torch

from qimpy import rc, log
from qimpy.mpi import ProcessGrid
from qimpy.io import CheckpointPath, CheckpointContext
from qimpy.lattice import Lattice
from .. import Material, fermi


class SingleBand(Material):
    """Single-band model for charge transport with energy resolution."""

    def __init__(
        self,
        *,
        lattice: Union[Lattice, dict],
        kmesh: Union[Sequence[int], np.ndarray],
        v: float = 0.0,
        m: float = 0.0,
        dispersion: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        mu: float,
        T: float,
        nT_below: float = 5,
        nT_above: float = 5,
        process_grid: ProcessGrid,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        """Initialize a single-band model material.

        Parameters
        ----------
        lattice
            :yaml:`Bulk lattice vectors / unit cell definition.`
        kmesh
            :yaml:`Uniform k-point mesh from which to select relevant k-points.`
        v
            :yaml:`Velocity specifying a linear dispersion model.`
            Exactly only one of v, m and dispersion must be specified.
        m
            :yaml:`Effective mass specifying a quadratic dispersion model.`
            Exactly only one of v, m and dispersion must be specified.
        dispersion
            In code, custom function that returns energy given Cartesian k
            (in atomic units) and the lattice. For convenience, the Cartesian k
            will be pre-wrapped aroung k = 0. In principle the energy function
            should be periodic on the reciprocal lattice, but this will not matter
            if the selected k are far from the Brillouin zone boundaries.
            This option currently does not support continuing from checkpoints.
            Exactly only one of v, m and dispersion must be specified.
        mu
            :yaml:`Backround/reference chemical potential of initial state`
        T
            :yaml:`Backround/reference temperature of initial state`
        nT_below
            :yaml:`Include states with energies starting at this many T below mu.`
        nT_above
            :yaml:`Include states with energies up to this many T above mu.`
        """
        super().__init__()
        self.add_child("lattice", Lattice, lattice, checkpoint_in)
        self.kmesh = kmesh
        self.mu = mu
        self.T = T
        self.nT_below = nT_below
        self.nT_above = nT_above

        # Prepare dispersion relation:
        if v:
            assert (not m) and (dispersion is None)
            self.dispersion = LinearDispersion(v=v)
        if m:
            assert (not v) and (dispersion is None)
            self.dispersion = QuadraticDispersion(m=m)
        if dispersion is not None:
            assert (not v) and (not m)
            assert callable(dispersion)
            self.dispersion = dispersion

        # Select k-points:
        k0, k1, k2 = [
            torch.cat(
                (
                    torch.arange((Nk + 1) // 2, device=rc.device),
                    torch.arange((Nk + 1) // 2 - Nk, 0, device=rc.device),
                )
            )
            / Nk
            for Nk in kmesh
        ]
        k12 = torch.stack(
            torch.meshgrid(k0[:1], k1, k2, indexing="ij"), dim=-1
        ).flatten(
            0, 2
        )  # k1-k2 slice with k0 = 0
        Emin = mu - nT_below * T
        Emax = mu + nT_above * T
        k_sel = []
        for k0_i in k0:
            k12[:, 0] = k0_i
            k_cart = k12 @ self.lattice.Gbasis.T
            E = self.dispersion(k_cart, self.lattice)
            sel = torch.where(torch.logical_and(E >= Emin, E <= Emax))[0]
            if len(sel):
                k_sel.append(k12[sel].clone())
        k_all = torch.cat(k_sel, dim=0)
        nk = len(k_all)
        wk = 2 / np.prod(kmesh)
        log.info(f"Selected {nk} k-points from k-mesh with dimensions {kmesh}")
        self.initialize(nk=nk, wk=wk, n_bands=1, n_dim=3, process_grid=process_grid)

        # Initialize further properties for selected k-points:
        self.k[:] = k_all[self.k_mine]
        k_cart = self.k @ self.lattice.Gbasis.T
        k_cart.requires_grad_(True)
        E = self.dispersion(k_cart, self.lattice)
        E.sum().backward()  # compute v = dE/dk in k_cart.grad
        self.E[:, 0] = E.detach()
        self.v[:, 0, :] = k_cart.grad.detach()
        self.rho0[:, 0, 0] = fermi(self.E[:, 0], mu, T)

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        attrs = cp_path.attrs
        attrs["kmesh"] = self.kmesh
        if isinstance(self.dispersion, LinearDispersion):
            attrs["v"] = self.dispersion.v
        if isinstance(self.dispersion, QuadraticDispersion):
            attrs["m"] = self.dispersion.m
        attrs["mu"] = self.mu
        attrs["T"] = self.T
        attrs["nT_below"] = self.nT_below
        attrs["nT_above"] = self.nT_above

        # Write k-points, velocities and energies:
        cp, path = cp_path
        assert cp is not None
        cp.write_slice(
            cp.create_dataset_real(f"{path}/k", (self.k_division.n_tot, 3)),
            (self.k_division.i_start, 0),
            self.k,
        )
        cp.write_slice(
            cp.create_dataset_real(f"{path}/v", (self.k_division.n_tot, 3)),
            (self.k_division.i_start, 0),
            self.v[:, 0, :],
        )
        cp.write_slice(
            cp.create_dataset_real(f"{path}/E", (self.k_division.n_tot,)),
            (self.k_division.i_start,),
            self.E[:, 0],
        )
        return list(attrs.keys()) + ["k", "v", "E"]

    def initialize_fields(
        self, rho: torch.Tensor, params: dict[str, torch.Tensor], patch_id: int
    ) -> None:
        pass  # No spatially-varying / parameter sweep fields yet

    def get_contactor(
        self, n: torch.Tensor, **kwargs
    ) -> Callable[[float], torch.Tensor]:
        return Contactor(self, n, **kwargs)

    def get_reflector(
        self, n: torch.Tensor
    ) -> Callable[[torch.Tensor], torch.Tensor]:  # absorbing boundary
        return torch.zeros_like

    def rho_dot(self, rho: torch.Tensor, t: float, patch_id: int) -> torch.Tensor:
        return torch.zeros_like(rho)  # TODO: add scattering

    def get_observable_names(self) -> list[str]:
        return ["n", "jx", "jy", "e"]  # density, fluxes and energy density

    @cache
    def get_observables(self, t: float) -> torch.Tensor:
        return torch.cat(
            (
                torch.ones((1, self.nk_mine), device=rc.device),
                self.transport_velocity.T,
                self.E.T,
            ),
            dim=0,
        )


class Contactor:
    rho_contact: torch.Tensor  #: Cached constant contact distribution

    def __init__(
        self,
        sb: SingleBand,
        n: torch.Tensor,
        *,
        dmu: float = 0.0,
        dT: float = 0.0,
        Eeff: float = 0.0,
    ) -> None:
        """Return contact distribution function for specified chemical potential
        shift `dmu`, temperature shift `dT` and Fermi surface tilt parameterized
        by effective electric field `Eeff` along the normal direction `n`.
        (Eeff is the Lagrange multiplier corresponding to a drift velocity.)"""
        self.rho_contact = fermi(
            sb.E[:, 0] + Eeff * (n @ sb.transport_velocity.T), sb.mu + dmu, sb.T + dT
        )

    def __call__(self, t):
        # TODO: add time dependence options
        return self.rho_contact


@dataclass
class LinearDispersion:
    v: float  #: velocity

    def __call__(self, k_cart: torch.Tensor, lattice: Lattice) -> torch.Tensor:
        return self.v * k_cart.norm(dim=-1)


@dataclass
class QuadraticDispersion:
    m: float  #: effective mass

    def __call__(self, k_cart: torch.Tensor, lattice: Lattice) -> torch.Tensor:
        return (0.5 / self.m) * (k_cart**2).sum(dim=-1)
