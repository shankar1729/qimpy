from __future__ import annotations
from typing import Callable
from functools import cache

import numpy as np
import torch

from qimpy import rc
from qimpy.mpi import ProcessGrid
from qimpy.profiler import stopwatch
from qimpy.io import CheckpointPath, CheckpointContext, InvalidInputException
from ..advect import limiter
from . import Material, fermi


class FermiCircle(Material):
    """Fermi-circle model material suitable for graphene and 2DEGs."""

    kF: float  #: Fermi wave-vector
    vF: float  #: Fermi velocity
    n_theta: int  #: Number of k along Fermi circle
    n_r: int  #: Number of radial k perpendicular to Fermi circle.`
    dk_r: float  #: Radial k grid spacing perpendicular to Fermi circle (for n_r > 1)
    m_star: float  #: Effective mass
    T: float  #: Temperature
    tau_inv_p: float  #: Momentum relaxation rate
    tau_inv_ee: float  #: Electron internal scattering rate (momentum-conserving)
    theta0: float  #: Initial angle in fermi circle grid
    specularity: float  #: Specularity of reflection at all boundaries
    r_c: float  #: Cyclotron radius for extenral magnetic field (disabled if infinity)
    F_theta: float  #: Angular force in grid coordinates

    def __init__(
        self,
        *,
        kF: float,
        vF: float,
        n_theta: int,
        n_r: int = 1,
        dk_r: float = 0.0,
        m_star: float = np.inf,
        T: float = 0.0,
        tau_p: float = np.inf,
        tau_ee: float = np.inf,
        r_c: float = np.inf,
        theta0: float = 0.0,
        specularity: float = 1.0,
        process_grid: ProcessGrid,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ):
        """
        Initialize Fermi-circle model material.

        Parameters
        ----------
        kF
            :yaml:`Fermi wave vector in atomic units.`
        vF
            :yaml:`Fermi velocity in atomic units.`
        n_theta
            :yaml:`Number of k along Fermi circle.`
        n_r
            :yaml:`Number of radial k perpendicular to Fermi circle.`
        dk_r
            :yaml:`Radial k grid spacing perpendicular to Fermi circle.`
            This is only relevant if `n_r` > 1, and should typically be ~ `T` / `vF`.
        m_star
            :yaml:`Longitudinal effective mass determining dispersion relation.`
            If `n_r` > 1, this controls the velocity and energy away from kF.
            The default infinite value sets a linear dispersion relation.
            Set to `kF` / `vF` for a parabolic dispersion relation.
        T
            :yaml:`Temperature of reference equilibrium state.`
            If `n_r` > 1, this controls the equilibrium Fermi distribution about
            which the change in occupations are referenced. It also controls the Fermi
            distributions used in defining the contacts and the `tau_p` relaxation.
            The `n_r` = 1 case does not resolve energies and is effectively T = 0.
        tau_p
            :yaml:`Momentum relaxation time.`
        tau_ee
            :yaml:`Electron-electron scattering time (momentum-conserving).`
        theta0
            :yaml:`Angle of first k-point.`
        r_c
            :yaml:`Cyclotron radius, corresponding to external magnetic field`.
            If infinite, disable magentic field (default).
        specularity
            :yaml:`Specularity of reflection at all surfaces.`
            Should be in the range of 0 for fully diffuse scattering,
            to 1 for perfectly specular reflection.
        """
        super().__init__()
        self.kF = kF
        self.vF = vF
        self.r_c = r_c
        self.tau_inv_p = 1.0 / tau_p
        self.tau_inv_ee = 1.0 / tau_ee

        # Initialize angular k grid:
        self.n_theta = n_theta
        self.theta0 = theta0
        dtheta = 2 * np.pi / n_theta
        theta = theta0 + torch.arange(n_theta, device=rc.device) * dtheta
        k_hat = torch.stack([theta.cos(), theta.sin()], dim=-1)

        # Initialize radial k grid (centered on kF):
        self.n_r = n_r
        self.dk_r = dk_r
        self.m_star = m_star
        self.T = T
        k_r = kF + dk_r * (torch.arange(n_r, device=rc.device) - 0.5 * (n_r - 1))
        v_r = vF + (k_r - kF) / m_star
        E_r = vF * (k_r - kF) + 0.5 * (k_r - kF).square() / m_star  # Ef = 0 WLOG
        wk = (2 / (2 * np.pi) ** 2) * (dtheta * k_r.repeat(n_theta))
        if n_r > 1:
            wk *= dk_r  # include radial grid spacing; n_r == 1 case is r-integrated
            if not dk_r:
                raise InvalidInputException("dk_r must be > 0 when n_r > 1")
            if not T:
                raise InvalidInputException("T must be > 0 when n_r > 1")
            assert vF * dk_r <= 2 * T  # Radial grid energy spacing should resolve T
            assert vF * dk_r * n_r >= 2 * T  # Energy range covers multiple T units

        self.initialize(
            wk=wk,
            nk=len(wk),
            n_bands=1,
            n_dim=2,
            process_grid=process_grid,
        )
        if self.group.size() > 1:
            raise NotImplementedError("FermiCircle does not parallelize over k")

        # Create theta grid on Fermi circle:
        self.k[:] = (k_hat[:, None] * k_r[:, None]).flatten(0, 1)
        self.v[:, 0] = (k_hat[:, None] * v_r[:, None]).flatten(0, 1)
        self.E[:, 0] = E_r.repeat(n_theta)
        self.specularity = specularity

        # Cached normalizations for collision intergal
        self.nk_inv = 1.0 / n_theta
        self.vv_inv = torch.linalg.inv(torch.einsum("...i, ...j -> ij", self.v, self.v))

        # Initialize F*drho/dk calculator, if needed:
        if np.isfinite(r_c):
            self.F_theta = self.vF / (self.r_c * dtheta)
            self.dt_max = 0.5 / abs(self.F_theta)
        else:
            self.F_theta = 0.0

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        attrs = cp_path.attrs
        attrs["kF"] = self.kF
        attrs["vF"] = self.vF
        attrs["n_theta"] = self.n_theta
        attrs["n_r"] = self.n_r
        attrs["dk_r"] = self.dk_r
        attrs["m_star"] = self.m_star
        attrs["T"] = self.T
        attrs["tau_p"] = (1.0 / self.tau_inv_p) if self.tau_inv_p else np.inf
        attrs["tau_ee"] = (1.0 / self.tau_inv_ee) if self.tau_inv_ee else np.inf
        attrs["r_c"] = self.r_c
        attrs["theta0"] = self.theta0
        attrs["specularity"] = self.specularity
        return list(attrs.keys())

    def initialize_fields(
        self, rho: torch.Tensor, params: dict[str, torch.Tensor], patch_id: int
    ) -> None:
        pass  # No spatially-varying / parameter sweep fields yet

    def get_contactor(
        self, n: torch.Tensor, **kwargs
    ) -> Callable[[float], torch.Tensor]:
        return Contactor(self, n, **kwargs)

    def get_reflector(self, n: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
        v = self.v[:, 0].unflatten(0, (self.n_theta, self.n_r))
        return Reflector(n, v, self.theta0, self.specularity)

    @stopwatch
    def rho_dot(self, rho: torch.Tensor, t: float, patch_id: int) -> torch.Tensor:
        result = self.rho_dot_collisions(rho)
        if self.F_theta:  # k-space advection due to magnetic fields:
            result += advect_theta(
                rho.unflatten(1, (self.n_theta, self.n_r)), self.F_theta
            ).flatten(1, 2)
        return result

    def rho_dot_collisions(self, rho: torch.Tensor) -> torch.Tensor:
        if not (self.tau_inv_p or self.tau_inv_ee):
            return torch.zeros_like(rho)  # no scattering

        # Compute stationary carrier density:
        rho_0 = rho.mean(dim=-1, keepdim=True)
        result = (rho_0 - rho) * (self.tau_inv_p + self.tau_inv_ee)

        # Compute moving equlibrium carrier density if needed:
        if self.tau_inv_ee:
            v = self.transport_velocity
            rho_v_sum = torch.einsum("...k, ki -> ...i", rho, v)
            rho_v = torch.einsum("...i, ij, kj -> ...k", rho_v_sum, self.vv_inv, v)
            result += rho_v * self.tau_inv_ee  # combines with rho_0 - rho above

        return result

    def get_observable_names(self) -> list[str]:
        return ["n"] if self.n_r == 1 else ["n", "E"]

    @cache
    def get_observables(self, t: float) -> torch.Tensor:
        observables = [torch.ones(self.nk_mine, device=rc.device)]  # for density
        if self.n_r > 1:
            observables.append(self.E.flatten())  # add energy
        return torch.stack(observables)

    def get_flux_indices(self) -> list[int]:
        return [0]


class Contactor:
    rho_contact: torch.Tensor  #: Cached constant contact distribution

    def __init__(self, fc: FermiCircle, n: torch.Tensor, *, dmu: float = 0.0) -> None:
        """Return contact distribution function for chemical potential shift `dmu`."""
        self.rho_contact = torch.empty((len(n), fc.rho0.numel()), device=rc.device)
        if fc.n_r == 1:
            self.rho_contact[:] = dmu / fc.vF  # include DOS factor for k_r integral
        else:
            self.rho_contact[:] = (fermi(fc.E, dmu, fc.T) - fermi(fc.E, 0.0, fc.T)).T

    def __call__(self, t):
        return self.rho_contact


class Reflector:
    """Reflect velocities accounting for specular and diffuse contributions.
    Specular reflections map reflected angle = incident angle (relative to normals),
    mapping reflected angle to finite grid using linear interpolation. Any small flux
    mismatch due to inexact angle mapping and diffuse contributions (specularity < 1)
    are mapped to a uniform incoming distribution to conserve net flux exactly."""

    def __init__(
        self,
        n: torch.Tensor,
        v: torch.Tensor,
        theta0: float,
        specularity: float,
    ) -> None:
        # Calculate linearly-interpolated reflection for theta0:
        angle_n = torch.atan2(n[:, 1], n[:, 0])
        dtheta0_reflected = 2 * (angle_n - theta0) + np.pi  # deflection of theta0
        n_theta, n_r, _ = v.shape
        x0_reflected = dtheta0_reflected * (n_theta / (2 * np.pi))
        i0_reflected = x0_reflected.floor().to(torch.long)
        w1 = x0_reflected - i0_reflected

        # Map the rest correspondingly:
        i0 = (i0_reflected[:, None] - torch.arange(n_theta, device=rc.device)) % n_theta
        i01 = torch.stack((i0, (i0 + 1) % n_theta))  # reflected theta interval indices
        v_n = (n[:, None] * v[:, 0]).sum(dim=-1)  # for one radius (len(n) x n_theta)
        outward = torch.where(v_n.flatten() > 0)[0]  # only need to reflect outward flux

        # Construct sparse matrix for remapping theta with linear interpolation:
        w1 = torch.repeat_interleave(w1, n_theta)[outward]
        w0 = 1.0 - w1
        n_flattened = v_n.numel()
        i_boundary = torch.arange(len(n), device=rc.device)[:, None]
        index_out = torch.cat((outward, outward))
        index_in = (i_boundary * n_theta + i01).view(2, -1)[:, outward].flatten()
        inward = torch.where(v_n.flatten()[index_in] < 0)[0]  # only compute inward flux
        self.specular_matrix = torch.sparse_coo_tensor(
            indices=torch.stack((index_in[inward], index_out[inward])),
            values=torch.cat((w0, w1))[inward] * specularity,
            size=(n_flattened, n_flattened),
        ).to_sparse_csr()

        # Compute flux in and out weights for diffuse scattering and norm correction:
        self.flux_out = torch.where(v_n > 0, v_n, 0.0)[..., None]  # add dim for radial
        self.flux_in = torch.where(v_n < 0, -v_n, 0.0)[..., None]
        norm_factor = 1 / self.flux_in.sum(dim=1, keepdim=True)
        self.flux_in *= norm_factor
        self.flux_out *= norm_factor

    def __call__(self, rho: torch.Tensor) -> torch.Tensor:
        # Compute specular contribution:
        n_boundary, n_theta, _ = self.flux_out.shape
        shape_nt_r = (n_boundary * n_theta, -1)  # flatten except n_r
        shape_n_t_r = (n_boundary, n_theta, -1)  # all dimensions separate
        out = (self.specular_matrix @ rho.reshape(shape_nt_r)).reshape(shape_n_t_r)
        # Fix difference between in and out fluxes diffusively:
        # (handles specularity < 1, and fixes norm errors for un-aligned walls)
        out += (rho.reshape(shape_n_t_r) * self.flux_out - out * self.flux_in).sum(
            dim=1, keepdim=True
        )
        return out.flatten(1, 2)  # recombine n_theta and n_r dimensions


def advect_theta(
    rho: torch.Tensor, F_theta: float, epsilon_sq: float = 0.0
) -> torch.Tensor:
    """Compute theta advection due to magnetic force.
    Applies smooth face-limiter and ensures upstream-only dependence of derivative."""
    # Compute +/- neighbor diffs and reconstruction to +/- faces along dim=0:
    drho = torch.stack((rho.roll(+1, dims=-2), rho.roll(-1, dims=-2))) - rho
    reconstruct = torch.tensor([+0.25, -0.25], device=rc.device).view(-1, 1, 1, 1)
    d = reconstruct * drho.diff(dim=0)
    lo, hi = drho.aminmax(dim=0, keepdim=True)
    rho_faces = rho + limiter(d, lo, hi, 0, epsilon_sq) * d
    if F_theta > 0.0:  # Use + face fluxes for +F.del calculation
        return F_theta * (rho_faces[0].roll(+1, dims=-2) - rho_faces[0])
    else:  # Use - face fluxes for -F.del calculation
        return F_theta * (rho_faces[1] - rho_faces[1].roll(-1, dims=-2))
