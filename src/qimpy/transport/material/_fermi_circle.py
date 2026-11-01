from __future__ import annotations
from typing import Callable
from functools import cache

import numpy as np
import torch

from qimpy import rc
from qimpy.mpi import ProcessGrid
from qimpy.profiler import stopwatch
from qimpy.io import CheckpointPath, CheckpointContext
from qimpy.transport.advect import Advect, N_GHOST
from . import Material


class FermiCircle(Material):
    """Fermi-circle model material suitable for graphene and 2DEGs."""

    kF: float  #: Fermi wave-vector
    vF: float  #: Fermi velocity
    tau_inv_p: float  #: Momentum relaxation rate
    tau_inv_ee: float  #: Electron internal scattering rate (momentum-conserving)
    theta0: float  #: Initial angle in fermi circle grid
    specularity: float  #: Specularity of reflection at all boundaries
    r_c: float  #: Cyclotron radius for extenral magnetic field (disabled if infinity)
    F_theta: float  #: Angular force in grid coordinates
    advect: torch.jit.ScriptModule  #: Momentum-space advection logic

    def __init__(
        self,
        *,
        kF: float,
        vF: float,
        n_theta: int,
        n_r: int = 1,
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
        self.initialize(
            wk=torch.tensor([1.0 / n_theta], device=rc.device),
            nk=n_theta,
            n_bands=1,
            n_dim=2,
            process_grid=process_grid,
        )
        if self.group.size() > 1:
            raise NotImplementedError("FermiCircle does not parallelize over k")
        assert n_r == 1  # TODO: merge n_r support from FermiSurface

        # Create theta grid on Fermi circle:
        dtheta = 2 * np.pi / n_theta
        theta = theta0 + torch.arange(n_theta, device=rc.device) * dtheta
        k_hat = torch.stack([theta.cos(), theta.sin()], dim=-1)
        self.k[:] = k_hat * kF
        self.v[:] = k_hat[:, None] * vF
        self.theta0 = theta0
        self.specularity = specularity

        # Cached normalizations for collision intergal
        self.nk_inv = 1.0 / n_theta
        self.vv_inv = torch.linalg.inv(torch.einsum("...i, ...j -> ij", self.v, self.v))

        # Initialize F*drho/dk calculator, if needed:
        if np.isfinite(r_c):
            self.F_theta = self.vF / (self.r_c * dtheta)
            self.dt_max = 0.5 / abs(self.F_theta)
            self.advect = torch.jit.script(Advect())
            self.riemann_mask = torch.tensor(
                (1.0, 0.0) if self.F_theta > 0 else (0.0, 1.0), device=rc.device
            ).view(
                1, 1, 1, -1
            )  # faster constant-F equivalent to get_riemann_mask
        else:
            self.F_theta = 0.0

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        attrs = cp_path.attrs
        attrs["kF"] = self.kF
        attrs["vF"] = self.vF
        attrs["n_theta"] = len(self.v)
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
        return SpecularReflector(n, self.v[:, 0], self.theta0, self.specularity)

    @stopwatch
    def rho_dot(self, rho: torch.Tensor, t: float, patch_id: int) -> torch.Tensor:
        result = self.rho_dot_collisions(rho)
        if self.F_theta:
            # k-space advection due to magnetic fields:
            result += self.accumulate_edges(
                *self.advect(
                    self.pad_ghost(rho) * self.F_theta,
                    self.riemann_mask,
                    -1,
                    True,
                    (True, True),
                )
            )
        return result

    def pad_ghost(self, rho: torch.Tensor) -> torch.Tensor:
        """Pad by ghost zones for monetum-space advection."""
        rho_padded = torch.nn.functional.pad(rho, (N_GHOST, N_GHOST))
        rho_padded[..., N_GHOST - 1] = rho[..., -1]
        rho_padded[..., -N_GHOST] = rho[..., 0]
        return rho_padded

    def accumulate_edges(
        self, out: torch.Tensor, outL: torch.Tensor, outR: torch.Tensor
    ) -> torch.Tensor:
        """Accumulate momentum-space advection edge contributions `outL` and `outR`
        into `out`, and return `out` for convenience."""
        out[..., 0] += outR
        out[..., -1] += outL
        return out

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
        return ["n", "jx", "jy"]  # density and fluxes

    @cache
    def get_observables(self, t: float) -> torch.Tensor:
        return torch.cat(
            (
                torch.ones((1, self.nk_mine), device=rc.device),
                self.transport_velocity.T,
            ),
            dim=0,
        )

    def get_flux_indices(self) -> list[int]:
        return [0]


class Contactor:
    rho_contact: torch.Tensor  #: Cached constant contact distribution

    def __init__(
        self, fc: FermiCircle, n: torch.Tensor, *, dmu: float = 0.0, vD: float = 0.0
    ) -> None:
        """Return contact distribution function for specified chemical potential
        shift and drift velocity. Note that positive vD corresponds to current
        flowing into the device (along -n), while negative vD flows out (along +n)."""
        v_hat = fc.transport_velocity / fc.vF
        self.rho_contact = dmu - (n @ v_hat.T) * (vD / fc.vF)  # TODO: check

    def __call__(self, t):
        # TODO: add time dependence options
        return self.rho_contact


class SpecularReflector:
    """Reflect velocities specularly i.e. with reflection angle = incidence angle.
    Optionally, account for diffuse scattering by replacing portion of the flux with a
    uniform distribution at all angles (matching total incoming and outgoing flux)."""

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
        n_theta = v.shape[0]
        x0_reflected = dtheta0_reflected * (n_theta / (2 * np.pi))
        i0_reflected = x0_reflected.floor().to(torch.long)
        w1 = x0_reflected - i0_reflected

        # Map the rest correspondingly:
        i0 = (i0_reflected[:, None] - torch.arange(n_theta, device=rc.device)) % n_theta
        i01 = torch.stack((i0, (i0 + 1) % n_theta))  # reflected theta interval indices
        v_n = (n[:, None] * v).sum(dim=-1)
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
        self.flux_out = torch.where(v_n > 0, v_n, 0.0)
        self.flux_in = torch.where(v_n < 0, -v_n, 0.0)
        norm_factor = 1 / self.flux_in.sum(dim=-1, keepdim=True)
        self.flux_in *= norm_factor
        self.flux_out *= norm_factor

    def __call__(self, rho: torch.Tensor) -> torch.Tensor:
        # Compute specular contribution:
        out = (self.specular_matrix @ rho.flatten()).reshape(rho.shape)
        # Fix difference between in and out fluxes diffusively:
        # (handles specularity < 1, and fixes norm errors for un-aligned walls)
        out -= (out * self.flux_in).sum(dim=-1, keepdim=True)
        out += (rho * self.flux_out).sum(dim=-1, keepdim=True)
        return out
