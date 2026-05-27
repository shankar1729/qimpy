from __future__ import annotations
from typing import Callable
from functools import cache

import numpy as np
import torch

from qimpy import rc
from qimpy.mpi import ProcessGrid
from qimpy.profiler import stopwatch
from qimpy.io import CheckpointPath, CheckpointContext, InvalidInputException
from ._angular_modes import AngularModes
from . import Material


class FermiCircleModes(Material):
    """Fermi-circle material in the angular-harmonic (modal) representation.

    Same physics as :class:`FermiCircle`, but the momentum-space (angular)
    degree of freedom is expanded in real Fourier harmonics up to order ``M``
    (the ``2M+1`` coefficients ``a_0, a_1, b_1, ...``) instead of ``N_theta``
    discrete directions. In this basis:

    * collisions are diagonal per harmonic (m=0 conserved, m=1 relaxes through
      tau_p, m>=2 through tau_p^-1 + tau_ee^-1) -- no quadrature, no MPI reduce;
    * the cyclotron (momentum-space) advection is an *exact* harmonic rotation
      omega_c G with no CFL limit;
    * real-space advection couples m <-> m+-1 and is carried out by the shared
      DG kernel as a coupled hyperbolic system (``coupling`` below), of which the
      discrete-ordinates per-channel advection is the diagonal special case.

    The harmonics are few and coupled, so they are never split across ranks
    (k-process-grid dimension must be 1); parallelism is purely spatial.
    """

    kF: float
    vF: float
    M: int  #: highest angular harmonic kept
    tau_inv_p: float
    tau_inv_ee: float
    r_c: float
    specularity: float
    k_speed: float  #: cyclotron rate omega_c = vF / r_c
    am: AngularModes
    coupling: AngularModes  #: flux operator consumed by the DG spatial advection

    def __init__(
        self,
        *,
        kF: float,
        vF: float,
        M: int,
        tau_p: float,
        tau_ee: float,
        r_c: float = np.inf,
        specularity: float = 1.0,
        process_grid: ProcessGrid,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ):
        """
        Initialize the modal Fermi-circle material.

        Parameters
        ----------
        kF
            :yaml:`Fermi wave vector in atomic units.`
        vF
            :yaml:`Fermi velocity in atomic units.`
        M
            :yaml:`Highest angular harmonic retained (state has 2M+1 modes).`
        r_c
            :yaml:`Cyclotron radius (external magnetic field); infinite disables it.`
            The cyclotron advection is exact in this basis (no time-step limit).
        specularity
            :yaml:`Specularity of boundary reflection (1 = specular).`
        """
        super().__init__()
        self.kF = kF
        self.vF = vF
        self.M = M
        self.r_c = r_c
        self.tau_inv_p = 1.0 / tau_p
        self.tau_inv_ee = 1.0 / tau_ee
        self.specularity = specularity
        dim = 2 * M + 1
        self.initialize(
            wk=1.0, nk=dim, n_bands=1, n_dim=2, process_grid=process_grid
        )
        if self.comm.size > 1:
            raise InvalidInputException(
                "FermiCircleModes couples harmonics, so the k process-grid "
                "dimension must be 1 (parallelize over space instead)."
            )
        self.am = AngularModes(M, vF, dtype=self.v.dtype)
        self.coupling = self.am
        self.rates = self.am.collision_rates(tau_p, tau_ee)  # (dim,) diagonal
        self.k_speed = (vF / r_c) if np.isfinite(r_c) else 0.0

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        attrs = cp_path.attrs
        attrs["kF"] = self.kF
        attrs["vF"] = self.vF
        attrs["M"] = self.M
        attrs["tau_p"] = (1.0 / self.tau_inv_p) if self.tau_inv_p else np.inf
        attrs["tau_ee"] = (1.0 / self.tau_inv_ee) if self.tau_inv_ee else np.inf
        attrs["r_c"] = self.r_c
        attrs["specularity"] = self.specularity
        return list(attrs.keys())

    def initialize_fields(
        self, rho: torch.Tensor, params: dict[str, torch.Tensor], patch_id: int
    ) -> None:
        pass

    def get_contactor(
        self, n: torch.Tensor, **kwargs
    ) -> Callable[[float], torch.Tensor]:
        return ModalContactor(self, n, **kwargs)

    def get_reflector(self, n: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
        return ModalReflector(self.am, n, self.specularity)

    @stopwatch
    def rho_dot(self, rho: torch.Tensor, t: float, patch_id: int) -> torch.Tensor:
        result = -self.rates * rho  # diagonal collisions (m=0 conserved)
        if self.k_speed:
            result = result + self.am.cyclotron(rho, self.k_speed)  # exact cyclotron
        return result

    def get_observable_names(self) -> list[str]:
        return ["n", "jx", "jy"]

    @cache
    def get_observables(self, t: float) -> torch.Tensor:
        # n = a_0 ; jx = (vF/2) a_1 ; jy = (vF/2) b_1  (with wk = 1)
        obs = torch.zeros((3, self.nk_mine), device=rc.device, dtype=self.v.dtype)
        obs[0, 0] = 1.0
        obs[1, 1] = 0.5 * self.vF
        obs[2, 2] = 0.5 * self.vF
        return obs


class ModalContactor:
    """Contact (Dirichlet) distribution in modes: an isotropic dmu shift plus a
    drift current set by the m=1 harmonic."""

    def __init__(
        self, fc: FermiCircleModes, n: torch.Tensor, *, dmu: float = 0.0, vD: float = 0.0
    ) -> None:
        phi = torch.atan2(n[:, 1], n[:, 0])
        contact = torch.zeros((len(n), fc.am.dim), device=rc.device, dtype=n.dtype)
        contact[:, 0] = dmu
        contact[:, 1] = -(vD / fc.vF) * torch.cos(phi)
        contact[:, 2] = -(vD / fc.vF) * torch.sin(phi)
        self.rho_contact = contact

    def __call__(self, t):
        return self.rho_contact


class ModalReflector:
    """Boundary reflection in modes, specular fraction ``s`` plus diffuse (1-s).

    Specular: a direction theta reflects to 2*phi_n - theta + pi (v -> v-2(v.n)n),
    so the distribution maps rho(theta) -> rho(2*phi_n + pi - theta) -- flip the
    sin components, then rotate harmonic m by m*(2*phi_n + pi). Exact at any wall
    angle (no on-grid constraint).

    Diffuse: the incident flux is re-emitted with an isotropic phase-space
    density D (only the m=0 mode), with D fixed by zero net normal current at
    the wall. Writing the upwind flux as f* = A_n^+ u^- + A_n^- u^+ with
    A_n^+- = (A_n +- |A_n|)/2, the m=0 (mass) flux vanishes pointwise when
        D = -(A_n^+ u^-)_0 / (A_n^- e_0)_0,
    so the combined ghost  s * specular + (1-s) * D e_0  conserves mass for any s
    and any normal direction.
    """

    def __init__(self, am: AngularModes, n: torch.Tensor, specularity: float) -> None:
        self.am = am
        self.s = float(specularity)
        self.phi = torch.atan2(n[:, 1], n[:, 0])      # outward-normal angle per node
        self.angle = 2 * self.phi + np.pi             # specular rotation 2*phi + pi

    def _specular(self, rho: torch.Tensor) -> torch.Tensor:
        flipped = rho.clone()
        for m in range(1, self.am.M + 1):
            flipped[..., 2 * m] = -flipped[..., 2 * m]  # negate sin(m theta) coeffs
        return self.am.rotate(flipped, self.angle)

    def __call__(self, rho: torch.Tensor) -> torch.Tensor:
        spec = self._specular(rho)
        if self.s >= 1.0:
            return spec
        am = self.am
        nx, ny = torch.cos(self.phi), torch.sin(self.phi)        # (Nsel,)
        # outgoing m=0 flux (A_n^+ rho)_0 = 1/2[(A_n rho)_0 + (|A_n| rho)_0]
        An_rho_0 = nx * (rho @ am.Ax[0]) + ny * (rho @ am.Ay[0])  # (..., Nsel)
        Ap_rho_0 = 0.5 * (An_rho_0 + am.abs_flux(rho, self.phi)[..., 0])
        # inflow m=0 capacity (A_n^- e0)_0 = 1/2[(A_n e0)_0 - (|A_n| e0)_0]  (<0)
        e0 = torch.zeros_like(rho); e0[..., 0] = 1.0
        An_e0_0 = nx * am.Ax[0, 0] + ny * am.Ay[0, 0]
        Am_e0_0 = 0.5 * (An_e0_0 - am.abs_flux(e0, self.phi)[..., 0])
        D = -Ap_rho_0 / Am_e0_0                                  # (..., Nsel)
        diff = torch.zeros_like(rho); diff[..., 0] = D
        return self.s * spec + (1.0 - self.s) * diff
