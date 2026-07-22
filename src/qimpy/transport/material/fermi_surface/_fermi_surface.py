"""FermiSurface: the physical model (a degenerate 2DEG Fermi liquid with e-e
collisions), independent of its numerical k-representation.

The model owns the physics -- kF, vF, m*, E_F, the angular/radial modal bases,
the microscopic e-e operator (EEScattering), the phenomenological relaxation
rates and the cyclotron generator.  The k-space discretization is a swappable
child `representation` (add_child_one_of):

    representation: {delta_k: {}}        # Fermi-circle / small-drift (default)
    representation: {cartesian: {...}}   # fixed 2D k-grid, large-drift full-f

The collision is applied representation-agnostically:
    a = rep.to_modes(rho);  a_dot = -rates*a + ee.a_dot(a) + k_speed*G(a);
    rho_dot = rep.from_modes(a_dot)
wrapped as `rep.apply_collision(rho, self._modal_collision)`.  This works for
both backends because `EEScattering.a_dot` is a pure modal operator, agnostic to
whether the coefficients came from the lab-centred tensor transform (delta-k) or
the drift-centred per-cell projection (cartesian).
"""
from __future__ import annotations
from typing import Callable, Optional, Union
import numpy as np
import torch

from qimpy import rc, TreeNode
from qimpy.mpi import ProcessGrid
from qimpy.profiler import stopwatch
from qimpy.io import CheckpointPath, CheckpointContext, InvalidInputException
from .scattering import EEScattering
from .._material import Material
from ._representation import DeltaK
from ._cartesian import Cartesian


# ----------------------------------------------------------------------------
# Angular basis: real Fourier on [0, 2*pi), nodal storage at midpoint nodes
# ----------------------------------------------------------------------------
class AngularBasis:
    """Real Fourier transforms with nodal (delta-k) storage.

    Mode ordering ``(a_0, a_1, b_1, ..., a_M, b_M)`` for ``2M+1`` modes; nodes are
    the midpoint grid ``theta_q = 2*pi*(q+1/2)/Nq``, symmetric under
    ``theta->-theta`` and (even Nq) ``theta->pi-theta`` -- the wall reflections.
    """

    def __init__(self, M: int, n_quad: int | None = None,
                 dtype: torch.dtype = torch.float64,
                 device: torch.device | None = None) -> None:
        Nq = (2 * M + 1) if n_quad is None else n_quad
        if Nq < 2 * M + 1:
            raise ValueError(f"AngularBasis n_quad={Nq} < 2M+1={2*M+1}")
        self.M = M
        self.dim = 2 * M + 1
        self.N_theta = Nq
        theta = 2.0 * np.pi * (np.arange(Nq) + 0.5) / Nq
        Tfm = np.zeros((Nq, self.dim))
        Tfm[:, 0] = 1.0
        for m in range(1, M + 1):
            Tfm[:, 2 * m - 1] = np.cos(m * theta)
            Tfm[:, 2 * m]     = np.sin(m * theta)
        Ttm = np.zeros((self.dim, Nq))
        Ttm[0, :] = 1.0 / Nq
        for m in range(1, M + 1):
            Ttm[2 * m - 1, :] = (2.0 / Nq) * np.cos(m * theta)
            Ttm[2 * m, :]     = (2.0 / Nq) * np.sin(m * theta)
        G = np.zeros((self.dim, self.dim))
        for m in range(1, M + 1):
            G[2 * m - 1, 2 * m] = -m
            G[2 * m,     2 * m - 1] = +m
        dev = device or rc.device
        self.theta        = torch.as_tensor(theta, dtype=dtype, device=dev)
        self.T_from_modes = torch.as_tensor(Tfm,   dtype=dtype, device=dev)
        self.T_to_modes   = torch.as_tensor(Ttm,   dtype=dtype, device=dev)
        self.G            = torch.as_tensor(G,     dtype=dtype, device=dev)


# ----------------------------------------------------------------------------
# Radial basis: polynomials in xi orthonormal under  w(xi) = (1/4T) sech^2(xi/2)
# ----------------------------------------------------------------------------
class RadialBasis:
    """Polynomial transforms orthonormal under ``w(xi) = (1/4T) sech^2(xi/2)``.
    ``Nr == 1`` collapses to the identity at ``xi = 0`` (pure Fermi circle)."""

    def __init__(self, Nr: int, T_temp: float = 1.0, xi_max: float = 6.0,
                 dtype: torch.dtype = torch.float64,
                 device: torch.device | None = None) -> None:
        self.Nr = Nr; self.T_temp = T_temp; self.xi_max = xi_max
        dev = device or rc.device
        if Nr == 1:
            self.xi           = torch.zeros(1, dtype=dtype, device=dev)
            self.quad_w       = torch.ones(1,  dtype=dtype, device=dev)
            self.T_from_modes = torch.ones((1, 1), dtype=dtype, device=dev)
            self.T_to_modes   = torch.ones((1, 1), dtype=dtype, device=dev)
            return
        x_std, w_std = np.polynomial.legendre.leggauss(Nr)
        xi  = xi_max * x_std
        w_x = xi_max * w_std
        w_eq = (1.0 / (4.0 * T_temp)) / np.cosh(0.5 * xi) ** 2
        w_q  = w_x * w_eq
        V  = np.vander(xi / xi_max, Nr, increasing=True)
        Vw = V * np.sqrt(w_q)[:, None]
        _Q, R = np.linalg.qr(Vw)
        sgn = np.sign(np.diag(R)); sgn[sgn == 0] = 1.0
        R = sgn[:, None] * R
        Tfm = V @ np.linalg.solve(R, np.eye(Nr))
        Ttm = Tfm.T * w_q
        err = float(np.max(np.abs(Ttm @ Tfm - np.eye(Nr))))
        if err > 1e-10:
            raise RuntimeError(f"RadialBasis orthonormality failed (err={err:.2e})")
        self.xi           = torch.as_tensor(xi,  dtype=dtype, device=dev)
        self.quad_w       = torch.as_tensor(w_q, dtype=dtype, device=dev)
        self.T_from_modes = torch.as_tensor(Tfm, dtype=dtype, device=dev)
        self.T_to_modes   = torch.as_tensor(Ttm, dtype=dtype, device=dev)


# ----------------------------------------------------------------------------
# FermiSurface model
# ----------------------------------------------------------------------------
class FermiSurface(Material):
    """Degenerate-2DEG Fermi-liquid transport model; the k-representation is a
    swappable child (`delta_k` default, or `cartesian` for large drift)."""

    kF: float; vF: float; m_star: float; mu: float
    M_theta: int; Nr: int; T_temp: float; xi_max: float
    tau_inv_p: float; tau_inv_ee: float; r_c: float; specularity: float
    k_speed: float
    angular: AngularBasis; radial: RadialBasis
    representation: object

    def __init__(
        self, *, kF: float, vF: float, M_theta: int,
        Nr: int = 1, T: float = 1.0, xi_max: float = 6.0,
        tau_p: float = np.inf, tau_ee: float = np.inf,
        r_c: float = np.inf, specularity: float = 1.0,
        delta_k: Optional[Union[dict, DeltaK]] = None,
        cartesian: Optional[Union[dict, Cartesian]] = None,
        ee_scattering: Optional[Union[EEScattering, dict]] = None,
        process_grid: ProcessGrid,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        super().__init__()
        self.kF, self.vF = kF, vF
        self.m_star = kF / vF
        self.mu = 0.5 * kF * vF                              # E_F (degenerate)
        self.M_theta, self.Nr = M_theta, Nr
        self.T_temp, self.xi_max = T, xi_max
        self.r_c = r_c
        self.tau_inv_p  = 1.0 / tau_p
        self.tau_inv_ee = 1.0 / tau_ee
        self.specularity = specularity
        self.k_speed = (vF / r_c) if np.isfinite(r_c) else 0.0

        # modal bases (shared by both representations and the e-e operator)
        dtype = torch.get_default_dtype()
        N_theta = -(-(2 * M_theta + 1) // 4) * 4             # even, multiple of 4, >= 2M+1
        self.angular = AngularBasis(M_theta, n_quad=N_theta, dtype=dtype)
        self.radial  = RadialBasis(Nr, T_temp=T, xi_max=xi_max, dtype=dtype)

        # k-representation (builds the grid; delta-k is the default first option)
        self.add_child_one_of(
            "representation", checkpoint_in,
            TreeNode.ChildOptions("delta-k", DeltaK, delta_k,
                                  fermi_surface=self, process_grid=process_grid),
            TreeNode.ChildOptions("cartesian", Cartesian, cartesian,
                                  fermi_surface=self, process_grid=process_grid),
            have_default=True,
        )
        rep = self.representation
        self.initialize(wk=rep.wk, nk=rep.Nk, n_bands=1, n_dim=2,
                        process_grid=process_grid)
        if self.comm.size > 1:
            raise InvalidInputException(
                "FermiSurface couples k-channels; the k process-grid dimension "
                "must be 1 (parallelize over space)."
            )
        self.v = rep.v                                       # transport velocity

        # phenomenological per-mode collision rates in flattened (n_r, m) ordering
        ang = np.zeros(self.angular.dim)
        for m in range(1, M_theta + 1):
            ang[2 * m - 1] = ang[2 * m] = (
                self.tau_inv_p if m == 1 else (self.tau_inv_p + self.tau_inv_ee))
        rad = np.zeros(Nr); rad[1:] = self.tau_inv_ee
        rates = ang[None, :] + rad[:, None]
        rates[0, 0] = 0.0                                    # particle conservation
        self.rates_modal = torch.as_tensor(
            rates.reshape(-1), dtype=self.v.dtype, device=rc.device)

        # microscopic e-e collisions (replaces the tau_ee placeholder)
        if (ee_scattering is not None) or checkpoint_in.member("ee_scattering"):
            if np.isfinite(tau_ee):
                raise InvalidInputException(
                    "Specify either phenomenological tau_ee or microscopic"
                    " ee_scattering, not both")
            self.add_child("ee_scattering", EEScattering, ee_scattering,
                           checkpoint_in, fermi_surface=self)

    def __getattr__(self, name):
        # Expose the modal transforms ONLY when the representation has them, so
        # hasattr(material, "to_modes") is True for delta-k (FiniteVolume de-aliases
        # its modal storage) and False for the Cartesian grid (which must NOT be
        # run through the modal de-alias path).  __getattr__ runs only on normal
        # lookup miss, so it never shadows real attributes.
        if name in ("to_modes", "from_modes"):
            rep = self.__dict__.get("representation")
            if rep is not None and hasattr(rep, name):
                return getattr(rep, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}")

    @property
    def transport_velocity(self) -> torch.Tensor:
        return self.v

    # ---- the representation-agnostic modal collision (the physics) ----
    def _modal_collision(self, a: torch.Tensor,
                         te2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Modal collision.  ``te2`` (optional, per-cell broadcastable) scales
        the microscopic e-e operator by (T_e/T)^2 -- the leading local-
        temperature law for ALL its blocks (L, allowed-Q, C alike; the
        particle-hole-forbidden Q sector's (T_e/T)^3 is O(T/E_F)-small).  The
        phenomenological tau_p/tau_ee rates are user-set constants and are NOT
        rescaled; cyclotron is magnetic, not collisional."""
        a_dot = -self.rates_modal * a
        if hasattr(self, "ee_scattering"):
            ee_term = self.ee_scattering.a_dot(a)
            if te2 is not None:
                ee_term = ee_term * te2.to(ee_term.dtype)
            a_dot = a_dot + ee_term
        if self.k_speed:
            Nr, dim_t = self.Nr, self.angular.dim
            a4 = a.reshape(*a.shape[:-1], Nr, dim_t)
            Ga4 = torch.einsum("dc,...nc->...nd", self.angular.G, a4)
            a_dot = a_dot + self.k_speed * Ga4.reshape(*a.shape)
        return a_dot

    @stopwatch
    def rho_dot(self, rho: torch.Tensor, t: float, patch_id: int) -> torch.Tensor:
        if (self.rates_modal.abs().sum() == 0 and self.k_speed == 0.0
                and not hasattr(self, "ee_scattering")):
            return torch.zeros_like(rho)                     # ballistic, no field
        return self.representation.apply_collision(rho, self._modal_collision)

    # ---- observables / boundaries: delegate to the representation ----
    def get_observable_names(self) -> list[str]:
        return ["density"]                               # row-0 weight (contact current)

    @stopwatch
    def get_observables(self, t: float) -> torch.Tensor:
        return self.representation.get_density_weight()[None, :]   # (1, Nk)

    def get_cell_scalar_names(self) -> list[str]:
        return self.representation.get_cell_scalar_names()

    @stopwatch
    def get_cell_scalars(self, rho: torch.Tensor, t: float) -> torch.Tensor:
        return self.representation.get_cell_scalars(rho)  # full local moments (K, n)

    def get_flux_names(self) -> list[str]:
        return self.representation.get_flux_names()      # fluxes (face-centred)

    def get_flux_weights(self) -> torch.Tensor:
        return self.representation.get_flux_weights()

    def get_contactor(self, n: torch.Tensor, **kwargs) -> Callable:
        return self.representation.get_contactor(n, **kwargs)

    def get_reflector(self, n: torch.Tensor) -> Callable:
        return self.representation.get_reflector(n)

    def initialize_fields(self, rho, params, patch_id) -> None:
        pass

    def _save_checkpoint(self, cp_path: CheckpointPath, context: CheckpointContext) -> list[str]:
        a = cp_path.attrs
        a["kF"], a["vF"] = self.kF, self.vF
        a["M_theta"], a["Nr"] = self.M_theta, self.Nr
        a["T"], a["xi_max"] = self.T_temp, self.xi_max
        a["tau_p"] = (1.0 / self.tau_inv_p) if self.tau_inv_p else np.inf
        a["tau_ee"] = (1.0 / self.tau_inv_ee) if self.tau_inv_ee else np.inf
        a["r_c"], a["specularity"] = self.r_c, self.specularity
        return list(a.keys())
