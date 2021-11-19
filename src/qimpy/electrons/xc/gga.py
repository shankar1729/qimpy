"""Internal GGA implementations."""
# List exported symbols for doc generation
__all__ = ["X_PBE", "C_PBE"]

from .functional import Functional
from .lda import get_rs, _C_PW, SpinInterpolate1, SpinInterpolate3
import numpy as np
import torch


class X_PBE(Functional):
    """PBE/PBEsol exchange."""

    def __init__(self, sol: bool, scale_factor: float = 1.0) -> None:
        super().__init__(
            has_exchange=True,
            needs_sigma=True,
            scale_factor=scale_factor,
            _apply=torch.jit.script(SpinScaled(_X_PBE, sol)),
        )
        self.report(f'PBE{"sol" if sol else ""} GGA exchange')


class C_PBE(Functional):
    """PBE/PBEsol correlation."""

    def __init__(self, sol: bool, scale_factor: float = 1.0) -> None:
        super().__init__(
            has_correlation=True,
            needs_sigma=True,
            scale_factor=scale_factor,
            _apply=torch.jit.script(SpinUnpolarized(_C_PBE, SpinInterpolate1, sol)),
            _apply_spin=torch.jit.script(SpinPolarized(_C_PBE, SpinInterpolate3, sol)),
        )
        self.report(f'PBE{"sol" if sol else ""} GGA correlation')


# ----- Internal exchange/kinetic implementations -----


class SpinScaled(torch.nn.Module):
    """Common wrapper for spin-scaled (exchange-like) GGA functionals."""

    def __init__(self, Compute: type, *args) -> None:
        super().__init__()
        self.compute = Compute(*args)

    def forward(
        self,
        n: torch.Tensor,
        sigma: torch.Tensor,
        lap: torch.Tensor,
        tau: torch.Tensor,
        requires_grad: bool,
        scale_factor: float,
    ) -> float:
        n_spins = n.shape[0]
        n.requires_grad_(requires_grad)
        sigma.requires_grad_(requires_grad)
        rs = ((n_spins * 4.0 * np.pi / 3.0) * n) ** (-1.0 / 3)  # rs for each spin
        s2 = ((18.0 * np.pi) ** (-2.0 / 3)) * sigma[::2] * (rs / n).square()
        e = self.compute(rs, s2)
        E = (e * n).sum() * scale_factor
        if requires_grad:
            E.backward()  # updates n.grad and sigma.grad
        return E.item()


def get_e_slater(rs: torch.Tensor) -> torch.Tensor:
    """Compute per-particle slater exchange energy."""
    return (-0.75 * ((1.5 / np.pi) ** (2.0 / 3))) / rs


class _X_PBE(torch.nn.Module):
    """Internal JIT-friendly implementation of PBE/PBEsol exchange."""

    mu: torch.jit.Final[float]

    def __init__(self, sol: bool) -> None:
        super().__init__()
        self.mu = 10.0 / 81 if sol else 0.2195149727645171

    def forward(self, rs: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        kappa = 0.804
        F = (1.0 + kappa) - kappa / (1.0 + (self.mu / kappa) * s2)  # GGA enhancement
        return F * get_e_slater(rs)


# ----- Internal correlation implementations -----


class SpinUnpolarized(torch.nn.Module):
    """Unpolarized GGA correlation wrapper."""

    def __init__(self, Get_ec: type, *args) -> None:
        super().__init__()
        self.get_ec = Get_ec(*args)

    def forward(
        self,
        n: torch.Tensor,
        sigma: torch.Tensor,
        lap: torch.Tensor,
        tau: torch.Tensor,
        requires_grad: bool,
        scale_factor: float,
    ) -> float:
        n.requires_grad_(requires_grad)
        sigma.requires_grad_(requires_grad)
        # Compute dimensionless parameters of correlation functionals:
        n_tot = n[0]
        rs = get_rs(n_tot)
        zeta = torch.zeros_like(rs)
        g = torch.ones_like(rs)
        sigma_tot = sigma[0]
        t2 = ((np.pi / 96) ** (2.0 / 3)) * sigma_tot * rs / n_tot.square()
        # Compute energy:
        E = (n_tot * self.get_ec(rs, zeta, g, t2)).sum() * scale_factor
        if requires_grad:
            E.backward()  # updates n.grad and sigma.grad
        return E.item()


class SpinPolarized(torch.nn.Module):
    """Polarized GGA correlation wrapper."""

    def __init__(self, Get_ec: type, *args) -> None:
        super().__init__()
        self.get_ec = Get_ec(*args)

    def forward(
        self,
        n: torch.Tensor,
        sigma: torch.Tensor,
        lap: torch.Tensor,
        tau: torch.Tensor,
        requires_grad: bool,
        scale_factor: float,
    ) -> float:
        n.requires_grad_(requires_grad)
        sigma.requires_grad_(requires_grad)
        # Compute dimensionless parameters of correlation functionals:
        n_tot = n[0] + n[1]
        rs = get_rs(n_tot)
        zeta = (n[0] - n[1]) / n_tot
        g = 0.5 * ((1.0 + zeta) ** (2.0 / 3) + (1.0 - zeta) ** (2.0 / 3))
        sigma_tot = sigma[0] + 2 * sigma[1] + sigma[2]
        t2 = ((np.pi / 96) ** (2.0 / 3)) * sigma_tot * rs / (g * n_tot).square()
        # Compute per-particle energy and total:
        E = (n_tot * self.get_ec(rs, zeta, g, t2)).sum() * scale_factor
        if requires_grad:
            E.backward()  # updates n.grad and sigma.grad
        return E.item()


def PW91_H0(
    gamma: float, beta: float, g3: torch.Tensor, t2: torch.Tensor, ec_unif: torch.Tensor
) -> torch.Tensor:
    """H0 function (equations 13, 14) of PW91.
    Same as the H function (equations 7, 8) of PBE.
    Using a mixed notation, picking the shortest of both references: using g
    from PW91 (phi in PBE) and gamma from PBE (beta^2/(2*alpha) in PW91).
    """
    beta_by_gamma = beta / gamma
    At2 = t2 * beta_by_gamma / (torch.exp(-ec_unif / (gamma * g3)) - 1)
    frac = (1.0 + At2) / (1.0 + At2 * (1.0 + At2))
    return gamma * g3 * (1.0 + beta_by_gamma * t2 * frac).log()


class _C_PBE(torch.nn.Module):
    """Internal JIT-friendly implementation of PBE/PBEsol correlation."""

    beta: torch.jit.Final[float]
    gamma: torch.jit.Final[float]

    def __init__(self, SpinInterpolate: type, sol: bool):
        super().__init__()
        self.ec_pw = SpinInterpolate(_C_PW)
        self.beta = 0.046 if sol else 0.06672455060314922
        self.gamma = (1.0 - np.log(2.0)) / (np.pi ** 2)

    def forward(
        self, rs: torch.Tensor, zeta: torch.Tensor, g: torch.Tensor, t2: torch.Tensor
    ) -> torch.Tensor:
        ec_unif = self.ec_pw(rs, zeta)  # underlying LDA correlation
        return ec_unif + PW91_H0(self.gamma, self.beta, g ** 3, t2, ec_unif)
