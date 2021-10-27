"""Internal GGA implementations."""
# List exported symbols for doc generation
__all__ = ["SpinScaled", "SpinInterpolated", "X_PBE", "C_PBE"]

from .functional import Functional
from .lda import C_PW
import numpy as np
import torch
from abc import abstractmethod


class SpinScaled(Functional):
    """Abstract base class of spin-scaled (exchange, KE) functionals."""

    def __init__(self, **kwargs) -> None:
        super().__init__(needs_sigma=True, **kwargs)

    def __call__(
        self, n: torch.Tensor, sigma: torch.Tensor, lap: torch.Tensor, tau: torch.Tensor
    ) -> float:
        n_spins = n.shape[0]
        n.requires_grad_()
        sigma.requires_grad_()
        rs = ((n_spins * 4.0 * np.pi / 3.0) * n) ** (-1.0 / 3)  # rs for each spin
        s2 = ((18.0 * np.pi) ** (-2.0 / 3)) * sigma[::2] * (rs / n).square()
        e = self.compute(rs, s2)
        E = (e * n).sum() * self.scale_factor
        E.backward()  # updates n.grad and sigma.grad
        return E.item()

    @abstractmethod
    def compute(self, rs: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        """Compute energy (per-particle) of spin-scaled functional."""


class X_PBE(SpinScaled):
    """PBE/PBEsol exchange."""

    __slots__ = ("sol",)
    sol: bool  # PBEsol if True; PBE otherwise

    def __init__(self, sol: bool, scale_factor: float = 1.0) -> None:
        super().__init__(
            has_exchange=True,
            scale_factor=scale_factor,
            name=f'PBE{"sol" if sol else ""} GGA exchange',
        )
        self.sol = sol

    def compute(self, rs: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        kappa = 0.804
        mu_by_kappa = (10.0 / 81 if self.sol else 0.2195149727645171) / kappa
        eSlater = (-0.75 * ((1.5 / np.pi) ** (2.0 / 3))) / rs
        F = (1.0 + kappa) - kappa / (1.0 + mu_by_kappa * s2)  # GGA enhancement
        return eSlater * F


class SpinInterpolated(Functional):
    """Abstract base class of spin-interpolated (correlation) functionals."""

    def __init__(self, **kwargs) -> None:
        super().__init__(needs_sigma=True, **kwargs)

    def __call__(
        self, n: torch.Tensor, sigma: torch.Tensor, lap: torch.Tensor, tau: torch.Tensor
    ) -> float:
        n_spins = n.shape[0]
        n.requires_grad_()
        sigma.requires_grad_()
        # Compute dimensionless parameters of correlation functionals:
        n_tot = n.sum(dim=0)
        rs = ((4.0 * np.pi / 3.0) * n_tot) ** (-1.0 / 3)
        if n_spins == 2:
            zeta = (n[0] - n[1]) / n_tot
            g = 0.5 * ((1.0 + zeta) ** (2.0 / 3) + (1.0 - zeta) ** (2.0 / 3))
            sigma_tot = sigma[0] + 2 * sigma[1] + sigma[2]
        else:
            zeta = torch.zeros(1, dtype=n.dtype, device=n.device)
            g = torch.ones(1, dtype=n.dtype, device=n.device)
            sigma_tot = sigma[0]
        t2 = ((np.pi / 96) ** (2.0 / 3)) * sigma_tot * rs / (g * n_tot).square()
        # Compute per-particle energy and total:
        e = self.compute(rs, zeta, g, t2)
        E = (e * n_tot).sum() * self.scale_factor
        E.backward()  # updates n.grad and sigma.grad
        return E.item()

    @abstractmethod
    def compute(
        self, rs: torch.Tensor, zeta: torch.Tensor, g: torch.Tensor, t2: torch.Tensor
    ) -> torch.Tensor:
        """Compute energy (per-particle) of spin-interpolated functional."""


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


class C_PBE(SpinInterpolated):
    """PBE/PBEsol correlation."""

    __slots__ = ("sol", "_pw")
    sol: bool  #: PBEsol if True; PBE otherwise
    _pw: C_PW  #: PW LDA correlation evaluator

    def __init__(self, sol: bool, scale_factor: float = 1.0) -> None:
        super().__init__(
            has_correlation=True,
            scale_factor=scale_factor,
            name=f'PBE{"sol" if sol else ""} GGA correlation',
        )
        self.sol = sol
        self._pw = C_PW(high_precision=True, helper=True)

    def compute(
        self, rs: torch.Tensor, zeta: torch.Tensor, g: torch.Tensor, t2: torch.Tensor
    ) -> torch.Tensor:
        beta = 0.046 if self.sol else 0.06672455060314922
        gamma = (1.0 - np.log(2.0)) / (np.pi ** 2)
        ec_unif = self._pw.get_ec(rs, zeta)  # underlying LDA correlation
        return ec_unif + PW91_H0(gamma, beta, g ** 3, t2, ec_unif)
