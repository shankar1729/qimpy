from .functional import Functional
import numpy as np
import torch
from abc import abstractmethod


class SpinScaled(Functional):
    """Abstract base class of spin-scaled (exchange, KE) functionals."""
    def __init__(self, **kwargs) -> None:
        super().__init__(needs_sigma=True, **kwargs)

    def __call__(self, n: torch.Tensor, sigma: torch.Tensor,
                 lap: torch.Tensor, tau: torch.Tensor) -> float:
        n_spins = n.shape[0]
        n.requires_grad_()
        sigma.requires_grad_()
        rs = ((n_spins * 4.*np.pi/3.) * n) ** (-1./3)  # rs for each spin
        s2 = ((18.*np.pi) ** (-2./3)) * sigma[::2] * (rs / n).square()
        e = self.compute(rs, s2)
        E = (e * n).sum() * self.scale_factor
        E.backward()  # updates n.grad and sigma.grad
        return E.item()

    @abstractmethod
    def compute(self, rs: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        """Compute energy (per-particle) of spin-scaled functional."""


class X_PBE(SpinScaled):
    """PBE/PBEsol exchange."""
    __slots__ = ('sol',)
    sol: bool  # PBEsol if True; PBE otherwise

    def __init__(self, sol: bool, scale_factor: float = 1.) -> None:
        """Initialize PBE exchange if `sol` is False and PBEsol if True."""
        super().__init__(has_exchange=True, scale_factor=scale_factor)
        self.sol = sol

    def compute(self, rs: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        kappa = 0.804
        mu_by_kappa = (10./81 if self.sol else 0.2195149727645171) / kappa
        eSlater = (-0.75 * ((1.5/np.pi)**(2./3))) / rs
        F = (1. + kappa) - kappa / (1. + mu_by_kappa * s2)  # GGA enhancement
        return eSlater * F
