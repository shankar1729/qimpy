"""Internal MGGA TPSS implementations."""
# List exported symbols for doc generation
__all__ = ["x_tpss"]

import numpy as np
import torch

from .functional import Functional


def x_tpss(rev: bool, scale_factor: float = 1.0) -> Functional:
    """Create TPSS/revTPSS exchange functional."""
    return Functional(
        name=f'{"rev" if rev else ""}TPSS MGGA exchange',
        has_exchange=True,
        needs_sigma=True,
        needs_lap=True,
        needs_tau=True,
        scale_factor=scale_factor,
        _apply=torch.jit.script(SpinScaled(_X_TPSS, rev)),
    )


# ----- Internal exchange/kinetic implementations -----


class SpinScaled(torch.nn.Module):
    """Common wrapper for spin-scaled (exchange-like) TPSS MGGA functionals."""

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
        lap.requires_grad_(requires_grad)
        tau.requires_grad_(requires_grad)
        rs = ((n_spins * 4.0 * np.pi / 3.0) * n) ** (-1.0 / 3)  # rs for each spin
        s2 = ((18.0 * np.pi) ** (-2.0 / 3)) * sigma[::2] * (rs / n).square()
        z = sigma[::2] / (8.0 * n * tau)
        z = torch.clip(z, max=1.0)
        y = 8.0 * ((18.0 * np.pi) ** (-2.0 / 3)) * tau * rs.square() / n  # y = s2 / z
        e = self.compute(rs, s2, z, y)
        E = (e * n).sum() * scale_factor
        if requires_grad:
            E.backward()
        return E.item()


def get_e_slater(rs: torch.Tensor) -> torch.Tensor:
    """Compute per-particle slater exchange energy."""
    return (-0.75 * ((1.5 / np.pi) ** (2.0 / 3))) / rs


class _X_TPSS(torch.nn.Module):
    """Internal JIT-friendly implementation of TPSS exchange."""

    mu: torch.jit.Final[float]
    c: torch.jit.Final[float]
    e: torch.jit.Final[float]
    rev: torch.jit.Final[bool]

    def __init__(self, rev: bool) -> None:
        super().__init__()
        self.mu = 0.14 if rev else 0.21951
        self.c = 2.35204 if rev else 1.59096
        self.e = 2.1677 if rev else 1.537
        self.sqrt_e = np.sqrt(2.1677) if rev else np.sqrt(1.537)
        self.rev = rev

    def forward(
        self,
        rs: torch.Tensor,
        s2: torch.Tensor,
        z: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        # Eqn. (7) of ref
        b = 0.4
        a = (5.0 / 3) * y * (1.0 - z) - 1.0  # a = alphazmz / z
        qb_den = 1.0 / torch.sqrt(1.0 + b * a * (a + 1))
        qb = 0.45 * a * qb_den + (2.0 / 3) * s2
        # Eqn. (10) of ref
        kappa = 0.804
        z2 = z**2
        s4 = s2**2
        # --- Term 1 of numerator
        x_num_1 = (
            10.0 / 81 + self.c * (z2 * z if self.rev else z2) / ((1.0 + z2) ** 2)
        ) * s2
        x_num_3 = (-73.0 / 405) * z * torch.sqrt(0.18 + 0.5 * y**2) * qb
        # --- Numerator
        x_num = (
            x_num_1
            + (146.0 / 2025) * qb**2
            + x_num_3
            + (100.0 / (6561 * kappa)) * s4
            + (4.0 * self.sqrt_e / 45) * z2
            + (self.e * self.mu) * s4 * s2
        )
        # --- Denominator
        x_den = 1.0 / (1.0 + self.sqrt_e * s2) ** 2
        # --- Eqn (10) for x:
        x = x_num * x_den
        # TPSS Enhancement factor:
        F = 1.0 + kappa - kappa**2 / (kappa + x)
        # TPSS Exchange energy per particle:
        return F * get_e_slater(rs)
