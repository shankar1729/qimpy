"""Internal MGGA TPSS implementations."""
# List exported symbols for doc generation
__all__ = ["x_tpss", "c_tpss"]

import numpy as np
import torch

from .functional import Functional
from .lda import get_rs, _C_PW, SpinInterpolate3


def x_tpss(rev: bool, scale_factor: float = 1.0) -> Functional:
    """Create TPSS/revTPSS exchange functional."""
    return Functional(
        name=f'{"rev" if rev else ""}TPSS MGGA exchange',
        has_exchange=True,
        needs_sigma=True,
        needs_tau=True,
        scale_factor=scale_factor,
        _apply=torch.jit.script(SpinScaled(_X_TPSS, rev)),
    )


def c_tpss(rev: bool, scale_factor: float = 1.0) -> Functional:
    """Create TPSS/revTPSS correlation functional."""
    return Functional(
        name=f'{"rev" if rev else ""}TPSS MGGA correlation',
        has_correlation=True,
        needs_sigma=True,
        needs_tau=True,
        scale_factor=scale_factor,
        _apply=torch.jit.script(SpinUnpolarized(_C_TPSS, SpinInterpolate3, rev)),
        _apply_spin=torch.jit.script(SpinPolarized(_C_TPSS, SpinInterpolate3, rev)),
    )


# ----- Internal exchange implementations -----


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
        tau.requires_grad_(requires_grad)
        rs = ((n_spins * 4.0 * np.pi / 3.0) * n) ** (-1.0 / 3)  # rs for each spin
        s2 = ((18.0 * np.pi) ** (-2.0 / 3)) * sigma[::2] * (rs / n).square()
        z = sigma[::2] / (8.0 * n * tau)
        z = torch.clip(z, max=1.0)
        y = 8.0 * ((18.0 * np.pi) ** (-2.0 / 3)) * tau * rs.square() / n  # y = s2 / z
        e = self.compute(rs, s2, z, y)
        E = (e * n).sum() * scale_factor
        if requires_grad:
            E.backward()  # updates n.grad, sigma.grad and tau.grad
        return E.item()


def get_e_slater(rs: torch.Tensor) -> torch.Tensor:
    """Compute per-particle slater exchange energy."""
    return (-0.75 * ((1.5 / np.pi) ** (2.0 / 3))) / rs


class _X_TPSS(torch.nn.Module):
    """Internal JIT-friendly implementation of TPSS exchange."""

    mu: torch.jit.Final[float]
    c: torch.jit.Final[float]
    e: torch.jit.Final[float]
    sqrt_e: torch.jit.Final[float]
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


# ----- Internal correlation implementations -----


class SpinUnpolarized(torch.nn.Module):
    """Unpolarized TPSS MGGA correlation wrapper."""

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
        tau.requires_grad_(requires_grad)
        # Compute dimensionless parameters of correlation functionals:
        n_tot = n[0]
        tau_tot = tau[0]
        sigma_tot = sigma[0]
        rs = get_rs(n_tot)
        zeta = torch.zeros_like(rs)
        g = torch.ones_like(rs)
        t2 = ((np.pi / 96) ** (2.0 / 3)) * sigma_tot * rs / n_tot.square()
        t2_up = 2 * t2
        t2_dn = t2_up.clone()  # or a copy? any changes to t2_up will change t2_dn
        zi2 = torch.zeros_like(rs)
        z = sigma_tot / (8.0 * n_tot * tau_tot)  # check jdftx and qimpy exchange z and this z
        z = torch.clip(z, max=1.0)
        # Compute energy:
        ec = self.get_ec(rs, zeta, g, t2, t2_up, t2_dn, zi2, z)
        E = (n_tot * ec).sum() * scale_factor
        if requires_grad:
            E.backward()  # updates n.grad, sigma.grad and tau.grad
        return E.item()


class SpinPolarized(torch.nn.Module):
    """Polarized TPSS MGGA correlation wrapper."""

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
        tau.requires_grad_(requires_grad)
        # Compute dimensionless parameters of correlation functionals:
        n_tot = n[0] + n[1]
        tau_tot = tau[0] + tau[1]
        sigma_tot = sigma[0] + 2 * sigma[1] + sigma[2]
        rs = get_rs(n_tot)
        rs_up = get_rs(n[0])
        rs_dn = get_rs(n[1])
        zeta = (n[0] - n[1]) / n_tot
        g = 0.5 * ((1.0 + zeta) ** (2.0 / 3) + (1.0 - zeta) ** (2.0 / 3))
        t2 = ((np.pi / 96) ** (2.0 / 3)) * sigma_tot * rs / (g * n_tot).square()
        t2_up = ((np.pi / 48) ** (2.0 / 3)) * sigma[0] * rs_up / n[0].square()
        t2_dn = ((np.pi / 48) ** (2.0 / 3)) * sigma[2] * rs_dn / n[1].square()
        sigma_diff = (n[1].square() * sigma[0]
                      - 2.0 * n[0] * n[1] * sigma[1]
                      + n[0].square() * sigma[2])
        zi2 = ((9.0 * np.pi / 4) ** (-2.0 / 3)) * rs.square() * sigma_diff / n_tot ** 4
        z = sigma_tot / (8.0 * n_tot * tau_tot)  # check jdftx and qimpy exchange z and this z
        z = torch.clip(z, max=1.0)
        # Compute per-particle energy and total:
        ec = self.get_ec(rs, zeta, g, t2, t2_up, t2_dn, zi2, z)
        E = (n_tot * ec).sum() * scale_factor
        if requires_grad:
            E.backward()  # updates n.grad, sigma.grad and tau.grad
        return E.item()


def PW91_H0(
    gamma: float, beta: torch.Tensor, g3: torch.Tensor, t2: torch.Tensor, ec_unif: torch.Tensor
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


class _C_TPSS(torch.nn.Module):
    """Internal JIT-friendly implementation of TPSS correlation."""

    gamma: torch.jit.Final[float]
    rev: torch.jit.Final[bool]

    def __init__(self, SpinInterpolate: type, rev: bool):
        super().__init__()
        self.ec_pw = SpinInterpolate(_C_PW)
        self.gamma = (1.0 - np.log(2.0)) / (np.pi**2)
        self.rev = rev

    def forward(
        self, rs: torch.Tensor, zeta: torch.Tensor, g: torch.Tensor, t2: torch.Tensor,
        t2_up: torch.Tensor, t2_dn: torch.Tensor, zi2: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        # Compute C(zeta,0) (eqn (13))
        C0 = 0.59 if self.rev else 0.53
        C1 = 0.9269 if self.rev else 0.87
        C2 = 0.6225 if self.rev else 0.50
        C3 = 2.1540 if self.rev else 2.26
        C_zeta_0 = C0 + C1 * zeta**2 + C2 * zeta**4 + C3 * zeta**6
        # Compute C(zeta,zi2) (eqn (14))
        zeta_p = 1.0 + zeta
        zeta_m = 1.0 - zeta
        C_num = (zeta_p * zeta_m) ** (4.0 / 3)
        C_den = C_num + 0.5 * zi2 * (zeta_p ** (4.0 / 3) + zeta_m ** (4.0 / 3))
        # if(!Cnum && !Cden) { C=Czeta0 } //Avoid 0/0 error
        C_num_den = torch.where(torch.logical_and(C_num == 0., C_den == 0.), 1., C_num / C_den)
        C = C_zeta_0 * C_num_den**4
        # Ingredients for eqn (12):
        # PBE correlation at target spin densities:
        beta = 0.066725 * (1.0 + 0.1 * rs) / (
                1.0 + 0.1778 * rs) if self.rev else 0.06672455060314922 * torch.ones_like(rs)
        ec_unif = self.ec_pw(rs, zeta)  # underlying LDA correlation
        H = PW91_H0(self.gamma, beta, g**3, t2, ec_unif)
        ec = ec_unif + H
        g_pol = 2.0 ** (-1.0 / 3)
        # PBE correlation with up-spins alone:
        rs_up = rs / (g_pol * zeta_p ** (1.0 / 3))
        beta_up = 0.066725 * (1.0 + 0.1 * rs_up) / (
            1.0 + 0.1778 * rs_up) if self.rev else 0.06672455060314922 * torch.ones_like(rs_up)
        ec_up_unif = self.ec_pw(rs_up, torch.ones_like(zeta))  # underlying LDA correlation
        H_up = PW91_H0(self.gamma, beta_up, g_pol**3 * torch.ones_like(g), t2_up, ec_up_unif)
        ec_up = ec_up_unif + H_up
        # PBE correlation with down-spins alone:
        # if(!zeta && t2up==t2dn) ec_dn=ec_up -> is it worth it to implement here?
        if torch.equal(t2_up, t2_dn) and torch.equal(zeta, torch.zeros_like(zeta)):
            ec_dn = ec_up
        else:
            rs_dn = rs / (g_pol * zeta_m ** (1.0 / 3))
            beta_dn = 0.066725 * (1.0 + 0.1 * rs_dn) / (
                1.0 + 0.1778 * rs_dn) if self.rev else 0.06672455060314922 * torch.ones_like(rs_dn)
            ec_dn_unif = self.ec_pw(rs_dn, torch.ones_like(zeta))  # underlying LDA correlation
            ec_dn = ec_dn_unif + PW91_H0(self.gamma, beta_dn, (g_pol * torch.ones_like(g)) ** 3, t2_dn, ec_dn_unif)
        # Compute ecTilde = 0.5*(1+zeta) max(ec, ecUp) + 0.5*(1-zeta) max(ec, ecDn):
        ec_max_up = torch.where(ec > ec_up, ec, ec_up)
        ec_max_dn = torch.where(ec > ec_dn, ec, ec_dn)
        ec_tilde = 0.5 * zeta_p * ec_max_up + 0.5 * zeta_m * ec_max_dn
        # Put together eqn. (12):
        ec_PKZB = ec * (1.0 + C * z**2) - ec_tilde * z**2 * (1.0 + C)
        # Put together the final correlation energy (eqn. (11)):
        d = 2.8
        return ec_PKZB * (1.0 + d * ec_PKZB * z**3)

    # rename x and c for mgga
