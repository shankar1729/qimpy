"""Internal LDA implementations."""
# List exported symbols for doc generation
__all__ = ["ke_tf", "x_slater", "c_pz", "c_pw", "c_vwn", "xc_teter"]

import numpy as np
import torch

from .functional import Functional


def ke_tf(scale_factor: float = 1.0) -> Functional:
    """Create Thomas-Fermi kinetic energy functional."""
    return Functional(
        name="Thomas-Fermi LDA KE",
        has_kinetic=True,
        scale_factor=scale_factor,
        _apply=torch.jit.script(_ke_tf),
    )


def x_slater(scale_factor: float = 1.0) -> Functional:
    """Create Slater exchange functional."""
    return Functional(
        name="Slater LDA exchange",
        has_exchange=True,
        scale_factor=scale_factor,
        _apply=torch.jit.script(_x_slater),
    )


def c_pz(scale_factor: float = 1.0) -> Functional:
    """Create Perdew-Zunger LDA correlation functional."""
    return Functional(
        name="Perdew-Zunger LDA correlation",
        has_correlation=True,
        scale_factor=scale_factor,
        _apply=torch.jit.script(SpinUnpolarized(SpinInterpolate1, _C_PZ)),
        _apply_spin=torch.jit.script(SpinPolarized(SpinInterpolate2, _C_PZ)),
    )


def c_pw(high_precision: bool, scale_factor: float = 1.0) -> Functional:
    """Create Perdew-Wang LDA correlation functional.
    Here, `high_precision` controls whether parameters are at the
    full precision (if True) as used within the PBE GGA, or at the
    original precision (if False) as in the original PW-LDA paper."""
    stiffness_scale = (
        0.0  # use the default full precision
        if high_precision
        else 1.0 / 1.709921  # limit to single precision
    )
    return Functional(
        name="Perdew-Zunger LDA correlation",
        has_correlation=True,
        scale_factor=scale_factor,
        _apply=torch.jit.script(
            SpinUnpolarized(SpinInterpolate1, _C_PW, high_precision)
        ),
        _apply_spin=torch.jit.script(
            SpinPolarized(SpinInterpolate3, _C_PW, stiffness_scale, high_precision)
        ),
    )


def c_vwn(scale_factor: float = 1.0) -> Functional:
    """Create Vosko-Wilk-Nusair LDA correlation functional."""
    return Functional(
        name="Vosko-Wilk-Nusair LDA correlation",
        has_correlation=True,
        scale_factor=scale_factor,
        _apply=torch.jit.script(SpinUnpolarized(SpinInterpolate1, _C_VWN)),
        _apply_spin=torch.jit.script(SpinPolarized(SpinInterpolate3, _C_VWN)),
    )


def xc_teter(scale_factor: float = 1.0) -> Functional:
    """Create Teter LSDA functional."""
    return Functional(
        name="Teter93 LSD exchange+correlation",
        has_exchange=True,
        has_correlation=True,
        scale_factor=scale_factor,
        _apply=torch.jit.script(SpinUnpolarized(_XC_TeterUnpolarized)),
        _apply_spin=torch.jit.script(SpinPolarized(_XC_TeterPolarized)),
    )


# ----- Internal exchange/kinetic implementations -----


def _ke_tf(
    n: torch.Tensor,
    sigma: torch.Tensor,
    lap: torch.Tensor,
    tau: torch.Tensor,
    requires_grad: bool,
    scale_factor: float,
) -> float:
    """Internal JIT-friendly implementation of Thomas-Fermi kinetic energy"""
    n_spins = n.shape[0]
    prefactor = 0.3 * ((3 * (np.pi**2) * n_spins) ** (2.0 / 3.0)) * scale_factor
    n.requires_grad_(requires_grad)
    E = prefactor * (n ** (5.0 / 3)).sum()
    if requires_grad:
        E.backward()  # updates n.grad
    return E.item()


def _x_slater(
    n: torch.Tensor,
    sigma: torch.Tensor,
    lap: torch.Tensor,
    tau: torch.Tensor,
    requires_grad: bool,
    scale_factor: float,
) -> float:
    """Internal JIT-friendly implementation of Slater exchange"""
    n_spins = n.shape[0]
    prefactor = -0.75 * ((3 * n_spins / np.pi) ** (1.0 / 3.0)) * scale_factor
    n.requires_grad_(requires_grad)
    E = prefactor * (n ** (4.0 / 3)).sum()
    if requires_grad:
        E.backward()  # updates n.grad
    return E.item()


def get_rs(n_tot: torch.Tensor) -> torch.Tensor:
    """Compute Wigner-Seitz radius `rs` from electron density `n_tot`."""
    return ((4.0 * np.pi / 3.0) * n_tot) ** (-1.0 / 3)


def get_spin_interp(zeta: torch.Tensor) -> torch.Tensor:
    """Compute spin interpolation function from fractional polarization `zeta`."""
    exponent = 4.0 / 3
    scale = 1.0 / (2.0**exponent - 2.0)
    return ((1.0 + zeta) ** exponent + (1.0 - zeta) ** exponent - 2.0) * scale


# ----- Internal correlation/combined-XC implementations -----


class SpinUnpolarized(torch.nn.Module):
    """Unpolarized LDA correlation wrapper."""

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
        # Compute rs:
        n_tot = n[0]
        rs = get_rs(n_tot)
        # Compute energy:
        E = (n_tot * self.get_ec(rs, rs)).sum() * scale_factor
        if requires_grad:
            E.backward()  # updates n.grad
        return E.item()


class SpinPolarized(torch.nn.Module):
    """Polarized LDA correlation wrapper."""

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
        # Compute rs and zeta:
        n_tot = n[0] + n[1]
        rs = get_rs(n_tot)
        zeta = (n[0] - n[1]) / n_tot
        # Compute energy:
        E = (n_tot * self.get_ec(rs, zeta)).sum() * scale_factor
        if requires_grad:
            E.backward()  # updates n.grad
        return E.item()


class SpinInterpolate1(torch.nn.Module):
    """Correlation with single spin channel: para (no interpolation)."""

    def __init__(self, Compute: type, *args) -> None:
        super().__init__()
        self.compute_para = Compute("para", *args)

    def forward(self, rs: torch.Tensor, zeta: torch.Tensor) -> torch.Tensor:
        return self.compute_para(rs)


class SpinInterpolate2(torch.nn.Module):
    """Spin interpolate correlation using 2 channels: para and ferro."""

    def __init__(self, Compute: type, *args) -> None:
        super().__init__()
        self.compute_para = Compute("para", *args)
        self.compute_ferro = Compute("ferro", *args)

    def forward(self, rs: torch.Tensor, zeta: torch.Tensor) -> torch.Tensor:
        # Calculate each spin channel:
        ec_para = self.compute_para(rs)
        ec_ferro = self.compute_ferro(rs)
        # Interpolate between spin channels:
        return ec_para + get_spin_interp(zeta) * (ec_ferro - ec_para)


class SpinInterpolate3(torch.nn.Module):
    """Spin interpolate correlation using 3 channels: para, ferro and stiffness."""

    stiffness_scale: torch.jit.Final[float]

    def __init__(self, Compute: type, stiffness_scale: float = 0.0, *args) -> None:
        super().__init__()
        self.stiffness_scale = (
            stiffness_scale if stiffness_scale else (9.0 / 4) * (2.0 ** (1.0 / 3) - 1)
        )
        self.compute_para = Compute("para", *args)
        self.compute_ferro = Compute("ferro", *args)
        self.compute_stiff = Compute("stiff", *args)

    def forward(self, rs: torch.Tensor, zeta: torch.Tensor) -> torch.Tensor:
        # Calculate each spin channel:
        ec_para = self.compute_para(rs)
        ec_ferro = self.compute_ferro(rs)
        ec_stiff = self.compute_stiff(rs)
        # Interpolate between spin channels:
        spin_interp = get_spin_interp(zeta)
        zeta4 = zeta**4
        w1 = zeta4 * spin_interp
        w2 = (zeta4 - 1.0) * spin_interp * self.stiffness_scale
        return ec_para + w1 * (ec_ferro - ec_para) + w2 * ec_stiff


class _C_PZ(torch.nn.Module):
    """Internal JIT-friendly implementation of Perdew-Zunger correlation."""

    a: torch.jit.Final[float]
    b: torch.jit.Final[float]
    c: torch.jit.Final[float]
    d: torch.jit.Final[float]
    gamma: torch.jit.Final[float]
    beta1: torch.jit.Final[float]
    beta2: torch.jit.Final[float]

    def __init__(self, spin_mode: str):
        super().__init__()
        self.a, self.b, self.c, self.d, self.gamma, self.beta1, self.beta2 = {
            "para": (0.0311, -0.0480, 0.0020, -0.0116, -0.1423, 1.0529, 0.3334),
            "ferro": (0.01555, -0.0269, 0.0007, -0.0048, -0.0843, 1.3981, 0.2611),
        }[spin_mode]

    def forward(self, rs: torch.Tensor) -> torch.Tensor:
        return torch.where(
            rs < 1.0,
            (self.a + self.c * rs) * rs.log() + self.b + self.d * rs,
            self.gamma / (1.0 + self.beta1 * rs.sqrt() + self.beta2 * rs),
        )


class _C_PW(torch.nn.Module):
    """Internal JIT-friendly implementation of Perdew-Wang correlation."""

    A2: torch.jit.Final[float]
    alpha: torch.jit.Final[float]
    beta1: torch.jit.Final[float]
    beta2: torch.jit.Final[float]
    beta3: torch.jit.Final[float]
    beta4: torch.jit.Final[float]

    def __init__(self, spin_mode: str, high_precision: bool = True):
        super().__init__()
        # Select A based on precision setting (and store 2A):
        self.A2 = (
            2.0
            * {
                "para": (0.0310907 if high_precision else 0.031091),
                "ferro": (0.01554535 if high_precision else 0.015545),
                "stiff": (0.0168869 if high_precision else 0.016887),
            }[spin_mode]
        )
        # Remaining paramters (alpha, beta[1-4]):
        self.alpha, self.beta1, self.beta2, self.beta3, self.beta4 = {
            "para": (0.21370, 7.5957, 3.5876, 1.6382, 0.49294),
            "ferro": (0.20548, 14.1189, 6.1977, 3.3662, 0.62517),
            "stiff": (0.11125, 10.357, 3.6231, 0.88026, 0.49671),
        }[spin_mode]

    def forward(self, rs: torch.Tensor) -> torch.Tensor:
        x = rs.sqrt()
        den = (
            self.A2
            * x
            * (self.beta1 + x * (self.beta2 + x * (self.beta3 + x * self.beta4)))
        )
        return -self.A2 * (1 + self.alpha * rs) * (1.0 + 1.0 / den).log()


class _C_VWN(torch.nn.Module):
    """Internal JIT-friendly implementation of Vosko-Wilk-Nusair correlation."""

    A: torch.jit.Final[float]
    b: torch.jit.Final[float]
    c: torch.jit.Final[float]
    x0: torch.jit.Final[float]
    Q: torch.jit.Final[float]

    def __init__(self, spin_mode: str):
        super().__init__()
        self.A, self.b, self.c, self.x0 = {
            "para": (0.0310907, 3.72744, 12.9352, -0.10498),
            "ferro": (0.01554535, 7.06042, 18.0578, -0.32500),
            "stiff": (1.0 / (6.0 * (np.pi**2)), 1.13107, 13.0045, -0.0047584),
        }[spin_mode]
        self.Q = np.sqrt(4.0 * self.c - self.b**2)

    def forward(self, rs: torch.Tensor) -> torch.Tensor:
        # Commonly used combinations of rs:
        X0 = self.c + self.x0 * (self.b + self.x0)
        x = rs.sqrt()
        X = self.c + x * (self.b + x)
        X_x = 2 * x + self.b
        # Three transcendental terms:
        atan_term = (2.0 / self.Q) * (self.Q / X_x).atan()
        log_term1 = (x.square() / X).log()
        log_term2 = ((x - self.x0).square() / X).log()
        # Final combination to correlation energy:
        return self.A * (
            log_term1
            + self.b
            * (
                atan_term
                - (self.x0 / X0) * (log_term2 + (self.b + 2 * self.x0) * atan_term)
            )
        )


class _XC_TeterUnpolarized(torch.nn.Module):
    """Internal JIT-friendly implementation of unpolarized Teter functional."""

    def forward(self, rs: torch.Tensor, zeta: torch.Tensor) -> torch.Tensor:
        # Constant parameters in unpolarized case:
        a0 = 0.4581652932831429
        a1 = 2.217058676663745
        a2 = 0.7405551735357053
        a3 = 0.01968227878617998
        b2 = 4.504130959426697
        b3 = 1.110667363742916
        b4 = 0.02359291751427506
        # Pade approximant:
        return -(a0 + rs * (a1 + rs * (a2 + rs * a3))) / (
            rs * (1.0 + rs * (b2 + rs * (b3 + rs * b4)))
        )


class _XC_TeterPolarized(torch.nn.Module):
    """Internal JIT-friendly implementation of spin-polarized Teter functional."""

    def forward(self, rs: torch.Tensor, zeta: torch.Tensor) -> torch.Tensor:
        spin_interp = get_spin_interp(zeta)
        # Spin-interpolate parameters:
        a0 = 0.4581652932831429 + spin_interp * 0.119086804055547
        a1 = 2.217058676663745 + spin_interp * 0.6157402568883345
        a2 = 0.7405551735357053 + spin_interp * 0.1574201515892867
        a3 = 0.01968227878617998 + spin_interp * 0.003532336663397157
        b2 = 4.504130959426697 + spin_interp * 0.2673612973836267
        b3 = 1.110667363742916 + spin_interp * 0.2052004607777787
        b4 = 0.02359291751427506 + spin_interp * 0.004200005045691381
        # Pade approximant:
        return -(a0 + rs * (a1 + rs * (a2 + rs * a3))) / (
            rs * (1.0 + rs * (b2 + rs * (b3 + rs * b4)))
        )
