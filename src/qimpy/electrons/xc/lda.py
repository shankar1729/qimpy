"""Internal LDA implementations."""
# List exported symbols for doc generation
__all__ = ['KE_TF', 'X_Slater', 'SpinInterpolated',
           'C_PZ', 'C_PW', 'C_VWN', 'XC_Teter']

from .functional import Functional
import numpy as np
import torch
from abc import abstractmethod


class KE_TF(Functional):
    """Thomas-Fermi kinetic energy functional."""
    def __init__(self, scale_factor: float = 1.) -> None:
        super().__init__(has_kinetic=True, scale_factor=scale_factor)

    def __call__(self, n: torch.Tensor, sigma: torch.Tensor,
                 lap: torch.Tensor, tau: torch.Tensor) -> float:
        n_spins = n.shape[0]
        prefactor = (0.3 * ((3*(np.pi**2) * n_spins) ** (2./3.))
                     * self.scale_factor)
        n.requires_grad_()
        E = prefactor * (n ** (5./3)).sum()
        E.backward()  # updates n.grad
        return E.item()


class X_Slater(Functional):
    """Slater exchange functional."""
    def __init__(self, scale_factor: float = 1.) -> None:
        super().__init__(has_exchange=True, scale_factor=scale_factor)

    def __call__(self, n: torch.Tensor, sigma: torch.Tensor,
                 lap: torch.Tensor, tau: torch.Tensor) -> float:
        n_spins = n.shape[0]
        prefactor = -0.75 * ((3*n_spins/np.pi) ** (1./3.)) * self.scale_factor
        n.requires_grad_()
        E = prefactor * (n ** (4./3)).sum()
        E.backward()  # updates n.grad
        return E.item()


class SpinInterpolated(Functional):
    """Abstract base class for spin-interpolated LDA functionals.
    This is typical for most LDA correlation functionals."""
    __slots__ = ('stiffness_scale',)
    stiffness_scale: float  #: scale factor for spin-stiffness term

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Forward remaining arguments
        self.stiffness_scale = (9./4) * (2.**(1./3) - 1)  # overridden in PW

    def __call__(self, n: torch.Tensor, sigma: torch.Tensor,
                 lap: torch.Tensor, tau: torch.Tensor) -> float:
        """Main interface function including gradient evaluation."""
        n_spins = n.shape[0]
        n.requires_grad_()
        n_tot = n.sum(dim=0)
        rs = ((4.*np.pi/3.) * n_tot) ** (-1./3)
        zeta = (((n[0] - n[1]) / n_tot) if (n_spins == 2)
                else torch.zeros(1, dtype=n.dtype, device=n.device))
        E = (n_tot * self.get_ec(rs, zeta)).sum() * self.scale_factor
        E.backward()  # updates n.grad
        return E.item()

    def get_ec(self, rs: torch.Tensor, zeta: torch.Tensor) -> torch.Tensor:
        """Internal spin interpolator, starting with `rs` and `zeta`."""
        spin_polarized = ((len(zeta) != 1) or (zeta.item() != 0.))
        ec_spins = self.compute(rs, spin_polarized)
        # Interpolate between spin channels:
        n_channels = ec_spins.shape[-1]
        if n_channels == 1:  # un-polarized: no spin interpolation needed
            return ec_spins[..., 0]
        else:
            spin_interp = (((1 + zeta)**(4./3) + (1 - zeta)**(4./3) - 2.)
                           / (2.**(4./3) - 2.))
            if n_channels == 2:  # interpolate between para and ferro
                ec_para, ec_ferro = ec_spins.unbind(dim=-1)
                return ec_para + spin_interp * (ec_ferro - ec_para)
            else:  # n_channels == 3: additionally include spin stiffness
                ec_para, ec_ferro, ec_stiff = ec_spins.unbind(dim=-1)
                zeta4 = zeta ** 4
                w1 = zeta4 * spin_interp
                w2 = (zeta4 - 1.) * spin_interp * self.stiffness_scale
                return ec_para + w1 * (ec_ferro - ec_para) + w2 * ec_stiff

    @abstractmethod
    def compute(self, rs: torch.Tensor, spin_polarized: bool) -> torch.Tensor:
        """Compute energy (per-particle) to be spin-interpolated.
        Output should have one extra dimension at the end beyond those of `rs`
        containing various channels to be spin-interpolated. This dimension
        should be of length 1 when `spin_polarized` is False, and of length
        2 or 3 when `spin_polarized` is True. The spin channels correspond
        to paramagnetic, ferromagnetic and optionally spin-stiffness."""


class C_PZ(SpinInterpolated):
    """Perdew-Zunger LDA correlation functional."""
    __slots__ = ('_params',)
    _params: torch.Tensor  # PZ functional parameters

    def __init__(self, scale_factor: float = 1.) -> None:
        super().__init__(has_correlation=True, scale_factor=scale_factor)
        self._params = torch.tensor([
            [0.0311, 0.01555],  # a
            [-0.0480, -0.0269],  # b
            [0.0020, 0.0007],  # c
            [-0.0116, -0.0048],  # d
            [-0.1423, -0.0843],  # gamma
            [1.0529, 1.3981],  # beta1
            [0.3334, 0.2611],  # beta2
        ])

    def compute(self, rs: torch.Tensor, spin_polarized: bool) -> torch.Tensor:
        n_channels = (2 if spin_polarized else 1)
        a, b, c, d, gamma, beta1, beta2 \
            = self._params[:, :n_channels].to(rs.device).unbind()
        e = torch.empty(rs.shape + (n_channels,))  # energy density
        # --- rs < 1 case:
        sel = torch.where(rs < 1.)
        if len(sel[0]):
            rs_sel = rs[sel][..., None]  # add dim for spin interpolation
            e[sel] = (a + c*rs_sel) * torch.log(rs_sel) + b + d*rs_sel
        # --- rs >= 1 case:
        sel = torch.where(rs >= 1.)
        if len(sel[0]):
            rs_sel = rs[sel][..., None]  # add dim for spin interpolation
            e[sel] = gamma / (1. + beta1*rs_sel.sqrt() + beta2*rs_sel)
        return e


class C_PW(SpinInterpolated):
    """Perdew-Wang LDA correlation functional."""
    __slots__ = ('_params',)
    _params: torch.Tensor  # PZ functional parameters

    def __init__(self, high_precision: bool, scale_factor: float = 1.) -> None:
        """Initialize PW correlation functional.
        Here, `high_precision` controls whether parameters are at the
        full precision (if True) as used within the PBE GGA, or at the
        original precision (if False) as in the original PW-LDA paper."""
        super().__init__(has_correlation=True, scale_factor=scale_factor)
        if not high_precision:
            self.stiffness_scale = 1./1.709921  # limit to single precision
        self._params = torch.tensor([
            ([0.0310907, 0.01554535, 0.0168869]  # A at full precision
             if high_precision else
             [0.031091, 0.015545, 0.016887]),  # A at original precision
            [0.21370, 0.20548, 0.11125],  # alpha
            [7.5957, 14.1189, 10.357],  # beta1
            [3.5876, 6.1977, 3.6231],  # beta2
            [1.6382, 3.3662, 0.88026],  # beta3
            [0.49294, 0.62517, 0.49671],  # beta4
        ])
        self._params[0] *= 2.  # Convert A to 2A used below

    def compute(self, rs: torch.Tensor, spin_polarized: bool) -> torch.Tensor:
        n_channels = (3 if spin_polarized else 1)
        A2, alpha, beta1, beta2, beta3, beta4 \
            = self._params[:, :n_channels].to(rs.device).unbind()
        x = rs.sqrt()[..., None]  # add dim for spin interpolation
        den = A2 * x*(beta1 + x*(beta2 + x*(beta3 + x*beta4)))
        return -A2 * (1 + alpha*rs[..., None]) * (1. + 1./den).log()


class C_VWN(SpinInterpolated):
    """Vosko-Wilk-Nusair LDA correlation functional."""
    __slots__ = ('_params',)
    _params: torch.Tensor  # VWN functional parameters

    def __init__(self, scale_factor: float = 1.) -> None:
        super().__init__(has_correlation=True, scale_factor=scale_factor)
        self._params = torch.tensor([
            [0.0310907, 0.01554535,  1./(6.*(np.pi**2))],  # A
            [3.72744, 7.06042, 1.13107],  # b
            [12.9352, 18.0578, 13.0045],  # c
            [-0.10498, -0.32500, -0.0047584],  # x0
        ])

    def compute(self, rs: torch.Tensor, spin_polarized: bool) -> torch.Tensor:
        n_channels = (3 if spin_polarized else 1)
        A, b, c, x0 = self._params[:, :n_channels].to(rs.device).unbind()
        # Commonly used combinations of rs:
        X0 = c + x0*(b + x0)
        Q = (4.*c - b*b).sqrt()
        x = rs.sqrt()[..., None]  # add dim for spin interpolation
        X = c + x*(b + x)
        X_x = 2*x + b
        # Three transcendental terms:
        atan_term = (2./Q) * (Q / X_x).atan()
        log_term1 = (x.square() / X).log()
        log_term2 = ((x - x0).square() / X).log()
        # Final combination to correlation energy:
        return A*(log_term1 + b*(atan_term - (x0/X0)*(log_term2
                                                      + (b+2*x0)*atan_term)))


class XC_Teter(Functional):
    """Teter LSDA functional."""
    __slots__ = ('_params',)
    _params: torch.Tensor  # Teter functional parameters

    def __init__(self, scale_factor: float = 1.) -> None:
        super().__init__(has_exchange=True, has_correlation=True,
                         scale_factor=scale_factor)
        self._params = torch.tensor([
            [0.4581652932831429, 0.119086804055547],  # a0  (para, ferro-para)
            [2.217058676663745, 0.6157402568883345],  # a1
            [0.7405551735357053, 0.1574201515892867],  # a2
            [0.01968227878617998, 0.003532336663397157],  # a3
            [4.504130959426697, 0.2673612973836267],  # b2
            [1.110667363742916, 0.2052004607777787],  # b3
            [0.02359291751427506, 0.004200005045691381]  # b4
        ])

    def __call__(self, n: torch.Tensor, sigma: torch.Tensor,
                 lap: torch.Tensor, tau: torch.Tensor) -> float:
        n_spins = n.shape[0]
        n.requires_grad_()
        n_tot = n.sum(dim=0)
        rs = ((4.*np.pi/3.) * n_tot) ** (-1./3)
        # Spin interpolate the parameters (if needed):
        if n_spins == 1:
            params = self._params[:, 0].to(n.device)
        else:
            zeta = (n[0] - n[1]) / n_tot
            spin_interp = (((1 + zeta)**(4./3) + (1 - zeta)**(4./3) - 2.)
                           / (2.**(4./3) - 2.))
            params_para, params_dferro = self._params.to(n.device
                                                         ).unbind(dim=-1)
            params = params_para + spin_interp[..., None] * params_dferro
        # Pade approximant with spin-interpolated parameters:
        a0, a1, a2, a3, b2, b3, b4 = params.unbind(dim=-1)
        minus_exc = ((a0 + rs*(a1 + rs*(a2 + rs*a3)))
                     / (rs*(1. + rs*(b2 + rs*(b3 + rs*b4)))))
        E = (minus_exc * n_tot).sum() * (-self.scale_factor)
        E.backward()  # updates n.grad
        return E.item()
