from .functional import Functional
import numpy as np
import torch


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


class C_PZ(Functional):
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

    def compute(self, rs: torch.Tensor, n_spins: int) -> torch.Tensor:
        """Compute LDA-PZ correlation for paramagnetic case
        if para=True and ferromagnetic if para=False.
        Return energy density for each rs point."""
        n_channels = (1 if (n_spins == 1) else 2)
        _params = self._params[:, :n_channels].to(rs.device)
        a, b, c, d, gamma, beta1, beta2 = _params.unbind()
        e = torch.empty(rs.shape + (n_channels,))  # energy density
        # --- rs < 1 case:
        sel = torch.where(rs < 1.)
        if len(sel[0]):
            rs_sel = rs[sel][..., None]
            log_rs = torch.log(rs_sel)
            e[sel] = (a + c*rs_sel) * torch.log(rs_sel) + b + d*rs_sel
        # --- rs >= 1 case:
        sel = torch.where(rs >= 1.)
        if len(sel[0]):
            rs_sel = rs[sel][..., None]
            rs_sqrt = torch.sqrt(rs_sel)
            e[sel] = gamma / (1. + beta1*rs_sel.sqrt() + beta2*rs_sel)
        return e

    def __call__(self, n: torch.Tensor, sigma: torch.Tensor,
                 lap: torch.Tensor, tau: torch.Tensor) -> float:
        n_spins = n.shape[0]
        n.requires_grad_()
        n_tot = n.sum(dim=0)
        rs = ((4.*np.pi/3.) * n_tot) ** (-1./3)
        ec_spins = self.compute(rs, n_spins)
        if n_spins == 1:
            ec = ec_spins[..., 0]
        else:
            ec_para, ec_ferro = ec_spins.unbind(dim=-1)
            zeta = (n[0] - n[1]) / n_tot
            spin_interp = (((1 + zeta)**(4./3) + (1 - zeta)**(4./3) - 2.)
                           / (2.**(4./3) - 2.))
            ec = ec_para + spin_interp * (ec_ferro - ec_para)
        E = (ec * n_tot).sum()
        E.backward()  # updates n.grad
        return E.item()
