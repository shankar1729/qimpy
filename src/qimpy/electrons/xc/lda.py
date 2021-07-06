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
    def __init__(self, scale_factor: float = 1.) -> None:
        super().__init__(has_correlation=True, scale_factor=scale_factor)

    PARAMS = {
        True: (0.0311, -0.0480, 0.0020, -0.0116, -0.1423, 1.0529, 0.3334),
        False: (0.01555, -0.0269, 0.0007, -0.0048, -0.0843, 1.3981, 0.2611)}

    @staticmethod
    def compute(rs: torch.Tensor, para: bool = True) -> torch.Tensor:
        """Compute LDA-PZ correlation for paramagnetic case
        if para=True and ferromagnetic if para=False.
        Return energy density for each rs point."""
        a, b, c, d, gamma, beta1, beta2 = C_PZ.PARAMS[para]
        e = torch.empty_like(rs)  # energy density
        # --- rs < 1 case:
        sel = torch.where(rs < 1.)
        if len(sel[0]):
            rs_sel = rs[sel]
            log_rs = torch.log(rs_sel)
            e[sel] = (a + c*rs_sel) * torch.log(rs_sel) + b + d*rs_sel
        # --- rs >= 1 case:
        sel = torch.where(rs >= 1.)
        if len(sel[0]):
            rs_sel = rs[sel]
            rs_sqrt = torch.sqrt(rs_sel)
            e[sel] = gamma / (1. + beta1*rs_sel.sqrt() + beta2*rs_sel)
        return e

    def __call__(self, n: torch.Tensor, sigma: torch.Tensor,
                 lap: torch.Tensor, tau: torch.Tensor) -> float:
        n_spins = n.shape[0]
        n.requires_grad_()
        n_tot = n.sum(dim=0)
        rs = ((4.*np.pi/3.) * n_tot) ** (-1./3)
        if n_spins == 1:
            ec = self.compute(rs, True)
        else:
            ec_para = self.compute(rs, True)
            ec_ferro = self.compute(rs, False)
            zeta = (n[0] - n[1]) / n_tot
            spin_interp = (((1 + zeta)**(4./3) + (1 - zeta)**(4./3) - 2.)
                           / (2.**(4./3) - 2.))
            ec = ec_para + spin_interp * (ec_ferro - ec_para)
        E = (ec * n_tot).sum()
        E.backward()  # updates n.grad
        return E.item()
