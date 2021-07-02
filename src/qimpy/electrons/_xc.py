import qimpy as qp
import numpy as np
import torch
from typing import TYPE_CHECKING, Tuple
if TYPE_CHECKING:
    from ..grid import FieldH


PZ_PARAMS = {
    True: (0.0311, -0.0480, 0.0020, -0.0116, -0.1423, 1.0529, 0.3334),
    False: (0.01555, -0.0269, 0.0007, -0.0048, -0.0843, 1.3981, 0.2611)}


class XC:
    """Exchange-correlation functional. Only LDA so far.
    TODO: add other functionals and interface with LibXC."""

    def __call__(self, n_t: 'FieldH') -> Tuple[float, 'FieldH']:
        """Compute exchange-correlation energy and potential."""
        grid = n_t.grid
        n_in = (~n_t).data
        n_densities = n_in.shape[0]

        # Transform to spin densities:
        if n_densities == 1:
            n = n_in
        elif n_densities == 2:
            n = torch.empty_like(n_in)
            n[0] = 0.5 * (n_in[0] + n_in[1])
            n[1] = 0.5 * (n_in[0] - n_in[1])
        else:  # n_densities == 4:
            n = torch.empty((2,) + grid.shapeR_mine, dtype=n_in.dtype,
                            device=n_in.device)
            Mvec = n_in[1:]
            Mmag = Mvec.norm(dim=0)
            n[0] = 0.5 * (n_in[0] + Mmag)
            n[1] = 0.5 * (n_in[0] - Mmag)

        # Clamp low densities for numerical stability:
        n_cut = 1e-16
        clamp_sel = torch.where(n < n_cut)
        n[clamp_sel] = n_cut
        n = n.requires_grad_()  # to get V = delta E / delta n

        # Slater exchange:
        n_spins = n.shape[0]
        Ex_prefac = -0.75 * ((3*n_spins/np.pi) ** (1./3.))
        Ex_by_dV = Ex_prefac * (n ** (4./3)).sum()

        # PZ correlation:
        def lda_pz_c(rs: torch.Tensor, para: bool = True) -> torch.Tensor:
            """Compute LDA-PZ correlation for paramagnetic case
            if para=True and ferromagnetic if para=False.
            Return energy density for each rs point."""
            a, b, c, d, gamma, beta1, beta2 = PZ_PARAMS[para]
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
        n_tot = n.sum(dim=0)
        rs = ((4.*np.pi/3.) * n_tot) ** (-1./3)
        if n_spins == 1:
            ec = lda_pz_c(rs, True)
        else:
            ec_para = lda_pz_c(rs, True)
            ec_ferro = lda_pz_c(rs, False)
            zeta = (n[0] - n[1]) / n_tot
            spin_interp = (((1 + zeta)**(4./3) + (1 - zeta)**(4./3) - 2.)
                           / (2.**(4./3) - 2.))
            ec = ec_para + spin_interp * (ec_ferro - ec_para)
        Ec_by_dV = (ec * n_tot).sum()

        # Collect results:
        Exc_by_dV = Ex_by_dV + Ec_by_dV
        Exc_by_dV.backward()  # compute derivative w.r.t n
        E_n = n.grad

        # Gradient propagation for potential:
        E_n[clamp_sel] = 0.  # account for any clamping
        # --- propagate to n_in (density, magnetization):
        if n_densities == 1:
            E_n_in = E_n
        elif n_densities == 2:
            E_n_in = torch.empty_like(n_in)
            E_n_in[0] = 0.5 * (E_n[0] + E_n[1])
            E_n_in[1] = 0.5 * (E_n[0] - E_n[1])
        else:  # n_densities == 4:
            E_n_in = torch.empty_like(n_in)
            E_n_in[0] = 0.5 * (E_n[0] + E_n[1])
            E_n_in[1:] = 0.5 * (E_n[0] - E_n[1]) * (Mvec / Mmag)

        Vxc = ~qp.grid.FieldR(grid, data=E_n_in)
        Exc = grid.dV * Exc_by_dV.item()
        if grid.comm is not None:
            Exc = grid.comm.allreduce(Exc, qp.MPI.SUM)
        return Exc, Vxc
