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
        n_count = n_t.data.shape[0]
        n_in = (~n_t).data.requires_grad_()  # to get V = delta E / delta n
        n = torch.clamp(n_in, min=1e-16)

        # Slater exchange:
        Ex_prefac = -np.pi/4 * ((3*n_count/np.pi) ** (4./3.))
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
        assert n_count == 1  # TODO: support spin interpolation here
        Ec_by_dV = (lda_pz_c(rs, True) * n_tot).sum()

        # Collect results:
        Exc_by_dV = Ex_by_dV + Ec_by_dV
        Exc_by_dV.backward()  # compute derivative w.r.t n
        Vxc = ~qp.grid.FieldR(grid, data=n_in.grad)
        Exc = grid.dV * Exc_by_dV.item()
        if grid.comm is not None:
            Exc = grid.comm.allreduce(Exc, qp.MPI.SUM)
        return Exc, Vxc
