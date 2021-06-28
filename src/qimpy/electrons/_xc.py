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

    def __call__(self, n: 'FieldH') -> Tuple[float, 'FieldH']:
        """Compute exchange-correlation energy and potential."""
        grid = n.grid
        n_count = n.data.shape[0]

        # Slater exchange:
        ns = torch.clamp((~n).data * n_count, min=1e-16)
        Vx = -((3/np.pi) ** (1./3.)) * (ns ** (1./3))
        Ex = (3./4) * (Vx * ns).sum().item() * grid.dV

        # PZ correlation:
        def lda_pz_c(rs, para=True):
            """Compute LDA-PZ correlation for paramagnetic case
            if para=True and ferromagnetic if para=False.
            Return energy density and potential for each rs point."""
            a, b, c, d, gamma, beta1, beta2 = PZ_PARAMS[para]
            e = torch.empty_like(rs)  # energy density
            e_rs = torch.empty_like(rs)  # it's derivative w.r.t rs
            # --- rs < 1 case:
            sel = torch.where(rs < 1.)
            if len(sel[0]):
                rs_sel = rs[sel]
                log_rs = torch.log(rs_sel)
                e[sel] = (a + c*rs_sel) * log_rs + b + d*rs_sel
                e_rs[sel] = a/rs_sel + c*(1 + log_rs) + d
            # --- rs >= 1 case:
            sel = torch.where(rs >= 1.)
            if len(sel[0]):
                rs_sel = rs[sel]
                rs_sqrt = torch.sqrt(rs_sel)
                den_inv = 1./(1. + beta1*rs_sqrt + beta2*rs_sel)
                den_rs = 0.5*beta1/rs_sqrt + beta2
                e[sel] = gamma * den_inv
                e_rs[sel] = -gamma * den_inv.square() * den_rs
            return e, e_rs
        n_tot = ns.mean(dim=0)
        rs = ((4.*np.pi/3.) * n_tot) ** (-1./3)
        assert n_count == 1  # TODO: support spin interpolation here
        e, e_rs = lda_pz_c(rs, True)
        Vc = e - (1./3) * e_rs * rs
        Ec = (e * n_tot).sum().item() * grid.dV

        # Collect results:
        Vxc = ~qp.grid.FieldR(grid, data=(Vx + Vc))
        Exc = Ex + Ec
        if grid.comm is not None:
            Exc = grid.comm.allreduce(Exc, qp.MPI.SUM)
        return Exc, Vxc
