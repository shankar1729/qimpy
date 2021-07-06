import qimpy as qp
import numpy as np
import torch
from . import lda
from typing import TYPE_CHECKING, Tuple, List
if TYPE_CHECKING:
    from ...grid import FieldH
    from .functional import Functional


class XC:
    """Exchange-correlation functional. Only LDA so far.
    TODO: add other functionals and interface with LibXC."""
    __slots__ = ('_functionals',)
    _functionals: List["Functional"]  #: list of functionals that add up to XC

    def __init__(self):
        """TODO: add selection of functionals here"""
        self._functionals = []
        self._functionals.append(lda.X_Slater())
        self._functionals.append(lda.C_PZ())

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
        sigma = torch.tensor(0., device=n.device)
        lap = torch.tensor(0., device=n.device)
        tau = torch.tensor(0., device=n.device)
        # TODO: compute sigma, lap and tau when needed

        # Clamp low densities for numerical stability:
        n_cut = 1e-16
        clamp_sel = torch.where(n < n_cut)
        n[clamp_sel] = n_cut

        # Evaluate functionals:
        n.grad = torch.zeros_like(n)
        sigma.grad = torch.zeros_like(sigma)
        lap.grad = torch.zeros_like(lap)
        tau.grad = torch.zeros_like(tau)
        E_by_dV = 0.
        for functional in self._functionals:
            E_by_dV += functional(n, sigma, lap, tau)

        # Gradient propagation for potential:
        E_n = n.grad
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
        # TODO: propagate sigma, lap and tau gradients when needed

        Vxc = ~qp.grid.FieldR(grid, data=E_n_in)
        Exc = grid.dV * E_by_dV
        if grid.comm is not None:
            Exc = grid.comm.allreduce(Exc, qp.MPI.SUM)
        return Exc, Vxc
