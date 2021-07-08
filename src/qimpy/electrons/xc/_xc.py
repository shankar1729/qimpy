import qimpy as qp
import numpy as np
import torch
from . import lda, gga
from typing import TYPE_CHECKING, Tuple, List, Optional
if TYPE_CHECKING:
    from ...grid import FieldH
    from .functional import Functional


N_CUT = 1e-16  # Regularization threshold for densities


class XC:
    """Exchange-correlation functional. Only LDA so far.
    TODO: add other functionals and interface with LibXC."""
    __slots__ = ('_functionals', 'need_sigma', 'need_lap', 'need_tau')
    _functionals: List["Functional"]  #: list of functionals that add up to XC
    need_sigma: bool  #: whether overall functional needs gradient
    need_lap: bool  #: whether overall functional needs laplacian
    need_tau: bool  #: whether overall functional needs KE density

    def __init__(self, *, name: str = 'lda-pz'):
        """TODO: add selection of functionals here"""
        self._functionals = []
        if name == 'lda-pz':
            self._functionals.append(lda.X_Slater())
            self._functionals.append(lda.C_PZ())
        elif name == 'lda-pw':
            self._functionals.append(lda.X_Slater())
            self._functionals.append(lda.C_PW(high_precision=False))
        elif name == 'lda-pw-prec':
            self._functionals.append(lda.X_Slater())
            self._functionals.append(lda.C_PW(high_precision=True))
        elif name == 'lda-vwn':
            self._functionals.append(lda.X_Slater())
            self._functionals.append(lda.C_VWN())
        elif name == 'lda-teter':
            self._functionals.append(lda.XC_Teter())
        elif name == 'gga-pbe':
            self._functionals.append(gga.X_PBE(sol=False))
            self._functionals.append(lda.C_PW(high_precision=True))  # TODO
        else:
            raise KeyError(f'Unknown XC functional {name}')

        # Collect overall needs:
        self.need_sigma = any(func.needs_sigma for func in self._functionals)
        self.need_lap = any(func.needs_lap for func in self._functionals)
        self.need_tau = any(func.needs_tau for func in self._functionals)

    def __call__(self, n_t: 'FieldH', tau_t: 'FieldH'
                 ) -> Tuple[float, 'FieldH', 'FieldH']:
        """Compute exchange-correlation energy and potential.
        Here, `n_t` and `tau_t` are the electron density and KE density
        (used if `need_tau` is True) in reciprocal space."""
        grid = n_t.grid
        watch = qp.utils.StopWatch('XC', grid.rc)
        n_in = (~n_t).data
        n_densities = n_in.shape[0]

        # Initialize local spin basis for vector-spin mode:
        if n_densities == 4:
            Mvec = n_in[1:]
            MmagInv = 1. / Mvec.norm(dim=0).clamp(min=N_CUT)
            Mhat = Mvec * MmagInv  # regularized unit vector

        # Get required quantities in local-spin basis:
        def from_magnetization(x_in: torch.Tensor) -> torch.Tensor:
            """Transform a quantity from magnetization to up/dn basis.
            First dimension of `x_in` must be the spin dimension."""
            if n_densities == 1:
                return x_in
            x = torch.empty((2,) + x_in.shape[1:], dtype=x_in.dtype,
                            device=x_in.device)
            if n_densities == 4:
                xM = (x_in[1:] * Mhat).sum(dim=0)
                x[0] = 0.5 * (x_in[0] + xM)
                x[1] = 0.5 * (x_in[0] - xM)
            else:  # n_densities == 2:
                x[0] = 0.5 * (x_in[0] + x_in[1])
                x[1] = 0.5 * (x_in[0] - x_in[1])
            return x
        n = from_magnetization(n_in)
        n_spins = n.shape[0]  # always 1 or 2 (local basis in vector case)
        # --- density gradient:
        if self.need_sigma:
            Dn_in = (~(n_t.gradient(dim=1))).data
            Dn = from_magnetization(Dn_in)
            sigma = torch.empty((2*n_spins-1,) + n.shape[1:], dtype=n.dtype,
                                device=n.device)
            for s1 in range(n_spins):
                for s2 in range(s1, n_spins):
                    sigma[s1+s2] = (Dn[s1] * Dn[s2]).sum(dim=0)
        else:
            sigma = torch.tensor(0., device=n.device)
        # --- laplacian:
        if self.need_lap:
            lap_in = (~(n_t.laplacian())).data
            lap = from_magnetization(lap_in)
        else:
            lap = torch.tensor(0., device=n.device)
        # --- KE density:
        if self.need_tau:
            tau_in = (~tau_t).data
            tau = from_magnetization(tau_in)
        else:
            tau = torch.tensor(0., device=n.device)

        # Clamp low densities for numerical stability:
        clamp_sel = torch.where(n < N_CUT)
        n[clamp_sel] = N_CUT

        # Evaluate functionals:
        n.grad = torch.zeros_like(n)
        sigma.grad = torch.zeros_like(sigma)
        lap.grad = torch.zeros_like(lap)
        tau.grad = torch.zeros_like(tau)
        E_by_dV = 0.
        for functional in self._functionals:
            E_by_dV += functional(n, sigma, lap, tau)

        # Gradient propagation for potential:
        def from_magnetization_grad(E_x: torch.Tensor,
                                    x_in: Optional[torch.Tensor] = None
                                    ) -> torch.Tensor:
            """Gradient propagation corresponding to `from_magnetization`.
            Returns the gradient contribution to `E_x_in` from `E_x`.
            In vector-spin mode, this also contributes to `E_n_in`, even
            when `x` is not `n` because `n` determines `Mhat`.
            Parameter `x_in` must be provided when x is not n."""
            if n_densities == 1:
                return E_x
            E_x_in = torch.empty((n_densities,) + E_x.shape[1:],
                                 dtype=E_x.dtype, device=E_x.device)
            E_x_in[0] = 0.5 * (E_x[0] + E_x[1])
            E_x_diff = 0.5 * (E_x[0] - E_x[1])
            if n_densities == 4:
                E_x_in[1:] = E_x_diff * Mhat
                if x_in is not None:
                    x_vec = x_in[1:]
                    E_n_in[1:] += E_x_diff * MmagInv * (
                        x_vec - Mhat * (Mhat * x_vec).sum(dim=0))
            else:  # n_densities == 2:
                E_x_in[1] = E_x_diff
            return E_x_in

        n.grad[clamp_sel] = 0.  # account for any clamping
        E_n_in = from_magnetization_grad(n.grad)
        # --- contributions from GGA gradients:
        if self.need_sigma:
            E_Dn = torch.zeros_like(Dn)
            for s1 in range(n_spins):
                for s2 in range(s1, n_spins):
                    E_Dn[s1] += sigma.grad[s1+s2] * Dn[s2]
                    E_Dn[s2] += sigma.grad[s1+s2] * Dn[s1]
            E_Dn_in = from_magnetization_grad(E_Dn, Dn_in)
            E_n_t = -(~qp.grid.FieldR(grid, data=E_Dn_in)).divergence(dim=1)
        else:
            E_n_t = n_t.zeros_like()
        # --- contributions from Laplacian:
        if self.need_lap:
            E_lap_in = from_magnetization_grad(lap.grad, lap_in)
            E_n_t += (~qp.grid.FieldR(grid, data=E_lap_in)).laplacian()
        # --- contributions from KE density:
        if self.need_tau:
            E_tau_in = from_magnetization_grad(tau.grad, tau_in)
            E_tau_t = ~qp.grid.FieldR(grid, data=E_tau_in)
        else:
            E_tau_t = tau_t.zeros_like()
        # --- direct n contributions
        E_n_t += ~qp.grid.FieldR(grid, data=E_n_in)

        # Collect energy
        E = grid.dV * E_by_dV
        if grid.comm is not None:
            E = grid.comm.allreduce(E, qp.MPI.SUM)
        watch.stop()
        return E, E_n_t, E_tau_t
