import qimpy as qp
import numpy as np
import torch
import collections
from scipy import optimize
from typing import Optional, Dict, Union, Callable, TYPE_CHECKING
if TYPE_CHECKING:
    from ..utils import RunConfig
    from ..ions import Ions
    from ._electrons import Electrons
    from .._system import System


SmearingResults = collections.namedtuple('SmearingResults',
                                         ['f', 'f_eig', 'S'])
SmearingFunc = Callable[[torch.Tensor, float, float], SmearingResults]


class Fillings:
    """Electron occupation factors (smearing)"""
    __slots__ = ('rc', 'n_electrons', 'n_bands_min', 'smearing',
                 'sigma', 'M_initial', 'M_constrain', '_smearing_func',
                 'mu', 'M', 'f')
    rc: 'RunConfig'  #: Current run configuration
    n_electrons: float  #: Number of electrons
    n_bands_min: int  #: Minimum number of bands to accomodate `n_electrons`
    smearing: Optional[str]  #: Smearing method name
    sigma: Optional[float]  #: Gaussian width (:math:`2k_BT` for Fermi)
    M_initial: float  #: Initial total magnetization
    M_constrain: float  #: Whether to constrain magnetization to `M_initial`
    mu: float  #: Electron chemical potential
    M: torch.Tensor  #: Total magnetization (vector if spinorial)
    f: torch.Tensor  #: Electronic occupations
    _smearing_func: Optional[SmearingFunc]  #: Smearing function calculator

    def __init__(self, *,
                 rc: 'RunConfig', ions: 'Ions', electrons: 'Electrons',
                 charge: float = 0., smearing: str = 'gauss',
                 sigma: Optional[float] = None,
                 kT: Optional[float] = None,
                 M_initial: float = 0., M_constrain: bool = False) -> None:
        """Initialize occupation factor (smearing) scheme.

        Parameters
        ----------
        charge : float, default: 0.
            Net charge of electrons + ions in e units, which determines
            n_electrons = ions.Z_tot - charge.
        smearing : {'gauss', 'fermi', 'cold', 'mp1', False}, default: 'gauss'
            Smearing method for setting electron occupations, where 'gauss',
            'fermi', 'cold', 'mp1' select Gaussian, Fermi-Dirac, Cold and
            first order Methfessel-Paxton (MP1) smearing respectively.
            Use False (or None) to disable smearing and keep the electron
            occupations fixed at their initial values.
        sigma : float, default: 0.002
            Width of the smearing function (in :math:`E_h`), corresponding
            to the Gaussian width :math:`\\sigma` in the Gaussian, Cold and
            M-P schemes, and to :math:`2k_BT` in the Fermi-Dirac scheme.
        kT : float, default: 0.001
            :math:`k_BT` for Fermi-Dirac occupations, amounting to
            :math:`\\sigma/2` for the other Gaussian-based smearing schemes.
            Specify only one of sigma or kT.
        M_initial: float, default: 0.
            Total magnetization (only for non-spinorial spin-polarized mode).
            This magnetization is assigned to the initial occupations and it
            may change when smearing is present depending on `M_constrain`.
        M_constrain: bool, default: False
            Whether to hold magnetization fixed to `M_initial` in occupation
            updates: this only matters when `smearing` is not None.
        """
        self.rc = rc

        # Number of electrons and bands:
        self.n_electrons = ions.Z_tot - charge
        self.n_bands_min = int(np.ceil(self.n_electrons / electrons.w_spin))

        # Smearing:
        self.smearing = smearing.lower() if smearing else None
        self.sigma = None
        if self.smearing:
            self._smearing_func = _smearing_funcs.get(self.smearing, None)
            if self._smearing_func is None:
                raise KeyError('smearing must be None/False or one of '
                               f'{_smearing_funcs.keys()}')
            if sigma and kT:
                raise ValueError('specify only one of sigma or kT')
            self.sigma = float(sigma if sigma  # get from sigma
                               else ((2*kT) if kT  # get from kT
                                     else 0.002))  # default value
            self.n_bands_min += 1  # need at least one extra empty band
        sigma_str = (f'{self.sigma:g} (equivalent kT: {0.5*self.sigma:g})'
                     if self.sigma else str(None))
        qp.log.info(f'n_electrons: {self.n_electrons:g}'
                    f'  n_bands_min: {self.n_bands_min}')
        qp.log.info(f'smearing: {self.smearing}  sigma: {sigma_str}')

        # Magnetization mode:
        if M_initial and ((not electrons.spin_polarized)
                          or electrons.spinorial):
            raise ValueError('M_initial only allowed for spin-polarized'
                             ' and non-spinorial calculations')
        self.M_initial = M_initial
        self.M_constrain = M_constrain
        if M_initial:
            qp.log.info(f'M: initial: {self.M_initial}'
                        f'  constrained: {self.M_constrain}')

    def update(self, system: 'System') -> None:
        """Update fillings and chemical potential, if needed.
        Set updated fillings in `system.electrons` and corresponding
        energy comonents in `system.energy`, initializing fillings
        if not already done so.
        """
        # Initialize fillings if necessary:
        electrons = system.electrons
        if not hasattr(electrons, "f"):
            # Filings sum for each spin channel:
            f_sums = (np.ones(electrons.n_spins) * self.n_electrons
                      / (electrons.w_spin * electrons.n_spins))
            if electrons.spin_polarized and self.M_initial:
                assert electrons.n_spins == 2  # must be noncollinear
                f_sums[0] += 0.5 * self.M_initial
                f_sums[1] -= 0.5 * self.M_initial
                if f_sums.min() < 0:
                    raise ValueError(f'n_electrons = {self.n_electrons:g}'
                                     f' insufficient to support M_initial'
                                     f' = {self.M_initial:g}')
                if f_sums.max() > electrons.n_bands:
                    raise ValueError(f'n_bands = {electrons.n_bands:g}'
                                     f' insufficient to support M_initial'
                                     f' = {self.M_initial:g}')
            # Initialize fillings based on sum in each channel:
            self.mu = np.nan
            self.M = torch.tensor(self.M_initial, device=self.rc.device)
            self.f = torch.zeros_like(electrons.eig)
            for i_spin, f_sum in enumerate(f_sums):
                n_full = int(np.floor(f_sum))  # number of fully filled bands
                self.f[i_spin, :, :n_full] = 1.  # full fillings
                if f_sum > n_full:
                    self.f[i_spin, :, n_full] = (f_sum - n_full)  # left overs

        # Update fillings if necessary:
        if (self.sigma is not None) and (not np.isnan(electrons.deig_max)):
            assert self._smearing_func is not None
            w_sk = electrons.basis.w_sk

            def error_n_M(params):
                """Root function for electron number and magnetization
                constraints. Here, params is the set of Legendre multipliers
                for these constraints, including chemical potential
                and magnetization. Returns error and its gradient.
                """
                mu = params[0]
                f, f_eig, _ = self._smearing_func(electrons.eig, mu, sigma_cur)
                n = self.rc.comm_k.allreduce((w_sk * f).sum().item(),
                                             qp.MPI.SUM)
                n_mu = -self.rc.comm_k.allreduce((w_sk * f_eig).sum().item(),
                                                 qp.MPI.SUM)
                # Broadcast across replica for machine-precision consistency:
                n_err = np.array([self.rc.comm_kb.bcast(n - self.n_electrons)])
                n_err_mu = np.array([self.rc.comm_kb.bcast(n_mu)])
                # qp.log.info(f'sigma_cur: {sigma_cur:f}  n_err: {n_err[0]}'
                #            f'  n_mu: {n_err_mu[0]}')
                return n_err, n_err_mu
            # Bracket mu over range of eigenvalues (with margin):
            params0 = np.array([self.mu if (not np.isnan(self.mu)) else 0.])
            eig_diff_max = self.rc.comm_k.allreduce(
                electrons.eig.diff(dim=-1).max(), qp.MPI.MAX)
            sigma_cur = max(self.sigma, min(0.1, eig_diff_max))
            final_step = False
            while not final_step:
                final_step = (sigma_cur == self.sigma)
                xtol = 1e-8 if final_step else (0.1 * sigma_cur)
                result = optimize.root(error_n_M, params0, jac=True,
                                       method='lm',
                                       options={'xtol': xtol, 'ftol': 1e-12})
                params0 = result.x
                sigma_cur = max(self.sigma, 0.5 * sigma_cur)
            self.mu = params0[0]
            # Update fillings and entropy accordingly:
            self.f, _, S = self._smearing_func(electrons.eig, self.mu,
                                               self.sigma)
            system.energy['-TS'] = -self.sigma * self.rc.comm_k.allreduce(
                (w_sk * S).sum().item(), qp.MPI.SUM)
            # --- compute magnetization
            M_str = ''
            if electrons.spin_polarized:
                assert self.f.shape[0] == 2  # TODO: support vector-spin
                n_each = (w_sk * self.f).sum(dim=(1, 2))
                self.M = n_each[1] - n_each[0]
                self.rc.comm_k.Allreduce(qp.MPI.IN_PLACE,
                                         qp.utils.BufferView(self.M),
                                         qp.MPI.SUM)
                M_str = f'  M: {self.M.item():.5f}'
            qp.log.info(f'  FillingsUpdate:  mu: {self.mu:.9f}'
                        f'  n_electrons: {self.n_electrons:.6f}{M_str}')


def _smearing_fermi(eig: torch.Tensor, mu: float,
                    sigma: float) -> SmearingResults:
    """Compute Fermi-Dirac occupations, its energy derivative and entropy.
    Note that sigma is taken as 2 kT to keep width consistent."""
    f = torch.sigmoid((mu - eig)/(0.5*sigma))
    f_eig = f * (1 - f) / (-0.5*sigma)
    S = -f.xlogy(f) - (1-f).xlogy(1-f)
    return SmearingResults(f, f_eig, S)


def _smearing_gauss(eig: torch.Tensor, mu: float,
                    sigma: float) -> SmearingResults:
    """Compute Gaussian (erfc) occupations, energy derivative and entropy."""
    x = (eig - mu) / sigma
    f = 0.5*torch.erfc(x)
    S = torch.exp(-x*x) / np.sqrt(np.pi)
    f_eig = (-1./sigma) * S
    return SmearingResults(f, f_eig, S)


def _smearing_mp1(eig: torch.Tensor, mu: float,
                  sigma: float) -> SmearingResults:
    """Compute first-order Methfessel-Paxton occupations, energy derivative
    and entropy."""
    x = (eig - mu) / sigma
    gaussian = torch.exp(-x*x) / np.sqrt(np.pi)
    f = 0.5*(torch.erfc(x) - x * gaussian)
    f_eig = (x*x - 1.5) * gaussian / sigma
    S = (0.5 - x*x) * gaussian
    return SmearingResults(f, f_eig, S)


def _smearing_cold(eig: torch.Tensor, mu: float,
                   sigma: float) -> SmearingResults:
    """Compute Cold smearing occupations, energy derivative and entropy."""
    x = (eig - mu) / sigma + np.sqrt(0.5)  # note: not centered at mu
    sqrt2 = np.sqrt(2.)
    gaussian = torch.exp(-x*x) / np.sqrt(np.pi)
    f = 0.5*(torch.erfc(x) + sqrt2*gaussian)
    f_eig = -gaussian * (1 + x*sqrt2) / sigma
    S = gaussian * x * sqrt2
    return SmearingResults(f, f_eig, S)


_smearing_funcs: Dict[str, SmearingFunc] = {
    'fermi': _smearing_fermi,
    'gauss': _smearing_gauss,
    'mp1': _smearing_mp1,
    'cold': _smearing_cold}


if __name__ == '__main__':
    def main():
        """Check derivatives and plot comparison of smearing functions."""
        import matplotlib.pyplot as plt
        torch.set_default_tensor_type(torch.DoubleTensor)
        mu = 0.3
        kT = 0.01
        eig = torch.linspace(mu-20*kT, mu+20*kT, 4001)
        deig = eig[1] - eig[0]
        for name, func in _smearing_funcs.items():
            f, f_eig, S = func(eig, mu, 2*kT)
            f_eig_num = (f[2:] - f[:-2])/(2*deig)
            f_eig_err = (f_eig[1:-1] - f_eig_num).norm() / f_eig_num.norm()
            print(f'{name:>5s}:  Err(f_eig): {f_eig_err:.2e}'
                  f'  integral(S): {S.sum()*deig:f}')
            e = (eig - mu)/kT  # dimensionless energy
            for i, (result, result_name) in enumerate([
                    (f, '$f$'), (f_eig, r'$df/d\varepsilon$'), (S, '$S$')]):
                plt.figure(i)
                plt.plot(e, result, label=name)
                plt.xlabel(r'$(\varepsilon-\mu)/k_BT$')
                plt.ylabel(result_name)
                plt.legend()
        plt.show()
    main()
