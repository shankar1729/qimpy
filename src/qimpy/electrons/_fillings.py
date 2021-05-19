import qimpy as qp
import numpy as np
import torch
import collections
from typing import Optional, Union, TYPE_CHECKING
if TYPE_CHECKING:
    from ..utils import RunConfig
    from ..ions import Ions
    from ._electrons import Electrons


SmearingResults = collections.namedtuple('SmearingResults', ['f', 'f_E', 'S'])


class Fillings:
    '''Electron occupation factors (smearing)'''
    __slots__ = ('rc', 'n_electrons', 'n_bands_min', 'smearing',
                 'sigma', '_smearing_func')
    rc: 'RunConfig'  #: Current run configuration
    n_electrons: float  #: Number of electrons
    n_bands_min: int  #: Minimum number of bands to accomodate `n_electrons`
    smearing: Optional[str]  #: Smearing method name
    sigma: Optional[float]  #: Gaussian width (:math:`2k_BT` for Fermi)

    def __init__(self, *,
                 rc: 'RunConfig', ions: 'Ions', electrons: 'Electrons',
                 charge: float = 0., smearing: str = 'gauss',
                 sigma: Optional[float] = None,
                 kT: Optional[float] = None) -> None:
        '''Initialize occupation factor (smearing) scheme.

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
        '''
        self.rc = rc

        # Number of electrons and bands:
        self.n_electrons = ions.Z_tot - charge
        self.n_bands_min = int(np.ceil(self.n_electrons / electrons.w_spin))
        qp.log.info(f'n_electrons: {self.n_electrons:g}'
                    f'  n_bands_min: {self.n_bands_min}')

        # Smearing:
        smear_options = _smearing_funcs.keys()
        self.smearing = smearing.lower() if smearing else None
        self.sigma = None
        if self.smearing:
            self._smearing_func = _smearing_funcs.get(self.smearing, None)
            if self._smearing_func is None:
                raise KeyError('smearing must be None/False or one of '
                               + str(_smearing_funcs.keys()))
            if sigma and kT:
                raise ValueError('specify only one of sigma or kT')
            self.sigma = float(sigma if sigma  # get from sigma
                               else ((2*kT) if kT  # get from kT
                                     else 0.002))  # default value
        sigma_str = (f'{self.sigma:g} (equivalent kT: {0.5*self.sigma:g})'
                     if self.sigma else str(None))
        qp.log.info(f'smearing: {self.smearing}  sigma: {sigma_str}')

    def compute(self, E: torch.Tensor, mu: float,
                extra_outputs=False) -> Union[SmearingResults, torch.Tensor]:
        '''Compute occupations for energies `E` and chemical potential `mu`.
        Optionally also return the energy derivative of the occupation factors
        and corresponding entropy contributions if `extra_outputs` = True.

        Parameters
        ----------
        E
            Orbital energies
        mu
            Electron chemical potential
        extra_outputs : bool, default: False
            Whether to also return energy derivative and entropy contribution
            corresponding to each occupation factor (in a named tuple)

        Returns
        -------
        f : Tensor
            Occupations
        f_E : Tensor, only if extra_outputs=True
            Energy derivative (df/dE)
        S : Tensor, only if extra_outputs=True
            Entropy contribution
        '''
        assert(self.sigma is not None)
        assert(self._smearing_func is not None)
        return self._smearing_func(E, mu, self.sigma, extra_outputs)


def _smearing_fermi(E: torch.Tensor, mu: float, sigma: float,
                    extra_outputs=False) -> Union[SmearingResults,
                                                  torch.Tensor]:
    '''Compute Fermi-Dirac occupations, and optionally also its
    derivative and entropy in a named tuple if extra_outputs=True.
    Note that sigma is taken as 2 kT to keep width consistent.'''
    f = torch.sigmoid((mu - E)/(0.5*sigma))
    if extra_outputs:
        f_E = f * (1 - f) / (-0.5*sigma)
        S = -f.xlogy(f) - (1-f).xlogy(1-f)
        return SmearingResults(f, f_E, S)
    return f


def _smearing_gauss(E: torch.Tensor, mu: float, sigma: float,
                    extra_outputs=False) -> Union[SmearingResults,
                                                  torch.Tensor]:
    '''Compute Gaussian (erfc) occupations, and optionally also its
    derivative and entropy in a named tuple if extra_outputs=True'''
    x = (E - mu) / sigma
    f = 0.5*torch.erfc(x)
    if extra_outputs:
        S = torch.exp(-x*x) / np.sqrt(np.pi)
        f_E = (-1./sigma) * S
        return SmearingResults(f, f_E, S)
    return f


def _smearing_mp1(E: torch.Tensor, mu: float, sigma: float,
                  extra_outputs=False) -> Union[SmearingResults, torch.Tensor]:
    '''Compute first-order Methfessel-Paxton occupations, and optionally also
    its derivative and entropy in a named tuple if extra_outputs=True'''
    x = (E - mu) / sigma
    gaussian = torch.exp(-x*x) / np.sqrt(np.pi)
    f = 0.5*(torch.erfc(x) - x * gaussian)
    if extra_outputs:
        f_E = (x*x - 1.5) * gaussian / sigma
        S = (0.5 - x*x) * gaussian
        return SmearingResults(f, f_E, S)
    return f


def _smearing_cold(E: torch.Tensor, mu: float, sigma: float,
                   extra_outputs=False) -> Union[SmearingResults,
                                                 torch.Tensor]:
    '''Compute Cold smearing occupations, and optionally also
    its derivative and entropy in a named tuple if extra_outputs=True'''
    x = (E - mu) / sigma + np.sqrt(0.5)  # note: not centered at mu
    sqrt2 = np.sqrt(2.)
    gaussian = torch.exp(-x*x) / np.sqrt(np.pi)
    f = 0.5*(torch.erfc(x) + sqrt2*gaussian)
    if extra_outputs:
        f_E = -gaussian * (1 + x*sqrt2) / sigma
        S = gaussian * x * sqrt2
        return SmearingResults(f, f_E, S)
    return f


_smearing_funcs = {
    'fermi': _smearing_fermi,
    'gauss': _smearing_gauss,
    'mp1': _smearing_mp1,
    'cold': _smearing_cold}


if __name__ == '__main__':
    # Check derivatives and plot comparison of smearing functions:
    import matplotlib.pyplot as plt
    torch.set_default_tensor_type(torch.DoubleTensor)
    Ef = 0.3
    kT = 0.01
    E = torch.linspace(Ef-20*kT, Ef+20*kT, 4001)
    dE = E[1] - E[0]
    for name, func in _smearing_funcs.items():
        f, f_E, S = func(E, Ef, 2*kT, extra_outputs=True)
        f_E_num = (f[2:] - f[:-2])/(2*dE)
        f_E_err = (f_E[1:-1] - f_E_num).norm() / f_E_num.norm()
        print(f'{name:>5s}:  Err(f_E): {f_E_err:.2e}'
              f'  integral(S): {S.sum()*dE:f}')
        e = (E - Ef)/kT  # dimensionless energy
        for i, (result, result_name) in enumerate([
                (f, '$f$'), (f_E, '$dF/dE$'), (S, '$S$')]):
            plt.figure(i)
            plt.plot(e, result, label=name)
            plt.xlabel(r'$(E-E_f)/k_BT$')
            plt.ylabel(result_name)
            plt.legend()
    plt.show()
