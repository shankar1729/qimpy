"""Internal interface for XC functionals."""
# List exported symbols for doc generation
__all__ = ['Functional', 'LIBXC_AVAILABLE', 'get_libxc_functional_names']

from abc import abstractmethod, ABC
import qimpy as qp
import torch

try:
    import pylibxc
    LIBXC_AVAILABLE: bool = True  #: Whether Libxc is available.
except ImportError:
    LIBXC_AVAILABLE = False

from typing import Set
from functools import lru_cache


class Functional(ABC):
    """Abstract base class for exchange-correlation functionals."""
    __slots__ = ('needs_sigma', 'needs_lap', 'needs_tau',
                 'has_exchange', 'has_correlation', 'has_kinetic',
                 'has_energy', 'scale_factor')
    needs_sigma: bool  #: Whether functional needs gradient :math:`\sigma`
    needs_lap: bool  #: Whether functional needs Laplacian :math:`\nabla^2 n`
    needs_tau: bool  #: Whether functional needs KE density :math:`\tau`
    has_exchange: bool  #: Whether functional includes exchange
    has_correlation: bool  #: Whether functional includes correlation
    has_kinetic: bool  #: Whether functional includes kinetic energy
    has_energy: bool  #: Whether functional has meaningful total energy

    def __init__(self, *, needs_sigma: bool = False, needs_lap: bool = False,
                 needs_tau: bool = False, has_exchange: bool = False,
                 has_correlation: bool = False, has_kinetic: bool = False,
                 has_energy: bool = True, scale_factor: float = 1.,
                 name: str = '') -> None:
        self.needs_sigma = needs_sigma
        self.needs_lap = needs_lap
        self.needs_tau = needs_tau
        self.has_exchange = has_exchange
        self.has_correlation = has_correlation
        self.has_kinetic = has_kinetic
        self.has_energy = has_energy
        self.scale_factor = scale_factor
        if name:
            scale_str = ('' if (scale_factor == 1.)
                         else f' (scaled by {scale_factor})')
            qp.log.info(f'  {name} functional{scale_str}.')

    @abstractmethod
    def __call__(self, n: torch.Tensor, sigma: torch.Tensor,
                 lap: torch.Tensor, tau: torch.Tensor) -> float:
        """Compute exchange/correlation/kinetic functional for several points.
        The first dimension of each tensor corresponds to spin channels,
        and all subsequent dimenions are grid points.
        Gradients with respect to each input should be accumulated to the
        corresponding `grad` fields (eg. `n.grad`), allowing convenient
        internal use of torch's autograd functionality wherever applicable.

        Parameters
        ----------
        n
            Electron density: 1 or 2 spin channels (up/dn)
        sigma
            Density gradient: 1 or 3 spin channels (up-up, up-dn, dn-dn)
        lap
            Laplacian: 1 or 2 spin channels
        tau
            Kinetic energy density: 1 or 2 spin channels

        Returns
        -------
        Total energy density, summed over all input grid points.
        """


@lru_cache
def get_libxc_functional_names() -> Set[str]:
    """Get set of available Libxc functionals.
    (Empty if Libxc is not available.)"""
    if LIBXC_AVAILABLE:
        return set(pylibxc.util.xc_available_functional_names())
    else:
        return set()
