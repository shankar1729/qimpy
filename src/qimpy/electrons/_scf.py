import qimpy as qp
import numpy as np
import torch
from typing import Optional, Sequence, TYPE_CHECKING
if TYPE_CHECKING:
    from ..utils import RunConfig
    from .._energy import Energy
    from ..grid import FieldH
    from .._system import System


class SCF(qp.utils.Pulay['FieldH']):
    """Electronic self-consistent field iteration."""
    __slots__ = ('mix_fraction_mag', 'q_kerker', 'q_metric', 'q_kappa',
                 'n_eig_steps', 'eig_threshold', 'mix_density',
                 'system', 'K_kerker', 'K_metric')
    mix_fraction_mag: float  #: Mix-fraction for magnetization
    q_kerker: float  #: Kerker-mixing wavevector
    q_metric: float  #: Wavevector controlling reciprocal-space metric
    q_kappa: Optional[float]  #: Debye wavevector (automatic if None)
    n_eig_steps: int  #: Number of eigenvalue steps per cycle
    eig_threshold: float  #: Eigenvalue convergence threshold
    mix_density: bool  #: Mix density if True, else mix potential
    system: 'System'  #: Current system being optimized
    K_kerker: torch.Tensor  #: Kernel for Kerker mixing (preconditioner)
    K_metric: torch.Tensor  #: Kernel for metric used in Pulay overlaps

    def __init__(self, *, rc: 'RunConfig', comm: qp.MPI.Comm,
                 n_iterations: int = 50, energy_threshold: float = 1e-8,
                 residual_threshold: float = 1e-7, n_history: int = 10,
                 mix_fraction: float = 0.5, mix_fraction_mag: float = 1.5,
                 q_kerker: float = 0.8, q_metric: float = 0.8,
                 q_kappa: Optional[float] = None, n_eig_steps: int = 2,
                 eig_threshold: float = 1e-8, mix_density: bool = True):
        self.mix_fraction_mag = mix_fraction_mag
        self.q_kerker = q_kerker
        self.q_metric = q_metric
        self.q_kappa = q_kappa
        self.n_eig_steps = n_eig_steps
        self.eig_threshold = eig_threshold
        self.mix_density = mix_density
        super().__init__(rc=rc, comm=comm, name='SCF',
                         n_iterations=n_iterations,
                         energy_threshold=energy_threshold,
                         residual_threshold=residual_threshold,
                         extra_thresholds={'|deig|': eig_threshold},
                         n_history=n_history, mix_fraction=mix_fraction)

    def update(self, system: 'System') -> None:
        self.system = system
        # Initialize preconditioner and metric:
        grid = system.grid
        iG = grid.get_mesh('H').to(torch.double)  # half-space
        Gsq = ((iG @ grid.lattice.Gbasis.T) ** 2).sum(dim=-1)
        # --- regularize Gsq by q_kappa or min(G!=0) as appropriate
        Gsq_min = self.comm.allreduce(Gsq[Gsq > 0.].min().item(), qp.MPI.MIN)
        q_kappa_sq = 0. if (self.q_kappa is None) else (self.q_kappa ** 2)
        Gsq_reg = ((Gsq + q_kappa_sq) if q_kappa_sq
                   else torch.clamp(Gsq, min=Gsq_min))
        # --- compute kernels
        q_kerker_sq = self.q_kerker ** 2
        q_metric_sq = self.q_metric ** 2
        self.K_kerker = Gsq_reg / (Gsq_reg + q_kerker_sq)
        self.K_metric = ((1. + q_metric_sq/Gsq_reg) if self.mix_density
                         else Gsq_reg / (Gsq_reg + q_metric_sq))
        # Initialize electronic energy for current state:
        system.electrons.update(system)

    def cycle(self, dEprev: float) -> Sequence[float]:
        electrons = self.system.electrons
        eig_prev = electrons.eig[..., :electrons.n_bands]
        eig_threshold_inner = min(1e-6, 0.1*abs(dEprev))
        electrons.diagonalize(n_iterations=self.n_eig_steps,
                              eig_threshold=eig_threshold_inner)
        electrons.update(self.system)  # update total energy
        # Compute eigenvalue difference for extra convergence threshold:
        deig = (electrons.eig[..., :electrons.n_bands] - eig_prev).abs()
        deig_max = self.rc.comm_kb.allreduce(deig.max().item(), qp.MPI.MAX)
        return [deig_max]

    @property
    def energy(self) -> 'Energy':
        return self.system.energy

    @property
    def variable(self) -> 'FieldH':
        """Get density or potential, depending on `mix_density`."""
        electrons = self.system.electrons
        return electrons.n if self.mix_density else electrons.V_ks

    @variable.setter
    def variable(self, v: 'FieldH') -> None:
        """Set density or potential, depending on `mix_density`."""
        electrons = self.system.electrons
        if self.mix_density:
            electrons.n = v
            electrons.update_potential(self.system)
        else:
            electrons.V_ks = v

    def precondition(self, v: 'FieldH') -> 'FieldH':
        result = qp.grid.FieldH(v.grid, data=(v.data * self.K_kerker))
        if result.data.shape[0] > 1:  # Different fraction for magnetization
            result.data[1:] *= (self.mix_fraction_mag / self.mix_fraction)
        return result

    def metric(self, v: 'FieldH') -> 'FieldH':
        return qp.grid.FieldH(v.grid, data=(v.data * self.K_metric))
