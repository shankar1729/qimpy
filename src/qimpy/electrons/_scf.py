from __future__ import annotations
import qimpy as qp
import numpy as np
import torch
from typing import Optional, Sequence


class SCF(qp.utils.Pulay[qp.grid.FieldH]):
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
    system: qp.System  #: Current system being optimized
    K_kerker: torch.Tensor  #: Kernel for Kerker mixing (preconditioner)
    K_metric: torch.Tensor  #: Kernel for metric used in Pulay overlaps

    def __init__(self, *, co: qp.ConstructOptions, comm: qp.MPI.Comm,
                 n_iterations: int = 50, energy_threshold: float = 1e-8,
                 residual_threshold: float = 1e-7, n_history: int = 10,
                 mix_fraction: float = 0.5, mix_fraction_mag: float = 1.5,
                 q_kerker: float = 0.8, q_metric: float = 0.8,
                 q_kappa: Optional[float] = None, n_eig_steps: int = 2,
                 eig_threshold: float = 1e-8, mix_density: bool = True):
        """Initialize parameters of self-consistent field iteration (SCF).

        Parameters
        ----------
        n_iterations
            Number of self-consistent field iterations / cycles.
            :yaml:`inputfile`
        energy_threshold
            Convergence threshold on energy difference (in :math:`E_h`) between
            consecutive iterations. :yaml:`inputfile`
        residual_threshold
            Convergence threshold on the norm of the residual i.e. difference
            in mixed variable (density / potential) between consecutive
            iterations. :yaml:`inputfile`
        n_history
            Number of previous residuals and variables to use in the Pulay
            mixing algorithm. :yaml:`inputfile`
        mix_fraction
            Fraction of new variable (density / poitential) to mix into the
            current variable. :yaml:`inputfile`
        mix_fraction_mag
            Different `mix_fraction` for magnetization degrees of freedom.
            :yaml:`inputfile`
        q_kerker
            Characteristic wavevector for Kerker mixing. :yaml:`inputfile`
        q_metric
            Characteristic wavevector controlling Pulay metric.
            :yaml:`inputfile`
        q_kappa
            Long-range cutoff wavevector important for grand-canonical SCF.
            If unspecified, set based on Debye screening length.
            :yaml:`inputfile`
        n_eig_steps
            Number of inner eigenvalue iterations for each SCF cycle.
            :yaml:`inputfile`
        eig_threshold
            Convergence threshold on maximum eigenvalue change (in :math:`E_h`)
            between SCF cycles. :yaml:`inputfile`
        mix_density
            Whether to mix density (if True) or potential (if False).
            :yaml:`inputfile`
        """
        self.mix_fraction_mag = mix_fraction_mag
        self.q_kerker = q_kerker
        self.q_metric = q_metric
        self.q_kappa = q_kappa
        self.n_eig_steps = n_eig_steps
        self.eig_threshold = float(eig_threshold)
        self.mix_density = mix_density
        super().__init__(co=co, comm=comm, name='SCF',
                         n_iterations=n_iterations,
                         energy_threshold=float(energy_threshold),
                         residual_threshold=float(residual_threshold),
                         extra_thresholds={'|deig|': self.eig_threshold},
                         n_history=n_history, mix_fraction=mix_fraction)

    def update(self, system: qp.System) -> None:
        self.system = system
        # Initialize preconditioner and metric:
        grid = system.grid
        iG = grid.get_mesh('H').to(torch.double)  # half-space
        Gsq = ((iG @ grid.lattice.Gbasis.T) ** 2).sum(dim=-1)
        # --- regularize Gsq by q_kappa or min(G!=0) as appropriate
        Gsq_min = qp.utils.globalreduce.min(Gsq[Gsq > 0.], self.comm)
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
        eig_prev = electrons.eig[..., :electrons.fillings.n_bands]
        eig_threshold_inner = min(1e-6, 0.1*abs(dEprev))
        electrons.diagonalize(n_iterations=self.n_eig_steps,
                              eig_threshold=eig_threshold_inner)
        electrons.update(self.system)  # update total energy
        # Compute eigenvalue difference for extra convergence threshold:
        eig_cur = electrons.eig[..., :electrons.fillings.n_bands]
        deig = (eig_cur - eig_prev).abs()
        deig_max = qp.utils.globalreduce.max(deig, self.rc.comm_kb)
        return [deig_max]

    @property
    def energy(self) -> qp.Energy:
        return self.system.energy

    @property
    def variable(self) -> qp.grid.FieldH:
        """Get density or potential, depending on `mix_density`."""
        electrons = self.system.electrons
        return electrons.n if self.mix_density else electrons.V_ks

    @variable.setter
    def variable(self, v: qp.grid.FieldH) -> None:
        """Set density or potential, depending on `mix_density`."""
        electrons = self.system.electrons
        if self.mix_density:
            electrons.n = v
            electrons.update_potential(self.system)
        else:
            electrons.V_ks = v

    def precondition(self, v: qp.grid.FieldH) -> qp.grid.FieldH:
        result = qp.grid.FieldH(v.grid, data=(v.data * self.K_kerker))
        if result.data.shape[0] > 1:  # Different fraction for magnetization
            result.data[1:] *= (self.mix_fraction_mag / self.mix_fraction)
        return result

    def metric(self, v: qp.grid.FieldH) -> qp.grid.FieldH:
        return qp.grid.FieldH(v.grid, data=(v.data * self.K_metric))
