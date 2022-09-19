from __future__ import annotations
import qimpy as qp
import torch
from typing import Optional, Sequence


class SCF(qp.utils.Pulay[qp.grid.FieldH]):
    """Electronic self-consistent field iteration."""

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

    def __init__(
        self,
        *,
        comm: qp.MPI.Comm,
        checkpoint_in: qp.utils.CpPath = qp.utils.CpPath(),
        n_iterations: int = 50,
        energy_threshold: float = 1e-8,
        residual_threshold: float = 1e-7,
        n_consecutive: int = 2,
        n_history: int = 10,
        mix_fraction: float = 0.5,
        mix_fraction_mag: float = 1.5,
        q_kerker: float = 0.8,
        q_metric: float = 0.8,
        q_kappa: Optional[float] = None,
        n_eig_steps: int = 2,
        eig_threshold: float = 1e-8,
        mix_density: bool = True
    ):
        """Initialize parameters of self-consistent field iteration (SCF).

        Parameters
        ----------
        n_iterations
            :yaml:`Number of self-consistent field iterations / cycles.`
        energy_threshold
            :yaml:`Energy convergence threshold in Hartrees.`
            Stop when energy difference between consecutive iterations falls
            below this threshold.
        residual_threshold
            :yaml:`Residual-norm convergence threshold.`
            Stop when the norm of the residual i.e. difference in mixed
            variable (density / potential) between consecutive iterations
            falls below this threshold.
        n_consecutive
            :yaml:`Number of consecutive iterations each threshold must be satisfied.`
        n_history
            :yaml:`History size for Pulay mixing.`
            This sets the number of previous residuals and variables to use
            in the Pulay mixing algorithm. Larger history could improve
            convergence, while requiring more memory.
        mix_fraction
            :yaml:`Fraction of new variable mixed into current variable.`
            Lower values (< 0.5) can improve stability, while higher values
            (0.5 - 1) attempt more aggressive convergence.
        mix_fraction_mag
            :yaml:`Different mix_fraction for magnetization components.`
            More aggressive fractions (> 1) are typically required to
            converge the magnetization degrees of freedom, because they
            tend to contribute less to the overall energy of the system
            (compared to the overall electron density / potential).
        q_kerker
            :yaml:`Characteristic wavevector for Kerker mixing.`
        q_metric
            :yaml:`Characteristic wavevector controlling Pulay metric.`
        q_kappa
            :yaml:`Long-range cutoff wavevector for grand-canonical SCF.`
            If unspecified, set based on Debye screening length.
        n_eig_steps
            :yaml:`Number of inner eigenvalue iterations for each SCF cycle.`
        eig_threshold
            :yaml:`Convergence threshold on eigenvalues in Hartrees.`
            Stop when the maximum change in any eigenvalue between SCF cycles
            falls below this threshold.
        mix_density
            :yaml:`Whether to mix density or potential.`
            Mix density if True, and potential if False.
        """
        self.mix_fraction_mag = mix_fraction_mag
        self.q_kerker = q_kerker
        self.q_metric = q_metric
        self.q_kappa = q_kappa
        self.n_eig_steps = n_eig_steps
        self.eig_threshold = float(eig_threshold)
        self.mix_density = mix_density
        super().__init__(
            comm=comm,
            name="SCF",
            checkpoint_in=checkpoint_in,
            n_iterations=n_iterations,
            energy_threshold=float(energy_threshold),
            residual_threshold=float(residual_threshold),
            n_consecutive=n_consecutive,
            extra_thresholds={"|deig|": self.eig_threshold},
            n_history=n_history,
            mix_fraction=mix_fraction,
        )

    def update(self, system: qp.System) -> None:
        self.system = system
        # Initialize preconditioner and metric:
        grid = system.grid
        iG = grid.get_mesh("H").to(torch.double)  # half-space
        Gsq = ((iG @ grid.lattice.Gbasis.T) ** 2).sum(dim=-1)
        # --- regularize Gsq by q_kappa or min(G!=0) as appropriate
        Gsq_min = qp.utils.globalreduce.min(Gsq[Gsq > 0.0], self.comm)
        q_kappa_sq = 0.0 if (self.q_kappa is None) else (self.q_kappa ** 2)
        Gsq_reg = (Gsq + q_kappa_sq) if q_kappa_sq else torch.clamp(Gsq, min=Gsq_min)
        # --- compute kernels
        q_kerker_sq = self.q_kerker ** 2
        q_metric_sq = self.q_metric ** 2
        self.K_kerker = Gsq_reg / (Gsq_reg + q_kerker_sq)
        self.K_metric = (
            (1.0 + q_metric_sq / Gsq_reg)
            if self.mix_density
            else Gsq_reg / (Gsq_reg + q_metric_sq)
        )
        self.K_metric *= grid.lattice.volume * grid.weight2H  # integration weight
        # Initialize electronic energy for current state:
        system.electrons.update(system)

    def cycle(self, dEprev: float) -> Sequence[float]:
        electrons = self.system.electrons
        eig_prev = electrons.eig[..., : electrons.fillings.n_bands]
        eig_threshold_inner = min(1e-6, 0.1 * abs(dEprev))
        electrons.diagonalize(
            n_iterations=self.n_eig_steps, eig_threshold=eig_threshold_inner
        )
        electrons.update(self.system)  # update total energy
        # Compute eigenvalue difference for extra convergence threshold:
        eig_cur = electrons.eig[..., : electrons.fillings.n_bands]
        deig = (eig_cur - eig_prev).abs()
        deig_max = qp.utils.globalreduce.max(deig, electrons.comm)
        return [deig_max]

    @property
    def energy(self) -> qp.Energy:
        return self.system.energy

    @property
    def variable(self) -> qp.grid.FieldH:
        """Get density or potential, depending on `mix_density`."""
        electrons = self.system.electrons
        return electrons.n_tilde if self.mix_density else electrons.n_tilde.grad

    @variable.setter
    def variable(self, v: qp.grid.FieldH) -> None:
        """Set density or potential, depending on `mix_density`."""
        electrons = self.system.electrons
        if self.mix_density:
            electrons.n_tilde = v
            electrons.update_potential(self.system)
        else:
            electrons.n_tilde.grad = v

    def precondition(self, v: qp.grid.FieldH) -> qp.grid.FieldH:
        result = v.convolve(self.K_kerker)
        if result.data.shape[0] > 1:  # Different fraction for magnetization
            result.data[1:] *= self.mix_fraction_mag / self.mix_fraction
        return result

    def metric(self, v: qp.grid.FieldH) -> qp.grid.FieldH:
        return v.convolve(self.K_metric)
