from __future__ import annotations
import qimpy as qp
import numpy as np
import torch
from ..utils import Minimize, MinimizeState, MatrixArray


class LCAO(Minimize[MatrixArray]):
    """Optimize electronic state in atomic-orbital subspace."""
    __slots__ = ('system', '_rot_prev')
    system: qp.System
    _rot_prev: torch.Tensor  #: accumulated rotations of subspace

    def __init__(self, *, co: qp.ConstructOptions, n_iterations: int = 30,
                 energy_threshold: float = 1E-6,
                 gradient_threshold: float = 1E-8) -> None:
        """Set stopping criteria for initial subspace optimization.

        Parameters
        ----------
        energy_threshold
            :yaml:`Energy convergence threshold in Hartrees.`
            Stop when energy difference between consecutive LCAO iterations
            falls below this threshold.
        gradient_threshold
            :yaml:`Subspace-gradient convergence threshold (dimensionless).`
            Stop when gradient of energy with respect to subspace Hamiltonian
            falls below this threshold.
        """
        super().__init__(co=co, comm=co.rc.comm_kb, name='LCAO', method='cg',
                         n_iterations=n_iterations,
                         extra_thresholds={'|grad|': gradient_threshold},
                         energy_threshold=energy_threshold)

    def update(self, system: qp.System) -> None:
        """Set wavefunctions to optimum subspace of atomic orbitals."""
        el = system.electrons
        # Initialize based on reference atomic density (or fixed-H density):
        if not el.fixed_H:
            el.n_t = system.ions.get_atomic_density(system.grid, el.fillings.M)
            el.tau_t = qp.grid.FieldH(system.grid, shape_batch=(0,))  # TODO
            el.update_potential(system)
        C_OC = el.C.dot_O(el.C)
        C_HC = el.C ^ el.hamiltonian(el.C)
        el.eig, V = qp.utils.eighg(C_HC, C_OC)
        el.C = el.C @ V  # Set to eigenvectors
        if (el.fillings.smearing is None) or el.fixed_H:
            return

        # Subspace optimization:
        el.deig_max = np.inf  # allow fillings to use these eigenvalues
        self.system = system
        self._rot_prev = torch.eye(V.shape[-1], dtype=V.dtype,
                                   device=V.device)[None, None]
        self.minimize()

    def step(self, direction: MatrixArray, step_size: float) -> None:
        el = self.system.electrons
        # Move auxiliary Hamiltonian accounting for accumulated rotations:
        H_aux = (el.eig.diag_embed()  # current Hamiltonian
                 + step_size * (qp.utils.dagger(self._rot_prev)
                                @ (direction.M @ self._rot_prev)))
        # Update rotations to re-diagonalize auxiliary Hamiltonian
        el.eig, V = torch.linalg.eigh(H_aux)
        self._rot_prev = self._rot_prev @ V
        el.C = el.C @ V

    def compute(self, state: MinimizeState[MatrixArray],
                energy_only: bool) -> None:
        system = self.system
        el = system.electrons
        # Compute energy and subspace Hamiltonian:
        el.update(system)
        state.energy = system.energy
        if energy_only:
            return
        C_HC = el.C ^ el.hamiltonian(el.C)
        # Compute fillings constraint gradients:
        dH_sub = C_HC - el.eig.diag_embed()  # difference in Hamiltonian
        dH_sub_diag = dH_sub.diagonal(dim1=-2, dim2=-1).real
        wf_eig = el.basis.w_sk * el.fillings.f_eig
        E_mu_num = (wf_eig * dH_sub_diag).sum(dim=(1, 2))
        E_mu_den = wf_eig.sum(dim=(1, 2))  # TODO: make this more general:
        self.rc.comm_k.Allreduce(qp.MPI.IN_PLACE,
                                 qp.utils.BufferView(E_mu_num), qp.MPI.SUM)
        self.rc.comm_k.Allreduce(qp.MPI.IN_PLACE,
                                 qp.utils.BufferView(E_mu_den), qp.MPI.SUM)
        E_mu_den.clamp_(max=-1e-20)  # avoid 0/0 in large-gap corner cases
        if (el.n_spins == 1) or el.fillings.M_constrain:
            E_mu = E_mu_num / E_mu_den  # N of each spin channel constrained
        else:
            E_mu = E_mu_num.sum() / E_mu_den.sum()  # only total N constrained
        E_mu = E_mu.view(-1, 1, 1, 1)
        E_f = dH_sub - (torch.eye(el.fillings.n_bands, device=self.rc.device
                                  )[None, None] * E_mu)
        # Compute auxiliary hamiltonian gradient:
        delta_f = el.fillings.f[..., None] - el.fillings.f[:, :, None, :]
        delta_eig = el.eig[..., None] - el.eig[:, :, None, :]
        f_eig_mat = torch.where(delta_eig.abs() < 1e-6,
                                el.fillings.f_eig.diag_embed(),
                                delta_f / delta_eig)
        E_H_aux = el.basis.w_sk[..., None] * (E_f * f_eig_mat)
        K_E_H_aux = -E_f  # drop f' and weights in preconditioned gradient
        # Transform back to original rotation (which CG remains in):
        dagger_rot_prev = qp.utils.dagger(self._rot_prev)
        E_H_aux = self._rot_prev @ E_H_aux @ dagger_rot_prev
        K_E_H_aux = self._rot_prev @ K_E_H_aux @ dagger_rot_prev
        # Store gradients:
        state.gradient = MatrixArray(M=E_H_aux, comm=self.rc.comm_k)
        state.K_gradient = MatrixArray(M=K_E_H_aux, comm=self.rc.comm_k)
        state.extra = [np.sqrt(state.gradient.overlap(state.gradient))]
