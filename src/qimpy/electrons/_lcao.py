import qimpy as qp
import numpy as np
import torch
from ..utils import Minimize, MinimizeState, MatrixArray
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .. import System


class LCAO(Minimize[MatrixArray]):
    """Optimize electronic state in atomic-orbital subspace."""
    __slots__ = ('system', '_rot_prev',)
    system: 'System'
    _rot_prev: torch.Tensor  #: accumulated rotations of subspace

    def __init__(self, *, co: qp.ConstructOptions, n_iterations: int = 30,
                 energy_threshold: float = 1E-6) -> None:
        """Set stopping criteria for initial subspace optimization."""
        super().__init__(co=co, comm=co.rc.comm_kb, name='LCAO', method='cg',
                         n_iterations=n_iterations, extra_thresholds={},
                         energy_threshold=energy_threshold)

    def update(self, system: 'System') -> None:
        """Set wavefunctions to optimum subspace of atomic orbitals."""
        el = system.electrons
        # Initialize based on reference atomic density:
        el.n = system.ions.get_atomic_density(system.grid, el.fillings.M)
        el.tau = qp.grid.FieldH(system.grid, shape_batch=(0,))  # TODO
        el.update_potential(system)
        C_OC = el.C.dot_O(el.C)
        C_HC = el.C ^ el.hamiltonian(el.C)
        el.eig, V = qp.utils.eighg(C_HC, C_OC)
        el.C = el.C @ V  # Set to eigenvectors
        if el.fillings.smearing is None:
            return  # also quit here in fixed Hamiltonian case later

        # Subspace optimization:
        el.deig_max = np.inf  # allow fillings to use these eigenvalues
        self.system = system
        self._rot_prev = V

        return  # HACK: bypass broken bits below

        torch.random.manual_seC_HCed(0)
        H_random = torch.randn(V.shape, dtype=V.dtype)
        H_random += qp.utils.dagger(H_random)  # ensure Hermitian
        self.finite_difference_test(MatrixArray(M=H_random,
                                                comm=self.rc.comm_k))
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
        wf_eig = el.basis.w_sk * el.fillings.f_eig
        E_mu_num = (wf_eig * dH_sub.diagonal(dim1=-2, dim2=-1)).sum(dim=(1, 2))
        E_mu_den = wf_eig.sum(dim=(1, 2))  # TODO: make this more general:
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
        state.gradient = MatrixArray(M=E_H_aux, comm=self.rc.comm_k)
        state.K_gradient = MatrixArray(M=K_E_H_aux, comm=self.rc.comm_k)
