import qimpy as qp
import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .. import System


class LCAO(qp.Constructable):
    """Optimize electronic state in atomic-orbital subspace."""
    __slots__ = ('n_iterations', 'energy_threshold')
    n_iterations: int  #: Number of iterations in atomic-orbital subspace
    energy_threshold: float  #: Convergence threshold on energy change

    def __init__(self, *, co: qp.ConstructOptions, n_iterations: int = 30,
                 energy_threshold: float = 1E-6) -> None:
        """Set stopping criteria for initial subspace optimization."""
        super().__init__(co=co)
        self.n_iterations = n_iterations
        self.energy_threshold = energy_threshold

    def update(self, system: 'System') -> None:
        """Set wavefunctions to optimum subspace of atomic orbitals."""
        el = system.electrons
        el.n = system.ions.get_atomic_density(system.grid, el.fillings.M)
        el.tau = qp.grid.FieldH(system.grid, shape_batch=(0,))  # TODO
        el.update_potential(system)
        C_OC = el.C.dot_O(el.C)
        C_HC = el.C ^ el.hamiltonian(el.C)
        el.eig, V = qp.utils.eighg(C_HC, C_OC)
        el.C = el.C @ V  # Set to eigenvectors
