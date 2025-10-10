"""Isodensity-cavity solvation model variants."""
import torch
import numpy as np

from qimpy import TreeNode, Energy
from qimpy.io import CheckpointPath
from qimpy.grid import FieldR, FieldH


class LA12(TreeNode):
    """Purely electrostatic model with an isodensity cavity, as defined in:

    K. Letchworth-Weaver and T.A. Arias, Phys. Rev. B 86, 075140 (2012).
    """

    nc: float  #: threshold electron density
    sigma: float  #: transition width in log(n)

    shape: FieldR  #: cavity shape function
    _erfc_arg: torch.Tensor  #: erfc argument cached for gradient propoagation

    def __init__(
        self,
        *,
        nc: float = 7e-4,
        sigma: float = 0.6,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ):
        super().__init__()
        self.nc = nc
        self.sigma = sigma

    def update_shape(self, n_tilde: FieldH) -> None:
        n = (~n_tilde[0]).data
        eta = np.sqrt(0.5) / self.sigma
        self._erfc_arg = eta * (n.abs() / self.nc).log()
        self.shape = FieldR(n_tilde.grid, data=0.5 * torch.erfc(self._erfc_arg))

    def propagate_shape_grad(self, n_tilde: FieldH) -> None:
        assert self.shape.grad is not None
        eta = np.sqrt(0.5) / self.sigma
        prefactor = (-eta / (self.nc * np.sqrt(np.pi))) * self.shape.grad.data
        exp_arg = -self._erfc_arg * (self._erfc_arg + 1.0 / eta)
        n_tilde.grad[0] += ~FieldR(n_tilde.grid, data=prefactor * torch.exp(exp_arg))

    def update_energy(self, energy: Energy) -> None:
        pass  # purely electrostatic model


class GLSSA13(LA12):
    """Electrostatic + surface tension model with an isodensity cavity, as defined in:

        D. Gunceler, K. Letchworth-Weaver, R. Sundararaman, K.A. Schwarz and T.A. Arias,
        Modelling Simul. Mater. Sci. Eng. 21, 074005 (2013).

    This model was subsequently implemented in VASP as VASPsol:

        K. Matthew, R. Sundararaman, K. Letchworth-Weaver, T.A. Arias and R. Hennig,
        J. Chem. Phys. 140, 084106 (2014).
    """

    cavity_tension: float  #: Cavitation energy per unit area

    def __init__(
        self,
        *,
        nc: float = 3.7e-4,
        sigma: float = 0.6,
        cavity_tension: float = 5.4e-6,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ):
        super().__init__(nc=nc, sigma=sigma)
        self.cavity_tension = cavity_tension

    def update_energy(self, energy: Energy) -> None:
        """Surface-area based cavitation energy."""
        Dshape = self.shape.gradient()
        surface_density = FieldR(Dshape.grid, data=Dshape.data.norm(dim=0))
        surface_area = surface_density.integral().item()
        energy["Acavity"] = self.cavity_tension * surface_area
        if self.shape.requires_grad:
            self.shape.grad -= (
                self.cavity_tension * (Dshape / surface_density).divergence()
            )
