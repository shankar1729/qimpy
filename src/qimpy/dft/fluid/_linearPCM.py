import numpy as np
import torch

from qimpy import log
from qimpy.algorithms import LinearSolve
from qimpy.grid import Grid, FieldH, FieldR
from qimpy.profiler import stopwatch


class LinearPCMFluidModel(LinearSolve[FieldH]):
    epsilon: FieldR  #: spatially varying dielectric constant
    enabled: bool  #: mask for fluid contributions (used during initialization)

    def __init__(
        self,
        *,
        grid: Grid,
        coulomb,
        ions,
        checkpoint_in=None,
        n_iterations=100,
        gradient_threshold=1E-8,
        eps_bulk=78.4,
        nc=3.7E-4,
        sigma=0.6,
        cavity_tension=5.4E-6,
    ):
        super().__init__(
            checkpoint_in=checkpoint_in,
            comm=grid.comm,
            n_iterations=n_iterations,
            gradient_threshold=gradient_threshold,
        )
        self.enabled = True
        self.grid = grid
        self.coulomb = coulomb
        self.ions = ions

        self.eps_bulk = eps_bulk
        self.nc = nc
        self.sigma = sigma
        self.cavity_tension = cavity_tension

        self.phi = FieldH(self.grid)

        # Initialize preconditioner:
        iG = grid.get_mesh("H").to(torch.double)
        Gsq = (iG @ grid.lattice.Gbasis.T).square().sum(dim=-1)
        GSQ_CUT = 1E-12  # regularization
        self.Kkernel = torch.clamp(Gsq, min=GSQ_CUT).reciprocal() / self.eps_bulk
        self.Kkernel[Gsq < GSQ_CUT] = 0.0  # project out null-space

    def compute_shape_function(self, n: FieldR) -> FieldR:
        nc = self.nc
        sigma = self.sigma
        return FieldR(
            self.grid,
            data=0.5 * torch.erfc((np.sqrt(0.5) / sigma) * (n.data.abs() / nc).log()),
        )

    def shape_gradient(self, n: FieldR, grad_shape: FieldR) -> FieldR:
        nc = self.nc
        sigma = self.sigma
        return FieldR(
            self.grid,
            data=(
                (-1.0 / (nc * sigma * np.sqrt(2 * np.pi))) * grad_shape.data
                * torch.exp(
                    0.5 * (sigma**2 - ((n.data.abs() / nc).log() / sigma + sigma)**2)
                )
            ),
        )

    def compute_cavity_energy(self, shape: FieldR) -> tuple[float, FieldR]:
        """Compute cavitation energy and its shape function gradient"""
        Dshape = shape.gradient()
        surface_density = FieldR(self.grid, data=Dshape.data.norm(dim=0))
        surface_area = surface_density.integral().item()
        Acavity = self.cavity_tension * surface_area
        Acavity_shape = self.cavity_tension * (Dshape / surface_density).divergence()
        return Acavity, Acavity_shape

    def hessian(self, phi: FieldH) -> FieldH:
        result = (~(~phi.gradient() * self.epsilon[None])).divergence()
        return (-1 / (4 * np.pi)) * result

    def precondition(self, vector: FieldH) -> FieldH:
        return vector.convolve(self.Kkernel)

    @stopwatch(name="LinearPCM.update")
    def compute_Adiel_and_potential(
        self, n_tilde: FieldH
    ) -> tuple[float, FieldH, FieldH]:
        rho_field = self.ions.rho_tilde + n_tilde[0]
        n_cavity = ~n_tilde[0]
        shape = self.compute_shape_function(n_cavity)
        self.epsilon = 1.0 + (self.eps_bulk - 1.0) * shape

        n_iter = self.solve(rho_field, self.phi)
        log.info(f"  Fluid: solve completed in {n_iter} iterations")

        # Electrostatic contributions:
        phi_ext = self.coulomb.kernel(rho_field)
        term1 = -0.5 * (self.phi ^ self.hessian(self.phi))
        term2 = (self.phi - (0.5 * phi_ext)) ^ rho_field
        Ael = term1 + term2
        grad_phi_sq = (~self.phi.gradient()).data.square().sum(dim=0)
        Ael_shape = FieldR(
            self.grid, data=(-(self.eps_bulk - 1) / (8 * np.pi)) * grad_phi_sq
        )

        # Cavitation terms:
        Acavity, Acavity_shape = self.compute_cavity_energy(shape)

        A_rho_tilde = self.phi - phi_ext
        A_n_cavity = self.shape_gradient(n_cavity, Ael_shape + Acavity_shape)
        A_n_tilde = A_rho_tilde + ~A_n_cavity
        return Ael + Acavity, A_n_tilde, A_rho_tilde
