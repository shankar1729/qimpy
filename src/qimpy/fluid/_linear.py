import numpy as np
import torch

from qimpy import log
from qimpy.algorithms import LinearSolve
from qimpy.grid import Grid, FieldH, FieldR
from qimpy.grid.coulomb import Coulomb
from qimpy.profiler import stopwatch


class Linear(LinearSolve[FieldH]):
    epsilon: FieldR  #: spatially varying dielectric constant

    def __init__(
        self,
        *,
        grid: Grid,
        coulomb: Coulomb,
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
        self.grid = grid
        self.coulomb = coulomb

        self.eps_bulk = eps_bulk
        self.nc = nc
        self.sigma = sigma
        self.cavity_tension = cavity_tension

        self.phi_tilde = FieldH(self.grid)

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

    def compute_cavity_energy(self, shape: FieldR) -> float:
        """Compute cavitation energy and its gradient if needed"""
        Dshape = shape.gradient()
        surface_density = FieldR(self.grid, data=Dshape.data.norm(dim=0))
        surface_area = surface_density.integral().item()
        Acavity = self.cavity_tension * surface_area
        if shape.requires_grad:
            shape.grad -= self.cavity_tension * (Dshape / surface_density).divergence()
        return Acavity

    def hessian(self, phi_tilde: FieldH) -> FieldH:
        result = (~(~phi_tilde.gradient() * self.epsilon[None])).divergence()
        return (-1 / (4 * np.pi)) * result

    def precondition(self, vector: FieldH) -> FieldH:
        return vector.convolve(self.Kkernel)

    @stopwatch(name="Linear.calculate")
    def update(self, n_tilde: FieldH, rho_tilde: FieldH) -> float:
        n_cavity = ~n_tilde[0]  # Cavity determining electron density
        shape = self.compute_shape_function(n_cavity)
        self.epsilon = 1.0 + (self.eps_bulk - 1.0) * shape

        n_iter = self.solve(rho_tilde, self.phi_tilde)
        log.info(f"  Fluid: solve completed in {n_iter} iterations")

        # Electrostatic contributions:
        phi_ext_tilde = self.coulomb.kernel(rho_tilde)
        Ael = (
            -0.5 * (self.phi_tilde ^ self.hessian(self.phi_tilde))
            + ((self.phi_tilde - 0.5 * phi_ext_tilde) ^ rho_tilde)
        )
        if n_tilde.requires_grad:
            grad_phi_sq = (~self.phi_tilde.gradient()).data.square().sum(dim=0)
            shape.requires_grad_(True)
            shape.grad = FieldR(
                self.grid, data=(-(self.eps_bulk - 1) / (8 * np.pi)) * grad_phi_sq
            )

        # Cavitation terms:
        Acavity = self.compute_cavity_energy(shape)

        # Propagate gradients as needed:
        if n_tilde.requires_grad:
            n_tilde.grad += ~self.shape_gradient(n_cavity, shape.grad)
        if rho_tilde.requires_grad:
            rho_tilde.grad += self.phi_tilde - phi_ext_tilde
        return Ael + Acavity
