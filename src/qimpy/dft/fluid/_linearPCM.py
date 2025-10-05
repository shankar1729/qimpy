# fluid/_linearPCM.py
import torch

from qimpy import log
from qimpy.io import CheckpointPath
from qimpy.algorithms import LinearSolve
from math import sqrt, pi
from qimpy.grid import Grid, FieldH, FieldR


class LinearPCMFluidModel(LinearSolve[FieldH]):

    def __init__(
        self,
        *,
        grid: Grid,
        coulomb,
        ions,
        fluid_params=None,
        checkpoint_in=None,
        n_iterations=100,
        gradient_threshold=1E-8,
    ):
        super().__init__(
            checkpoint_in=checkpoint_in,
            comm=grid.comm,
            n_iterations=n_iterations,
            gradient_threshold=gradient_threshold,
        )
        
        self.grid = grid
        self.coulomb = coulomb
        self.ions = ions
        
        self.params = fluid_params or {
            "epsBulk": 78.4,
            "pMol": 0.92466,
            "k2factor": 0,
            "nc": 7e-4,
            "sigma": 0.6,
            "cavityTension": 5.4e-6,
        }

        self.epsilon_val = None
        self.kappa_val = None
        self.phi = FieldH(self.grid)

        # Initialize preconditioner:
        iG = grid.get_mesh("H").to(torch.double)
        Gsq = (iG @ grid.lattice.Gbasis.T).square().sum(dim=-1)
        GSQ_CUT = 1E-12  # regularization
        self.Kkernel = torch.clamp(Gsq, min=GSQ_CUT).reciprocal()
        self.Kkernel[Gsq < GSQ_CUT] = 0.0  # project out null-space

    def write(self, path: CheckpointPath) -> None:
        path["params"] = self.params

    @classmethod
    def read(cls, path: CheckpointPath, grid, coulomb) -> "LinearPCMFluidModel":
        params = path.get("params", None)
        return cls(grid=grid, coulomb=coulomb, ions=None, fluid_params=params)

    def compute_shape_function(self, n: torch.Tensor) -> torch.Tensor:
        nc = self.params["nc"]
        sigma = self.params["sigma"]
        return 0.5 * torch.erfc((1 / (sigma * sqrt(2.0))) * torch.log(n.abs() / nc))

    def calculate_epsilon_kappa(self, nTilde: FieldH):
        S = self.compute_shape_function(~nTilde.data[0])
        epsilon = 1.0 + (-1.0 + self.params["epsBulk"]) * S  # placeholder water
        kappa = 0.01 * S
        return epsilon, kappa

    def shape_gradient(self, n: torch.Tensor, grad_shape: torch.Tensor) -> torch.Tensor:
        nc = self.params["nc"]
        sigma = self.params["sigma"]
        alpha = 1.0 / (sigma * sqrt(2.0))
        logterm = torch.log(torch.abs(n) / nc)
        u = alpha * logterm
        grad_n = -alpha / (torch.abs(n) * sqrt(pi)) * torch.exp(-u**2) * torch.sign(n)
        return grad_n

    def propagate_gradient(self, n: FieldH, E_shape: FieldR) -> FieldH:
        grad_n = self.shape_gradient(n.data, E_shape.data)
        E_n = FieldH(self.grid, data=grad_n)
        # print(E_n.data, "E_n propagated gradient")
        return E_n

    def compute_cavity_energy(self, s_field: FieldR) -> float:
        grad_s = s_field.gradient().data
        l1_norm = torch.abs(grad_s[0]) + torch.abs(grad_s[1]) + torch.abs(grad_s[2])
        norm_field = FieldR(self.grid, data=l1_norm)
        surface_area = norm_field.integral()
        return self.params["cavityTension"] * surface_area

    def hessian(self, phi: FieldH) -> FieldH:
        result = (~(~phi.gradient() * self.epsilon_val[None])).divergence()
        return (-1 / (4 * pi)) * result

    def precondition(self, vector: FieldH) -> FieldH:
        return vector.convolve(self.Kkernel)

    def compute_Adiel_and_potential(self, n_tilde: FieldH) -> tuple[float, FieldH, FieldH]:
        rho_field = self.ions.rho_tilde + n_tilde[0]

        S = FieldR(self.grid, data=self.compute_shape_function((~n_tilde[0]).data))

        epsilon_val = 1.0 + (-1.0 + self.params["epsBulk"]) * S
        kappa_val = 0.01 * S
        self.epsilon_val = epsilon_val
        self.kappa_val = kappa_val

        n_iter = self.solve(rho_field, self.phi)
        log.info(f"  Fluid: solve completed in {n_iter} iterations")
        phi_ext = self.coulomb.kernel(rho_field)

        term1 = -0.5 * (self.phi ^ self.hessian(self.phi))
        term2 = (self.phi - (0.5 * phi_ext)) ^ (rho_field)
        Adiel_electrostatic = (term1 + term2)

        cavity_energy = self.compute_cavity_energy(S)
        total_energy = Adiel_electrostatic  # + cavity_energy

        Adiel_rhoTilde = self.phi - phi_ext
        V_fluid = Adiel_rhoTilde  # + cavity_potential

        return total_energy, V_fluid, Adiel_rhoTilde
