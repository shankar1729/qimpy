# fluid/_linearPCM.py
from qimpy import TreeNode
from qimpy.io import CheckpointPath
from math import sqrt, pi
from qimpy.grid import FieldH, FieldR
import torch


class LinearPCMFluidModel(TreeNode):
    def __init__(self, *, grid, coulomb, ions, fluid_params=None, checkpoint_in=None):
        super().__init__()
        
        iG = grid.get_mesh("H").to(torch.double)  # get Gsq manually
        self.Gsq = (iG @ grid.lattice.Gbasis.T).square().sum(dim=-1)
        self.grid = grid
        self.coulomb = coulomb
        self.ions = ions
        
        self.params = fluid_params or {
            "epsBulk": 78.4,
            "pMol": 0.92466,
            "epsInf": 1.77,
            "Rvdw": 1.385,
            "k2factor": 0,
            "nc": 7e-4,
            "sigma": 0.6,
            "cavityTension": 5.4e-6,
            "tol": 1e-6,
            "max_iterations": 1000
        }

        self.epsilon_val = None
        self.kappa_val = None

    def write(self, path: CheckpointPath) -> None:
        path["params"] = self.params

    @classmethod
    def read(cls, path: CheckpointPath, grid, coulomb) -> "LinearPCMFluidModel":
        params = path.get("params", None)
        return cls(grid=grid, coulomb=coulomb, ions=None, fluid_params=params)

    def compute_shape_function(self, n: torch.Tensor) -> torch.Tensor:
        # nc = self.params["nc"]
        # sigma = self.params["sigma"]
        return torch.erfc((1 / (0.6 * sqrt(2.0))) * torch.log(torch.abs(n) / 7e-4)) * 0.5

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

    def hessian(self, vector: FieldH) -> FieldH:
        grad = (-1 / (4 * pi)) * (~(~vector.gradient() * self.epsilon_val[None])).divergence()
        H_vec = grad + 1e-6 * vector #small regularization factor for test
        return H_vec

    def preconditioner(self, vector: FieldH) -> FieldH:
        preconditioner_factor = 1 / (self.Gsq + 0.01)
        preconditioned_data = vector.convolve(preconditioner_factor)
        return preconditioned_data

    def solve(self, rhs: FieldH, params: dict = None) -> FieldH:
        params = params or self.params
        tol = params.get("tol", 1e-6)
        max_iterations = params.get("max_iterations", 1000)

        x = FieldH(self.grid, data=torch.zeros_like(rhs.data))
        r = rhs.clone()
        z = self.preconditioner(r)
        d = z.clone()
        rdotz = r.dot(z)

        for i in range(max_iterations):
            Hd = self.hessian(d)
            Hd_dot_d = d.dot(Hd)
            alpha = rdotz / Hd_dot_d
            x = FieldH(self.grid, data=(x.data + alpha * d.data))
            r_new = FieldH(self.grid, data=(r.data - alpha * Hd.data))
            z_new = self.preconditioner(r_new)
            rdotz_new = r_new.dot(z_new)
            # print(r_new.norm())
            if r_new.norm() < tol:
                return x

            beta = rdotz_new / rdotz
            d = FieldH(self.grid, data=(z_new.data + beta * d.data))
            r = r_new
            z = z_new
            rdotz = rdotz_new
        return x  # fieldH

    def compute_Adiel_and_potential(self, n_tilde: FieldH) -> tuple[float, FieldH, FieldH]:
        rho_field = self.ions.rho_tilde + n_tilde[0]

        S = FieldR(self.grid, data=self.compute_shape_function((~n_tilde[0]).data))

        epsilon_val = 1.0 + (-1.0 + self.params["epsBulk"]) * S
        kappa_val = 0.01 * S
        self.epsilon_val = epsilon_val
        self.kappa_val = kappa_val

        phi = self.solve(rhs=rho_field)
        phi_ext = self.coulomb.kernel(rho_field)

        term1 = -0.5 * (phi ^ self.hessian(phi))
        term2 = (phi - (0.5 * phi_ext)) ^ (rho_field)
        Adiel_electrostatic = (term1 + term2)

        cavity_energy = self.compute_cavity_energy(S)
        total_energy = Adiel_electrostatic  # + cavity_energy

        Adiel_rhoTilde = phi - phi_ext
        V_fluid = Adiel_rhoTilde  # + cavity_potential

        return total_energy, V_fluid, Adiel_rhoTilde
