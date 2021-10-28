import qimpy as qp
import torch
import pytest
from typing import Sequence


class RandomFunction(qp.utils.Minimize[qp.grid.FieldR]):  # type: ignore
    """Random objective function to test minimization algorithms."""

    grid: qp.grid.Grid  #: Dummy grid for the fields below
    x: qp.grid.FieldR  #: State of test system
    x0: qp.grid.FieldR  #: True solution
    E0: float  #: True minimum energy
    M: Sequence[torch.Tensor]  #: Matrices defining even terms in energy
    K: torch.Tensor

    def __init__(
        self,
        rc: qp.utils.RunConfig,
        n_dim: int,
        method: str,
        checkpoint_in: qp.utils.CpPath = qp.utils.CpPath(),
    ):
        super().__init__(
            rc=rc,
            comm=rc.comm,
            checkpoint_in=checkpoint_in,
            name="TestMinimize",
            n_iterations=100,
            energy_threshold=1e-9,
            extra_thresholds={"|grad|": 1e-8},
            method=method,
        )
        lattice = qp.lattice.Lattice(rc=rc, system="Orthorhombic", a=1.0, b=1.0, c=10.0)
        ions = qp.ions.Ions(rc=rc, pseudopotentials=[], coordinates=[])
        symmetries = qp.symmetries.Symmetries(rc=rc, lattice=lattice, ions=ions)
        grid = qp.grid.Grid(
            rc=rc,
            lattice=lattice,
            symmetries=symmetries,
            shape=(1, 1, n_dim),
            comm=self.rc.comm,
        )
        self.grid = grid
        x0 = torch.arange(n_dim, dtype=torch.float64, device=self.rc.device)
        self.x0 = qp.grid.FieldR(grid, data=x0.view(grid.shape))
        self.E0 = -5.0
        torch.random.manual_seed(0)
        self.x = self.x0 + 0.1 * self.random_direction()
        self.M = []
        for i_M in range(2):
            self.M.append(torch.randn((n_dim, n_dim), device=self.rc.device))
        # Preconditioner (inexact inverse):
        Kreg = 0.1 * (self.M[0] ** 2).sum() * torch.eye(n_dim, device=self.rc.device)
        self.K = torch.linalg.inv(Kreg + self.M[0].T @ self.M[0])

    def step(self, direction: qp.grid.FieldR, step_size: float) -> None:
        self.x += step_size * direction

    def compute(self, state, energy_only):  # type: ignore
        E = self.E0
        E_x = torch.zeros_like(self.x0.data)
        for i_M, M in enumerate(self.M):
            v = M @ (self.x - self.x0).data[0, 0]
            v_norm_sq = (v ** 2).sum().item()
            E += v_norm_sq ** (i_M + 1) * self.grid.dV
            E_x[0, 0] += (2 * (i_M + 1) * (v_norm_sq ** i_M)) * (M.T @ v)
        K_E_x = (self.K @ E_x[0, 0]).view(E_x.shape)
        state.energy["E"] = E
        if not energy_only:
            state.gradient = qp.grid.FieldR(self.grid, data=E_x)
            state.K_gradient = qp.grid.FieldR(self.grid, data=K_E_x)
            state.extra = [state.gradient.norm()]

    def random_direction(self) -> qp.grid.FieldR:
        data = torch.randn(self.grid.shape, device=self.rc.device)
        return qp.grid.FieldR(self.grid, data=data)


@pytest.mark.mpi_skip
@pytest.mark.parametrize(
    "n_dim, method", [(10, "cg"), (100, "cg"), (10, "l-bfgs"), (100, "l-bfgs")]
)
def test_minimize(n_dim: int, method: str, rc: qp.utils.RunConfig):
    n_dim = 100
    method = "cg"
    rf = RandomFunction(rc=rc, n_dim=n_dim, method=method)
    rf.finite_difference_test(rf.random_direction())
    E = float(rf.minimize())
    assert abs(E - rf.E0) < (n_dim * rf.energy_threshold)
