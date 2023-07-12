from typing import Sequence

import numpy as np
import torch
import pytest
from mpi4py import MPI

from qimpy import rc
from qimpy.io import CheckpointPath, log_config
from qimpy.mpi import BufferView
from qimpy.lattice import Lattice
from qimpy.symmetries import Symmetries
from qimpy.grid import Grid, FieldR
from . import Minimize


class RandomFunction(Minimize[FieldR]):  # type: ignore
    """Random objective function to test minimization algorithms."""

    grid: Grid  #: Dummy grid for the fields below
    x: FieldR  #: State of test system
    x0: FieldR  #: True solution
    E0: float  #: True minimum energy
    M: Sequence[torch.Tensor]  #: Matrices defining even terms in energy
    K: torch.Tensor

    def __init__(
        self,
        n_dim: int,
        method: str,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ):
        super().__init__(
            comm=rc.comm,
            checkpoint_in=checkpoint_in,
            name="TestMinimize",
            n_iterations=100,
            energy_threshold=1e-9,
            extra_thresholds={"|grad|": 1e-8},
            n_consecutive=1,
            method=method,
        )
        lattice = Lattice(system="Orthorhombic", a=10.0, b=1.0, c=1.0)
        symmetries = Symmetries(lattice=lattice)
        grid = Grid(
            lattice=lattice,
            symmetries=symmetries,
            shape=(n_dim, 1, 1),
            comm=rc.comm,
        )
        self.grid = grid
        self.i0slice = slice(grid.split0.i_start, grid.split0.i_stop)
        x0 = torch.arange(
            self.i0slice.start,
            self.i0slice.stop,
            dtype=torch.float64,
            device=rc.device,
        )
        self.x0 = FieldR(grid, data=x0.view(grid.shapeR_mine))
        self.E0 = -5.0
        torch.random.manual_seed(0)
        self.x = self.x0 + 0.1 * self.random_direction()
        M_all = []  # full matrices
        self.M = []  # process-divided matrix
        for i_M in range(2):
            M_all.append(torch.randn((n_dim, n_dim), device=rc.device))
            self.M.append(M_all[-1][:, self.i0slice])
        # Preconditioner (inexact inverse):
        Kreg = 0.1 * (M_all[0] ** 2).sum() * torch.eye(n_dim, device=rc.device)
        self.K = torch.linalg.inv(Kreg + M_all[0].T @ M_all[0])[:, self.i0slice]
        self.K *= grid.dV  # include integration weights in preconditioner

    def step(self, direction: FieldR, step_size: float) -> None:
        self.x += step_size * direction

    def compute(self, state, energy_only):  # type: ignore
        grid = self.grid
        if not energy_only:
            self.x.data.requires_grad = True
            self.x.data.grad = None

        E = torch.tensor(self.E0, device=rc.device)
        for i_M, M in enumerate(self.M):
            v = M @ (self.x - self.x0).data.flatten()  # partial results, full array
            rc.current_stream_synchronize()
            self.comm.Allreduce(MPI.IN_PLACE, BufferView(v), MPI.SUM)
            E += (v**2).sum() ** (i_M + 1) * grid.dV
        state.energy["E"] = E.item()

        if not energy_only:
            E.backward()
            E_x = self.x.data.grad
            # Apply preconditioner:
            K_E_x = self.K @ E_x.flatten()  # partial results, full array
            self.comm.Allreduce(MPI.IN_PLACE, BufferView(K_E_x), MPI.SUM)
            K_E_x = K_E_x.view(grid.shape)[self.i0slice]  # full results, partial array
            # Convert to fields:
            state.gradient = FieldR(grid, data=E_x)
            state.K_gradient = FieldR(grid, data=K_E_x)
            state.extra = [np.sqrt(state.gradient.vdot(state.K_gradient))]
            self.x.data.requires_grad = False

    def random_direction(self) -> FieldR:
        data = torch.randn(self.grid.shape, device=rc.device)
        return FieldR(self.grid, data=data[self.i0slice])


@pytest.mark.parametrize("n_dim, method", [(10, "cg"), (100, "cg"), (100, "l-bfgs")])
def test_minimize(n_dim: int, method: str):
    n_dim = 100
    method = "cg"
    rf = RandomFunction(n_dim=n_dim, method=method)
    E = float(rf.minimize())
    assert abs(E - rf.E0) < (n_dim * rf.energy_threshold)


def main():
    """Manually run a test with full output"""
    log_config()
    rc.init()
    rf = RandomFunction(n_dim=100, method="cg")
    rf.finite_difference_test(rf.random_direction())
    rf.minimize()


if __name__ == "__main__":
    main()
