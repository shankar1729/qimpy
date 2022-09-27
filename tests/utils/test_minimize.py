import qimpy as qp
import numpy as np
import torch
import pytest
from typing import Sequence
from qimpy.rc import MPI


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
        n_dim: int,
        method: str,
        checkpoint_in: qp.utils.CpPath = qp.utils.CpPath(),
    ):
        super().__init__(
            comm=qp.rc.comm,
            checkpoint_in=checkpoint_in,
            name="TestMinimize",
            n_iterations=100,
            energy_threshold=1e-9,
            extra_thresholds={"|grad|": 1e-8},
            n_consecutive=1,
            method=method,
        )
        process_grid = qp.utils.ProcessGrid(self.comm, "rkb")
        lattice = qp.lattice.Lattice(system="Orthorhombic", a=10.0, b=1.0, c=1.0)
        ions = qp.ions.Ions(
            lattice=lattice,
            pseudopotentials=[],
            coordinates=[],
            process_grid=process_grid,
        )
        symmetries = qp.symmetries.Symmetries(lattice=lattice, ions=ions)
        grid = qp.grid.Grid(
            lattice=lattice,
            symmetries=symmetries,
            shape=(n_dim, 1, 1),
            comm=qp.rc.comm,
        )
        self.grid = grid
        self.i0slice = slice(grid.split0.i_start, grid.split0.i_stop)
        x0 = torch.arange(
            self.i0slice.start,
            self.i0slice.stop,
            dtype=torch.float64,
            device=qp.rc.device,
        )
        self.x0 = qp.grid.FieldR(grid, data=x0.view(grid.shapeR_mine))
        self.E0 = -5.0
        torch.random.manual_seed(0)
        self.x = self.x0 + 0.1 * self.random_direction()
        M_all = []  # full matrices
        self.M = []  # process-divided matrix
        for i_M in range(2):
            M_all.append(torch.randn((n_dim, n_dim), device=qp.rc.device))
            self.M.append(M_all[-1][:, self.i0slice])
        # Preconditioner (inexact inverse):
        Kreg = 0.1 * (M_all[0] ** 2).sum() * torch.eye(n_dim, device=qp.rc.device)
        self.K = torch.linalg.inv(Kreg + M_all[0].T @ M_all[0])[:, self.i0slice]
        self.K *= grid.dV  # include integration weights in preconditioner

    def step(self, direction: qp.grid.FieldR, step_size: float) -> None:
        self.x += step_size * direction

    def compute(self, state, energy_only):  # type: ignore
        grid = self.grid
        if not energy_only:
            self.x.data.requires_grad = True
            self.x.data.grad = None

        E = torch.tensor(self.E0, device=qp.rc.device)
        for i_M, M in enumerate(self.M):
            v = M @ (self.x - self.x0).data.flatten()  # partial results, full array
            qp.rc.current_stream_synchronize()
            self.comm.Allreduce(MPI.IN_PLACE, qp.utils.BufferView(v), MPI.SUM)
            E += (v ** 2).sum() ** (i_M + 1) * grid.dV
        state.energy["E"] = E.item()

        if not energy_only:
            E.backward()
            E_x = self.x.data.grad
            # Apply preconditioner:
            K_E_x = self.K @ E_x.flatten()  # partial results, full array
            self.comm.Allreduce(MPI.IN_PLACE, qp.utils.BufferView(K_E_x), MPI.SUM)
            K_E_x = K_E_x.view(grid.shape)[self.i0slice]  # full results, partial array
            # Convert to fields:
            state.gradient = qp.grid.FieldR(grid, data=E_x)
            state.K_gradient = qp.grid.FieldR(grid, data=K_E_x)
            state.extra = [np.sqrt(state.gradient.vdot(state.K_gradient))]
            self.x.data.requires_grad = False

    def random_direction(self) -> qp.grid.FieldR:
        data = torch.randn(self.grid.shape, device=qp.rc.device)
        return qp.grid.FieldR(self.grid, data=data[self.i0slice])


@pytest.mark.parametrize("n_dim, method", [(10, "cg"), (100, "cg"), (100, "l-bfgs")])
def test_minimize(n_dim: int, method: str):
    n_dim = 100
    method = "cg"
    rf = RandomFunction(n_dim=n_dim, method=method)
    E = float(rf.minimize())
    assert abs(E - rf.E0) < (n_dim * rf.energy_threshold)


def main():
    """Manually run a test with full output"""
    qp.utils.log_config()
    qp.rc.init()
    rf = RandomFunction(n_dim=100, method="cg")
    rf.finite_difference_test(rf.random_direction())
    rf.minimize()


if __name__ == "__main__":
    main()
