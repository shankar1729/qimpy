import qimpy as qp
import numpy as np
from typing import Callable, Dict, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from ._minimize import Minimize, MinimizeState, Vector


LineMinimize = Callable[['Minimize[Vector]', 'Vector', float,
                         'MinimizeState[Vector]'], Tuple[float, float, bool]]


def _constant(self: 'Minimize[Vector]', direction: 'Vector',
              step_size_test: float, state: 'MinimizeState[Vector]'
              ) -> Tuple[float, float, bool]:
    """Take a pre-specified step size."""
    step_size = step_size_test  # Constant specified step size
    self.step(direction, step_size)
    self.compute(state, energy_only=False)
    E = self._sync(float(state.energy))
    if not np.isfinite(E):
        qp.log.info(f'{self.name}: constant step failed with'
                    f' {state.energy.name} = {E}')
        return E, step_size, False
    return E, step_size, True


LINE_MINIMIZE: Dict[str, LineMinimize] = {
    'constant': _constant
}


if __name__ == '__main__':
    import numpy as np
    import torch
    from typing import Sequence

    class TestFunction(qp.utils.Minimize[qp.grid.FieldR]):  # type: ignore
        grid: qp.grid.Grid  #: Dummy grid for the fields below
        x: qp.grid.FieldR  #: State of test system
        x0: qp.grid.FieldR  #: True solution
        E0: float  #: True minimum energy
        M: Sequence[torch.Tensor]  #: Matrices defining even terms in energy

        def __init__(self, co: qp.ConstructOptions):
            super().__init__(co=co, comm=co.rc.comm,
                             name='TestMinimize', n_iterations=40,
                             energy_threshold=1e-6, extra_thresholds={},
                             method='gradient', line_minimize='constant',
                             step_size={'initial': 0.1})
            lattice = qp.lattice.Lattice(co=co, system='Orthorhombic',
                                         a=1., b=1., c=10.)
            ions = qp.ions.Ions(co=co, pseudopotentials=[], coordinates=[])
            symmetries = qp.symmetries.Symmetries(co=co, lattice=lattice,
                                                  ions=ions)
            grid = qp.grid.Grid(co=co, lattice=lattice, symmetries=symmetries,
                                shape=(1, 1, 3), comm=self.rc.comm)
            self.grid = grid
            self.x = qp.grid.FieldR(grid, data=torch.tensor(
                [1., 4., 5.], device=self.rc.device).view(grid.shape))
            self.x0 = qp.grid.FieldR(grid, data=torch.tensor(
                [2., 3., 4.], device=self.rc.device).view(grid.shape))
            self.E0 = 7.
            torch.random.manual_seed(0)
            self.M = []
            for i_M in range(2):
                self.M.append(torch.randn((3, 3), device=self.rc.device))

        def step(self, direction: qp.grid.FieldR, step_size: float) -> None:
            self.x += step_size * direction

        def compute(self, state, energy_only):  # type: ignore
            E = self.E0
            E_x = torch.zeros_like(self.x0.data)
            for i_M, M in enumerate(self.M):
                v = M @ (self.x - self.x0).data[0, 0]
                v_norm_sq = (v**2).sum().item()
                E += v_norm_sq ** (i_M + 1)
                E_x[0, 0] += ((i_M + 1) * (v_norm_sq ** i_M)) * (M.T @ v)
            state.energy['E'] = E
            if not energy_only:
                state.gradient = qp.grid.FieldR(self.grid, data=E_x)
                state.K_gradient = qp.grid.FieldR(self.grid, data=E_x)

    def main():
        qp.utils.log_config()
        rc = qp.utils.RunConfig()
        co = qp.ConstructOptions(rc=rc)
        tf = TestFunction(co=co)
        tf.minimize()
    main()
