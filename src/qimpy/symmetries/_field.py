import qimpy as qp
import numpy as np
import torch
from typing import Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from ..grid import Grid, FieldH


class FieldSymmetrizer:
    """Space group symmetrization of reciprocal-space :class:`FieldH`'s."""
    grid: 'Grid'

    def __init__(self, grid: 'Grid') -> None:
        """Initialize symmetrization for fields on `grid`."""
        self.grid = grid
        rc = grid.rc
        shapeH = grid.shapeH
        rot = grid.symmetries.rot.to(torch.long)  # rotations (lattice coords)
        trans = grid.symmetries.trans

        def get_index(iH: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Unique index in half-reciprocal space.
            Also returns whether conjugation was required."""
            iH_wrapped = iH % shapeR
            is_conj = (iH_wrapped[..., 2] >= shapeH[2])  # in redundant half
            iH_wrapped[is_conj] = (-iH_wrapped[is_conj]) % shapeR
            return iH_wrapped @ strideH, is_conj

        # Find symmetry-reduced set:
        iH = grid.get_mesh('H', mine=False).view(-1, 3)  # global mesh
        strideH = torch.tensor([shapeH[1] * shapeH[2], shapeH[2], 1],
                               dtype=torch.long, device=rc.device)
        shapeR = torch.tensor(grid.shape, dtype=torch.long, device=rc.device)
        min_equiv_index = get_index(iH)[0]  # lowest equivalent index
        for rot_i in rot:
            # iH transforms by rot.T, so no transpose on right-multiply:
            min_equiv_index = torch.minimum(min_equiv_index,
                                            get_index(iH @ rot_i)[0])
        iH_reduced = iH[min_equiv_index.unique()]

        # Set up indices, phases of orbits of each point in reduced set:
        index, is_conj = get_index(iH_reduced @ rot)
        phase = qp.utils.cis((-2*np.pi) * (iH_reduced * trans[:, None]
                                           ).sum(dim=-1))
        _, multiplicity = index.unique(sorted=True, return_counts=True)

        self.index = index
        self.is_conj = is_conj
        self.phase = phase
        self.inv_multiplicity = 1./multiplicity

        # Set up MPI split over orbits:
        if grid.n_procs > 1:
            raise NotImplementedError('Distributed symmetrization')

        # division = qp.utils.TaskDivision(n_tot=iH_reduced.shape,
        #                                  n_procs=grid.n_procs,
        #                                  i_proc=grid.i_proc)
        # mine = slice(division.i_start, division.i_stop)
        # TODO

    def __call__(self, v: 'FieldH') -> None:
        """Symmetrize field `v` in-place."""
        grid = self.grid
        assert v.grid == grid
        n_batch = np.prod(v.data.shape[:-3])
        n_grid = np.prod(grid.shapeH_mine)
        v_data = v.data.view((n_batch, n_grid))  # flatten batch, grid

        # Collect data by orbits, transfering over MPI as needed:
        # TODO

        # Symmetrize in each orbit:
        v_orbits = v_data[:, self.index]
        v_orbits[:, self.is_conj] = v_orbits[:, self.is_conj].conj()

        v_sym = (v_orbits * self.phase[None]).mean(dim=1)
        v_orbits = v_sym[:, None] * self.phase[None].conj()

        v_orbits[:, self.is_conj] = v_orbits[:, self.is_conj].conj()
        v_data[0] = 0.
        v_data[0].index_put_((self.index,), v_orbits[0], accumulate=True)
        v_data *= self.inv_multiplicity[None]

        # Set results back to original grid, transfering over MPI as needed:
        # TODO
