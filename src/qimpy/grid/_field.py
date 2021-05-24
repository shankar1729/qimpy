import qimpy as qp
import torch
from abc import ABCMeta, abstractmethod
from typing import Tuple, Optional, Sequence, TYPE_CHECKING
if TYPE_CHECKING:
    from ._grid import Grid


class Field(metaclass=ABCMeta):
    '''Abstract base class for scalar/vector fields in real/reciprocal space.
    Provides common operators for fields in either space, but any fields
    used must specifically be in real (:class:`FieldR` and :class:`FieldC`),
    full-reciprocal (:class:`FieldG`) or half-reciprocal (:class:`FieldH`)
    space.'''

    __slots__ = ('grid', 'data')
    grid: 'Grid'  #: Associated grid, which determines dimensions of the field
    data: torch.Tensor  #: Underlying data, with last three dimensions on grid

    @abstractmethod
    def __init__(self, grid: 'Grid',
                 dtype: torch.dtype, shape_grid: Tuple[int, ...], *,
                 shape_batch: Sequence[int] = tuple(),
                 data: Optional[torch.Tensor] = None) -> None:
        '''Initialize data common to all field types. This abstract method
        should only be called as a helper from subclass initializers.

        Parameters
        ----------
        grid
            Associated grid, which determines last three dimensions of data
        dtype
            Data type, which may be real or complex for real space fields,
            but will always be complex in reciprocal space
        shape_grid
            Last three dimensions of data, based on grid
        shape_batch
            Optional preceding batch dimensions for vector fields, arrays of
            scalar fields etc.
        data
            Initial data if provided; initialize to zero otherwise
        '''
        self.grid = grid
        if data is None:
            # Initialize to zero:
            self.data = torch.zeros(shape_grid + tuple(shape_batch),
                                    dtype=dtype, device=grid.rc.device)
        else:
            # Initialize to provided data:
            assert data.shape[-3:] == shape_grid
            assert data.dtype == dtype
            self.data = data


class FieldR(Field):
    '''Real fields in real space.'''
    def __init__(self, grid: 'Grid', *, shape_batch: Sequence[int] = tuple(),
                 data: Optional[torch.Tensor] = None) -> None:
        '''Initialize real-space real fields with zeros / specified data.

        Parameters
        ----------
        shape_batch
            Optional preceding batch dimensions for vector fields, arrays of
            scalar fields etc.
        data
            Initial data if provided; initialize to zero otherwise
        '''
        super().__init__(grid, torch.double, grid.shapeR_mine,
                         shape_batch=shape_batch, data=data)


class FieldC(Field):
    '''Complex fields in real space.'''
    def __init__(self, grid: 'Grid', *, shape_batch: Sequence[int] = tuple(),
                 data: Optional[torch.Tensor] = None) -> None:
        '''Initialize real-space complex fields with zeros / specified data.

        Parameters
        ----------
        shape_batch
            Optional preceding batch dimensions for vector fields, arrays of
            scalar fields etc.
        data
            Initial data if provided; initialize to zero otherwise
        '''
        super().__init__(grid, torch.cdouble, grid.shapeR_mine,
                         shape_batch=shape_batch, data=data)


class FieldH(Field):
    '''Real fields in (half) reciprocal space. Note that the underlying
    data is complex in reciprocal space, but reduced to one half of
    reciprocal space using Hermitian symmetry.'''
    def __init__(self, grid: 'Grid', *, shape_batch: Sequence[int] = tuple(),
                 data: Optional[torch.Tensor] = None) -> None:
        '''Initialize half-reciprocal-space fields with zeros / specified data.

        Parameters
        ----------
        shape_batch
            Optional preceding batch dimensions for vector fields, arrays of
            scalar fields etc.
        data
            Initial data if provided; initialize to zero otherwise
        '''
        super().__init__(grid, torch.cdouble, grid.shapeH_mine,
                         shape_batch=shape_batch, data=data)


class FieldG(Field):
    '''Complex fields in (full) reciprocal space.'''
    def __init__(self, grid: 'Grid', *, shape_batch: Sequence[int] = tuple(),
                 data: Optional[torch.Tensor] = None) -> None:
        '''Initialize full-reciprocal-space fields with zeros / specified data.

        Parameters
        ----------
        shape_batch
            Optional preceding batch dimensions for vector fields, arrays of
            scalar fields etc.
        data
            Initial data if provided; initialize to zero otherwise
        '''
        super().__init__(grid, torch.cdouble, grid.shapeG_mine,
                         shape_batch=shape_batch, data=data)


# Test field construction / operations:
if __name__ == "__main__":
    qp.utils.log_config()
    qp.log.info('*'*15 + ' QimPy ' + qp.__version__ + ' ' + '*'*15)
    rc = qp.utils.RunConfig()
    # Prepare a grid for testing:
    lattice = qp.lattice.Lattice(
        rc=rc, system='triclinic', a=2.1, b=2.2, c=2.3,
        alpha=75, beta=80, gamma=85)  # pick one with no symmetries
    ions = qp.ions.Ions(rc=rc, pseudopotentials=[], coordinates=[])
    symmetries = qp.symmetries.Symmetries(rc=rc, lattice=lattice, ions=ions)
    grid = qp.grid.Grid(rc=rc, lattice=lattice, symmetries=symmetries,
                        shape=(96, 108, 112), comm=rc.comm)
    # Tests:
    v1R = FieldR(grid)
    v2C = FieldC(grid)
    v3H = FieldH(grid)
    v4G = FieldG(grid)
    print('v1R', v1R.data.shape, v1R.data.dtype)
    print('v2C', v2C.data.shape, v2C.data.dtype)
    print('v3H', v3H.data.shape, v3H.data.dtype)
    print('v4G', v4G.data.shape, v4G.data.dtype)

    qp.utils.StopWatch.print_stats()
