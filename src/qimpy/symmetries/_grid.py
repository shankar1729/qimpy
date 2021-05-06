import qimpy as qp
import torch


def _check_grid_shape(self, shape):
    '''Check that grid dimensions are compatible with symmetries

    Parameters
    ----------
    shape : tuple of 3 ints
        Grid dimensions

    Raises
    ------
    ValueError
        If incommensurate, raise ValueError with the error string
        including the reduced symmetry of specified grid shape
    '''

    # Compute rotation matrix in mesh coordinates
    S = torch.tensor(shape, dtype=float, device=self.rc.device)
    rot_mesh = S.view(1, 3, 1) * self.rot * (1./S).view(1, 1, 3)

    # Commensurate => matrix should still be an integer:
    err = (rot_mesh - rot_mesh.round()).abs().sum(dim=(-2, -1))
    i_sym = torch.where(err <= self.tolerance)[0]
    if len(i_sym) < self.n_sym:
        raise ValueError(
            'Grid dimensions {:s} commensurate only with a sub-group of '
            'symmetries with indices (0-based): {:s}'.format(
                str(shape), str(i_sym.tolist())))


def _get_grid_shape(self, shape_min):
    '''Determine smallest FFT-suitable grid dimensions >= shape_min
    that are compatible with symmetries

    Parameters
    ----------
    shape_min : tuple of 3 ints
        Minimum grid dimensions

    Returns
    -------
    tuple of 3 ints
    '''
    pass
