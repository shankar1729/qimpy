import qimpy as qp
import numpy as np
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

    # Determine constraints on S due to symmetries:
    rot = self.rot.to(dtype=int, device=self.rc.cpu).numpy()
    ratios = np.gcd.reduce(rot, axis=0)

    # Recursive function to set grid shapes compatible with one dimension
    def process(Sb, j):
        '''Given an integer vector Sb where Sb[j] is known to be non-zero,
        return smallest integer vector that would be commensurate with
        symmetries, by setting connected dimensions to j as appopriate.'''
        # Check dimensions constrained to j:
        k_linked = np.logical_or(ratios[j, :], ratios[:, j])
        k_linked[j] = False  # no need to check j against j
        for k in np.where(k_linked)[0]:
            if Sb[k]:  # pre-existing entry
                if(((ratios[j, k] * Sb[j]) % Sb[k])
                   or ((ratios[k, j] * Sb[k]) % Sb[j])):
                    # Sb violates constraints between j and k
                    qp.log.info('could not find anisotropic shape'
                                ' commensurate with symmetries')
                    return np.ones(3, dtype=int)  # fall-back solution
            else:  # add k to this basis entry
                if ratios[k, j]:
                    # scale remaining dimensions
                    Sb_k_new = Sb[j] * np.maximum(1, ratios[j, k])
                    Sb *= ratios[k, j]
                    Sb[k] = Sb_k_new
                else:  # ratios[j, k] must be non-zero (since j, k linked)
                    Sb[k] = ratios[j, k] * Sb[j]
                Sb = process(Sb, k)  # recursively process now non-zero dim k
        return Sb

    # Expand symmetry-compatible grid dimension-by-dimension:
    shape = np.zeros(3, dtype=int)
    for j in range(3):
        if not shape[j]:
            # start with a unit basis vector along j:
            Sb = np.zeros(3, dtype=int)
            Sb[j] = 1
            # make it symmetry-compatible:
            Sb = process(Sb, j)
            Sb //= np.gcd.reduce(Sb)  # remove common factors
            # check FFT suitability of Sb:
            i_nz = np.where(Sb)[0]
            if not np.logical_and.reduce([qp.utils.fft_suitable(s)
                                          for s in Sb[i_nz]]):
                qp.log.info('could not find anisotropic shape with'
                            ' FFT-suitable factors')
                Sb[i_nz] = 1
            # determine smallest fft-suitable scale factor to reach shape_min
            scale_Sb = 2*qp.utils.ceildiv(shape_min[i_nz], 2*Sb[i_nz]).max()
            while not qp.utils.fft_suitable(scale_Sb):
                scale_Sb += 2  # move through even numbers
            shape += scale_Sb * Sb
    return shape
