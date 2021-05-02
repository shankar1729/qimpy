import qimpy as qp
import numpy as np


class Kpoints:
    'Set of k-points in Brillouin zone'
    def __init__(self, rc, k, wk):
        '''Create Kpoints from explicit list of k and weights wk.
        Typically use derived classes Kmesh or Kpath instead.'''
        self.rc = rc
        self.k = k
        self.wk = wk


class Kmesh(Kpoints):
    'Uniform k-mesh sampling of Brillouin zone'
    def __init__(self, *, rc,
                 offset=(0., 0., 0.),
                 folding=(1, 1, 1)):
        # Check types and sizes:
        offset = np.array(offset)
        folding = np.array(folding)
        assert((offset.shape == (3,)) and (offset.dtype == float))
        assert((folding.shape == (3,)) and (folding.dtype == int))
        qp.log.info('Initializing {:d} x {:d} x {:d} uniform k-mesh '.format(
            *tuple(folding)) + (
                'centered at Gamma'
                if (np.linalg.norm(offset) == 0.)
                else ('offset by ' + np.array2string(offset, separator=', '))))


class Kpath(Kpoints):
    '''Path of k-points traversing Brillouin zone
    (typically for band structure calculations)'''
    def __init__(self, *, rc, dk, points):
        # TODO
        pass
