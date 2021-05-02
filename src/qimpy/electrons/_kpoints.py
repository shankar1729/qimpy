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
    def __init__(self, *, rc, offset=None, folding=None):
        # Assign defaults:
        if offset is None:
            offset = [0., 0., 0.]
        if folding is None:
            folding = [1, 1, 1]
        # TODO


class Kpath(Kpoints):
    '''Path of k-points traversing Brillouin zone
    (typically for band structure calculations)'''
    def __init__(self, *, rc, dk, points):
        # TODO
        pass
