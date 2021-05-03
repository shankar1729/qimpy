import qimpy as qp


class Electrons:
    'TODO: document class Electrons'

    def __init__(self, *, rc, lattice, symmetries,
                 k_mesh=None, k_path=None):
        '''TODO: document Electrons constructor'''
        self.rc = rc
        qp.log.info('\n--- Initializing Electrons ---')

        # Initialize k-points:
        if k_mesh is None:
            if k_path is None:
                self.kpoints = qp.electrons.Kmesh(  # Gamma-only
                    rc=rc, symmetries=symmetries, lattice=lattice)
            else:
                self.kpoints = qp.construct(
                    qp.electrons.Kpath, k_path, 'k_path',
                    rc=rc, lattice=lattice)
        else:
            if k_path is None:
                self.kpoints = qp.construct(
                    qp.electrons.Kmesh, k_mesh, 'k_mesh',
                    rc=rc, symmetries=symmetries, lattice=lattice)
            else:
                raise ValueError('Cannot use both k-mesh and k-path')
