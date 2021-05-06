import qimpy as qp


class Grid:
    'TODO: document class Grid'

    def __init__(self, *, rc, lattice, symmetries, ke_cutoff_orbitals=None,
                 ke_cutoff=None, shape=None):
        '''
        Parameters
        ----------
        rc : qimpy.utils.RunConfig
            Current run configuration
        lattice : qimpy.lattice.Lattice
            Lattice whose reciprocal lattice vectors define plane-wave basis
        symmetries : qimpy.symmetries.Symmetries
            Symmetries with which grid dimensions will be made commensurate,
            or checked if specified explicitly by shape below.
        ke_cutoff_orbitals : float, optional
            Plane-wave kinetic-energy cutoff in :math:`E_h` for any electronic
            orbitals to be used with this grid. This is an internally set
            parameter (should not be specified in dict / YAML input) that
            effectively sets the default for ke_cutoff.
        ke_cutoff : float, default: 4 * ke_cutoff_orbitals (if available)
            Plane-wave kinetic-energy cutoff in :math:`E_h` for the grid
            (i.e. the charge-density cutoff). This supercedes the default
            set by ke_cutoff_orbitals (if any), but may be superceded in turn
            by shape, if explicitly specified
        shape : list of 3 ints, optional
            Explicit grid dimensions. Highest precedence, and will supercede
            either ke_cutoff and ke_cutoff_orbitals, if specified
        '''
        self.rc = rc
        qp.log.info('TODO: initialize grid with ke_cutoff_orbitals: {:s}  '
                    'ke_cutoff: {:s}  shape:  {:s}'.format(
                        str(ke_cutoff_orbitals), str(ke_cutoff), str(shape)))
