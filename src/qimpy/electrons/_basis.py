import qimpy as qp


class Basis:
    'TODO: document class Basis'

    def __init__(self, *, rc, lattice, symmetries, kpoints, spinorial,
                 ke_cutoff=20., real_orbitals=False, grid=None):
        '''
        Parameters
        ----------
        rc : qimpy.utils.RunConfig
            Current run configuration.
        lattice : qimpy.lattice.Lattice
            Lattice whose reciprocal lattice vectors define plane-wave basis.
        symmetries : qimpy.symmetries.Symmetries
            Symmetries with which the orbital grid should be commensurate
        kpoints : qimpy.electrons.Kpoints
            Set of k-points to initialize basis for. Note that the basis is
            only initialized for k-points to be operated on by current process
            i.e. for k = kpoints.k[kpoints.i_start : kpoints.i_stop]
        spinorial : bool
            Whether the basis should support spinorial calculations.
            This is essentially used only to check real_orbitals.
        ke_cutoff : float, default: 20
            Plane-wave kinetic-energy cutoff in :math:`E_h`
        real_orbitals : bool, default: False
            If True, use wavefunctions that are real, instead of the
            default complex wavefunctions. This is only supported for
            non-spinorial, Gamma-point-only calculations.
        grid : dict, optional
            Optionally override parameters (such as shape or ke_cutoff)
            of the grid (qimpy.grid.Grid) used for wavefunction operations.
        '''
        self.rc = rc

        # Select subset of k-points relevant on this process:
        k_mine = slice(kpoints.i_start, kpoints.i_stop)
        self.k = kpoints.k[k_mine]
        self.wk = kpoints.wk[k_mine]

        # Check real orbital support:
        self.real_orbitals = real_orbitals
        if self.real_orbitals:
            if spinorial:
                raise ValueError('real-orbitals not compatible'
                                 ' with spinorial calculations')
            if kpoints.k.norm().item():  # i.e. not all k = 0 (Gamma)
                raise ValueError('real-orbitals only compatible with'
                                 ' Gamma-point-only calculations')

        # Initialize grid to match cutoff:
        self.ke_cutoff = float(ke_cutoff)
        qp.log.info('Initializing orbital grid:')
        self.grid = qp.grid.Grid(
            rc=rc, lattice=lattice, symmetries=symmetries,
            comm=None,  # always process-local
            ke_cutoff_orbital=self.ke_cutoff,
            **(qp.dict_input_cleanup(grid) if grid else {}))
