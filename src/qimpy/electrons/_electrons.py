import qimpy as qp


class Electrons:
    'TODO: document class Electrons'

    def __init__(self, *, rc, lattice, ions, symmetries,
                 k_mesh=None, k_path=None,
                 spin_polarized=False, spinorial=False,
                 fillings=None):
        '''
        Parameters
        ----------
        rc : qimpy.utils.RunConfig
            Current run configuration
        lattice : qimpy.lattice.Lattice
            Lattice (unit cell) to associate with electronic wave functions
        ions : qimpy.ions.Ions
            Ionic system interacting with the electrons
        symmetries : qimpy.symmetries.Symmetries
            Symmetries for k-point reduction and density symmetrization
        k_mesh : qimpy.electrons.Kmesh or dict, optional
            Uniform k-point mesh for Brillouin-zone integration.
            Specify only one of k_mesh or k_path.
            Default: use default qimpy.electrons.Kmesh()
        k_path : qimpy.electrons.Kpath or dict, optional
            Path of k-points through Brillouin zone, typically for band
            structure calculations. Specify only one of k_mesh or k_path.
            Default: None
        spin_polarized : bool, optional
            True, if electronic system has spin polarization / magnetization
            (i.e. breaks time reversal symmetry), else False.
            Spin polarization is treated explicitly with two sets of orbitals
            for up and down spins if spinorial = False, and implicitly by the
            spinorial wavefunctions if spinorial = True.
            Default: False
        spinorial : bool, optional
            True, if relativistic / spin-orbit calculations which require
            2-component spinorial wavefunctions, else False.
            Default: False
        fillings : qimpy.electrons.Fillings or None, optional
            Electron occupations and charge / chemical potential control.
            Default: use default qimpy.electrons.Fillings()
        '''
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

        # Initialize spin:
        self.spin_polarized = spin_polarized
        self.spinorial = spinorial
        # --- set # spinor components, # spin channels and weight
        self.n_spinor = (2 if spinorial else 1)
        self.n_spins = (2 if (spin_polarized and not spinorial) else 1)
        self.w_spin = 2 // (self.n_spins * self.n_spinor)  # spin weight
        qp.log.info('n_spins: {:d}  n_spinor: {:d}  w_spin: {:d}'.format(
            self.n_spins, self.n_spinor, self.w_spin))

        # Initialize fillings:
        self.fillings = qp.construct(
            qp.electrons.Fillings, fillings, 'fillings',
            rc=rc, ions=ions, electrons=self)
