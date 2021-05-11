import qimpy as qp
import numpy as np


class Electrons:
    'TODO: document class Electrons'

    def __init__(self, *, rc, lattice, ions, symmetries,
                 k_mesh=None, k_path=None,
                 spin_polarized=False, spinorial=False,
                 fillings=None, basis=None, n_bands=None):
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
            for up and down spins if spinorial = False, and implicitly by each
            orbital being spinorial if spinorial = True.
            Default: False
        spinorial : bool, optional
            True, if relativistic / spin-orbit calculations which require
            2-component spinorial wavefunctions, else False.
            Default: False
        fillings : qimpy.electrons.Fillings or None, optional
            Electron occupations and charge / chemical potential control.
            Default: use default qimpy.electrons.Fillings()
        basis : qimpy.electrons.Basis or None, optional
            Wavefunction basis set (plane waves).
            Default: use default qimpy.electrons.Basis()
        n_bands : {'x<scale>', 'atomic', int}, default: 'x1.'
            Number of bands, specified as a scale relative to the minimum
            number of bands to accommodate electrons i.e. 'x1.5' implies
            use 1.5 times the minimum number. Alternately, 'atomic' sets
            the number of bands to the number of atomic orbitals. Finally,
            an integer explicitly sets the number of bands.
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

        # Determine number of bands:
        if n_bands is None:
            n_bands = 'x1'
        if isinstance(n_bands, int):
            self.n_bands = n_bands
            assert(self.n_bands >= 1)
            n_bands_method = 'set explicitly'
        else:
            assert isinstance(n_bands, str)
            if n_bands == 'atomic':
                raise NotImplementedError('n_bands from atomic orbitals')
                n_bands_method = 'atomic'
            else:
                assert n_bands.startswith('x')
                n_bands_scale = float(n_bands[1:])
                if n_bands_scale < 1.:
                    raise ValueError('<scale> must be >=1 in n_bands')
                self.n_bands = max(1, int(np.ceil(self.fillings.n_bands_min
                                                  * n_bands_scale)))
                n_bands_method = 'n_bands_min x ' + n_bands[1:]
        qp.log.info('n_bands: {:d} ({:s})'.format(self.n_bands,
                                                  n_bands_method))

        # Initialize wave-function basis:
        self.basis = qp.construct(
            qp.electrons.Basis, basis, 'basis',
            rc=rc, lattice=lattice, ions=ions, symmetries=symmetries,
            kpoints=self.kpoints, n_spins=self.n_spins, n_spinor=self.n_spinor)

        # Initial wavefunctions:
        qp.log.info('Initializing wavefunctions:'
                    ' bandwidth-limited random numbers')
        self.C = qp.electrons.Wavefunction(self.basis, n_bands=self.n_bands)
        self.C.randomize()
        self.C = self.C.orthonormalize()
