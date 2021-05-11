import qimpy as qp
import numpy as np
import torch


class Basis(qp.utils.TaskDivision):
    'TODO: document class Basis'

    def __init__(self, *, rc, lattice, ions, symmetries,
                 kpoints, n_spins, n_spinor,
                 ke_cutoff=20., real_wavefunctions=False, grid=None):
        '''
        Parameters
        ----------
        rc : qimpy.utils.RunConfig
            Current run configuration.
        lattice : qimpy.lattice.Lattice
            Lattice whose reciprocal lattice vectors define plane-wave basis.
        ions : qimpy.ions.Ions
            Ions that specify the pseudopotential portion of the basis;
            the basis implicitly depends on the ion positions for ultrasoft
            or PAW due to the augmentation of all operators at each ion
        symmetries : qimpy.symmetries.Symmetries
            Symmetries with which the wavefunction grid should be commensurate
        kpoints : qimpy.electrons.Kpoints
            Set of k-points to initialize basis for. Note that the basis is
            only initialized for k-points to be operated on by current process
            i.e. for k = kpoints.k[kpoints.i_start : kpoints.i_stop]
        n_spins : int
            Default number of spin channels for wavefunctions in this basis
        n_spinor : int
            Default number of spinor components for wavefunctions in this
            basis. Also used only to check support for real_wavefunctions
        ke_cutoff : float, default: 20
            Plane-wave kinetic-energy cutoff for wavefunctions in :math:`E_h`
        real_wavefunctions : bool, default: False
            If True, use wavefunctions that are real, instead of the
            default complex wavefunctions. This is only supported for
            non-spinorial, Gamma-point-only calculations.
        grid : dict, optional
            Optionally override parameters (such as shape or ke_cutoff)
            of the grid (qimpy.grid.Grid) used for wavefunction operations.
        '''
        self.rc = rc
        self.lattice = lattice
        self.ions = ions
        self.kpoints = kpoints
        self.n_spins = n_spins
        self.n_spinor = n_spinor

        # Select subset of k-points relevant on this process:
        k_mine = slice(kpoints.i_start, kpoints.i_stop)
        self.k = kpoints.k[k_mine]
        self.wk = kpoints.wk[k_mine]

        # Check real wavefunction support:
        self.real_wavefunctions = real_wavefunctions
        if self.real_wavefunctions:
            if n_spinor == 2:
                raise ValueError('real-wavefunctions not compatible'
                                 ' with spinorial calculations')
            if kpoints.k.norm().item():  # i.e. not all k = 0 (Gamma)
                raise ValueError('real-wavefunctions only compatible with'
                                 ' Gamma-point-only calculations')

        # Initialize grid to match cutoff:
        self.ke_cutoff = float(ke_cutoff)
        qp.log.info('\nInitializing wavefunction grid:')
        self.grid = qp.grid.Grid(
            rc=rc, lattice=lattice, symmetries=symmetries,
            comm=None,  # always process-local
            ke_cutoff_wavefunction=self.ke_cutoff,
            **(qp.dict_input_cleanup(grid) if grid else {}))

        # Initialize basis:
        self.iG = self.grid.get_mesh('H' if self.real_wavefunctions
                                     else 'G').reshape((3, -1)).T
        within_cutoff = (self.get_ke() < ke_cutoff)  # mask of which iG to keep
        # --- determine max and avg n_basis across all k:
        self.n_basis = within_cutoff.count_nonzero(dim=1)
        n_basis_max = rc.comm_k.allreduce(self.n_basis.max().item(),
                                          qp.MPI.MAX)
        self.n_tot = (qp.utils.ceildiv(n_basis_max, rc.n_procs_b)
                      * rc.n_procs_b)  # padded to be multiple of n_procs
        n_basis_avg = rc.comm_k.allreduce(
            (self.n_basis.to(float) @ self.wk).item(), qp.MPI.SUM)
        n_basis_ideal = ((2.*ke_cutoff)**1.5) * lattice.volume / (6 * np.pi**2)
        qp.log.info('n_basis:  max: {:d}  avg: {:.3f}  ideal: {:.3f}'.format(
            n_basis_max, n_basis_avg, n_basis_ideal))
        # --- create indices from basis set to FFT grid:
        n_fft = self.iG.shape[0]  # number of points on FFT grid
        assert(self.n_tot <= n_fft)  # make sure padding doesn't exceed grid
        fft_range = torch.arange(n_fft, device=self.rc.device)
        self.fft_index = (torch.where(within_cutoff, 0, n_fft)
                          + fft_range[None, :]).argsort(  # ke<cutoff to front
                              dim=1)[:, :self.n_tot]  # same count all k
        self.iG = self.iG[self.fft_index]  # basis plane waves for each k
        self.pad_index = torch.where(
            fft_range[None, :self.n_tot]
            > self.n_basis[:, None])  # index to padded entries
        self.pad_index = (
            slice(None), self.pad_index[0], slice(None), slice(None),
            self.pad_index[1])  # add spin, band and spinor dims

        # Divide basis on comm_b:
        super().__init__(self.n_tot, rc.n_procs_b, rc.i_proc_b, 'padded basis')
        self.mine = slice(self.i_start, self.i_stop)
        # --- initialize local pad index separately (not trivially sliceable):
        self.pad_index_mine = torch.where(
            fft_range[None, self.i_start:self.i_stop]
            > self.n_basis[:, None])  # index to local padded entries
        self.pad_index_mine = (
            slice(None), self.pad_index_mine[0], slice(None), slice(None),
            self.pad_index_mine[1])  # add spin, band and spinor dims

        # Extra book-keeping for real-wavefunction basis:
        if self.real_wavefunctions and kpoints.n_mine:
            # Find conjugate pairs with iG_z = 0:
            self.index_z0 = torch.where(self.iG[0, :, 2] == 0)[0]
            # --- compute index of each point and conjugate in iG_z = 0 plane:
            shapeH = self.grid.shapeH_mine
            plane_index = self.fft_index[0, self.index_z0].div(
                shapeH[2], rounding_mode='floor')
            iG_conj = (-self.iG[0, self.index_z0, :2]) % torch.tensor(
                shapeH[:2], device=rc.device)[None, :]
            plane_index_conj = iG_conj[:, 0] * shapeH[1] + iG_conj[:, 1]
            # --- map plane_index_conj to basis using full plane for look-up:
            plane = torch.zeros(shapeH[0] * shapeH[1],
                                dtype=int, device=rc.device)
            plane[plane_index] = self.index_z0
            self.index_z0_conj = plane[plane_index_conj].clone().detach()
            # Weight by element for overlaps (only for this process portion):
            self.Gweight_mine = torch.zeros(self.n_each, device=self.rc.device)
            self.Gweight_mine[:self.n_mine] = torch.where(
                self.iG[0, self.i_start:self.i_stop, 2] == 0, 1., 2.)
            qp.log.info('basis weight sum: {:g}'.format(
                rc.comm_b.allreduce(self.Gweight_mine.sum().item(),
                                    qp.MPI.SUM)))

    def get_ke(self, basis_slice=slice(None)):
        '''Kinetic energy (KE) of each plane wave in basis in :math:`E_h`

        Parameters
        ----------
        basis_slice : slice, default: slice(None)
            Selection of basis functions to get KE for (default: full basis)

        Returns
        -------
        torch.Tensor (nk_mine x len(basis_slice), float)
        '''
        return 0.5 * (((self.iG[:, basis_slice] + self.k[:, None, :])
                       @ self.lattice.Gbasis.T) ** 2).sum(dim=-1)
