import qimpy as qp
import numpy as np
import torch


class Grid:
    'TODO: document class Grid'

    def __init__(self, *, rc, lattice, symmetries, ke_cutoff_orbital=None,
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
        ke_cutoff_orbital : float, optional
            Plane-wave kinetic-energy cutoff in :math:`E_h` for any electronic
            orbitals to be used with this grid. This is an internally set
            parameter (should not be specified in dict / YAML input) that
            effectively sets the default for ke_cutoff.
        ke_cutoff : float, default: 4 * ke_cutoff_orbital (if available)
            Plane-wave kinetic-energy cutoff in :math:`E_h` for the grid
            (i.e. the charge-density cutoff). This supercedes the default
            set by ke_cutoff_orbital (if any), but may be superceded in turn
            by shape, if explicitly specified
        shape : list of 3 ints, optional
            Explicit grid dimensions. Highest precedence, and will supercede
            either ke_cutoff and ke_cutoff_orbital, if specified
        '''
        self.rc = rc

        # Select the relevant ke-cutoff:
        self.ke_cutoff = ke_cutoff
        if ke_cutoff_orbital:
            if not ke_cutoff:  # note that ke_cutoff takes precedence
                self.ke_cutoff = 4*ke_cutoff_orbital
            # Make sure specified cutoff is sufficient to resolve orbitals:
            if self.ke_cutoff < ke_cutoff_orbital:
                raise ValueError('ke_cutoff (={:g}) must be >= '
                                 'ke_cutoff_orbital (={:g})'.format(
                                    self.ke_cutoff, ke_cutoff_orbital))
            elif self.ke_cutoff < 4*ke_cutoff_orbital:
                qp.log.info('Note: ke_cutoff (={:g}) < 4*ke_cutoff_orbital '
                            '(={:g}) truncates high wave vectors in density '
                            'calculation'.format(self.ke_cutoff,
                                                 4*ke_cutoff_orbital))

        # Compute minimum grid dimensions for cutoff:
        shape_min = None
        if self.ke_cutoff:
            Gmax = np.sqrt(2.*self.ke_cutoff)  # G-sphere radius at cutoff
            # This sphere should be within shape_min/2 in each direction x
            # corresponding spacing between reciprocal lattice planes (2pi/R).
            # Therefore shape_min >= 2 * Gmax / (2pi/R)
            shape_min = (lattice.Rbasis.norm(dim=0) * (Gmax / np.pi)).tolist()
            qp.log.info('minimum shape for ke-cutoff: '
                        '[{:.2f}, {:.2f}, {:.2f}]'.format(*tuple(shape_min)))
            # Align to multiple of 4 for FFT efficiency:
            shape_min = 4 * np.ceil(np.array(shape_min) / 4).astype(int)
            qp.log.info('minimum multiple-of-4 shape: '
                        '[{:d}, {:d}, {:d}]'.format(*tuple(shape_min)))

        if shape:
            self.shape = tuple(shape)
            # Check symmetries and cutoff of specified shape:
            symmetries.check_grid_shape(self.shape)
            if ((shape_min is not None)
                    and np.any(np.array(self.shape) < shape_min)):
                raise ValueError('Specified shape [{:d}, {:d}, {:d}] < '
                                 'minimum shape'.format(*self.shape))
        else:
            if shape_min is None:
                raise KeyError('At least one of ke-cutoff, ke-cutoff-orbital '
                               'or shape must be specified')
