import qimpy as qp
import numpy as np
import torch


class Kpoints:
    'Set of k-points in Brillouin zone'
    def __init__(self, rc, k, wk):
        '''
        Construct from explicit list of k-points and integration weights.
        Typically, this should be used only by derived classes
        such as qimpy.electrons.Kmesh or qimpy.electrons.Kpath

        Parameters
        ----------
        rc : qimpy.utils.RunConfig
            Current run configuration.
        k : torch.Tensor (N x 3)
            Explicit list of k-points for Brillouin zone integration.
        wk : torch.Tensor (N)
            Corresponding Brillouin zone integration weights (should add to 1).

'''
        self.rc = rc
        self.k = k
        self.wk = wk
        assert(abs(wk.sum() - 1.) < 1e-14)


class Kmesh(Kpoints):
    'Uniform k-mesh sampling of Brillouin zone'

    def __init__(self, *, rc, symmetries, lattice,
                 offset=(0., 0., 0.),
                 size=(1, 1, 1)):
        '''
        Parameters
        ----------
        rc : qimpy.utils.RunConfig
            Current run configuration.
        symmetries : qimpy.symmetries.Symmetries
            Symmetry group used to reduce k-points to irreducible set.
        lattice : qimpy.lattice.Lattice
            Lattice specification used for automatic size determination.
        offset : list of 3 floats, optional
            Offset k-point mesh by this amount in k-mesh coordinates
            i.e. by offset / size in fractional reciprocal coordinates.
            For example, use [0.5, 0.5, 0.5] for the Monkhorst-Pack scheme.
            Default: [0., 0., 0.] selects Gamma-centered mesh.
        size: list of 3 ints, or float, optional
            If given as a list of 3 integers, number of k-points along each
            reciprocal lattice direction. Instead, a single float specifies
            the minimum real-space size of the k-point sampled supercell
            i.e. pick number of k-points along dimension i = ceil(size / L_i),
            where L_i is the length of lattice vector i (in bohrs).
            Default: [1, 1, 1] selects a single k-point = offset.'''

        # Select size from real-space dimension if needed:
        if isinstance(size, float) or isinstance(size, int):
            sup_length = float(size)
            L_i = torch.linalg.norm(lattice.Rbasis, dim=0)  # lattice lengths
            size = torch.ceil(sup_length / L_i).to(int).tolist()
            qp.log.info(
                'Selecting {:d} x {:d} x {:d} k-mesh '.format(*tuple(size))
                + 'for supercell size >= {:g} bohrs'.format(sup_length))

        # Check types and sizes:
        offset = np.array(offset)
        size = np.array(size)
        assert((offset.shape == (3,)) and (offset.dtype == float))
        assert((size.shape == (3,)) and (size.dtype == int))
        qp.log.info('Creating {:d} x {:d} x {:d} uniform k-mesh '.format(
            *tuple(size)) + (
                'centered at Gamma'
                if (np.linalg.norm(offset) == 0.)
                else ('offset by ' + np.array2string(offset, separator=', '))))

        # Create mesh:
        # kgrids1d = [ for ]


class Kpath(Kpoints):
    '''Path of k-points traversing Brillouin zone
    (typically for band structure calculations)'''

    def __init__(self, *, rc, lattice, dk, points):
        '''
        Parameters
        ----------
        rc : qimpy.utils.RunConfig
            Current run configuration.
        lattice : qimpy.lattice.Lattice
            Lattice specification for converting k-points from
            reciprocal fractional coordinates (input) to Cartesian
            for determining path lengths.
        dk : float
            Maximum distance (in bohr^-1) between adjacent points on k-path
        points : list of lists
            List of special k-points along path: each k-point should contain
            three fractional coordinates (float) and optionally a string
            label for this point for use in band structure plots.
        '''

        # Check types, sizes and separate labels from points:
        dk = float(dk)
        labels = [(point[3] if (len(point) > 3) else '') for point in points]
        points = torch.tensor([point[:3] for point in points],
                              dtype=float, device=rc.device)
        qp.log.info('Creating k-path with dk = '
                    + '{:g} connecting {:d} special points'.format(
                        dk, points.shape[0]))

        # Create path one segment at a time:
        k = [points[:1]]
        self.labels = {0: labels[0]}
        k_length = [torch.zeros((1,), dtype=float, device=rc.device)]
        nk_tot = 1
        distance_tot = 0.
        dpoints = points.diff(dim=0)
        distances = torch.sqrt(((dpoints @ lattice.Gbasis.T)**2).sum(dim=1))
        for i, distance in enumerate(distances):
            nk = torch.ceil(distance / dk).to(int)  # for this segment
            t = torch.arange(1, nk+1, device=rc.device) / nk
            k.append(points[i] + t[:, None] * dpoints[i])
            nk_tot += nk.item()
            self.labels[nk_tot - 1] = labels[i+1]  # label at end of segment
            k_length.append(distance_tot + distance * t)
            distance_tot += distance
        k = torch.cat(k)
        wk = torch.full((nk_tot,),  1./nk_tot, device=rc.device)
        self.k_length = torch.cat(k_length)  # cumulative length on path
        qp.log.info('Created {:d} k-points on k-path of length {:g}'.format(
            nk_tot, distance_tot.item()))

        # Initialize base class:
        super().__init__(rc, k, wk)
