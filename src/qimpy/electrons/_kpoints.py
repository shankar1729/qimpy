import qimpy as qp
import numpy as np
import torch


class Kpoints:
    'Set of k-points in Brillouin zone'
    def __init__(self, rc, k, wk):
        '''Create Kpoints from explicit list of k and weights wk.
        Typically use derived classes Kmesh or Kpath instead.'''
        self.rc = rc
        self.k = k
        self.wk = wk
        qp.log.info(rc.fmt(k))


class Kmesh(Kpoints):
    'Uniform k-mesh sampling of Brillouin zone'
    def __init__(self, *, rc, symmetries, lattice,
                 offset=(0., 0., 0.),
                 size=(1, 1, 1)):
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
        '''TODO: document Kpath constructor'''

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
