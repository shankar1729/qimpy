import qimpy as qp
import torch


class Lattice:

    def __init__(self, *, vector1, vector2, vector3, scale=None):
        print('\n--- Initializing Lattice ---')

        # Get unscaled lattice vectors:
        self.Rbasis = torch.tensor([vector1, vector2, vector3]).T

        # Apply scale if needed:
        if scale:
            scale = torch.tensor(scale).flatten()
            assert((len(scale) == 1) or (len(scale) == 3))
            self.Rbasis = scale[:, None] * self.Rbasis
        print('Rbasis (real-space basis):\n', qp.fmt(self.Rbasis))

        # Compute reciprocal lattice vectors:
        self.Gbasis = torch.linalg.inv(self.Rbasis.T)
        print('Gbasis (reciprocal-space basis):\n', qp.fmt(self.Gbasis))
