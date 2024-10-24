from __future__ import annotations

import torch
import numpy as np

from . import Grid, FieldR
from qimpy import rc
from qimpy.symmetries import Symmetries
from qimpy.lattice import Lattice
from qimpy.lattice._wigner_seitz import WignerSeitz


class CoulombEmbedder:
    """Class for embedding a center for truncated Coulomb interactions"""

    periodic: tuple[bool, bool, bool]  #: Periodicity of each lattice vector
    gridOrig: Grid  #: Original grid
    gridEmbed: Grid  #: Embedded grid
    bMap: torch.Tensor  #: Transformation matrix from original --> embedding grid
    #ion_width: float  #: Ion-charge gaussian width for embedding TODO: implement

    def __init__(self, grid: Grid) -> None:
        """Initialize coulomb embedding.

        Parameters
        ----------
        grid
            Original data grid before embedding
        coulomb
            Relevant Coulomb interaction information (as of now, only uses ion_width)
        """

        self.gridOrig = grid
        self.periodic = grid.lattice.periodic

        gridOrig = self.gridOrig

        # Initialize embedding grid
        meshOrig = gridOrig.get_mesh("R", mine=True)
        Sorig = torch.tensor(gridOrig.shape)
        RbasisOrig = gridOrig.lattice.Rbasis
        # Extend cell in non-periodic directions
        dimScale = np.where(self.periodic, 1, 2)
        latticeEmbed = Lattice(Rbasis=(RbasisOrig.cpu() @ np.diag(dimScale)))
        gridEmbed = Grid(
            lattice=latticeEmbed,
            symmetries=Symmetries(lattice=latticeEmbed),
            shape=tuple(dimScale * gridOrig.shape),
            comm=rc.comm,
        )
        self.gridEmbed = gridEmbed
        Sembed = torch.tensor(gridEmbed.shape, device=rc.device)
        # Report embedding center in various coordinate systems:
        latticeCenter = np.array(grid.lattice.center)
        rCenter = RbasisOrig.cpu() @ latticeCenter
        ivCenter = np.round(
            latticeCenter @ (1.0 * np.diag(gridOrig.shape))
        ).astype(int)
        print("Integer grid location selected as the embedding center:")
        print("\tGrid: {:6} {:6} {:6}".format(*tuple(ivCenter)))
        print("\tLattice: {:6.3f} {:6.3f} {:6.3f}".format(*tuple(latticeCenter)))
        print("\tCartesian: {:6.3f} {:6.3f} {:6.3f}".format(*tuple(rCenter)))
        # Setup Wigner-Seitz cells of original and embed meshes
        wsOrig = WignerSeitz(gridOrig.lattice.Rbasis)
        wsEmbed = WignerSeitz(gridEmbed.lattice.Rbasis)
        # Reduce indices of embedded mesh with respect to its Wigner-Seitz cell
        ivEmbed = gridEmbed.get_mesh("R", mine=True).reshape(-1, 3)
        ivEmbed_wsOrig = wsEmbed.reduce_index(ivEmbed, Sembed)
        # Shift original mesh to be centered about the origin
        shifts = np.round(latticeCenter * meshOrig.shape[0:3]).astype(int)
        ivEquivOrig = (ivEmbed_wsOrig.cpu() + shifts) % Sorig[None, :]
        # Setup mapping between original and embedding meshes
        iEmbed = ivEmbed[:, 2] + Sembed[2] * (ivEmbed[:, 1] + Sembed[1] * ivEmbed[:, 0])
        iEquivOrig = ivEquivOrig[:, 2] + Sorig[2] * (
            ivEquivOrig[:, 1] + Sorig[1] * ivEquivOrig[:, 0]
        )
        # Symmetrize points on boundary using weight function "smoothTheta" function
        diagSembedInv = torch.diag(1 / torch.tensor(gridEmbed.shape, device=rc.device))
        xWS = (1.0 * ivEmbed_wsOrig @ diagSembedInv) @ gridEmbed.lattice.Rbasis.T
        weights = smoothTheta(wsOrig.ws_boundary_distance(xWS))
        bMap = torch.sparse_coo_tensor(
            np.array([iEquivOrig, iEmbed.cpu()]), weights, device=rc.device
        )
        colSums = torch.sparse.sum(bMap, dim=1).to_dense()
        colNorms = torch.sparse.spdiags(
            1.0 / colSums, offsets=torch.tensor([0]), shape=(Sorig.prod(), Sorig.prod())
        )
        self.bMap = torch.sparse.mm(colNorms, bMap)

    def embedExpand(self, fieldOrig: FieldR) -> FieldR:
        """Expand real-space field 'fieldOrig' within larger embedding cell."""
        dataOrig = fieldOrig.data
        dataEmbed = (dataOrig.reshape(-1) @ self.bMap).reshape(self.gridEmbed.shape)
        return FieldR(self.gridEmbed, data=dataEmbed)

    def embedShrink(self, fieldEmbed: FieldR) -> FieldR:
        """Shrink real-space field 'fieldEmbed' to original cell"""
        dataEmbed = fieldEmbed.data
        # Shrink operation is dagger of embedExpand (bMap is real-valued)
        dataOrig = (dataEmbed.reshape(-1) @ self.bMap.T).reshape(self.gridOrig.shape)
        return FieldR(self.gridOrig, data=dataOrig)


def smoothTheta(x: torch.Tensor) -> torch.Tensor:
    return torch.where(
        x <= -1, 1.0, torch.where(x >= 1, 0.0, 0.25 * (2.0 - x * (3.0 - x**2)))
    )
