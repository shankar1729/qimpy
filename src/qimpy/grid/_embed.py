from __future__ import annotations

import torch
import numpy as np

from . import Grid, FieldR
from qimpy import rc, log, MPI
from qimpy.mpi._sparse_matrices import SparseMatrixRight
from qimpy.symmetries import Symmetries
from qimpy.lattice import Lattice
from qimpy.lattice._wigner_seitz import WignerSeitz
from qimpy.grid._change import gather, scatter


class CoulombEmbedder:
    """Class for embedding a center for truncated Coulomb interactions"""

    periodic: tuple[bool, bool, bool]  #: Periodicity of each lattice vector
    gridOrig: Grid  #: Original grid
    gridEmbed: Grid  #: Embedded grid
    bMap: SparseMatrixRight  #: Transformation matrix from original --> embedding grid
    bMapT: SparseMatrixRight

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
        Sorig = torch.tensor(gridOrig.shape, device=rc.device)
        RbasisOrig = gridOrig.lattice.Rbasis
        # Extend cell in non-periodic directions
        dimScale = np.where(self.periodic, 1, 2)
        dimScale_t = torch.from_numpy(dimScale).to(rc.device)
        latticeEmbed = Lattice(Rbasis=(RbasisOrig * dimScale_t))
        gridEmbed = Grid(
            lattice=latticeEmbed,
            symmetries=Symmetries(lattice=latticeEmbed),
            shape=tuple(dimScale * gridOrig.shape),
            comm=gridOrig.comm,
        )
        self.gridEmbed = gridEmbed
        Sembed = torch.tensor(gridEmbed.shape, device=rc.device)
        # Report embedding center in various coordinate systems:
        lattice_center = grid.lattice.center
        rCenter = RbasisOrig @ lattice_center
        ivCenter = torch.round(
            lattice_center * torch.tensor(gridOrig.shape, device=rc.device)
        ).to(torch.int)
        log.info("Integer grid location selected as the embedding center:")
        log.info("\tGrid: {:6} {:6} {:6}".format(*ivCenter.tolist()))
        log.info("\tLattice: {:6.3f} {:6.3f} {:6.3f}".format(*lattice_center.tolist()))
        log.info("\tCartesian: {:6.3f} {:6.3f} {:6.3f}".format(*rCenter.tolist()))
        # Setup Wigner-Seitz cells of original and embed meshes
        wsOrig = WignerSeitz(gridOrig.lattice.Rbasis)
        wsEmbed = WignerSeitz(gridEmbed.lattice.Rbasis)
        # Reduce indices of embedded mesh with respect to its Wigner-Seitz cell
        ivEmbed = gridEmbed.get_mesh("R", mine=False).reshape(-1, 3)
        ivEmbed_wsOrig = wsEmbed.reduce_index(ivEmbed, Sembed)
        # Shift original mesh to be centered about the origin
        shifts = torch.round(lattice_center * Sorig).to(torch.int)
        ivEquivOrig = (ivEmbed_wsOrig + shifts) % Sorig

        # Setup mapping between original and embedding meshes
        iEmbed = ivEmbed[:, 2] + Sembed[2] * (ivEmbed[:, 1] + Sembed[1] * ivEmbed[:, 0])
        iEquivOrig = ivEquivOrig[:, 2] + Sorig[2] * (
            ivEquivOrig[:, 1] + Sorig[1] * ivEquivOrig[:, 0]
        )
        # Symmetrize points on boundary using weight function "smoothTheta" function
        invSembed = 1 / torch.tensor(gridEmbed.shape, device=rc.device)
        xWS = (invSembed * ivEmbed_wsOrig) @ gridEmbed.lattice.Rbasis.T
        weights = smoothTheta(wsOrig.ws_boundary_distance(xWS))
        # --- normalize weights:
        weight_sum = torch.zeros(Sorig.prod(), device=rc.device)
        weight_sum.index_add_(0, iEquivOrig, weights)
        weights *= 1.0 / weight_sum[iEquivOrig]

        self.bMap = SparseMatrixRight(torch.stack([iEquivOrig, iEmbed]), weights,
                                      comm=gridOrig.comm)
        self.bMapT = SparseMatrixRight(torch.stack([iEmbed, iEquivOrig]), weights,
                                      comm=gridOrig.comm)


    def embedExpand(self, fieldOrig: FieldR) -> FieldR:
        """Expand real-space field 'fieldOrig' within larger embedding cell."""
        #dataEmbed = (dataOrig.reshape(-1) @ self.bMap).reshape(self.gridEmbed.shape)
        dataOrig = gather(fieldOrig.data, self.gridOrig.split0, self.gridOrig.comm, 0)
        dataEmbed = self.bMap.vecTimesMatrix(dataOrig.reshape(-1))
        dataEmbedMine = scatter(dataEmbed.reshape(self.gridEmbed.shape),
                                     self.gridEmbed.split0, 0)
        return FieldR(self.gridEmbed, data=dataEmbedMine)

    def embedShrink(self, fieldEmbed: FieldR) -> FieldR:
        """Shrink real-space field 'fieldEmbed' to original cell"""
        dataEmbed = gather(fieldEmbed.data, self.gridEmbed.split0, self.gridEmbed.comm, 0)
        # Shrink operation is dagger of embedExpand (bMap is real-valued)
        #dataOrig = (dataEmbed.reshape(-1) @ self.bMap.T).reshape(self.gridOrig.shape)
        dataOrig = self.bMapT.vecTimesMatrix(dataEmbed.reshape(-1))
        dataOrigMine = scatter(dataOrig.reshape(self.gridOrig.shape),
                                self.gridOrig.split0, 0)
        return FieldR(self.gridOrig, data=dataOrigMine)


def smoothTheta(x: torch.Tensor) -> torch.Tensor:
    return torch.where(
        x <= -1, 1.0, torch.where(x >= 1, 0.0, 0.25 * (2.0 - x * (3.0 - x**2)))
    )
