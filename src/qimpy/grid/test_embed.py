import matplotlib.pyplot as plt
import numpy as np
import torch

from qimpy import rc, MPI
from qimpy.lattice import Lattice
from qimpy.grid import Grid, FieldR
from qimpy.symmetries import Symmetries
from qimpy.io import log_config
from qimpy.lattice._wigner_seitz import WignerSeitz
from qimpy.grid._embed import CoulombEmbedder


def check_embed(grid: Grid) -> None:
    """Check Coulomb embedding procedure on test system"""
    lattice_center = grid.lattice.center
    embedder = CoulombEmbedder(grid)
    # Create fake data
    r = get_r(grid, torch.eye(3, device=rc.device), space="R")  # In mesh coords
    sigma_r = 0.005
    blob = torch.exp(-torch.sum((r - lattice_center) ** 2, dim=-1) / (2 * sigma_r))
    blob /= np.sqrt(2 * np.pi * sigma_r**2)
    field1 = FieldR(grid, data=blob)
    field2 = embedder.embedExpand(field1)
    field3 = embedder.embedShrink(field2)
    # data1, data2, data3 = extend_grid(blob, grid, periodic, torch.tensor(latticeCenter))
    if rc.is_head:
        fig, axs = plt.subplots(1, 3)
        im = []
        im.append(show(axs[0], field1.data.sum(dim=0), "Original data"))
        im.append(show(axs[1], field2.data.sum(dim=0), "Embedded data"))
        im.append(show(axs[2], field3.data.sum(dim=0), "Embedded->Original data"))
        for _im in im:
            fig.colorbar(_im, orientation="horizontal")
        plt.show()


def extend_grid(
    dataOrig: torch.Tensor,
    gridOrig: Grid,
    periodic: np.ndarray,
    latticeCenter: torch.Tensor,
):
    # Initialize embedding grid
    meshOrig = gridOrig.get_mesh("R", mine=True)
    Sorig = torch.tensor(gridOrig.shape)
    RbasisOrig = gridOrig.lattice.Rbasis
    dimScale = (1, 1, 1) + np.where(periodic, 0, 1)  # extend in non-periodic directions
    latticeEmbed = Lattice(Rbasis=(RbasisOrig @ np.diag(dimScale)))
    gridEmbed = Grid(
        lattice=latticeEmbed,
        symmetries=Symmetries(lattice=latticeEmbed),
        shape=tuple(dimScale * gridOrig.shape),
        comm=rc.comm,
    )
    Sembed = torch.tensor(gridEmbed.shape)
    # Shift center to origin and report embedding center in various coordinate systems:
    latticeCenter = torch.tensor(latticeCenter)
    shifts = torch.round(latticeCenter * torch.tensor(meshOrig.shape[0:3])).to(int)
    rCenter = RbasisOrig @ latticeCenter
    ivCenter = torch.round(
        latticeCenter @ (1.0 * torch.diag(torch.tensor(gridOrig.shape)))
    ).to(int)
    print("Integer grid location selected as the embedding center:")
    print("\tGrid: {:6} {:6} {:6}".format(*tuple(ivCenter)))
    print("\tLattice: {:6.3f} {:6.3f} {:6.3f}".format(*tuple(latticeCenter)))
    print("\tCartesian: {:6.3f} {:6.3f} {:6.3f}".format(*tuple(rCenter)))
    # Setup Wigner-Seitz cells of original and embed meshes
    wsOrig = WignerSeitz(gridOrig.lattice.Rbasis)
    wsEmbed = WignerSeitz(gridEmbed.lattice.Rbasis)
    # Setup mapping between original and embedding meshes
    ivEmbed = gridEmbed.get_mesh("R", mine=True).reshape(-1, 3)
    diagSembedInv = torch.diag(1 / torch.tensor(gridEmbed.shape))
    ivEmbed_wsOrig = wsEmbed.reduce_index(ivEmbed, Sembed)
    ivEquivOrig = (ivEmbed_wsOrig + shifts) % Sorig[None, :]
    iEmbed = ivEmbed[:, 2] + Sembed[2] * (ivEmbed[:, 1] + Sembed[1] * ivEmbed[:, 0])
    iEquivOrig = ivEquivOrig[:, 2] + Sorig[2] * (
        ivEquivOrig[:, 1] + Sorig[1] * ivEquivOrig[:, 0]
    )
    # Symmetrize points on boundary using weight function "smoothTheta"
    xWS = (1.0 * ivEmbed_wsOrig @ diagSembedInv) @ gridEmbed.lattice.Rbasis.T
    weights = smoothTheta(wsOrig.ws_boundary_distance(xWS))
    bMap = torch.sparse_coo_tensor(
        np.array([iEquivOrig, iEmbed]), weights, device=rc.device
    )
    colSums = torch.sparse.sum(bMap, dim=1).to_dense()
    colNorms = torch.sparse.spdiags(
        1.0 / colSums, offsets=torch.tensor([0]), shape=(Sorig.prod(), Sorig.prod())
    )
    bMap = torch.sparse.mm(colNorms, bMap)
    dataEmbed = (dataOrig.reshape(-1) @ bMap).reshape(gridEmbed.shape)
    dataOrig2 = (dataEmbed.reshape(-1) @ bMap.T).reshape(gridOrig.shape)
    return dataOrig, dataEmbed, dataOrig2


def smoothTheta(x) -> torch.Tensor:
    return torch.where(
        x <= -1, 1.0, torch.where(x >= 1, 0.0, 0.25 * (2.0 - x * (3.0 - x**2)))
    )


def get_r(grid, R, space="R"):
    mesh = grid.get_mesh(space, mine=True)
    diagSinv = torch.diag(1 / torch.tensor(mesh.shape[0:3], device=rc.device))
    M = mesh.to(torch.double)
    return M @ diagSinv @ R.T


def show(ax, data, title=None):
    # ax.plot(center[1] * data.shape[0], center[2] * data.shape[1], 'rx')
    if title:
        ax.set_title(title)
    return ax.imshow(data.to(rc.cpu).numpy(), origin="lower")


def main():
    log_config()
    rc.init()

    shape = (24, 24, 126)
    lattice = Lattice(
        system=dict(name="tetragonal", a=3.3, c=15.1),
        periodic=(True, True, False),
        center=(0.75, 0.75, 0.75),
    )
    grid = Grid(
        lattice=lattice,
        symmetries=Symmetries(lattice=lattice, override="identity"),
        shape=shape,
        comm=rc.comm,
    )
    check_embed(grid)


if __name__ == "__main__":
    main()
