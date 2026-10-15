from __future__ import annotations
from typing import TypeAlias

import numpy as np

from qimpy import TreeNode
from qimpy.io import Checkpoint, CheckpointPath, CheckpointContext
from qimpy.transport import geometry

__all__ = ("File", "Line", "Rect", "Options")


class File(TreeNode):
    def __init__(
        self,
        *,
        name: str,
        mesh: geometry.Mesh,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        """Load mesh from file.

        Parameters
        ----------
        name
            :yaml:`Filename of HDF5 file to load mesh from.`
        """
        super().__init__()
        self.name = name
        if checkpoint_in:
            return  # data will be loaded from checkpoint instead
        with Checkpoint(name) as mesh_file:
            mesh.load(CheckpointPath(mesh_file, ""))

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        cp_path.attrs["name"] = self.name
        return list(cp_path.attrs.keys())


class Line(TreeNode):
    def __init__(
        self,
        *,
        L: float,
        N: int,
        mesh: geometry.Mesh,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        """Make a 1D mesh with 'source' and 'drain' contacts at the ends.

        Parameters
        ----------
        L
            :yaml:`Length of the 1D domain.`
        N
            :yaml:`Number of intervals to divide the domain into.`
        """
        super().__init__()
        self.L = L
        self.N = N
        if checkpoint_in:
            return  # data will be loaded from checkpoint instead
        mesh.vertices = np.column_stack((np.linspace(0, L, N + 1), np.zeros(N + 1)))
        mesh.cells = np.column_stack(
            (np.arange(N), np.arange(1, N + 1), np.full(N, -1))
        )
        mesh.boundary_edges = np.array([[0, 0], [N, 1]])
        mesh.region_names = []
        mesh.boundary_names = ["source", "drain"]
        mesh.lattice_vectors = None

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        cp_path.attrs["L"] = self.L
        cp_path.attrs["N"] = self.N
        return list(cp_path.attrs.keys())


class Rect(TreeNode):
    def __init__(
        self,
        *,
        Lx: float,
        Ly: float,
        Nx: int,
        Ny: int,
        contacts: dict[str, tuple[float, ...]],
        regions: dict[str, tuple[float, ...]] | None = None,
        lattice_vectors: list[list[float]] | None = None,
        mesh: geometry.Mesh,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        """Make a 2D rectangular domain with a body-centered rectangular mesh.
        The overall rectangle is divided into rectanglular tiles in each direction,
        and each tile is divided into four triangular cells using its center.

        Parameters
        ----------
        Lx
            :yaml:`Length of the first dimensinon of rectangular domain.`
        Ly
            :yaml:`Length of the second dimensinon of rectangular domain.`
        Nx
            :yaml:`Number of intervals to divide the first dimension into.`
        Ny
            :yaml:`Number of intervals to divide the second dimension into.`
        contacts
            :yaml:`Named regions on the rectangle boundary to use as contacts.`
            Each name is associated with a circle (x0, y0, r) or axis-aligned
            ellipse (x0, y0, rx, ry): any edges on the boundary with center
            within this circle or ellipse is associated to that name.
        regions
            :yaml:`Named regions in the interior to use as probes.`
            Each name is associated with a circle or ellipse as in `contacts`.
        lattice
            :yaml:`Lattice vectors for periodic boundary conditions.`
            If specified, it should be a 2 x 2 matrix with each row corresponding
            to a lattice vector. If desirable to only have one periodic direction,
            just make the second lattice vector longer than the geometry.
        """
        super().__init__()
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        if checkpoint_in:
            return  # data will be loaded from checkpoint instead

        # Construct vertices for body-centered rectangular mesh:
        ix = np.arange(Nx + 1)
        iy = np.arange(Ny + 1)
        i_mesh = np.stack(np.meshgrid(ix, iy, indexing="ij"))  # 2 x (Nx+1) x (Ny+1)
        h = np.array([Lx / Nx, Ly / Ny])  # grid spacings
        corners = i_mesh.reshape(2, -1).T * h
        centers = (i_mesh[:, :-1, :-1] + 0.5).reshape(2, -1).T * h
        mesh.vertices = np.vstack((corners, centers))

        # Construct corresponding cells
        i_corners = i_mesh[0] * (Ny + 1) + i_mesh[1]  # vertex indices for corners
        i_centers = len(corners) + i_mesh[0, :-1, :-1] * Ny + i_mesh[1, :-1, :-1]
        i00 = i_corners[:-1, :-1]
        i01 = i_corners[:-1, 1:]
        i10 = i_corners[1:, :-1]
        i11 = i_corners[1:, 1:]
        null = np.full_like(i11, -1)
        mesh.cells = np.stack(
            (
                np.stack((i00, i10, i_centers, null), axis=-1),
                np.stack((i10, i11, i_centers, null), axis=-1),
                np.stack((i11, i01, i_centers, null), axis=-1),
                np.stack((i01, i00, i_centers, null), axis=-1),
            ),
            axis=1,
        ).reshape(-1, 4)

        # Label interior regions if available:
        mesh.region_names = []
        if regions is not None:
            cell_centers = mesh.vertices[mesh.cells[:, :-1]].mean(axis=1)
            for i_region, (region_name, ellipse) in enumerate(regions.items()):
                mesh.region_names.append(region_name)
                mesh.cells[within_ellipse(cell_centers, ellipse), -1] = i_region

        # Add boundaries and label contacts:
        null_x = np.full(Nx, -1)
        null_y = np.full(Ny, -1)
        mesh.boundary_edges = np.concatenate(
            (
                np.stack((i_corners[:-1, 0], i_corners[1:, 0], null_x), axis=-1),
                np.stack((i_corners[:-1, -1], i_corners[1:, -1], null_x), axis=-1),
                np.stack((i_corners[0, :-1], i_corners[0, 1:], null_y), axis=-1),
                np.stack((i_corners[-1, :-1], i_corners[-1, 1:], null_y), axis=-1),
            ),
            axis=0,
        )
        edge_centers = mesh.vertices[mesh.boundary_edges[:, :-1]].mean(axis=1)
        mesh.boundary_names = []
        for i_contact, (contact_name, ellipse) in enumerate(contacts.items()):
            mesh.boundary_names.append(contact_name)
            mesh.boundary_edges[within_ellipse(edge_centers, ellipse), -1] = i_contact

        mesh.lattice_vectors = (
            None if lattice_vectors is None else np.array(lattice_vectors)
        )

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        cp_path.attrs["Lx"] = self.Lx
        cp_path.attrs["Ly"] = self.Ly
        cp_path.attrs["Nx"] = self.Nx
        cp_path.attrs["Ny"] = self.Ny
        return list(cp_path.attrs.keys())


def within_ellipse(points: np.ndarray, ellipse: tuple[float, ...]) -> np.ndarray:
    center = ellipse[:2]
    r = ellipse[2:]  # 1 or 2 values, broadcasts to get circle or ellipse
    return np.linalg.norm((points - center) / r, axis=1) <= 1.0


Options: TypeAlias = File | Line | Rect
