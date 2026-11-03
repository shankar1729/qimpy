from __future__ import annotations
from typing import TypeAlias, Any

import numpy as np
import triangle
import torch
import torch.distributed as dist

from qimpy import io, rc, TreeNode
from qimpy.io import (
    Checkpoint,
    CheckpointPath,
    CheckpointContext,
    InvalidInputException,
    cast_tensor,
)
from qimpy.transport import geometry

__all__ = ("File", "Line", "Rect", "Triangulate", "Options")


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
        mesh.boundary_faces = np.array([[0, 0], [N, 1]])
        mesh.region_names = []
        mesh.boundary_names = ["source", "drain"]

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
        contacts: dict[str, tuple[float, ...]] | None = None,
        regions: dict[str, tuple[float, ...]] | None = None,
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
            ellipse (x0, y0, rx, ry): any faces on the boundary with center
            within this circle or ellipse is associated to that name.
        regions
            :yaml:`Named regions in the interior to use as probes.`
            Each name is associated with a circle or ellipse as in `contacts`.
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
        mesh.boundary_faces = np.concatenate(
            (
                np.stack((i_corners[:-1, 0], i_corners[1:, 0], null_x), axis=-1),
                np.stack((i_corners[:-1, -1], i_corners[1:, -1], null_x), axis=-1),
                np.stack((i_corners[0, :-1], i_corners[0, 1:], null_y), axis=-1),
                np.stack((i_corners[-1, :-1], i_corners[-1, 1:], null_y), axis=-1),
            ),
            axis=0,
        )
        face_centers = mesh.vertices[mesh.boundary_faces[:, :-1]].mean(axis=1)
        mesh.boundary_names = []
        assert contacts is not None
        for i_contact, (contact_name, ellipse) in enumerate(contacts.items()):
            mesh.boundary_names.append(contact_name)
            mesh.boundary_faces[within_ellipse(face_centers, ellipse), -1] = i_contact

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        cp_path.attrs["Lx"] = self.Lx
        cp_path.attrs["Ly"] = self.Ly
        cp_path.attrs["Nx"] = self.Nx
        cp_path.attrs["Ny"] = self.Ny
        return list(cp_path.attrs.keys())


class Triangulate(TreeNode):
    def __init__(
        self,
        *,
        vertices: list[tuple[float, float]] | np.ndarray,
        segments: list[tuple[int, int]] | np.ndarray,
        max_area: float,
        min_angle: float = 30.0,
        holes: list[tuple[float, float]] | np.ndarray | None = None,
        regions: dict[str, tuple[float, float]] | None = None,
        contacts: dict[str, tuple[float, ...]] | None = None,
        mesh: geometry.Mesh,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        """Generate a triangulation using the triangle python library.
        Specify the geometry in terms of vertices, segments that define polygons,
        optionally lists of points which make some of these polygons into holes,
        or define them with distinct region IDs. See the examples in the documentation
        of the triangle library to understand how these options work.

        Parameters
        ----------
        vertices
            :yaml:`Vertex coordinates that defines the polygonal geometry.`
        segments
            :yaml:`Pairs of vertex indices for each polygon segment.`
        max_area
            :yaml:`Maximum area of triangle in mesh.`
        min_angle
            :yaml:`Minimum angle (degrees) of triangle in mesh.`
        holes
            :yaml:`Optional coordinates of points defining holes.`
            If any point listed here is within a polygon defined by `segments`,
            than that polygon becomes a hole in the geometry.
        regions
            :yaml:`Optional named regions in the geometry.`
            Each name is associated with the coordinates of a point: the polygon
            containing that point then becomes a named region for probe output.
        contacts
            :yaml:`Named regions on the boundary to use as contacts.`
            Each name is associated with a circle (x0, y0, r) or axis-aligned
            ellipse (x0, y0, rx, ry): any faces on the boundary with center
            within this circle or ellipse is associated to that name.
        """
        super().__init__()
        self.vertices = np.array(vertices)
        self.segments = np.array(segments)
        self.max_area = float(max_area)
        self.min_angle = float(min_angle)
        if checkpoint_in:
            return  # data will be loaded from checkpoint instead

        # Prepare inputs for triangle library:
        triangle_input = dict[str, Any](vertices=self.vertices, segments=self.segments)
        triangle_command = f"pq{self.min_angle}a{self.max_area}e"
        self.holes: np.ndarray | None = None
        if holes is not None:
            triangle_input["holes"] = np.array(holes)
        mesh.region_names = list[str]()
        if regions is not None:
            region_list = []
            for i_region, (region_name, (x, y)) in enumerate(regions.items()):
                mesh.region_names.append(region_name)
                region_list.append([x, y, i_region, 0.0])
            triangle_input["regions"] = region_list
            triangle_command += "A"

        # Triangulate on head and broadcast results (mesh must be consistent):
        object_list = [
            (
                triangle.triangulate(triangle_input, triangle_command)
                if rc.is_head
                else None
            )
        ]
        if rc.n_procs > 1:
            dist.broadcast_object_list(object_list, src=0)
        results = object_list[0]

        # Extract required properties for mesh:
        mesh.vertices = results["vertices"]
        triangles = results["triangles"]
        if (cell_regions := results.get("triangle_attributes")) is None:
            cell_regions = np.full((len(triangles), 1), -1)
        mesh.cells = np.hstack((triangles, cell_regions)).astype(int)

        # Assign contacts:
        edges = results["edges"].astype(int)
        edge_markers = results["edge_markers"].astype(int)[:, 0]
        edges = edges[edge_markers == 1]  # keep only boundary edges
        mesh.boundary_faces = np.hstack((edges, np.full((len(edges), 1), -1)))
        mesh.boundary_names = []
        face_centers = mesh.vertices[edges].mean(axis=1)
        assert contacts is not None
        for i_contact, (contact_name, ellipse) in enumerate(contacts.items()):
            mesh.boundary_names.append(contact_name)
            mesh.boundary_faces[within_ellipse(face_centers, ellipse), -1] = i_contact

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        cp_path.attrs["vertices"] = self.vertices
        cp_path.attrs["segments"] = self.segments
        cp_path.attrs["max_area"] = self.max_area
        cp_path.attrs["min_angle"] = self.min_angle
        return list(cp_path.attrs.keys())


def within_ellipse(points: np.ndarray, ellipse: tuple[float, ...]) -> np.ndarray:
    center = ellipse[:2]
    r = ellipse[2:]  # 1 or 2 values, broadcasts to get circle or ellipse
    return np.linalg.norm((points - center) / r, axis=1) <= 1.0


class ParameterGrid(TreeNode):
    def __init__(
        self,
        *,
        N: int,
        mesh: geometry.Mesh,
        checkpoint_in: CheckpointPath = CheckpointPath(),
        **kwargs,
    ) -> None:
        """Make a parameter grid for independent dynamics (no spatial transport).

        Parameters
        ----------
        N
            :yaml:`Number of intervals to divide the domain into.`
        **kwargs
            :yaml:`Names of parameters to sweep and their corresponding values.`
            Each entry is a mapping from `sweep` or `loop` to a list, where sweep
            creates a linear variation between endpoints, while loop goes over
            a list of exactly `N` provided values. The name of the parameter must
            be recognized by the chosen material.
        """
        super().__init__()
        self.N = N
        if checkpoint_in:
            return  # data will be loaded from checkpoint instead
        mesh.vertices = np.column_stack(
            (np.tile(np.arange(N), 2), np.repeat([0.0, 1.0], N))
        )
        mesh.cells = np.column_stack(
            (np.arange(N), np.arange(N, 2 * N), np.full(N, -1))
        )
        mesh.boundary_faces = np.zeros((0, 2), dtype=int)
        mesh.region_names = []
        mesh.boundary_names = []

        # Initialize mesh fields:
        for key, values in io.dict.key_cleanup(kwargs).items():
            mesh.fields[key] = self.create_values(**values)

    def create_values(
        self, *, loop: list | None = None, sweep: list | None = None
    ) -> torch.Tensor:
        if (loop is None) == (sweep is None):
            raise InvalidInputException(
                "Exactly one of loop or sweep should be specified."
            )

        # Explicitly specified list of praamter values:
        if loop is not None:
            values = cast_tensor(loop)
            if len(values) != self.N:
                raise InvalidInputException(
                    f"Number of entries in loop = {len(values)} should match"
                    f" length {self.N} of parameter grid"
                )
            return values

        # Uniform sweep between endpoints:
        assert sweep is not None
        limits = cast_tensor(sweep)
        if len(limits) != 2:
            raise InvalidInputException(
                f"Number of entries in sweep = {len(limits)} must be 2"
            )
        t = torch.linspace(0, 1, self.N, device=rc.device)
        return limits[0] + torch.einsum("t,...->t...", t, limits[1] - limits[0])

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        cp_path.attrs["N"] = self.N
        return list(cp_path.attrs.keys())


Options: TypeAlias = File | Line | Rect | Triangulate | ParameterGrid
