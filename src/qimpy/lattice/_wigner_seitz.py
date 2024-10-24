from __future__ import annotations
from xml.etree import ElementTree as etree

import torch
import numpy as np


class WignerSeitz:
    """Wigner-Seitz cell / Brillouin zone of lattice in real / reciprocal space."""

    v_basis: torch.Tensor  # Input basis vectors in real-space or reciprocal space
    edges: torch.Tensor  # Edges of Wigner-Seitz lattice
    vertices: torch.Tensor  # Vertices of Wigner-Seitz lattice (Cartesian coords)
    faces: torch.Tensor  # Faces of Wigner-Seitz lattice (Cartesian coords)
    i_faces: torch.Tensor  # Faces of Wigner-Seitz lattice (Lattice coords)
    tol: float  # Relative error threshold in detecting geometric properties

    def __init__(self, v_basis: torch.Tensor, tol: float = 1e-6) -> None:
        """Initialize Wigner-Seitz lattice.

        Parameters
        ----------
        v_basis
            Basis vectors (i.e. lattice vectors in real-space or reciprocal space)
        """

        self.v_basis = v_basis
        self.tol = tol
        self.i_faces, self.faces = get_grid(v_basis, tol)  # initial guess
        # Down-select to planes that lie on WS boundary:
        touchWS = torch.where(self.ws_boundary_distance(0.5 * self.faces) < tol)[0]
        self.faces = self.faces[touchWS]
        self.i_faces = self.i_faces[touchWS]

        # Down-select to WS faces with finite area:
        finiteArea = self.ws_plane_count(0.5 * self.faces) == 1
        self.faces = self.faces[finiteArea]  # must be only face with 0 distance
        self.i_faces = self.i_faces[finiteArea]
        n_faces = len(self.faces)
        assert 6 <= n_faces <= 14
        assert n_faces % 2 == 0

        # Compute vertices by 3-way intersection of planes
        i_face_triplets = get_combinations(n_faces, 3)
        Lhs = 0.5 * self.faces[i_face_triplets]  # n_triplets x 3 x 3
        intersect = torch.abs(torch.linalg.det(Lhs)) > tol
        Lhs = Lhs[intersect]
        rhs = (Lhs**2).sum(dim=-1)
        vertices = torch.linalg.solve(Lhs, rhs)

        # Down-select vertices to WS boundary:
        on_boundary = torch.abs(self.ws_boundary_distance(vertices)) < tol
        vertices = vertices[on_boundary]
        vertices = weld_points(vertices, tol)
        self.vertices = vertices
        n_vertices = len(vertices)

        # Find edges of WS boundary:
        i_vertex_pairs = get_combinations(n_vertices, 2)
        edge_midpoints = vertices[i_vertex_pairs].mean(axis=1)
        on_boundary = self.ws_plane_count(edge_midpoints) == 2
        edges = i_vertex_pairs[on_boundary]
        self.edges = edges

    def get_plane_distance(self, r):
        """Signed distance of r from each perpendicular bisector plane from 0->faces."""
        bgrid_mag = torch.linalg.norm(self.faces, axis=-1)
        bgrid_hat = (
            self.faces / bgrid_mag[:, None]
        )  # outward normal of the perpendicular bisector planes
        return r @ bgrid_hat.T - 0.5 * bgrid_mag

    def ws_boundary_distance(self, r):
        """Distance to WS boundary for Cartesian coords in last axis of `r`.
        Result is negative for points inside and positive outside."""
        return torch.max(self.get_plane_distance(r), dim=-1)[0]

    def ws_plane_count(self, r):
        """Number of planes each point in `r` is on."""
        return torch.count_nonzero(
            torch.abs(self.get_plane_distance(r)) < self.tol, axis=-1
        )

    def reduce_vector(self, r, tol=1.0e-8):
        """Find the point within the Wigner-Seitz cell equivalent to x
        (Cartesian coords)."""
        # TODO: vectorize this function and reduce_index function
        changed = True
        while changed:
            changed = False
            for face in self.faces:  # TO-DO: simplify by considering only half-faces
                # equation of plane given by eqn.x==1 (x in Cartesian coords)
                feqn = face / torch.sum(face**2)
                fdotr = torch.dot(feqn, r)
                if torch.abs(fdotr) > 1 + tol:  # not in fundamental zone
                    fimg = (
                        2 * face
                    )  # image of origin through WS face (Cartesian coords)
                    r -= torch.floor(0.5 * (1 + fdotr)) * fimg
                    changed = True
        return r

    def reduce_index(self, iv0, S, tol=1.0e-8):
        """Find the point within the Wigner-Seitz cell equivalent to iv
        (mesh coordinates with sample count S)."""
        changed = True
        iv = torch.clone(iv0)
        i_faces = self.i_faces
        R = self.v_basis
        RTR = R.T @ R
        while changed:
            changed = False
            for i_face in i_faces:  # TO-DO: simplify by considering only half-faces
                # equation of plane given by eqn.x==1 (x in Lattice coords)
                feqn = 2 / (metric_length_squared(RTR, i_face)) * (RTR @ i_face)
                fdotr = torch.einsum("i,vi->v", feqn, iv / S[None, :])
                outside = torch.greater(torch.abs(fdotr), 1 + tol)
                fimg = i_face.to(int)  # image of origin on WS face (Lattice coords)
                iv[outside, :] -= (
                    torch.floor(0.5 * (1 + fdotr)).to(int)[outside, None]
                    * (fimg * S)[None, :]
                )
                changed = torch.sum(outside) > 0
        return iv

    def write_x3d(self, filename: str) -> None:
        """Construct x3d file to visualize the constructed Wigner-Seitz cell"""
        edges = np.array(self.edges)
        vertices = np.array(self.vertices)
        max_lw = np.iinfo(np.int64).max
        xsd_url = "http://www.web3d.org/specifications/x3d-3.2.xsd"
        data = etree.Element(
            "X3D",
            attrib={
                "xsd:noNamespaceSchemaLocation": xsd_url,
                "profile": "Interchange",
                "version": "3.2",
                "xmlns:xsd": "http://www.w3.org/2001/XMLSchema-instance",
            },
        )
        Scene = etree.SubElement(data, "Scene")
        etree.SubElement(Scene, "Background", {"skyColor": "1 1 1"})
        Shape = etree.SubElement(Scene, "Shape", {"DEF": "BZ"})
        Appearance = etree.SubElement(Shape, "Appearance")
        etree.SubElement(
            Appearance, "Material", {"diffuseColor": "0 0 0", "specularColor": "0 0 0"}
        )
        coordIndex = np.array2string(
            np.hstack((edges, np.full((edges.shape[0], 1), -1))).flatten(),
            max_line_width=max_lw,
        )[1:-1]
        Lineset = etree.SubElement(Shape, "IndexedLineSet", {"coordIndex": coordIndex})
        points_str = np.array2string(vertices.flatten(), max_line_width=max_lw)[1:-1]
        etree.SubElement(Lineset, "Coordinate", {"point": points_str})
        with open(filename, "wb") as f:
            f.write(etree.tostring(data, xml_declaration=True, encoding="UTF-8"))


def get_combinations(N: int, p: int) -> torch.Tensor:
    """Return unique p-tuple indices upto N (N_C_p x p array)."""
    i_single = torch.arange(N)
    i_tuples = torch.stack(torch.meshgrid(*([i_single] * p), indexing="ij")).reshape(
        p, -1
    )
    unique = torch.all(torch.less(i_tuples[:-1], i_tuples[1:]), dim=0)
    return i_tuples.T[unique]


def weld_points(points: torch.Tensor, tol: float) -> torch.Tensor:
    """Return unique points based on distance < tol from N x d tensor."""
    distances = torch.linalg.norm(points[:, None] - points[None, :], dim=-1)
    lowest_equivalent_index = torch.argmax(torch.where(distances < tol, 1, 0), dim=0)
    _, inverse, counts = torch.unique(
        lowest_equivalent_index, return_inverse=True, return_counts=True
    )
    # Compute the centroid of each set of equivalent vertices:
    points_uniq = torch.zeros((len(counts), points.shape[-1]), device=points.device)
    points_uniq.index_add_(0, inverse, points)  # sum equivalent coordinates
    points_uniq *= (1.0 / counts)[:, None]  # convert to mean
    return points_uniq


def get_grid(v_basis: torch.Tensor, tol: float) -> tuple[torch.Tensor, ...]:
    """Return grid containing linear combinations of basis vectors sufficient
    to cover Wigner-Seitz cell"""
    # Find maximum radius of sphere within parallelopiped from basis lattice vectors
    combinations = torch.tensor(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        device=v_basis.device,
        dtype=v_basis.dtype,
    )  # TODO: check if these combinations suffice for any lattice angles
    Rmax = (v_basis @ combinations.T).norm(dim=0).max().item()
    # This sphere should be within shape_min in each direction
    # Therefore shape_min >= Rmax * (R^-1 or G^-1)
    shape_min = Rmax * torch.linalg.inv(v_basis).norm(dim=0)
    shape_min = torch.ceil(shape_min)

    # Identify which lattice vectors are perpendicular to others
    overlap = v_basis.T @ v_basis
    overlap.fill_diagonal_(0.0)  # ignore self overlaps
    overlap_max, _ = torch.abs(overlap).max(dim=0)
    is_ortho = torch.less(overlap_max, tol)

    # Create individual grids:
    i_grids_1d = []
    for shape_min_i, is_ortho_i in zip(shape_min.tolist(), is_ortho.tolist()):
        if is_ortho_i:
            shape_min_i = 1  # minimum grid suffices if orthogonal
        i_grids_1d.append(
            torch.arange(-shape_min_i, shape_min_i + 1, device=v_basis.device)
        )

    # Create net grids:
    i_grid = torch.stack(torch.meshgrid(*i_grids_1d, indexing="ij"), dim=-1).flatten(
        0, -2
    )
    i_grid = i_grid[torch.abs(i_grid).sum(dim=-1) > 0.0].to(v_basis.dtype)  # eliminate origin
    return i_grid, i_grid @ v_basis.T


def metric_length_squared(M, v) -> torch.Tensor:
    result = torch.sum(v**2 * torch.diag(M))
    result += 2 * (
        v[0] * v[1] * M[0, 1] + v[0] * v[2] * M[0, 2] + v[1] * v[2] * M[1, 2]
    )
    return result
