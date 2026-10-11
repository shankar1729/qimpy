from __future__ import annotations
from dataclasses import dataclass

from scipy.spatial import KDTree
import numpy as np
import torch

from qimpy import rc, log, TreeNode
from qimpy.io import CheckpointPath, CheckpointContext
from . import mesh_source


class Mesh(TreeNode):
    """Definition of mesh geometry (dimension, d = 2 or 1)"""

    source: mesh_source.Options  #: Method by which mesh is constructed
    vertices: np.ndarray  #: vertex coordinates (Nv, 2)
    cells: np.ndarray  #: vertex indices and region ID in cells (Nc, d+2)
    region_names: list[str]  #: names of interior regions (for cell IDs >= 0)
    boundary_edges: np.ndarray  #: sorted vertex indices, ID of boundary edges (Ne, d+1)
    boundary_names: list[str]  #: names of boundary regions (for edge IDs >= 0)
    lattice_vectors: np.ndarray | None  #: lattice vectors in rows if periodic

    def __init__(
        self,
        *,
        file: mesh_source.File | dict | None = None,
        line: mesh_source.Line | dict | None = None,
        rect: mesh_source.Rect | dict | None = None,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        """
        Initialize mesh geometry.

        Parameters
        ----------
        file
            :yaml:`Load mesh from HDF5 file.`
            Only one source for the mesh must be specified.
        line
            :yaml:`Construct uniform mesh for a 1D domain.`
            Only one source for the mesh must be specified.
        rect
            :yaml:`Construct triangular mesh for a 2D rectangular domain.`
            Only one source for the mesh must be specified.
        """
        super().__init__()
        self.add_child_one_of(
            "source",
            checkpoint_in,
            TreeNode.ChildOptions("file", mesh_source.File, file, mesh=self),
            TreeNode.ChildOptions("line", mesh_source.Line, line, mesh=self),
            TreeNode.ChildOptions("file", mesh_source.Rect, rect, mesh=self),
            have_default=False,
        )
        if checkpoint_in:
            self.load(checkpoint_in)

    def load(self, cp: CheckpointPath) -> None:
        self.vertices = cp.read_np("vertices")
        self.cells = cp.read_np("cells")
        self.region_names = cp.read_str_list("region_names")
        self.boundary_edges = cp.read_np("boundary_edges")
        self.boundary_names = cp.read_str_list("boundary_names")
        self.lattice_vectors = cp.read_optional_np("lattice_vectors")

    def save(self, cp: CheckpointPath) -> list[str]:
        """Save within h5 file and return names of saved variables."""
        saved_list = [
            cp.write("vertices", self.vertices),
            cp.write("cells", self.cells),
            cp.write_str("region_names", ",".join(self.region_names)),
            cp.write("boundary_edges", self.boundary_edges),
            cp.write_str("boundary_names", ",".join(self.boundary_names)),
        ]
        if self.lattice_vectors is not None:
            saved_list.append(cp.write("lattice_vectors", self.lattice_vectors))
        return saved_list

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        return self.save(cp_path)

    @property
    def is1D(self) -> bool:
        return self.cells.shape[1] == 3  # cells coontain d + 1 vertices, 1 region index

    def map_edges(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find interior and boundary edges, and map them to the cells.

        Returns
        -------
        edges
            Nc x (d + 1) x d indices of the 2 vertices per edge in 2D and 1 in 1D,
            for each of the 3 edges per cell in 2D and 2 edges per cell in 1D.
        interior
            Ni x 2 x 2 indices: cell number and edge number within cell (last dimension)
            for each of two cells (middle dimension) that each interior edge belongs to.
        boundary
            Nb x 3 indices: cell number, edge number within cell for the single cell
            that each boundary edge belongs to, along with the boundary ID of the edge.
        """
        if self.is1D:
            edges = self.cells[:, :-1, None]
            edge_ids = edges[..., 0]  # edge defined by single vertex in 1D
            boundary_edge_ids = self.boundary_edges[..., 0]
        else:
            edges = self.cells[:, np.array([(0, 1), (1, 2), (2, 0)])]

            if self.lattice_vectors is None:
                edges_weld = edges
                boundary_edges_weld = self.boundary_edges[:, :2]
                n_vertices_weld = len(self.vertices)
            else:
                # Identify vertices separated by lattice vectors:
                frac_coords = self.vertices @ np.linalg.inv(self.lattice_vectors)
                frac_coords -= np.floor(frac_coords)  # wrap to [0, 1)
                tree = KDTree(frac_coords, boxsize=(1.0, 1.0))  # periodic lookup
                MAX_EQUIVALENT = 4
                _, i_equivalents = tree.query(
                    frac_coords, k=MAX_EQUIVALENT, distance_upper_bound=1.0e-5
                )
                lowest_equivalent_index = np.minimum(
                    i_equivalents.min(axis=1), np.arange(len(self.vertices))
                )
                _, vertex_map = np.unique(lowest_equivalent_index, return_inverse=True)
                edges_weld = vertex_map[edges]
                boundary_edges_weld = vertex_map[self.boundary_edges[:, :2]]
                n_vertices_weld = vertex_map.max() + 1
                if n_vertices_weld == len(self.vertices):
                    log.info("Lattice vectors have no effect: no equivalent vertices.")

            # Construct edge ID that is equal for edge with different vertex orders:
            stride = np.array((n_vertices_weld, 1))
            edge_ids = np.sort(edges_weld, axis=-1) @ stride
            boundary_edge_ids = np.sort(boundary_edges_weld, axis=-1) @ stride

        # Find unique edge IDs and counts:
        unique_ids, unique_indices, unique_inverse, counts = np.unique(
            edge_ids, return_index=True, return_inverse=True, return_counts=True
        )
        assert counts.max() <= 2
        unique_rank = np.ones_like(unique_inverse)  # rank within equivalence class
        unique_rank[unique_indices] = 0  # and others are 1 because count <= 2
        edge_map = np.zeros_like(unique_inverse, shape=(len(unique_ids), 2, 2))
        edge_map[unique_inverse, unique_rank] = np.stack(
            np.meshgrid(
                np.arange(edges.shape[0]), np.arange(edges.shape[1]), indexing="ij"
            ),
            axis=-1,
        ).reshape(
            -1, 2
        )  # cell and edge within cell, for each copy of each unique edge

        # Split edge map into interior and boundary
        interior = edge_map[np.where(counts == 2)[0]]
        sel_boundary = np.where(counts == 1)[0]

        # Find boundary region index for each boundary edge:
        i_unique_boundary = np.searchsorted(unique_ids, boundary_edge_ids)
        boundary_region_ids = np.full(len(unique_ids), -1)
        boundary_region_ids[i_unique_boundary] = self.boundary_edges[:, -1]
        boundary = np.hstack(
            (edge_map[sel_boundary, 0], boundary_region_ids[sel_boundary, None])
        )
        return edges, interior, boundary


@dataclass
class FVGeom:
    """Static FV geometry from a triangle (2D) or line (1D) mesh; hot-path arrays
    are torch tensors.  ``area`` is the cell measure (triangle area / interval
    length) and ``elen``/``blen`` the face measure (edge length / 1 for a 1D
    point face).

    Faces split into interior (shared by cells ``eL``/``eR``, normal ``en`` points
    out of ``eL``) and boundary (cell ``bcell``, outward normal ``bn``, ``bmark``
    names the wall/contact). ``eLF``/``eRF``/``bF`` are flat ``cell*n_face +
    localface`` indices (``n_face`` = 3 triangles, 2 line cells) used to gather
    the reconstructed face value of the adjacent cell.
    Periodic faces are paired through the lattice and stored as interior edges
    (the streaming neighbour is the periodic image).
    """

    area: torch.Tensor
    inradius: torch.Tensor  # (K,)
    centroid_np: np.ndarray
    eL: torch.Tensor
    eR: torch.Tensor
    eLF: torch.Tensor
    eRF: torch.Tensor  # (Ne,)
    en: torch.Tensor
    elen: torch.Tensor  # (Ne,2),(Ne,)
    bcell: torch.Tensor
    bF: torch.Tensor
    bmark: torch.Tensor  # (Nb,)
    bn: torch.Tensor
    blen: torch.Tensor  # (Nb,2),(Nb,)
    nbr: torch.Tensor  # (K, Nmax) vertex-neighbor cells (self-padded)
    recon: torch.Tensor  # (K, 3, Nmax) face-increment op: d_face = recon @ (u_nbr - u)


def build_fv_geom(mesh: Mesh, *, dtype: torch.dtype = torch.float64) -> FVGeom:
    """Build the FV geometry from a loaded mesh (``_mesh.MeshResult``).

    Dispatches on cell type: 3 vertices/cell -> 2D triangles, 2 vertices/cell ->
    a 1D line mesh (interval cells; see :func:`_build_fv_geom_1d`).
    """
    tri = mesh.cells[:, :-1]
    if tri.shape[1] == 2:
        return _build_fv_geom_1d(mesh, dtype=dtype)
    V = mesh.vertices
    K = len(tri)
    p = V[tri]  # (K, 3, 2)
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    area = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    if np.any(area <= 0.0):
        raise ValueError("triangle mesh must be CCW with positive area")
    centroid = p.mean(axis=1)

    fa, fb = np.array([0, 1, 2]), np.array([1, 2, 0])
    Pa, Pb = p[:, fa], p[:, fb]
    fmid = 0.5 * (Pa + Pb)
    tvec = Pb - Pa
    flen = np.linalg.norm(tvec, axis=2)
    fnrm = np.stack([tvec[..., 1], -tvec[..., 0]], axis=2) / flen[..., None]  # outward
    inradius = area / (0.5 * flen.sum(axis=1))

    # Deduplicate physical edges -> interior (two cells) / boundary (one).
    _, interior, boundary = mesh.map_edges()
    (kL, fL), (kR, fR) = np.transpose(interior, (1, 2, 0))
    bk, bf, bmark = boundary.T

    # Reconstruction stencil: vertex-neighbors (every cell sharing a vertex), which
    # stays full-rank and well-conditioned on distorted/irregular/boundary cells
    # where the 3 face-neighbors alone are too few or near-collinear.
    v2c: dict[int, list[int]] = {}
    for k in range(K):
        for vtx in tri[k]:
            v2c.setdefault(int(vtx), []).append(k)
    vnbr = [
        sorted({c for vtx in tri[k] for c in v2c[int(vtx)]} - {k}) for k in range(K)
    ]
    Nmax = max((len(s) for s in vnbr), default=1)
    nbr = np.arange(K)[:, None].repeat(Nmax, axis=1)  # pad slots with self
    # Inverse-distance-weighted least-squares gradient operator per cell:
    #   grad_i = (D^T W D)^{-1} D^T W (u_nbr - u_i),  w_j = 1 / |c_j - c_i|^2.
    # pinv handles any residual rank-deficiency gracefully (min-norm gradient).
    grad_op = np.zeros((K, 2, Nmax))
    for i, js in enumerate(vnbr):
        if len(js) < 2:
            continue
        nbr[i, : len(js)] = js
        D = centroid[js] - centroid[i]  # (n, 2)
        w = 1.0 / np.maximum((D**2).sum(1), 1e-300)  # inverse-distance^2
        sw = np.sqrt(w)
        grad_op[i, :, : len(js)] = np.linalg.pinv(sw[:, None] * D) * sw[None, :]
    # Fuse gradient + centroid->face offsets so a step reconstructs face
    # increments with one (3 x Nmax) @ (Nmax x Nk) matmul per cell.
    face_off = fmid - centroid[:, None]  # (K, 3, 2)
    recon = np.einsum("kfx,kxg->kfg", face_off, grad_op)  # (K, 3, Nmax)

    def t(a, long=False):
        return torch.tensor(
            np.ascontiguousarray(a),
            device=rc.device,
            dtype=torch.long if long else dtype,
        )

    area_t = t(area)
    return FVGeom(
        area=area_t,
        inradius=t(inradius),
        centroid_np=centroid,
        eL=t(kL, long=True),
        eR=t(kR, long=True),
        eLF=t(kL * 3 + fL, long=True),
        eRF=t(kR * 3 + fR, long=True),
        en=t(fnrm[kL, fL]),
        elen=t(flen[kL, fL]),
        bcell=t(bk, long=True),
        bF=t(bk * 3 + bf, long=True),
        bmark=t(bmark, long=True),
        bn=t(fnrm[bk, bf]),
        blen=t(flen[bk, bf]),
        nbr=t(nbr, long=True),
        recon=t(recon),
    )


def _build_fv_geom_1d(mesh: Mesh, *, dtype: torch.dtype = torch.float64) -> FVGeom:
    """Build the FV geometry for a 1D line mesh: interval cells on a line.

    Each cell is an interval with two endpoint "faces" (local face 0 = left
    vertex, 1 = right vertex).  The cell measure is its length L (the FV update
    divides by it, so ``area``:=L), the outward face normals are +/- x (unit), and
    point faces have unit measure (``elen``/``blen``:=1, the 1D divergence
    theorem).  The reconstruction reuses the same inverse-distance least-squares
    gradient as 2D; on a line the centroid offsets are purely x, so ``pinv``
    returns the x-gradient and a zero y-gradient (min-norm).  Adjacency is by
    shared vertex: a vertex in two cells is an interior face, in one a boundary
    face (the domain ends), whose marker is looked up as ``(v, v)``.  The material
    is untouched -- velocities stay 2D; only ``v_x = v.n`` streams along the line.
    """
    V = mesh.vertices
    seg = mesh.cells[:, :-1]
    K = len(seg)
    p = V[seg]  # (K, 2, 2): endpoints
    centroid = p.mean(axis=1)  # (K, 2)
    L = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)  # (K,) cell length
    if np.any(L <= 0.0):
        raise ValueError("1D line mesh has a zero-length cell")
    area = L  # FV cell measure
    inradius = L  # dt = cfl * L / vmax
    fmid = p  # face = the endpoint vertex
    face_off = fmid - centroid[:, None]  # (K, 2, 2) centroid->face
    fnrm = face_off / np.linalg.norm(
        face_off, axis=2, keepdims=True
    )  # +/- x unit normal
    flen = np.ones((K, 2))  # point face: unit measure

    # Interior / boundary by shared-vertex dedup.
    _, interior, boundary = mesh.map_edges()
    (kL, fL), (kR, fR) = np.transpose(interior, (1, 2, 0))
    bk, bf, bmark = boundary.T

    # Inverse-distance least-squares gradient over shared-vertex neighbors.
    v2c: dict[int, list[int]] = {}
    for k in range(K):
        for vtx in seg[k]:
            v2c.setdefault(int(vtx), []).append(k)
    vnbr = [
        sorted({c for vtx in seg[k] for c in v2c[int(vtx)]} - {k}) for k in range(K)
    ]
    Nmax = max((len(s) for s in vnbr), default=1)
    nbr = np.arange(K)[:, None].repeat(Nmax, axis=1)
    grad_op = np.zeros((K, 2, Nmax))
    for i, js in enumerate(vnbr):
        if not js:
            continue
        nbr[i, : len(js)] = js
        D = centroid[js] - centroid[i]  # (n, 2), y ~ 0
        w = 1.0 / np.maximum((D**2).sum(1), 1e-300)
        sw = np.sqrt(w)
        grad_op[i, :, : len(js)] = np.linalg.pinv(sw[:, None] * D) * sw[None, :]
    recon = np.einsum("kfx,kxg->kfg", face_off, grad_op)  # (K, 2, Nmax)

    def t(a, long=False):
        return torch.tensor(
            np.ascontiguousarray(a),
            device=rc.device,
            dtype=torch.long if long else dtype,
        )

    area_t = t(area)
    return FVGeom(
        area=area_t,
        inradius=t(inradius),
        centroid_np=centroid,
        eL=t(kL, long=True),
        eR=t(kR, long=True),
        eLF=t(kL * 2 + fL, long=True),
        eRF=t(kR * 2 + fR, long=True),
        en=t(fnrm[kL, fL]),
        elen=t(flen[kL, fL]),
        bcell=t(bk, long=True),
        bF=t(bk * 2 + bf, long=True),
        bmark=t(bmark, long=True),
        bn=t(fnrm[bk, bf]),
        blen=t(flen[bk, bf]),
        nbr=t(nbr, long=True),
        recon=t(recon),
    )
