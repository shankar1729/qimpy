from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch

from qimpy import rc, TreeNode
from qimpy.io import Checkpoint, CheckpointPath, CheckpointContext


class Mesh(TreeNode):
    """Definition of mesh geometry (2D or 1D)"""

    source: File | Line | Rect  #: Method by which mesh is constructed
    vertices: np.ndarray  #: vertex coordinates (Nv, 2)
    cells: np.ndarray  #: vertex indices and region ID in cells: (Nc, 3 if 1D else 4)
    edges: np.ndarray  #: sorted vertex indices and ID of boundary edges: (Ne, 3)
    region_names: list[str]  #: names of interior regions (for cell IDs >= 0)
    boundary_names: list[str]  #: names of boundary regions (for edge IDs >= 0)
    lattice_vectors: np.ndarray | None  #: lattice vectors in rows if periodic

    def __init__(
        self,
        *,
        file: File | dict | None = None,
        line: Line | dict | None = None,
        rect: Rect | dict | None = None,
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
            TreeNode.ChildOptions("file", Mesh.File, file, mesh=self),
            TreeNode.ChildOptions("line", Mesh.Line, line, mesh=self),
            TreeNode.ChildOptions("file", Mesh.Rect, rect, mesh=self),
            have_default=False,
        )
        if checkpoint_in:
            self.load(checkpoint_in)

    def load(self, cp: CheckpointPath) -> None:
        self.vertices = cp.read_np("vertices")
        self.cells = cp.read_np("cells")
        self.edges = cp.read_np("edges")
        self.region_names = cp.read_str_list("region_names")
        self.boundary_names = cp.read_str_list("boundary_names")
        self.lattice_vectors = cp.read_optional_np("lattice_vectors")

    def save(self, cp: CheckpointPath) -> list[str]:
        """Save within h5 file and return names of saved variables."""
        saved_list = [
            cp.write("vertices", self.vertices),
            cp.write("cells", self.cells),
            cp.write("edges", self.edges),
            cp.write_str("region_names", ",".join(self.region_names)),
            cp.write_str("boundary_names", ",".join(self.boundary_names)),
        ]
        if self.lattice_vectors is not None:
            saved_list.append(cp.write("lattice_vectors", self.lattice_vectors))
        return saved_list

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        return self.save(cp_path)

    class File(TreeNode):
        def __init__(
            self,
            *,
            name: str,
            mesh: Mesh,
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
            mesh: Mesh,
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
            mesh.edges = np.array([[0, 0, 0], [N, N, 1]])
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
            contacts: dict[str, tuple[float, float, float]],
            mesh: Mesh,
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
                Each name is associated with a circle (x0, y0, r): any edges on the
                boundary with center within this circle is associated to that name.
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
            mesh.region_names = []

            # Add boundaries and label contacts:
            null_x = np.full(Nx, -1)
            null_y = np.full(Ny, -1)
            mesh.edges = np.concatenate(
                (
                    np.stack((i_corners[:-1, 0], i_corners[1:, 0], null_x), axis=-1),
                    np.stack((i_corners[:-1, -1], i_corners[1:, -1], null_x), axis=-1),
                    np.stack((i_corners[0, :-1], i_corners[0, 1:], null_y), axis=-1),
                    np.stack((i_corners[-1, :-1], i_corners[-1, 1:], null_y), axis=-1),
                ),
                axis=0,
            )
            edge_centers = mesh.vertices[mesh.edges[:, :-1]].mean(axis=1)
            mesh.boundary_names = []
            for i_contact, (contact_name, (x0, y0, r)) in enumerate(contacts.items()):
                mesh.boundary_names.append(contact_name)
                within = np.linalg.norm(edge_centers - (x0, y0), axis=1) <= r
                mesh.edges[within, -1] = i_contact

            mesh.lattice_vectors = None

        def _save_checkpoint(
            self, cp_path: CheckpointPath, context: CheckpointContext
        ) -> list[str]:
            cp_path.attrs["Lx"] = self.Lx
            cp_path.attrs["Ly"] = self.Ly
            cp_path.attrs["Nx"] = self.Nx
            cp_path.attrs["Ny"] = self.Ny
            return list(cp_path.attrs.keys())


_FACE = np.array([[0, 1], [1, 2], [2, 0]])  # local vertex pairs of the 3 faces (CCW)


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

    fa, fb = _FACE[:, 0], _FACE[:, 1]
    Pa, Pb = p[:, fa], p[:, fb]
    fmid = 0.5 * (Pa + Pb)
    tvec = Pb - Pa
    flen = np.linalg.norm(tvec, axis=2)
    fnrm = np.stack([tvec[..., 1], -tvec[..., 0]], axis=2) / flen[..., None]  # outward
    inradius = area / (0.5 * flen.sum(axis=1))

    # Deduplicate physical edges -> interior (two cells) / boundary (one).
    edge_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for k in range(K):
        for f in range(3):
            key = (
                min(int(tri[k, fa[f]]), int(tri[k, fb[f]])),
                max(int(tri[k, fa[f]]), int(tri[k, fb[f]])),
            )
            edge_map.setdefault(key, []).append((k, f))
    interior, boundary = [], []
    edge_marker = {(v1, v2): m for v1, v2, m in mesh.edges}
    for key, hits in edge_map.items():
        if len(hits) == 2:
            (kL, fL), (kR, fR) = hits
            interior.append((kL, fL, kR, fR))
        else:
            ((k, f),) = hits
            boundary.append((k, f, edge_marker.get(key, 0)))
    interior = np.array(interior, int).reshape(-1, 4)
    boundary = np.array(boundary, int).reshape(-1, 3)

    # Periodic faces: pair leftover boundary edges across each lattice vector and
    # promote them to interior edges (streaming neighbour = periodic image). Match
    # face midpoints with a KD-tree for robustness on distorted/irregular meshes.
    lattice = getattr(mesh, "_lattice", None)
    if lattice is not None and len(boundary):
        from scipy.spatial import cKDTree

        bmid = fmid[boundary[:, 0], boundary[:, 1]]  # (Nb, 2)
        tol = 1e-6 * float(max(flen.max(), 1.0))
        tree = cKDTree(bmid)
        used = np.zeros(len(boundary), bool)
        paired = []
        for L in np.atleast_2d(np.asarray(lattice, float)):
            dist, j = tree.query(bmid + L, distance_upper_bound=tol)
            for i, (di, ji) in enumerate(zip(dist, j)):
                if (
                    di <= tol
                    and ji < len(bmid)
                    and i != ji
                    and not used[i]
                    and not used[ji]
                ):
                    used[i] = used[ji] = True
                    paired.append(
                        (
                            boundary[i, 0],
                            boundary[i, 1],
                            boundary[ji, 0],
                            boundary[ji, 1],
                        )
                    )
        if paired:
            interior = np.vstack([interior, np.array(paired, int)])
            boundary = boundary[~used]

    kL, fL, kR, fR = interior.T if len(interior) else (np.empty(0, int),) * 4
    bk, bf, bmark = boundary.T if len(boundary) else (np.empty(0, int),) * 3

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
    edge_marker = {(v1, v2): m for v1, v2, m in mesh.edges}
    vmap: dict[int, list[tuple[int, int]]] = {}
    for k in range(K):
        for f in range(2):
            vmap.setdefault(int(seg[k, f]), []).append((k, f))
    interior, boundary = [], []
    for v, hits in vmap.items():
        if len(hits) == 2:
            (kL, fL), (kR, fR) = hits
            interior.append((kL, fL, kR, fR))
        else:
            ((k, f),) = hits
            boundary.append((k, f, edge_marker.get((v, v), 0)))
    interior = np.array(interior, int).reshape(-1, 4)
    boundary = np.array(boundary, int).reshape(-1, 3)
    kL, fL, kR, fR = interior.T if len(interior) else (np.empty(0, int),) * 4
    bk, bf, bmark = boundary.T if len(boundary) else (np.empty(0, int),) * 3

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
