"""Cell-centered 2nd-order finite-volume transport on an unstructured triangle mesh.

One cell average per triangle per momentum channel, ``u: (K, Nk)``. MUSCL
reconstruction (least-squares cell gradient, Venkatakrishnan limited) feeds a
scalar per-channel upwind flux ``F_c = (v_c.n) u_upwind`` -- each delta-k channel
streams with its own Fermi velocity, so the upwind side of each edge is fixed by
sign(v_c.n). Walls/contacts supply the exterior trace via the (reused)
FermiSurface reflector/contactor; collisions come from the material. Time
stepping is plain RK2 (see _time_evolution); no positivity limiter.

Boundary conditions: reflective walls, fixed-voltage/drift contacts, floating
(zero-current) probes and current sources (per-step scalar level solve), and
periodic faces paired through the mesh lattice. A METIS spatial decomposition
splits cells across the ``r`` comm with a thin halo exchange (FermiSurface
couples k-channels, so k is never split); see :class:`SpatialDecomp` below.

All velocity-independent per-edge weights are precomputed once, so a step is just
one limited-reconstruction pass, two gathers, and three scatters.
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from qimpy import rc, TreeNode
from qimpy.rc import MPI
from qimpy.io import CheckpointPath, CheckpointContext
from qimpy.mpi import ProcessGrid
from ..material import Material
from . import TensorList, Geometry
from ._mesh import load_mesh

_FACE = np.array([[0, 1], [1, 2], [2, 0]])   # local vertex pairs of the 3 faces (CCW)


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

    area: torch.Tensor; inv_area: torch.Tensor; inradius: torch.Tensor   # (K,)
    centroid_np: np.ndarray; vertices_np: np.ndarray; triangles_np: np.ndarray
    eL: torch.Tensor; eR: torch.Tensor; eLF: torch.Tensor; eRF: torch.Tensor  # (Ne,)
    en: torch.Tensor; elen: torch.Tensor                                  # (Ne,2),(Ne,)
    bcell: torch.Tensor; bF: torch.Tensor; bmark: torch.Tensor            # (Nb,)
    bn: torch.Tensor; blen: torch.Tensor                                  # (Nb,2),(Nb,)
    marker_names: list
    nbr: torch.Tensor          # (K, Nmax) vertex-neighbor cells (self-padded)
    recon: torch.Tensor        # (K, 3, Nmax) face-increment op: d_face = recon @ (u_nbr - u)


def build_fv_geom(mesh, *, dtype: torch.dtype = torch.float64) -> FVGeom:
    """Build the FV geometry from a loaded mesh (``_mesh.MeshResult``).

    Dispatches on cell type: 3 vertices/cell -> 2D triangles, 2 vertices/cell ->
    a 1D line mesh (interval cells; see :func:`_build_fv_geom_1d`).
    """
    tri = np.asarray(mesh.EToV, dtype=int)
    if tri.shape[1] == 2:
        return _build_fv_geom_1d(mesh, dtype=dtype)
    V = np.stack([mesh.VX, mesh.VY], axis=1).astype(float)
    K = len(tri)
    p = V[tri]                                                # (K, 3, 2)
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
            key = (min(int(tri[k, fa[f]]), int(tri[k, fb[f]])),
                   max(int(tri[k, fa[f]]), int(tri[k, fb[f]])))
            edge_map.setdefault(key, []).append((k, f))
    interior, boundary = [], []
    for key, hits in edge_map.items():
        if len(hits) == 2:
            (kL, fL), (kR, fR) = hits
            interior.append((kL, fL, kR, fR))
        else:
            (k, f), = hits
            boundary.append((k, f, mesh.edge_marker.get(key, 0)))
    interior = np.array(interior, int).reshape(-1, 4)
    boundary = np.array(boundary, int).reshape(-1, 3)

    # Periodic faces: pair leftover boundary edges across each lattice vector and
    # promote them to interior edges (streaming neighbour = periodic image). Match
    # face midpoints with a KD-tree for robustness on distorted/irregular meshes.
    lattice = getattr(mesh, "_lattice", None)
    if lattice is not None and len(boundary):
        from scipy.spatial import cKDTree
        bmid = fmid[boundary[:, 0], boundary[:, 1]]           # (Nb, 2)
        tol = 1e-6 * float(max(flen.max(), 1.0))
        tree = cKDTree(bmid)
        used = np.zeros(len(boundary), bool)
        paired = []
        for L in np.atleast_2d(np.asarray(lattice, float)):
            dist, j = tree.query(bmid + L, distance_upper_bound=tol)
            for i, (di, ji) in enumerate(zip(dist, j)):
                if di <= tol and ji < len(bmid) and i != ji \
                        and not used[i] and not used[ji]:
                    used[i] = used[ji] = True
                    paired.append((boundary[i, 0], boundary[i, 1],
                                   boundary[ji, 0], boundary[ji, 1]))
        if paired:
            interior = np.vstack([interior, np.array(paired, int)])
            boundary = boundary[~used]

    kL, fL, kR, fR = (interior.T if len(interior) else (np.empty(0, int),) * 4)
    bk, bf, bmark = (boundary.T if len(boundary) else (np.empty(0, int),) * 3)

    # Reconstruction stencil: vertex-neighbors (every cell sharing a vertex), which
    # stays full-rank and well-conditioned on distorted/irregular/boundary cells
    # where the 3 face-neighbors alone are too few or near-collinear.
    v2c: dict[int, list[int]] = {}
    for k in range(K):
        for vtx in tri[k]:
            v2c.setdefault(int(vtx), []).append(k)
    vnbr = [sorted({c for vtx in tri[k] for c in v2c[int(vtx)]} - {k}) for k in range(K)]
    Nmax = max((len(s) for s in vnbr), default=1)
    nbr = np.arange(K)[:, None].repeat(Nmax, axis=1)         # pad slots with self
    # Inverse-distance-weighted least-squares gradient operator per cell:
    #   grad_i = (D^T W D)^{-1} D^T W (u_nbr - u_i),  w_j = 1 / |c_j - c_i|^2.
    # pinv handles any residual rank-deficiency gracefully (min-norm gradient).
    grad_op = np.zeros((K, 2, Nmax))
    for i, js in enumerate(vnbr):
        if len(js) < 2:
            continue
        nbr[i, :len(js)] = js
        D = centroid[js] - centroid[i]                       # (n, 2)
        w = 1.0 / np.maximum((D ** 2).sum(1), 1e-300)        # inverse-distance^2
        sw = np.sqrt(w)
        grad_op[i, :, :len(js)] = np.linalg.pinv(sw[:, None] * D) * sw[None, :]
    # Fuse gradient + centroid->face offsets so a step reconstructs face
    # increments with one (3 x Nmax) @ (Nmax x Nk) matmul per cell.
    face_off = fmid - centroid[:, None]                      # (K, 3, 2)
    recon = np.einsum("kfx,kxg->kfg", face_off, grad_op)     # (K, 3, Nmax)

    def t(a, long=False):
        return torch.tensor(np.ascontiguousarray(a), device=rc.device,
                            dtype=torch.long if long else dtype)

    area_t = t(area)
    return FVGeom(
        area=area_t, inv_area=1.0 / area_t, inradius=t(inradius),
        centroid_np=centroid, vertices_np=V, triangles_np=tri,
        eL=t(kL, long=True), eR=t(kR, long=True),
        eLF=t(kL * 3 + fL, long=True), eRF=t(kR * 3 + fR, long=True),
        en=t(fnrm[kL, fL]), elen=t(flen[kL, fL]),
        bcell=t(bk, long=True), bF=t(bk * 3 + bf, long=True), bmark=t(bmark, long=True),
        bn=t(fnrm[bk, bf]), blen=t(flen[bk, bf]), marker_names=list(mesh.marker_names),
        nbr=t(nbr, long=True), recon=t(recon),
    )


def _build_fv_geom_1d(mesh, *, dtype: torch.dtype = torch.float64) -> FVGeom:
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
    V = np.stack([mesh.VX, mesh.VY], axis=1).astype(float)    # (Nv, 2), VY ~ 0
    seg = np.asarray(mesh.EToV, dtype=int)                    # (K, 2): [v_left, v_right]
    K = len(seg)
    p = V[seg]                                                # (K, 2, 2): endpoints
    centroid = p.mean(axis=1)                                 # (K, 2)
    L = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)             # (K,) cell length
    if np.any(L <= 0.0):
        raise ValueError("1D line mesh has a zero-length cell")
    area = L                                                  # FV cell measure
    inradius = L                                              # dt = cfl * L / vmax
    fmid = p                                                  # face = the endpoint vertex
    face_off = fmid - centroid[:, None]                       # (K, 2, 2) centroid->face
    fnrm = face_off / np.linalg.norm(face_off, axis=2, keepdims=True)  # +/- x unit normal
    flen = np.ones((K, 2))                                    # point face: unit measure

    # Interior / boundary by shared-vertex dedup.
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
            (k, f), = hits
            boundary.append((k, f, mesh.edge_marker.get((v, v), 0)))
    interior = np.array(interior, int).reshape(-1, 4)
    boundary = np.array(boundary, int).reshape(-1, 3)
    kL, fL, kR, fR = (interior.T if len(interior) else (np.empty(0, int),) * 4)
    bk, bf, bmark = (boundary.T if len(boundary) else (np.empty(0, int),) * 3)

    # Inverse-distance least-squares gradient over shared-vertex neighbors.
    v2c: dict[int, list[int]] = {}
    for k in range(K):
        for vtx in seg[k]:
            v2c.setdefault(int(vtx), []).append(k)
    vnbr = [sorted({c for vtx in seg[k] for c in v2c[int(vtx)]} - {k}) for k in range(K)]
    Nmax = max((len(s) for s in vnbr), default=1)
    nbr = np.arange(K)[:, None].repeat(Nmax, axis=1)
    grad_op = np.zeros((K, 2, Nmax))
    for i, js in enumerate(vnbr):
        if not js:
            continue
        nbr[i, :len(js)] = js
        D = centroid[js] - centroid[i]                        # (n, 2), y ~ 0
        w = 1.0 / np.maximum((D ** 2).sum(1), 1e-300)
        sw = np.sqrt(w)
        grad_op[i, :, :len(js)] = np.linalg.pinv(sw[:, None] * D) * sw[None, :]
    recon = np.einsum("kfx,kxg->kfg", face_off, grad_op)      # (K, 2, Nmax)

    def t(a, long=False):
        return torch.tensor(np.ascontiguousarray(a), device=rc.device,
                            dtype=torch.long if long else dtype)

    area_t = t(area)
    return FVGeom(
        area=area_t, inv_area=1.0 / area_t, inradius=t(inradius),
        centroid_np=centroid, vertices_np=V, triangles_np=seg,
        eL=t(kL, long=True), eR=t(kR, long=True),
        eLF=t(kL * 2 + fL, long=True), eRF=t(kR * 2 + fR, long=True),
        en=t(fnrm[kL, fL]), elen=t(flen[kL, fL]),
        bcell=t(bk, long=True), bF=t(bk * 2 + bf, long=True), bmark=t(bmark, long=True),
        bn=t(fnrm[bk, bf]), blen=t(flen[bk, bf]), marker_names=list(mesh.marker_names),
        nbr=t(nbr, long=True), recon=t(recon),
    )


# --------------------------------------------------------------------------- #
#  Spatial domain decomposition: each rank owns a contiguous block of cells and
#  does O(local) work per step, exchanging only a thin 2-ring halo of ghost-cell
#  averages. Cells are partitioned by METIS (min-cut on the face-neighbour dual
#  graph) and renumbered so every rank's block -- and its checkpoint slice -- is
#  contiguous. The MUSCL stencil needs two rings (a cell's face value uses its
#  1-ring gradient; the flux on its face also uses the neighbour's reconstructed
#  value, hence the neighbour's 1-ring), so the halo is the 2-ring vertex closure.
# --------------------------------------------------------------------------- #
def _dual_graph(EToV) -> list[list[int]]:
    """Face-neighbour adjacency (the FV dual graph): cells sharing an edge."""
    e2c: dict[tuple[int, int], list[int]] = defaultdict(list)
    for k, tri in enumerate(EToV):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            e2c[tuple(sorted((int(tri[a]), int(tri[b]))))].append(k)
    nbr: list[set[int]] = [set() for _ in range(len(EToV))]
    for cells in e2c.values():
        if len(cells) == 2:
            i, j = cells
            nbr[i].add(j)
            nbr[j].add(i)
    return [sorted(s) for s in nbr]


def _coordinate_part(mesh, nparts: int) -> np.ndarray:
    """Fallback partition (no METIS): sort cells along the longer axis into
    equal-count blocks. Correct but with poorer locality on branchy meshes."""
    V = np.stack([mesh.VX, mesh.VY], axis=1)
    cen = V[np.asarray(mesh.EToV, int)].mean(axis=1)
    axis = 0 if cen[:, 0].ptp() >= cen[:, 1].ptp() else 1
    order = np.argsort(cen[:, axis], kind="stable")
    part = np.empty(len(order), np.int32)
    part[order] = np.minimum((np.arange(len(order)) * nparts) // len(order), nparts - 1)
    return part


def partition(mesh, comm: MPI.Comm) -> tuple[np.ndarray, np.ndarray]:
    """Renumber cells into contiguous per-rank blocks.

    Returns ``(perm, bounds)``: applying ``EToV[perm]`` places rank ``r``'s cells
    in the contiguous slice ``[bounds[r], bounds[r+1])``. The partition is a METIS
    min-cut of the face-neighbour dual graph, computed on the head and broadcast
    so every rank agrees exactly; falls back to a coordinate sort if pymetis is
    not installed.
    """
    K = len(mesh.EToV)
    nparts = comm.size
    if nparts == 1:
        return np.arange(K), np.array([0, K], int)
    part = None
    if comm.rank == 0:
        try:
            import pymetis
            _, p = pymetis.part_graph(nparts, adjacency=_dual_graph(mesh.EToV))
            part = np.asarray(p, np.int32)
        except ImportError:
            part = _coordinate_part(mesh, nparts)
    part = comm.bcast(part, root=0)
    perm = np.argsort(part, kind="stable")              # group cells by rank
    bounds = np.concatenate([[0], np.cumsum(np.bincount(part, minlength=nparts))])
    return perm, bounds.astype(int)


class SpatialDecomp:
    """Owned/ghost bookkeeping and halo exchange over a renumbered cell mesh.

    Construct after the cells have been renumbered by :func:`partition` and the
    geometry built, passing the vertex-neighbour table ``nbr`` (global, in the
    renumbered order) and the per-rank ``bounds``. Owned cells of rank ``r`` are
    the contiguous slice ``[bounds[r], bounds[r+1])``.
    """

    def __init__(self, nbr_np: np.ndarray, bounds: np.ndarray,
                 comm: MPI.Comm) -> None:
        self.comm = comm
        self.size = comm.size
        self.rank = comm.rank
        self.K = nbr_np.shape[0]
        self.offset = np.asarray(bounds, int)
        self.start = int(self.offset[self.rank])
        self.stop = int(self.offset[self.rank + 1])
        self.owned = np.arange(self.start, self.stop)

        def ring(cells):
            """1-ring vertex closure of a set of cells (cells + their neighbours)."""
            if not cells:
                return set()
            return set(cells) | set(
                nbr_np[np.asarray(sorted(cells), int)].ravel().tolist())

        own = set(self.owned.tolist())
        ring1 = ring(own)                       # cells to reconstruct (owned + 1-ring)
        ring2 = ring(ring1)                     # cells whose u must be current (2-ring)
        self.recon_rows = np.array(sorted(ring1), int)
        ghosts = np.array(sorted(ring2 - own), int)

        # Halo plans: receive each ghost from its owning rank; send the owned
        # cells that another rank needs (its 2-ring minus its own block).
        self.recv: dict[int, np.ndarray] = {}
        for q in range(self.size):
            sel = ghosts[(ghosts >= self.offset[q]) & (ghosts < self.offset[q + 1])]
            if len(sel) and q != self.rank:
                self.recv[q] = sel
        self.send: dict[int, np.ndarray] = {}
        for q in range(self.size):
            if q == self.rank:
                continue
            owned_q = set(range(int(self.offset[q]), int(self.offset[q + 1])))
            need_q = ring(ring(owned_q)) - owned_q
            sel = self.owned[np.isin(self.owned, np.fromiter(need_q, int, len(need_q)))]
            if len(sel):
                self.send[q] = sel

    def exchange(self, u: torch.Tensor) -> None:
        """Fill this rank's ghost rows of ``u`` (K, Nk) with their owners' values."""
        if not self.recv and not self.send:
            return
        un = u.detach().to("cpu").numpy()
        reqs = []
        recv_bufs = {}
        for q, idx in self.recv.items():
            buf = np.empty((len(idx), un.shape[1]), un.dtype)
            recv_bufs[q] = (buf, idx)
            reqs.append(self.comm.Irecv(buf, source=q, tag=11))
        send_bufs = []
        for q, idx in self.send.items():
            sb = np.ascontiguousarray(un[idx])
            send_bufs.append(sb)
            reqs.append(self.comm.Isend(sb, dest=q, tag=11))
        MPI.Request.Waitall(reqs)
        for q, (buf, idx) in recv_bufs.items():
            u[torch.as_tensor(idx, device=u.device)] = torch.as_tensor(
                buf, device=u.device, dtype=u.dtype)


@dataclass
class _Contact:
    """One boundary contact. ``fixed`` holds a prescribed ghost; a feedback
    contact (``floating`` probe or ``current`` source) solves a scalar level
    each evaluation so its net current hits ``target`` (0 for floating)."""

    name: str
    idx: torch.Tensor                 # boundary-edge indices of this contact
    cur: torch.Tensor                 # (Nsel, Nk) outward number-flux operator
    kind: str = "fixed"
    ghost: Optional[torch.Tensor] = None       # fixed: prescribed exterior trace
    unit: Optional[torch.Tensor] = None        # feedback: ghost per unit level
    drift: Optional[torch.Tensor] = None       # feedback: prescribed drift part
    cur_out: Optional[torch.Tensor] = None     # feedback: outflow-only flux op
    den: float = 1.0                           # feedback: inflow capacity / level
    base: float = 0.0                          # feedback: drift inflow current
    target: float = 0.0                        # feedback: desired net current
    level: float = 0.0                         # feedback: last solved level


class FiniteVolume(Geometry):
    """Cell-centered finite-volume geometry on an external triangle mesh."""

    def __init__(
        self,
        *,
        material: Material,
        mesh_file: str,
        contacts: dict[str, Optional[dict]],
        cfl: float = 0.4,
        vk_eps2: float = 0.0,
        compile: bool = False,
        save_rho: bool = False,
        process_grid: ProcessGrid,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ):
        """
        Parameters
        ----------
        mesh_file
            :yaml:`Path to an external triangle mesh (.npz) to solve on.`
        contacts
            :yaml:`Dictionary of contact names to parameters (match mesh markers).`
            Each value selects the contact kind: ``{dmu, vD}`` a fixed
            voltage/drift source, ``{floating: true}`` a zero-current probe, and
            ``{I_set: <current>}`` a current source (with optional ``vD``).
        cfl
            :yaml:`CFL number for the explicit step (dt = cfl * inradius / vmax).`
        vk_eps2
            :yaml:`Venkatakrishnan threshold in field^2 units (0 = pure smooth limiter).`
            Set to ~(mesh-scale variation)^2 to stop limiting smooth/low-amplitude
            data and recover strict linearity preservation.
        compile
            :yaml:`torch.compile the limited reconstruction (fuses the per-step
            limiter kernels).` ~3x faster steps on GPU at the cost of a one-time
            compile; leave off for short runs and the test suite.
        """
        TreeNode.__init__(self)
        self.material = material
        self.comm = process_grid.get_comm("r")
        self.mesh_file = mesh_file
        self.contacts = contacts
        self.save_rho = save_rho
        self._vk_eps2 = float(vk_eps2)

        self.mesh = load_mesh(mesh_file)
        self._mpi = self.comm.size > 1
        if self._mpi:
            # METIS min-cut partition, renumbered so each rank owns a contiguous
            # block (compact halos + direct checkpoint slices). Keep the
            # permutation so the renumbered solution maps back to the input order.
            self._perm, bounds = partition(self.mesh, self.comm)
            self.mesh.EToV = np.asarray(self.mesh.EToV, int)[self._perm]
        else:
            self._perm, bounds = None, None
        g = build_fv_geom(self.mesh, dtype=material.transport_velocity.dtype)
        self.geom = g
        v = material.transport_velocity                       # (Nk, 2)
        self.Nk = v.shape[0]
        self.K = int(g.area.shape[0])
        self._nf = int(g.recon.shape[1])                      # faces/cell: 3 (tri) or 2 (1D)

        # Spatial decomposition: owned cell block, reconstruction rows (owned +
        # 1-ring), owned-incident edges and the halo exchange (see SpatialDecomp).
        if self._mpi:
            self._decomp = SpatialDecomp(g.nbr.detach().cpu().numpy(), bounds, self.comm)
            self._own_start, self._own_stop = self._decomp.start, self._decomp.stop
            self._R = torch.as_tensor(self._decomp.recon_rows,
                                      device=rc.device, dtype=torch.long)
            self._owned_mask = torch.zeros(self.K, 1, dtype=torch.bool, device=rc.device)
            self._owned_mask[self._own_start:self._own_stop] = True
        else:
            self._decomp = None
            self._own_start, self._own_stop = 0, self.K
            self._R = None
            self._owned_mask = None

        # Precompute velocity-weighted, area-scaled edge operators (constant):
        #   into eL: -a*elen/area_L,  into eR: +a*elen/area_R,  a = v_c.n
        a_int = g.en @ v.t()                                  # (Ne, Nk)
        self._maskL = a_int > 0
        self._wL = -a_int * (g.elen * g.inv_area[g.eL])[:, None]
        self._wR = a_int * (g.elen * g.inv_area[g.eR])[:, None]
        self._a_bnd = g.bn @ v.t()                            # (Nb, Nk)
        self._maskB = self._a_bnd > 0
        self._wB = -self._a_bnd * (g.blen * g.inv_area[g.bcell])[:, None]
        # Per-channel density weight; outward number-flux operator per boundary edge.
        self._ncoef = material.get_observables(0.0)[0]        # (Nk,)
        self._cur_b = (g.blen[:, None] * self._ncoef[None, :]) * self._a_bnd  # (Nb,Nk)
        # Restrict per-step work to cells/edges this rank owns (all of them serially).
        lo, hi = self._own_start, self._own_stop
        if self._mpi:
            own_e = ((g.eL >= lo) & (g.eL < hi)) | ((g.eR >= lo) & (g.eR < hi))
            self._eloc = torch.where(own_e)[0]
            self._bloc = torch.where((g.bcell >= lo) & (g.bcell < hi))[0]
        else:
            self._eloc = None
            self._bloc = None
        self._setup_boundary(material)

        # De-aliasing projector.  When the material's angular quadrature oversamples
        # its modes -- FermiSurface uses an even N_theta > 2M+1 so the velocity set
        # is mirror-symmetric (left-right contact symmetry) -- there are nodal DOFs
        # carrying no represented harmonic.  The per-channel MUSCL limiter is
        # nonlinear and excites them; the modal collision operator cannot damp them,
        # so under strong collisions they grow without bound.  Projecting the state
        # onto the represented modes each evaluation removes that aliased content;
        # it is exactly the identity on n, the current, and every retained harmonic.
        # No-op for square transforms or materials without modal transforms
        # (single_band, ab_initio).  Applied as  u @ self._proj.
        # The unrepresented ("ghost") subspace has tiny rank r = N_theta - (2M+1)
        # (1-3), so we remove it with a rank-r update  u -= (u @ A) @ B^T  rather
        # than a dense Nk*Nk matmul -- numerically identical, ~Nk/r times cheaper.
        # Always on: the ghost lies in ker(to_modes) (collision cannot damp it) yet
        # the nonlinear limiter excites it, so it must be projected out each step.
        self._dl_A = self._dl_B = None
        if hasattr(material, "to_modes") and hasattr(material, "from_modes"):
            eye = torch.eye(self.Nk, device=rc.device, dtype=v.dtype)
            proj = material.from_modes(material.to_modes(eye))   # (Nk, Nk) projector
            ghost = eye - proj                                   # onto unrepresented DOFs
            if ghost.abs().max() > 1e-10:                        # only if oversampled
                U, S, Vh = torch.linalg.svd(ghost)
                r = int((S > 1e-8).sum())
                self._dl_A = (U[:, :r] * S[:r]).contiguous()     # (Nk, r)
                self._dl_B = Vh[:r].T.contiguous()               # (Nk, r)

        # Ballistic short-circuit: when the material's collision+field operator is
        # identically zero (rates and cyclotron speed both zero) its rho_dot is a
        # no-op, so skip the per-step call entirely -- it otherwise allocates a
        # zero tensor and (via the rates check) forces a GPU->CPU sync each step.
        rm = getattr(material, "rates_modal", None)
        ks = getattr(material, "k_speed", 0.0)
        self._skip_collision = bool(
            rm is not None and float(rm.abs().sum()) == 0.0 and float(ks) == 0.0
        )

        # Optionally fuse the per-step kernels with torch.compile.  Serial: compile
        # the whole spatial RHS (reconstruction + limiter + flux scatter + boundary)
        # so the flux glue fuses into the reconstruction graph and the kernel-launch
        # overhead collapses.  MPI: shapes are dynamic (owned-edge gathers), so only
        # the reconstruction is compiled.
        self._faces_fn = self._faces
        self._srhs_fn = self._spatial_rhs
        if compile:
            try:
                if self._mpi:
                    self._faces_fn = torch.compile(self._faces, mode="max-autotune")
                else:
                    self._srhs_fn = torch.compile(self._spatial_rhs, mode="max-autotune")
            except Exception:               # older torch / no backend -> eager
                self._faces_fn, self._srhs_fn = self._faces, self._spatial_rhs

        vmax = float(v.norm(dim=1).max())
        dt_local = float(cfl) * float(g.inradius.min()) / max(vmax, 1e-300)
        self.dt_max = self.comm.allreduce(dt_local, op=MPI.MIN)

        rho0 = getattr(material, "rho0", None)
        if rho0 is not None:
            self._u = rho0.flatten().to(rc.device, v.dtype)[None, :].repeat(self.K, 1)
        else:
            self._u = torch.zeros(self.K, self.Nk, device=rc.device, dtype=v.dtype)
        self._stash_t, self._stash_i, self._stash_obs = [], [], []

    def _setup_boundary(self, material: Material) -> None:
        """Group boundary edges by marker into a wall (reflector) set and one
        contact object per parametrized marker; unparametrized markers reflect.

        Under decomposition each rank keeps only the boundary edges on its owned
        cells; a feedback contact's capacity (``den``/``base``) is summed across
        ranks so every rank solves the same global level."""
        g = self.geom
        names = g.marker_names
        name_of = [names[m] if 0 <= m < len(names) else "wall"
                   for m in g.bmark.tolist()]
        lo, hi = self._own_start, self._own_stop
        bcell = g.bcell.tolist()
        owns = [lo <= bcell[i] < hi for i in range(len(name_of))]  # this rank's edges
        is_c = lambda nm: (nm in self.contacts) and (self.contacts[nm] is not None)
        wall = np.array([i for i, nm in enumerate(name_of)
                         if owns[i] and not is_c(nm)], int)
        self._wall = torch.as_tensor(wall, device=rc.device, dtype=torch.long)
        self._reflector = material.get_reflector(g.bn[self._wall]) if wall.size else None
        # The reflector is linear in the wall trace, so collapse its per-step,
        # O(M_theta)-Python-loop modal transforms into a single per-edge matrix
        # applied as one batched matmul. Build it once by reflecting the Nk basis
        # vectors: refl_mat[e, i, c] = reflector(e_c)[e, i].
        self._refl_mat = None
        if self._reflector is not None:
            nw = self._wall.numel()
            bn_wall = g.bn[self._wall]
            eye = torch.eye(self.Nk, device=rc.device, dtype=g.area.dtype)
            # Build the per-edge matrix in chunks over wall edges into a preallocated
            # buffer: fully materializing the (Nk, nw, Nk) basis at once OOMs at
            # large Nk (the matrix itself is nw*Nk^2).  Cap the transient at ~200 MB.
            chunk = max(1, int(2.0e8 // (self.Nk * self.Nk * 8)))
            self._refl_mat = torch.empty(nw, self.Nk, self.Nk,
                                         device=rc.device, dtype=g.area.dtype)
            for s in range(0, nw, chunk):
                cs = min(chunk, nw - s)
                refl_c = material.get_reflector(bn_wall[s:s + cs])
                basis_c = eye[:, None, :].expand(self.Nk, cs, self.Nk)
                self._refl_mat[s:s + cs] = refl_c(basis_c).permute(1, 2, 0)

        def allreduce(x: float) -> float:
            return self.comm.allreduce(x) if self._mpi else x

        self._contacts: list[_Contact] = []
        for nm, params in self.contacts.items():
            if params is None:
                continue
            sel = np.array([i for i, x in enumerate(name_of)
                            if x == nm and owns[i]], int)
            ci = torch.as_tensor(sel, device=rc.device, dtype=torch.long)
            cur = self._cur_b[ci]                             # (Nsel, Nk) (maybe empty)
            params = dict(params)
            floating = bool(params.pop("floating", False))
            i_set = params.pop("I_set", None)
            vD = float(params.get("vD", 0.0))
            if floating or (i_set is not None):
                # Contactor ghost is affine in dmu: g(dmu) = dmu*unit + drift.
                bn_ci = g.bn[ci]
                unit = material.get_contactor(bn_ci, dmu=1.0)(0.0)
                drift = (material.get_contactor(bn_ci, vD=vD)(0.0)
                         if vD else torch.zeros_like(unit))
                cur_in = torch.where(self._a_bnd[ci] < 0, cur, torch.zeros_like(cur))
                cur_out = torch.where(self._a_bnd[ci] > 0, cur, torch.zeros_like(cur))
                den = allreduce(float(-(cur_in * unit).sum()))   # global inflow capacity
                base = allreduce(float((cur_in * drift).sum()))  # global drift inflow
                self._contacts.append(_Contact(
                    name=nm, idx=ci, cur=cur, kind="current",
                    unit=unit, drift=drift, cur_out=cur_out,
                    den=(den if abs(den) > 1e-300 else 1e-300), base=base,
                    target=(0.0 if floating else float(i_set))))
            else:
                ghost = material.get_contactor(g.bn[ci], **params)(0.0)
                self._contacts.append(_Contact(
                    name=nm, idx=ci, cur=cur, kind="fixed", ghost=ghost))

    def _limited_faces(self, uc: torch.Tensor, un: torch.Tensor,
                       recon: torch.Tensor) -> torch.Tensor:
        """Venkatakrishnan-limited face values for a set of cells.

        ``uc`` (n, Nk) cell averages, ``un`` (n, Nmax, Nk) their vertex-neighbor
        averages, ``recon`` (n, 3, Nmax) the fused gradient->face operator. Smooth
        (differentiable) limiter -> clean steady-state convergence; per face, with
        increment d = u_face-u and same-sign headroom D1 (to the neighbor max/min):
            phi = (D1^2 + 2 D1 d + e) / (D1^2 + D1 d + 2 d^2 + e),  e = vk_eps2,
        capped at 1 (never amplify the LSQ gradient) and min-ed over the 3 faces.
        """
        d = torch.einsum("nfg,ngc->nfc", recon, un - uc[:, None])   # (n, 3, Nk)
        hi = (torch.maximum(uc, un.amax(1)) - uc)[:, None]    # headroom up   (>= 0)
        lo = (torch.minimum(uc, un.amin(1)) - uc)[:, None]    # headroom down (<= 0)
        D1 = torch.where(d >= 0, hi, lo)                      # same sign as d
        e = self._vk_eps2
        phi = torch.where(
            d != 0,
            ((D1 * D1 + 2 * D1 * d + e)
             / (D1 * D1 + D1 * d + 2 * d * d + e)).clamp(max=1.0),
            torch.ones_like(d),
        ).amin(1)[:, None]                                    # (n, 1, Nk)
        return uc[:, None] + phi * d

    def _faces(self, u: torch.Tensor) -> torch.Tensor:
        """Reconstructed face values, (K, n_face, Nk). Serial reconstructs every
        cell; under decomposition only the rows this rank needs (owned + 1-ring)
        are filled, the rest left zero (their faces are never read)."""
        g = self.geom
        if self._R is None:
            return self._limited_faces(u, u[g.nbr], g.recon)
        R = self._R
        uf = u.new_zeros(self.K, self._nf, self.Nk)
        uf[R] = self._limited_faces(u[R], u[g.nbr[R]], g.recon[R])
        return uf

    def _exterior(self, uMb: torch.Tensor, t: float) -> torch.Tensor:
        """Exterior ghost at boundary edges: reflector on walls, prescribed or
        feedback-solved contactor on contacts. Feedback contacts solve a scalar
        level so the net current equals their target; only inflow channels are
        consumed downstream by the upwind flux."""
        uP = uMb.clone()
        if self._refl_mat is not None:
            uP[self._wall] = torch.einsum(
                "eic,ec->ei", self._refl_mat, uMb[self._wall])
        for c in self._contacts:
            if c.kind == "fixed":
                uP[c.idx] = c.ghost.to(uP)
            else:
                # I_net(level) = num_out + base - level*den; solve = target. The
                # outflow term is summed across ranks so the level is global.
                num = float((c.cur_out * uMb[c.idx]).sum())
                if self._mpi:
                    num = self.comm.allreduce(num)
                c.level = (num + c.base - c.target) / c.den
                uP[c.idx] = (c.level * c.unit + c.drift).to(uP)
        return uP

    def _spatial_rhs(self, u: torch.Tensor, t: float) -> torch.Tensor:
        g = self.geom
        u = self._dealias(u)                                  # de-alias (fused into graph)
        uf = self._faces_fn(u).reshape(-1, self.Nk)           # (K*3, Nk)
        dudt = torch.zeros_like(u)
        e = self._eloc                                        # owned-incident edges
        eL = g.eL if e is None else g.eL[e]
        eR = g.eR if e is None else g.eR[e]
        eLF = g.eLF if e is None else g.eLF[e]
        eRF = g.eRF if e is None else g.eRF[e]
        maskL = self._maskL if e is None else self._maskL[e]
        wL = self._wL if e is None else self._wL[e]
        wR = self._wR if e is None else self._wR[e]
        uup = torch.where(maskL, uf[eLF], uf[eRF])            # interior upwind trace
        dudt.index_add_(0, eL, wL * uup)
        dudt.index_add_(0, eR, wR * uup)
        if g.bcell.numel():
            uMb = uf[g.bF]
            uup_b = torch.where(self._maskB, uMb, self._exterior(uMb, t))
            wbu = self._wB * uup_b
            if self._bloc is None:
                dudt.index_add_(0, g.bcell, wbu)
            else:
                b = self._bloc
                dudt.index_add_(0, g.bcell[b], wbu[b])
        return dudt

    def _dealias(self, u: torch.Tensor) -> torch.Tensor:
        """Project onto the represented angular modes (rank-r ghost removal)."""
        if self._dl_A is None:
            return u
        return u - (u @ self._dl_A) @ self._dl_B.T

    # ---- qimpy Geometry contract ----
    def rho_dot(self, rho: TensorList, t: float) -> TensorList:
        u = rho[0]
        if self._decomp is not None:
            self._decomp.exchange(u)                          # fill halo ghost rows
        out = self._srhs_fn(u, t)                             # de-alias + spatial RHS (fused)
        if not self._skip_collision:                          # ballistic: collision is exactly 0
            # collision = from_modes(-rates * to_modes(.)); its to_modes already
            # annihilates the ghost, so the raw (un-de-aliased) u is exact here.
            lo, hi = self._own_start, self._own_stop
            out[lo:hi] = out[lo:hi] + self.material.rho_dot(u[lo:hi], t, id(self))
        if self._owned_mask is not None:
            out = out * self._owned_mask
        return TensorList([out])

    @property
    def rho(self) -> TensorList:
        return TensorList([self._u])

    @rho.setter
    def rho(self, rho_new: TensorList) -> None:
        self._u = self._dealias(rho_new[0])

    @property
    def density(self) -> torch.Tensor:
        return self._u

    # ---- contact diagnostics ----
    def contact_currents(self, t: float = 0.0) -> dict[str, float]:
        """Net outward number-current through each contact (positive = out of the
        device). Floating probes read ~0; current sources read their I_set; fixed
        contacts read a response. Sum over all boundaries = -d/dt of mass."""
        if self._decomp is not None:
            self._decomp.exchange(self._u)
        uf = self._faces(self._u).reshape(-1, self.Nk)
        uMb = uf[self.geom.bF]
        uup_b = torch.where(self._maskB, uMb, self._exterior(uMb, t))
        out = {}
        for c in self._contacts:
            I = float((c.cur * uup_b[c.idx]).sum())           # this rank's edges
            out[c.name] = self.comm.allreduce(I) if self._mpi else I
        return out

    def contact_potentials(self) -> dict[str, float]:
        """Self-adjusting level of each feedback contact, from the last evaluation."""
        return {c.name: c.level for c in self._contacts if c.kind != "fixed"}

    def update_stash(self, i_step: int, t: float) -> None:
        # Stash observables for this rank's owned cells (its checkpoint slice).
        u_own = self._u[self._own_start:self._own_stop]
        obs = torch.einsum("oc,kc->ko", self.material.get_observables(t), u_own)
        self._stash_i.append(i_step)
        self._stash_t.append(t)
        self._stash_obs.append(obs.detach().cpu().numpy())

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        g = self.geom
        names = self.material.get_observable_names()
        cp_path.attrs["order"] = 0                            # piecewise-constant FV
        cp_path.attrs["mesh_file"] = self.mesh_file
        saved = [
            cp_path.write("mesh_vertices", torch.from_numpy(g.vertices_np)),
            cp_path.write("mesh_triangles",
                          torch.from_numpy(g.triangles_np.astype(np.int64))),
            cp_path.write("cell_centroid", torch.from_numpy(g.centroid_np)),
            "fv_observables",
        ]
        cp_path.write_str("contact_names", ",".join(self.contacts.keys()))
        cp_path.write_str("observable_names", ",".join(names))
        cp_path["t"] = np.array(self._stash_t)
        cp_path["i_step"] = np.array(self._stash_i)
        checkpoint, path = cp_path
        n_stash = len(self._stash_t)
        # Collective: every rank creates the global dataset, then writes the
        # slice of cells it owns ([own_start, own_stop)). Serially this is the
        # whole array; under MPI it is each rank's contiguous block (needs an
        # mpio-enabled h5py to run multi-rank).
        CheckpointPath(checkpoint, path).create_dataset(
            "fv_observables", (n_stash, self.K, len(names)), np.float64)
        if checkpoint is not None and n_stash:
            checkpoint.write_slice(checkpoint[f"{path}/fv_observables"],
                                   (0, self._own_start, 0),
                                   torch.from_numpy(np.stack(self._stash_obs)))
        if self.save_rho:
            # Raw per-cell state (n_cells, n_channels), for exact restart /
            # steady-state warm start. Each rank writes its owned cell block.
            u_own = self._u[self._own_start:self._own_stop].detach().cpu().numpy()
            CheckpointPath(checkpoint, path).create_dataset(
                "fv_rho", (self.K, self.Nk), u_own.dtype)
            if checkpoint is not None:
                checkpoint.write_slice(checkpoint[f"{path}/fv_rho"],
                                       (self._own_start, 0),
                                       torch.from_numpy(u_own))
            saved.append("fv_rho")
        self._stash_t, self._stash_i, self._stash_obs = [], [], []
        return saved
