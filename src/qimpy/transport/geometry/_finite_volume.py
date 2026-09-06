"""Cell-centered 2nd-order finite-volume transport on an unstructured triangle mesh.

One cell average per triangle per momentum channel, ``u: (K, Nk)``. MUSCL
reconstruction (least-squares cell gradient, Venkatakrishnan limited) feeds a
scalar per-channel upwind flux ``F_c = (v_c.n) u_upwind`` -- each delta-k channel
streams with its own Fermi velocity, so the upwind side of each edge is fixed by
sign(v_c.n). Walls/contacts supply the exterior trace via the (reused)
FermiSurface reflector/contactor; collisions come from the material. Time
stepping is plain RK2 (see _time_evolution).

Boundary conditions: reflective walls, fixed-voltage/drift contacts, floating
(zero-current) probes and current sources (per-step scalar level solve), and
periodic faces paired through the mesh lattice. A METIS spatial decomposition
splits cells across the ``r`` comm with a thin halo exchange (FermiSurface
couples k-channels, so k is never split); see :class:`SpatialDecomp` below.

All velocity-independent per-edge weights are precomputed once, so a step is just
one limited-reconstruction pass, two gathers, and three scatters.
"""

from __future__ import annotations
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist

from qimpy import rc, log, TreeNode
from qimpy.rc import MPI
from qimpy.io import CheckpointPath, InvalidInputException, CheckpointContext
from qimpy.mpi import ProcessGrid, all_reduce_scalars
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
    # ⛔ NumPy 2.0 removed ndarray.ptp; the free function still exists.
    # This one line broke EVERY multi-rank run -- the decomposition could
    # not even be built -- so the halo exchange had never executed under
    # any numpy >= 2, and test_decomp_matches_serial failed on the
    # pristine tree for the same reason.
    axis = 0 if np.ptp(cen[:, 0]) >= np.ptp(cen[:, 1]) else 1
    order = np.argsort(cen[:, axis], kind="stable")
    part = np.empty(len(order), np.int32)
    part[order] = np.minimum((np.arange(len(order)) * nparts) // len(order), nparts - 1)
    return part


def partition(mesh, group: "dist.ProcessGroup") -> tuple[np.ndarray, np.ndarray]:
    """Renumber cells into contiguous per-rank blocks.

    Returns ``(perm, bounds)``: applying ``EToV[perm]`` places rank ``r``'s cells
    in the contiguous slice ``[bounds[r], bounds[r+1])``. The partition is a METIS
    min-cut of the face-neighbour dual graph, computed on the head and broadcast
    so every rank agrees exactly; falls back to a coordinate sort if pymetis is
    not installed.
    """
    K = len(mesh.EToV)
    nparts = group.size()
    if nparts == 1:
        return np.arange(K), np.array([0, K], int)
    part = None
    if dist.get_rank(group) == 0:
        try:
            import pymetis
            _, p = pymetis.part_graph(nparts, adjacency=_dual_graph(mesh.EToV))
            part = np.asarray(p, np.int32)
        except ImportError:
            part = _coordinate_part(mesh, nparts)
    # ⛔ dist has no bcast for python objects that returns a value; use the
    # object-list form and read element 0 back out.
    box = [part]
    dist.broadcast_object_list(box, src=0, group=group)
    part = box[0]
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
                 group: "dist.ProcessGroup") -> None:
        self.group = group
        self.size = group.size()
        self.rank = dist.get_rank(group)
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
        # torch.distributed point-to-point.  ⛔ Unlike mpi4py's Irecv/Isend on
        # numpy buffers, dist.irecv/isend take TENSORS and (for nccl) they must
        # live on the GPU, so the halo stays on u.device instead of round-
        # tripping through numpy.  Ranks are GROUP-relative, matching
        # self.recv/self.send which were built from group ranks.
        recv_bufs = {}
        ops = []
        for q, idx in self.recv.items():
            buf = torch.empty((len(idx), u.shape[1]), dtype=u.dtype,
                              device=u.device)
            recv_bufs[q] = (buf, idx)
            ops.append((dist.irecv, buf, q))
        send_bufs = []
        for q, idx in self.send.items():
            sb = u[torch.as_tensor(idx, device=u.device)].contiguous()
            send_bufs.append(sb)
            ops.append((dist.isend, sb, q))

        # ⛔⛔ NCCL POINT-TO-POINT MUST BE GROUPED OR IT DEADLOCKS.
        # mpi4py's Irecv/Isend are genuinely asynchronous, so posting every
        # receive and then every send is fine.  NCCL's isend/irecv are not:
        # each enqueues a ncclSend/ncclRecv that is matched and serialized on
        # the stream, so a rank that posts all of its receives first blocks
        # before issuing the sends its peers are waiting for -- symmetric
        # deadlock, every rank stuck, no error.  dist.batch_isend_irecv wraps
        # the whole set in ncclGroupStart/ncclGroupEnd so NCCL matches them
        # together.
        # ⛔ And this is invisible on CPU: gloo tolerates the ungrouped form,
        # so the 36-run rank sweep passed 27/27 on gloo while every cross-node
        # NCCL run hung at the first exchange.  A CPU-only parallel test cannot
        # validate this path.
        # ⛔ batch_isend_irecv is NCCL-only, hence the branch rather than using
        # it unconditionally.
        if dist.get_backend(self.group) == "nccl":
            p2p = [dist.P2POp(fn, t, peer, group=self.group) for fn, t, peer in ops]
            for r in dist.batch_isend_irecv(p2p):
                r.wait()
        else:
            for r in [fn(t, peer, group=self.group, tag=11)
                      for fn, t, peer in ops]:
                r.wait()
        for q, (buf, idx) in recv_bufs.items():
            u[torch.as_tensor(idx, device=u.device)] = buf


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
    cur_in: Optional[torch.Tensor] = None      # feedback(nonlinear): inflow-only flux op
    bn: Optional[torch.Tensor] = None          # feedback(nonlinear): boundary normals
    vD: float = 0.0                            # feedback: contact drift velocity
    nonlinear: bool = False                    # feedback: full-FD Newton (vs affine)
    hmu: float = 1e-3                          # feedback(nonlinear): Newton FD step in dmu


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
        save_terms: bool = False,
        probe_interval: int = 0,
        probe_file: str = "",
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
        self.group = process_grid.get_group("r")
        self.mesh_file = mesh_file
        self.contacts = contacts
        self.save_rho = save_rho
        self.save_terms = save_terms
        self._vk_eps2 = float(vk_eps2)

        self.mesh = load_mesh(mesh_file)
        self._mpi = self.group.size() > 1
        if self._mpi:
            # METIS min-cut partition, renumbered so each rank owns a contiguous
            # block (compact halos + direct checkpoint slices). Keep the
            # permutation so the renumbered solution maps back to the input order.
            self._perm, bounds = partition(self.mesh, self.group)
            self.mesh.EToV = np.asarray(self.mesh.EToV, int)[self._perm]
        else:
            self._perm, bounds = None, None
        g = build_fv_geom(self.mesh, dtype=material.transport_velocity.dtype)
        self.geom = g
        v = material.transport_velocity                       # (Nk, 2)
        self.Nk = v.shape[0]
        self.K = int(g.area.shape[0])
        self._nf = int(g.recon.shape[1])                      # faces/cell: 3 (tri) or 2 (1D)
        self._face_budget_gb = float(os.environ.get("QIMPY_FACE_BUDGET_GB", "3.0"))

        # Spatial decomposition: owned cell block, reconstruction rows (owned +
        # 1-ring), owned-incident edges and the halo exchange (see SpatialDecomp).
        if self._mpi:
            self._decomp = SpatialDecomp(g.nbr.detach().cpu().numpy(), bounds, self.group)
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

        # Face-flux output: fluxes (currents, heat) are emitted on edges as the
        # face-normal flux F = elen * sum_k u_face_k (v_k.n) g_k -- the SAME upwind
        # trace the scheme streams with, so F is the conserved flux (exactly 0
        # through specular walls).  Scalars stay cell-centred.  (Serial only.)
        self._flux_names = list(material.get_flux_names())
        self._flux_g = material.get_flux_weights()           # (Nflux, Nk) or None
        self._do_flux = (self._flux_g is not None) and (self._decomp is None)
        if self._do_flux:
            self._a_int = a_int                              # (Ne, Nk) interior v.n
            eLc = g.eL.detach().cpu().numpy(); eRc = g.eR.detach().cpu().numpy()
            bc = g.bcell.detach().cpu().numpy()
            self._edge_cells = np.concatenate([               # (n_edge, 2), eR=-1 on bnd
                np.stack([eLc, eRc], 1), np.stack([bc, -np.ones_like(bc)], 1)]).astype(np.int64)
            self._edge_normal = np.concatenate([              # (n_edge, 2) out of first cell
                g.en.detach().cpu().numpy(), g.bn.detach().cpu().numpy()])
            self._edge_len = np.concatenate([                 # (n_edge,)
                g.elen.detach().cpu().numpy(), g.blen.detach().cpu().numpy()])
            self._stash_flux = []
        # Deterministic flux assembly (serial): every face of every cell is
        # exactly one (interior-edge, side) or boundary-edge contribution, so
        # divergence = fixed-order sum over the cell's _nf face slots of a
        # concatenated contribution array [wL*uup; wR*uup; wB*uup_b].  This
        # replaces the atomic index_add_ scatter, whose accumulation order is
        # nondeterministic on CUDA (repeated evals differed at ~1 ulp).
        self._slot = None
        if not self._mpi:
            Ne = int(g.eL.shape[0])
            Nb = int(g.bcell.shape[0])
            slot = torch.full((self.K * self._nf,), -1, dtype=torch.long)
            eL = g.eL.cpu(); eR = g.eR.cpu()
            eLF = g.eLF.cpu(); eRF = g.eRF.cpu()
            slot[eLF] = torch.arange(Ne)                  # side L -> row e
            slot[eRF] = torch.arange(Ne) + Ne             # side R -> row Ne+e
            if Nb:
                slot[g.bF.cpu()] = torch.arange(Nb) + 2 * Ne
            if int((slot < 0).sum()):
                raise RuntimeError("deterministic flux slots incomplete")
            self._slot = slot.view(self.K, self._nf).to(rc.device)
            # Persistent contribution buffer (fully overwritten every eval;
            # per-call torch.empty of this GB-scale block fragments small GPUs):
            self._contrib = torch.empty(2 * Ne + Nb, self.Nk, device=rc.device,
                                        dtype=material.transport_velocity.dtype)

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
                # Numerical-rank cutoff scaled to the working precision. An
                # absolute threshold (1e-8) sits below the fp32 modal round-trip
                # floor (~1e-6), so in fp32 it counts ~Nk noise singular values as
                # ghost directions. sqrt(eps) lands safely between the noise floor
                # and the O(1) genuine ghost in both fp32 and fp64 (fp64: ~1.5e-8,
                # matching the old cutoff; fp32: ~3.4e-4).
                tol = float(S[0]) * torch.finfo(v.dtype).eps ** 0.5
                r = int((S > tol).sum())
                self._dl_A = (U[:, :r] * S[:r]).contiguous()     # (Nk, r)
                self._dl_B = Vh[:r].T.contiguous()               # (Nk, r)

        # Ballistic short-circuit: when the material's collision+field operator is
        # identically zero (rates and cyclotron speed both zero) its rho_dot is a
        # no-op, so skip the per-step call entirely -- it otherwise allocates a
        # zero tensor and (via the rates check) forces a GPU->CPU sync each step.
        # ---- dense in-run probe (every probe_interval steps) -------------
        # dt_save is sized by checkpoint cost (rho is ~0.4 GB/frame), so the
        # only V(t) / I(t) record a CLI run leaves behind is one point per
        # checkpoint.  That is far too sparse to watch a transient relax.  This
        # writes a small CSV every probe_interval steps instead: per-contact
        # currents plus the mean chemical potential over each named region.
        # Cost is bounded by evaluating get_cell_scalars on the PROBE CELLS
        # ONLY (the per-cell frame Newton is the expensive part), so a typical
        # 4-region mixer probe every 10 steps is ~1% of a collision step.
        self.probe_interval = int(probe_interval)
        self.probe_file = str(probe_file)
        self._probe_regions = {}
        self._probe_idx = None
        self._probe_header = False
        if self.probe_interval > 0:
            names = self.mesh.cell_regions
            if names is None:
                raise InvalidInputException(
                    f"probe_interval > 0 but {self.mesh_file} defines no"
                    " cell_regions; label the probe cells in the mesh"
                    " generator (regions are geometry, not run config)")
            lo, hi = self._own_start, self._own_stop
            local = np.asarray(names)[lo:hi]
            uniq = sorted({n for n in local.tolist() if n})
            sel_all = np.zeros(hi - lo, dtype=bool)
            for n in uniq:
                m = local == n
                self._probe_regions[n] = m
                sel_all |= m
            idx = np.flatnonzero(sel_all)
            # self._u does not exist yet at this point in __init__; realise the
            # index tensor lazily on the first probe instead.
            self._probe_idx = idx
            pos = {int(g): i for i, g in enumerate(idx)}
            self._probe_regions = {
                n: np.array([pos[int(g)] for g in np.flatnonzero(m)], dtype=int)
                for n, m in self._probe_regions.items()}
            log.info(f"Probing regions {uniq} ({len(idx)} cells) every "
                     f"{self.probe_interval} steps -> {self.probe_file}")

        # Operator-splitting support: TimeEvolution suppresses the collision
        # during the streaming stages when collision_interval > 1, and applies
        # it separately over the longer sub-step via collision_dot below.
        self.collision_enabled = True
        rm = getattr(material, "rates_modal", None)
        ks = getattr(material, "k_speed", 0.0)
        has_ee = hasattr(material, "ee_scattering")   # microscopic e-e: NOT in rates_modal
        self._skip_collision = bool(
            rm is not None and float(rm.abs().sum()) == 0.0 and float(ks) == 0.0
            and not has_ee
        )

        # Optionally fuse the per-step kernels with torch.compile.  Serial: compile
        # the whole spatial RHS (reconstruction + limiter + flux scatter + boundary)
        # so the flux glue fuses into the reconstruction graph and the kernel-launch
        # overhead collapses.  MPI: shapes are dynamic (owned-edge gathers), so only
        # the reconstruction is compiled.
        self._faces_fn = self._faces
        self._srhs_fn = self._spatial_rhs
        # ★ Two reconstruction kernels, and which one wins depends ENTIRELY on
        # whether Inductor gets to fuse it.  `_limited_faces_slot` accumulates
        # the neighbour contributions slot by slot and never builds the
        # (n, Nmax, Nk) tensor; `_limited_faces_gather` builds it and hands it
        # to a batched gemm.  Measured on the production mixer
        # (K=1792, Nk=40320, Nmax=16), whole spatial RHS:
        #     compiled   gather 48.2 ms   slot 28.8 ms   (slot 1.67x FASTER)
        #     eager      gather 140.6 ms  slot 243.5 ms  (slot 1.73x SLOWER)
        # so the choice has to follow the compile flag, not be picked once.
        # QIMPY_LIMITER=gather|slot forces one path, for A/B timing.
        _lim = os.environ.get("QIMPY_LIMITER", "")
        self._limited_faces = (
            self._limited_faces_slot if _lim == "slot"
            else self._limited_faces_gather if _lim == "gather"
            else (self._limited_faces_slot if compile
                  else self._limited_faces_gather))
        if compile:
            # This workload is kernel-launch- and bandwidth-bound (a long chain of
            # small elementwise ops). Benchmarked on a T4: "max-autotune" (kernel
            # fusion + tuning) is fastest; "reduce-overhead" (CUDA graphs) is a touch
            # slower here because the RK2 stages feed changing inputs, so the
            # cudagraph input copies offset the launch-overhead savings. Override
            # with QIMPY_COMPILE_MODE if a workload benefits from a different mode.
            mode = os.environ.get("QIMPY_COMPILE_MODE", "max-autotune")
            # Whole-RHS compilation is fastest when the working set fits; at
            # device-scale Cartesian (K*nf*Nk state ~ GB) Inductor's extra
            # buffers OOM small cards, so compile only the limiter hotspot
            # (85% of the step) there.  Threshold via QIMPY_COMPILE_WHOLE_GB.
            whole_gb = float(os.environ.get("QIMPY_COMPILE_WHOLE_GB", "1.0"))
            state_gb = self.K * self._nf * self.Nk * 8 / 2 ** 30
            try:
                if self._mpi:
                    self._faces_fn = torch.compile(self._faces, mode=mode)
                elif state_gb <= whole_gb:
                    self._srhs_fn = torch.compile(self._spatial_rhs, mode=mode)
                else:
                    self._limited_faces = torch.compile(self._limited_faces,
                                                        mode=mode)
                    log.info(
                        f"compile: limiter-only (face state {state_gb:.1f} GB >"
                        f" {whole_gb:.1f} GB whole-RHS threshold)")
            except Exception:               # older torch / no backend -> eager
                self._faces_fn, self._srhs_fn = self._faces, self._spatial_rhs

        vmax = float(v.norm(dim=1).max())
        dt_local = float(cfl) * float(g.inradius.min()) / max(vmax, 1e-300)
        self.dt_max = all_reduce_scalars(dt_local, dist.ReduceOp.MIN, self.group)

        rho0 = getattr(material, "rho0", None)
        if (self._decomp is None) and checkpoint_in and checkpoint_in.member("rho"):
            cp, path = checkpoint_in                          # warm start (serial)
            self._u = torch.as_tensor(np.array(cp[f"{path}/rho"]),
                                      device=rc.device, dtype=v.dtype)
            if tuple(self._u.shape) != (self.K, self.Nk):
                raise InvalidInputException(
                    f"checkpoint rho shape {tuple(self._u.shape)} != (K, Nk) ="
                    f" ({self.K}, {self.Nk}): k-representation config (n_k /"
                    f" circular / annulus_xi) must match the checkpoint")
        elif rho0 is not None:
            self._u = rho0.flatten().to(rc.device, v.dtype)[None, :].repeat(self.K, 1)
        else:
            self._u = torch.zeros(self.K, self.Nk, device=rc.device, dtype=v.dtype)
        self._stash_t, self._stash_i, self._stash_obs = [], [], []
        self._stash_terms = []   # per-frame (4, K_own, Nr*dim): [a, lin, quad, cub]

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
        # Dense (nw, Nk, Nk) reflection matrix collapses the reflector to one matmul.
        # Measured tradeoff (mixer, A100X-20C): at small Nk the dense matrix is tiny
        # and either path is fast; at large Nk the dense matmul is BANDWIDTH-bound
        # (it re-reads nw*Nk^2 every step: 5.4 GB at M_theta=1024 -> 0.019 s/step)
        # and the vectorized per-step reflector, which touches only the (nw, Nk)
        # wall trace, is faster (0.015 s/step).  For grid materials dense is
        # infeasible outright (nw*Nk^2 ~ PB).  Budget via QIMPY_REFL_BUDGET_GB
        # (default 4 GB) -- large-M modal materials land on the per-step path,
        # which is the right choice there since its harmonic loop was vectorized.
        self._refl_mat = None
        refl_budget = float(os.environ.get("QIMPY_REFL_BUDGET_GB", "4.0")) * 2 ** 30
        if self._reflector is not None:
            dense_gb = self._wall.numel() * self.Nk ** 2 * 8 / 2 ** 30
            if dense_gb * 2 ** 30 <= refl_budget:
                nw = self._wall.numel()
                bn_wall = g.bn[self._wall]
                eye = torch.eye(self.Nk, device=rc.device, dtype=g.area.dtype)
                # Build the per-edge matrix in chunks over wall edges into a
                # preallocated buffer: fully materializing the (Nk, nw, Nk) basis at
                # once OOMs at large Nk (the matrix itself is nw*Nk^2).  Cap the
                # transient at ~200 MB.
                chunk = max(1, int(2.0e8 // (self.Nk * self.Nk * 8)))
                self._refl_mat = torch.empty(nw, self.Nk, self.Nk,
                                             device=rc.device, dtype=g.area.dtype)
                for s in range(0, nw, chunk):
                    cs = min(chunk, nw - s)
                    refl_c = material.get_reflector(bn_wall[s:s + cs])
                    basis_c = eye[:, None, :].expand(self.Nk, cs, self.Nk)
                    self._refl_mat[s:s + cs] = refl_c(basis_c).permute(1, 2, 0)
                log.info(
                    f"Wall reflector: dense matrix ({dense_gb:.2f} GB, "
                    f"{nw} edges x {self.Nk}^2) -> one batched matmul per step")
            else:
                log.info(
                    f"Wall reflector: PER-STEP apply -- dense matrix would need "
                    f"{dense_gb:.1f} GB > QIMPY_REFL_BUDGET_GB "
                    f"({refl_budget / 2 ** 30:.1f} GB). This path is launch-bound "
                    f"for modal materials at large M_theta; raise the budget if "
                    f"the dense matrix fits GPU memory.")

        def allreduce(x: float) -> float:
            return (all_reduce_scalars(x, dist.ReduceOp.SUM, self.group)
                    if self._mpi else x)

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
            # nonlinear: for feedback (I_set/floating) contacts it selects the
            # Newton-on-exact-FD level solve; for fixed contacts it passes
            # through to the representation's contactor (exact reservoir vs its
            # linearization).  None = each contactor's own default.
            nl_key = params.pop("nonlinear", None)
            nonlinear = bool(nl_key)
            vD = float(params.get("vD", 0.0))
            if floating or (i_set is not None):
                # Feedback ghost is affine in dmu: g(dmu) = dmu*unit + drift, so the
                # unit is d(ghost)/d(dmu).  Take it as the LINEAR slope from a small
                # dmu: exact for the (linear-in-dmu) delta-k contactor, and the correct
                # df0/dmu for the Cartesian full-f Fermi-Dirac -- where dmu=1 would
                # saturate the FD and inject current into the inert empty high-energy
                # tail instead of the Fermi surface (no drive at all).
                bn_ci = g.bn[ci]
                eps = 0.1 * float(getattr(material, "T_temp", 1.0))
                unit = material.get_contactor(bn_ci, dmu=eps)(0.0) / eps
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
                    target=(0.0 if floating else float(i_set)),
                    cur_in=cur_in, bn=bn_ci, vD=vD, nonlinear=nonlinear,
                    hmu=0.01 * float(getattr(material, "T_temp", 1.0))))
            else:
                if nl_key is not None:
                    params["nonlinear"] = bool(nl_key)
                ghost = material.get_contactor(g.bn[ci], **params)(0.0)
                self._contacts.append(_Contact(
                    name=nm, idx=ci, cur=cur, kind="fixed", ghost=ghost))

    def _limited_faces_slot(self, uc: torch.Tensor, u: torch.Tensor,
                            nbr: torch.Tensor, recon: torch.Tensor
                            ) -> torch.Tensor:
        """Venkatakrishnan-limited face values for a set of cells.

        ``uc`` (n, Nk) cell averages, ``u`` (K, Nk) the full state, ``nbr``
        (n, Nmax) their vertex-neighbour cell indices, ``recon`` (n, 3, Nmax) the
        fused gradient->face operator. Smooth (differentiable) limiter -> clean
        steady-state convergence; per face, with increment d = u_face-u and
        same-sign headroom D1 (to the neighbor max/min):
            phi = (D1^2 + 2 D1 d + e) / (D1^2 + D1 d + 2 d^2 + e),  e = vk_eps2,
        capped at 1 (never amplify the LSQ gradient) and min-ed over the 3 faces.

        ★ The neighbour values are accumulated SLOT BY SLOT rather than gathered
        into an (n, Nmax, Nk) tensor.  That tensor is 9.2 GB at device scale and
        exists only to be consumed three times (the einsum, amax, amin); summing
        over the slot index instead needs the same total traffic but never
        materialises it, and leaves d/hi/lo as one fusable pass.  Measured on the
        production mixer (K=1792, Nk=40320, Nmax=16): compiled 52.5 -> 18.5 ms,
        a 2.84x speedup, agreeing with the gathered form to 2.9e-16.
        ⛔ EAGER it is 87% SLOWER (16 Python iterations, ~5 kernels each) -- this
        form is only worth it under torch.compile, which is the shipped default.
        """
        d = torch.zeros(uc.shape[0], recon.shape[1], uc.shape[1],
                        device=uc.device, dtype=uc.dtype)
        hi_n = uc
        lo_n = uc
        for gi in range(nbr.shape[1]):
            ung = u[nbr[:, gi]]                               # (n, Nk)
            # ⛔ Accumulate recon * (ung - uc), NOT recon * ung with a single
            # `- uc sum_g recon` at the end.  The two are algebraically equal,
            # but the deferred form subtracts two large nearly-equal sums and
            # loses the cancellation: measured 1.7e-11 relative on the full
            # rho_dot against 2.9e-16 for this form, which would break the
            # 1e-12 conservation assertions.  Differencing per slot costs one
            # extra (n, Nk) subtract and keeps the accuracy of the gathered
            # einsum exactly.
            d = d + recon[:, :, gi].unsqueeze(-1) * (ung - uc).unsqueeze(1)
            hi_n = torch.maximum(hi_n, ung)
            lo_n = torch.minimum(lo_n, ung)
        hi = (hi_n - uc)[:, None]                             # headroom up   (>= 0)
        lo = (lo_n - uc)[:, None]                             # headroom down (<= 0)
        D1 = torch.where(d >= 0, hi, lo)                      # same sign as d
        e = self._vk_eps2
        # Denominator is D1^2 + D1 d + 2 d^2 + e >= 2 d^2 > 0 for d != 0 in exact
        # arithmetic, but in fp32 it can underflow to a subnormal/zero at locally
        # flat cells (d a denormal-tiny roundoff with D1 == 0), giving 0/0 = NaN.
        # Guard on the denominator being a normal float rather than on d != 0:
        # where it isn't, the cell is flat and the limiter is 1 (no limiting).
        num = D1 * D1 + 2 * D1 * d + e
        den = D1 * D1 + D1 * d + 2 * d * d + e
        phi = torch.where(
            den > torch.finfo(d.dtype).tiny,
            (num / den).clamp(max=1.0),
            torch.ones_like(d),
        ).amin(1)[:, None]                                    # (n, 1, Nk)
        return uc[:, None] + phi * d

    def _limited_faces_gather(self, uc: torch.Tensor, u: torch.Tensor,
                              nbr: torch.Tensor, recon: torch.Tensor
                              ) -> torch.Tensor:
        """The same limiter, but gathering (n, Nmax, Nk) and using one batched
        gemm.  Faster EAGER (140.6 ms vs 243.5 ms for the whole RHS), slower
        compiled (48.2 vs 28.8) -- selected in __init__ by the compile flag.
        Agrees with the slot form to 2.9e-16."""
        un = u[nbr]                                           # (n, Nmax, Nk)
        d = torch.einsum("nfg,ngc->nfc", recon, un - uc[:, None])
        hi = (torch.maximum(uc, un.amax(1)) - uc)[:, None]
        lo = (torch.minimum(uc, un.amin(1)) - uc)[:, None]
        D1 = torch.where(d >= 0, hi, lo)
        e = self._vk_eps2
        num = D1 * D1 + 2 * D1 * d + e
        den = D1 * D1 + D1 * d + 2 * d * d + e
        phi = torch.where(
            den > torch.finfo(d.dtype).tiny,
            (num / den).clamp(max=1.0),
            torch.ones_like(d),
        ).amin(1)[:, None]
        return uc[:, None] + phi * d

    def _faces(self, u: torch.Tensor) -> torch.Tensor:
        """Reconstructed face values, (K, n_face, Nk). Serial reconstructs every
        cell; under decomposition only the rows this rank needs (owned + 1-ring)
        are filled, the rest left zero (their faces are never read).

        Chunked over cells so the peak transient stays within
        ``_face_budget_gb``.  ``_limited_faces`` no longer materialises the
        (rows, Nmax, Nk) neighbour tensor, so the budget now bounds the
        per-slot working set rather than one huge gather; the chunking is kept
        because the (n, 3, Nk) accumulators still scale with the row count.
        Small-Nk runs take the single-shot path (chunk >= K)."""
        g = self.geom
        rows = self._R                                       # None (serial) or owned+1ring
        n_rows = self.K if rows is None else int(rows.shape[0])
        per_row = g.nbr.shape[1] * self.Nk * 8               # neighbour-gather bytes/cell
        chunk = max(1, int(self._face_budget_gb * (2 ** 30) / max(per_row, 1)))
        if rows is None and chunk >= self.K:
            return self._limited_faces(u, u, g.nbr, g.recon)          # single-shot
        idx = torch.arange(self.K, device=u.device) if rows is None else rows
        # Serial chunked path fills every row -> skip the (K, nf, Nk) zero-fill
        # (3.6 GB/eval at device-scale Cartesian); MPI leaves unused rows unread
        # but must keep them defined, so retain zeros there.
        if rows is None:
            if not hasattr(self, "_uf_buf"):
                self._uf_buf = torch.empty(self.K, self._nf, self.Nk,
                                           device=u.device, dtype=u.dtype)
            uf = self._uf_buf
        else:
            uf = u.new_zeros(self.K, self._nf, self.Nk)
        for lo in range(0, n_rows, chunk):
            ci = idx[lo:lo + chunk]
            uf[ci] = self._limited_faces(u[ci], u, g.nbr[ci], g.recon[ci])
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
        elif self._reflector is not None:                         # sparse reflector (large Nk)
            uP[self._wall] = self._reflector(uMb[self._wall][None])[0]
        for c in self._contacts:
            if c.kind == "fixed":
                uP[c.idx] = c.ghost.to(uP)
            else:
                # Feedback: solve the contact level so the net current hits target.
                # I_net = num_out + inflow(ghost); num_out is the interior outflow.
                num = float((c.cur_out * uMb[c.idx]).sum())
                if self._mpi:
                    num = all_reduce_scalars(num, dist.ReduceOp.SUM, self.group)
                if c.nonlinear:
                    # Full physical contact: ghost is the exact Pauli-bounded Fermi-
                    # Dirac deviation FD(mu0+dmu)-f0, NONLINEAR in dmu.  Newton-solve
                    # dmu (warm-started from last step) so I_net(dmu)=target -- captures
                    # the contact nonlinearity the affine linear-response feedback drops.
                    dmu = c.level
                    for _ in range(8):
                        inflow = self.material.get_contactor(c.bn, dmu=dmu, vD=c.vD)(0.0)
                        i0 = float((c.cur_in * inflow).sum())
                        gh = self.material.get_contactor(c.bn, dmu=dmu + c.hmu, vD=c.vD)(0.0)
                        ih = float((c.cur_in * gh).sum())
                        if self._mpi:
                            i0 = all_reduce_scalars(i0, dist.ReduceOp.SUM, self.group)
                            ih = all_reduce_scalars(ih, dist.ReduceOp.SUM, self.group)
                        dIdmu = (ih - i0) / c.hmu
                        if abs(dIdmu) < 1e-300:
                            break
                        d = (num + i0 - c.target) / dIdmu
                        dmu = dmu - d
                        if abs(d) <= 1e-6 * (abs(dmu) + c.hmu):
                            break
                    c.level = dmu
                    uP[c.idx] = self.material.get_contactor(
                        c.bn, dmu=dmu, vD=c.vD)(0.0).to(uP)
                else:
                    c.level = (num + c.base - c.target) / c.den
                    uP[c.idx] = (c.level * c.unit + c.drift).to(uP)
        return uP

    def _spatial_rhs(self, u: torch.Tensor, t: float) -> torch.Tensor:
        g = self.geom
        u = self._dealias(u)                                  # de-alias (fused into graph)
        uf = self._faces_fn(u).reshape(-1, self.Nk)           # (K*3, Nk)
        dudt = None if self._slot is not None else torch.zeros_like(u)
        e = self._eloc                                        # owned-incident edges
        eL = g.eL if e is None else g.eL[e]
        eR = g.eR if e is None else g.eR[e]
        eLF = g.eLF if e is None else g.eLF[e]
        eRF = g.eRF if e is None else g.eRF[e]
        maskL = self._maskL if e is None else self._maskL[e]
        wL = self._wL if e is None else self._wL[e]
        wR = self._wR if e is None else self._wR[e]
        if self._slot is not None:
            # Deterministic fixed-order assembly: gather each cell's _nf face
            # contributions and sum in face order (no atomics).  Contributions
            # are written in place, chunked over edges: elementwise ops are
            # chunk-invariant (bit-exact) and the peak transient stays small
            # (full uup + cat copies OOM device-scale Cartesian on 20 GB).
            Ne = int(eLF.shape[0])
            Nb = int(g.bcell.numel())
            contrib = self._contrib
            ecs = max(1, int(self._face_budget_gb * (2 ** 30)
                             / max(6 * self.Nk * 8, 1)))
            for s0 in range(0, Ne, ecs):
                sl = slice(s0, min(s0 + ecs, Ne))
                uupc = torch.where(maskL[sl], uf[eLF[sl]], uf[eRF[sl]])
                torch.mul(wL[sl], uupc, out=contrib[s0:s0 + uupc.shape[0]])
                torch.mul(wR[sl], uupc,
                          out=contrib[Ne + s0:Ne + s0 + uupc.shape[0]])
                del uupc
            if Nb:
                uMb = uf[g.bF]
                uup_b = torch.where(self._maskB, uMb, self._exterior(uMb, t))
                torch.mul(self._wB, uup_b, out=contrib[2 * Ne:])
                del uup_b
            del uf
            acc = contrib[self._slot[:, 0]]
            for f in range(1, self._nf):
                acc = acc + contrib[self._slot[:, f]]
            return acc
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

    def rho_dot(self, rho: TensorList, t: float) -> TensorList:
        u = rho[0]
        if self._decomp is not None:
            self._decomp.exchange(u)                          # fill halo ghost rows
        out = self._srhs_fn(u, t)                             # de-alias + spatial RHS (fused)
        if self.collision_enabled and not self._skip_collision:   # ballistic: exactly 0
            # collision = from_modes(-rates * to_modes(.)); its to_modes already
            # annihilates the ghost, so the raw (un-de-aliased) u is exact here.
            lo, hi = self._own_start, self._own_stop
            out[lo:hi] = out[lo:hi] + self.material.rho_dot(u[lo:hi], t, id(self))
        if self._owned_mask is not None:
            out = out * self._owned_mask
        return TensorList([out])

    def collision_dot(self, rho: TensorList) -> TensorList:
        """The material collision term ALONE, with no spatial RHS.

        Used by operator splitting (``collision_interval > 1``).  The collision
        is local to a cell, so unlike :meth:`rho_dot` this needs no halo
        exchange, and no de-aliasing either -- the representation's
        ``to_modes`` already annihilates the rank-r ghost.
        """
        u = rho[0]
        out = torch.zeros_like(u)
        if not self._skip_collision:
            lo, hi = self._own_start, self._own_stop
            out[lo:hi] = self.material.rho_dot(u[lo:hi], 0.0, id(self))
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
            out[c.name] = (all_reduce_scalars(I, dist.ReduceOp.SUM, self.group)
                           if self._mpi else I)
        return out

    def maybe_probe(self, i_step: int, t: float) -> None:
        """Append one probe row every ``probe_interval`` steps (else a no-op)."""
        if self.probe_interval <= 0 or self._probe_idx is None:
            return
        if i_step % self.probe_interval:
            return
        if not torch.is_tensor(self._probe_idx):
            self._probe_idx = torch.as_tensor(
                np.asarray(self._probe_idx), dtype=torch.long,
                device=self._u.device)
        with torch.no_grad():
            cur = self.contact_currents(t)
            names = self.material.get_cell_scalar_names()
            i_mu = names.index("chemical_potential")
            sc = self.material.get_cell_scalars(
                self._u[self._probe_idx], t)[:, i_mu].detach().cpu().numpy()
        row = {"i_step": i_step, "t": t}
        row.update({f"I_{k}": v for k, v in cur.items()})
        for name, sel in self._probe_regions.items():
            if self._mpi:
                tot = all_reduce_scalars(float(sc[sel].sum()), dist.ReduceOp.SUM, self.group)
                cnt = all_reduce_scalars(int(sel.size), dist.ReduceOp.SUM, self.group)
            else:
                tot, cnt = float(sc[sel].sum()), int(sel.size)
            row[f"V_{name}"] = tot / max(cnt, 1)
        if self._mpi and dist.get_rank(self.group):
            return
        new = not self._probe_header
        with open(self.probe_file, "a") as fh:
            if new:
                fh.write(",".join(row.keys()) + "\n")
                self._probe_header = True
            fh.write(",".join(f"{v!r}" if isinstance(v, int) else f"{v:.10e}"
                              for v in row.values()) + "\n")

    def contact_potentials(self) -> dict[str, float]:
        """Self-adjusting level of each feedback contact, from the last evaluation."""
        return {c.name: c.level for c in self._contacts if c.kind != "fixed"}

    def update_stash(self, i_step: int, t: float) -> None:
        ce = getattr(getattr(self.material, "representation", None),
                     "check_envelope", None)
        if ce is not None:
            ce(self._u[self._own_start:self._own_stop])
        # Stash cell-centred scalar fields for this rank's owned cells.
        u_own = self._u[self._own_start:self._own_stop]
        obs = self.material.get_cell_scalars(u_own, t)
        self._stash_i.append(i_step)
        self._stash_t.append(t)
        self._stash_obs.append(obs.detach().cpu().numpy())
        if self._do_flux:
            # Face-normal fluxes on every edge, from the upwind face trace.
            uf = self._faces(self._u).reshape(-1, self.Nk)
            g = self.geom
            uup_i = torch.where(self._maskL, uf[g.eLF], uf[g.eRF])      # (Ne, Nk)
            uMb = uf[g.bF]
            uup_b = torch.where(self._maskB, uMb, self._exterior(uMb, t))   # (Nb, Nk)
            Fi = g.elen[None, :] * (self._flux_g @ (uup_i * self._a_int).t())  # (Nflux, Ne)
            Fb = g.blen[None, :] * (self._flux_g @ (uup_b * self._a_bnd).t())  # (Nflux, Nb)
            F = torch.cat([Fi, Fb], dim=1).t()                         # (n_edge, Nflux)
            self._stash_flux.append(F.detach().cpu().numpy())
        if self.save_terms and hasattr(self.material, "ee_scattering"):
            # Per-(m,l) collision breakdown, computed WARM in the evolution loop
            # (the same context as the in-run apply) to dodge the cold
            # standalone-apply slow path. a is the modal distribution; lin/quad/
            # cub are the linear, quadratic-in-deltaf and cubic-in-deltaf parts
            # of the e-e operator. All are modal, flattenable to (Nr, dim).
            with torch.no_grad():
                a = self.material.to_modes(u_own)            # (K_own, Nr*dim)
                lin, quad, cub = self.material.ee_scattering.a_dot_breakdown(a)
                self._stash_terms.append(np.stack([
                    a.detach().cpu().numpy(),
                    lin.detach().cpu().numpy(),
                    quad.detach().cpu().numpy(),
                    cub.detach().cpu().numpy(),
                ], axis=0))                                  # (4, K_own, Nr*dim)

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        g = self.geom
        names = self.material.get_cell_scalar_names()
        # NOTE: checkpoint attrs are fed back as constructor kwargs on restart
        # (qimpy convention: attrs == constructor params), so only real
        # constructor arguments may be written here.
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
        if self._do_flux and n_stash:
            # Flux observables live on edges: geometry + face-normal flux per frame.
            saved += [
                cp_path.write("edge_cells", torch.from_numpy(self._edge_cells)),
                cp_path.write("edge_normal", torch.from_numpy(self._edge_normal)),
                cp_path.write("edge_len", torch.from_numpy(self._edge_len)),
                "fv_edge_flux",
            ]
            cp_path.write_str("flux_names", ",".join(self._flux_names))
            n_edge = self._edge_len.shape[0]
            CheckpointPath(checkpoint, path).create_dataset(
                "fv_edge_flux", (n_stash, n_edge, len(self._flux_names)), np.float64)
            if checkpoint is not None:
                checkpoint.write_slice(checkpoint[f"{path}/fv_edge_flux"],
                                       (0, 0, 0), torch.from_numpy(np.stack(self._stash_flux)))
        if self.save_rho:
            # Raw per-cell state (n_cells, n_channels), for exact restart /
            # steady-state warm start. Each rank writes its owned cell block.
            u_own = self._u[self._own_start:self._own_stop].detach().cpu().numpy()
            CheckpointPath(checkpoint, path).create_dataset(
                "rho", (self.K, self.Nk), u_own.dtype)
            if checkpoint is not None:
                checkpoint.write_slice(checkpoint[f"{path}/rho"],
                                       (self._own_start, 0),
                                       torch.from_numpy(u_own))
            saved.append("rho")
        if self.save_terms and len(self._stash_terms):
            # Per-(m,l) collision breakdown stacked over saved frames:
            # (n_stash, 4, K, Nr*dim) with channel order [a, lin, quad, cub].
            # Reshape the last axis to (Nr, dim) offline for the radial/angular
            # (l, m) contributions. dim = 2*M_theta+1 (m = -M..M).
            fs = self.material
            Nr_m = int(fs.Nr); dim_m = int(fs.angular.dim)
            n_modal = Nr_m * dim_m
            # (Nr, dim) as a dataset, NOT attrs -- attrs round-trip into
            # constructor kwargs on restart and these are not constructor args.
            cp_path.write("terms_shape",
                          torch.tensor([Nr_m, dim_m], dtype=torch.int64))
            cp_path.write_str("terms_channels", "a,lin,quad,cub")
            terms_own = np.stack(self._stash_terms)          # (n_stash,4,K_own,n_modal)
            CheckpointPath(checkpoint, path).create_dataset(
                "fv_terms", (n_stash, 4, self.K, n_modal), terms_own.dtype)
            if checkpoint is not None:
                checkpoint.write_slice(checkpoint[f"{path}/fv_terms"],
                                       (0, 0, self._own_start, 0),
                                       torch.from_numpy(terms_own))
            saved.append("fv_terms")
        self._stash_t, self._stash_i, self._stash_obs = [], [], []
        self._stash_terms = []
        if self._do_flux:
            self._stash_flux = []
        return saved
