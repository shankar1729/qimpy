from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property

from scipy.spatial import KDTree
from scipy.sparse import csr_array
import numpy as np
import torch
import torch.distributed as dist

from qimpy import rc, log, TreeNode
from qimpy.io import CheckpointPath, CheckpointContext
from qimpy.mpi import TaskDivisionCustom
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
    cell_permutation: np.ndarray | None  #: Reordering of cells relative to the source

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
        self.cell_permutation = None
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
        self.cell_permutation = cp.read_optional_np("cell_permutation")

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
        if self.cell_permutation is not None:
            saved_list.append(cp.write("cell_permutation", self.cell_permutation))
        return saved_list

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        return self.save(cp_path)

    @property
    def is1D(self) -> bool:
        return self.cells.shape[1] == 3  # cells coontain d + 1 vertices, 1 region index

    def partition_cells(self, group: dist.ProcessGroup) -> CellDivision:
        """Optimally partition the mesh cells for communication over `group`.
        This reorders `cells` such that contiguous ranges of cells can be owned by
        each process within `group`, and returns a helper class to track the
        division and communicate ghost cells across processes.
        Correspondingly, `cell_permutation` tracks any reordering of the cells."""
        size = group.size()
        if size == 1:
            counts = np.array([len(self.cells)])  # no partitioning required
            return CellDivision(self, counts, group)

        # Compute partition on overall head process:
        # (Partition must be stable across process grid, not just the "r" dimension.)
        partition = np.empty(len(self.cells), dtype=np.int32)
        if rc.is_head:
            try:
                import pymetis

                adj = self.cell_adjacency_vertex
                csr_adj = pymetis.CSRAdjacency(adj.indptr, adj.indices)
                partition = np.asarray(pymetis.part_graph(size, csr_adj)[1], np.int32)
            except ImportError:
                # Fallback: divide equally along longest axis:
                centroids = self.vertices[self.cells[:, :-1]].mean(axis=1)
                long_axis = np.ptp(centroids, axis=0).argmin()
                order = np.argsort(centroids[:, long_axis], kind="stable")
                partition[order] = np.minimum(
                    (np.arange(len(order)) * size) // len(order), size - 1
                )

        # Synchronize partition across all processes:
        partition_t = torch.from_numpy(partition).to(rc.device)
        dist.broadcast(partition_t, src=0)  # on world, not just on group
        partition = partition_t.to(rc.cpu).numpy()

        # Determine permutation and bounds:
        permutation = np.argsort(partition, kind="stable")  # group cells by rank
        counts = np.bincount(partition, minlength=size)

        # Apply and track permutation:
        self.cells = self.cells[permutation]
        if self.cell_permutation is None:
            self.cell_permutation = permutation
        else:
            self.cell_permutation = self.cell_permutation[permutation]

        # Invalidate cached maps involving cells:
        self.__dict__.pop("cell_adjacency_vertex", None)
        self.__dict__.pop("edge_maps", None)
        return CellDivision(self, counts, group)

    @cached_property
    def cell_adjacency_vertex(self) -> csr_array:
        """Adjacency matrix of cells that share a vertex."""
        cells = self.cells[:, :-1]
        cell_to_vertex = csr_array(
            (
                np.ones_like(cells).flatten(),
                (np.repeat(np.arange(len(cells)), cells.shape[1]), cells.flatten()),
            ),
        )
        return make_adjacency(cell_to_vertex)

    @cached_property
    def cell_adjacency_edge(self) -> csr_array:
        """Adjacency matrix of cells that share an edge."""
        _, interior, _ = self.edge_maps
        cell_pairs = interior[..., 0]
        i_cell, j_cell = np.vstack((cell_pairs, cell_pairs[:, ::-1])).T  # both orders
        n_cells = len(self.cells)
        return csr_array((np.ones_like(i_cell), (i_cell, j_cell)), (n_cells, n_cells))

    @cached_property
    def edge_maps(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interior and boundary edges, mapped to cells.

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

            if vertex_map := self.lattice_vertex_map() is None:
                edges_weld = edges
                boundary_edges_weld = self.boundary_edges[:, :2]
                n_vertices_weld = len(self.vertices)
            else:
                # Identify vertices separated by lattice vectors:
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

    def lattice_vertex_map(self) -> np.ndarray | None:
        """Get mapping of vertices to reduced set under periodic boundary conditions,
        if any. Returns None if no periodicity."""
        if self.is1D or self.lattice_vectors is None:
            return None
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
        return np.unique(lowest_equivalent_index, return_inverse=True)[1]

    def get_fv_geom(self, *, dtype: torch.dtype = torch.float64) -> FVGeom:
        """Initialize finite-volume parameters for mesh."""
        # Compute cell area and centroid:
        cells = self.cells[:, :-1]  # ignore region ID
        n_cells = len(cells)
        cell_vertices = self.vertices[cells]
        if self.is1D:
            area = np.linalg.norm(cell_vertices[:, 1] - cell_vertices[:, 0], axis=-1)
        else:
            area = 0.5 * np.linalg.det(cell_vertices[:, 1:] - cell_vertices[:, :1])
            if np.any(area <= 0.0):
                raise ValueError("Areas must be positive (counter-clockwise triangles)")
        centroid = cell_vertices.mean(axis=1)

        # Identify interior and boundary edges:
        edges, interior, boundary = self.edge_maps
        (kL, fL), (kR, fR) = np.transpose(interior, (1, 2, 0))
        bk, bf, bmark = boundary.T

        # Compute edge properties:
        edge_vertices = self.vertices[edges]
        edge_offset = (
            edge_vertices.mean(axis=2) - centroid[:, None]
        )  # cell to edge center
        if self.is1D:
            edge_lengths = np.ones_like(edge_offset[..., 0])
            edge_normals = edge_offset / np.linalg.norm(
                edge_offset, axis=-1, keepdims=True
            )
            inradius = area  # as used for dt determination with Courant condition
        else:
            edge_vectors = edge_vertices[:, :, 1] - edge_vertices[:, :, 0]
            edge_lengths = np.linalg.norm(edge_vectors, axis=2)
            edge_normals = (
                np.stack([edge_vectors[..., 1], -edge_vectors[..., 0]], axis=2)
                / edge_lengths[..., None]
            )
            inradius = area / (0.5 * edge_lengths.sum(axis=1))

        # Inverse-distance-weighted least-squares gradient operator per cell:
        adjacency = self.cell_adjacency_vertex
        neighbor_indices = adjacency.indices
        index_ranges = adjacency.indptr
        Nmax = np.diff(index_ranges).max()
        nbr = np.arange(n_cells)[:, None].repeat(Nmax, axis=1)  # pad slots with self
        grad_op = np.zeros((n_cells, 2, Nmax))
        Nmin = 1 if self.is1D else 2
        for i in range(n_cells):
            js = neighbor_indices[index_ranges[i] : index_ranges[i + 1]]
            if len(js) < Nmin:
                continue
            nbr[i, : len(js)] = js
            D = centroid[js] - centroid[i]
            w = 1.0 / np.linalg.norm(D, axis=1)
            grad_op[i, :, : len(js)] = np.linalg.pinv(w[:, None] * D) * w[None, :]

        # Fuse gradient + centroid -> face offsets to a single matrix multiply:
        recon = np.einsum("kfx, kxg -> kfg", edge_offset, grad_op)

        def t(a, long=False):
            return torch.tensor(
                np.ascontiguousarray(a),
                device=rc.device,
                dtype=torch.long if long else dtype,
            )

        n_cell_edges = 2 if self.is1D else 3
        return FVGeom(
            area=t(area),
            inradius=t(inradius),
            centroid_np=centroid,
            eL=t(kL, long=True),
            eR=t(kR, long=True),
            eLF=t(kL * n_cell_edges + fL, long=True),
            eRF=t(kR * n_cell_edges + fR, long=True),
            en=t(edge_normals[kL, fL]),
            elen=t(edge_lengths[kL, fL]),
            bcell=t(bk, long=True),
            bF=t(bk * n_cell_edges + bf, long=True),
            bmark=t(bmark, long=True),
            bn=t(edge_normals[bk, bf]),
            blen=t(edge_lengths[bk, bf]),
            nbr=t(nbr, long=True),
            recon=t(recon),
        )


@dataclass
class FVGeom:
    """Finite-volume geometry parameters for mesh."""

    area: torch.Tensor  #: Cell areas (interval lengths in 1D)
    inradius: torch.Tensor  #: Corresponding in-radii for time step determination
    centroid_np: np.ndarray  #: Cell centroids
    eL: torch.Tensor  #: First cell index for interior edges
    eR: torch.Tensor  #: Second cell index for interior edges
    eLF: torch.Tensor  #: Flattened first cell-edge index for interior edges
    eRF: torch.Tensor  #: Flattened second cell-edge index for interior edges
    en: torch.Tensor  #: Outward normal directions for each edge (in first cell)
    elen: torch.Tensor  #: Edge lengths
    bcell: torch.Tensor  #: Cell index for boundary edges
    bF: torch.Tensor  #: Flattened cell-edge index for boundary edges
    bmark: torch.Tensor  #: Boundary region IDs for boundary edges
    bn: torch.Tensor  #: Outward normal directions for boundary edges
    blen: torch.Tensor  #: Boundary edge lengths
    nbr: torch.Tensor  #: Neighbor-indices by cell (n_cells, Nmax)
    recon: torch.Tensor  #: Face-increment  operator (n_cells, 3, Nmax)


def make_adjacency(x_to_y: csr_array) -> csr_array:
    """Construct adjacency matrix of x based on shared property y,
    given the mapping from x to y."""
    n_x = x_to_y.shape[0]
    adjacency_coo = (x_to_y @ x_to_y.T).tocoo()
    row = adjacency_coo.row
    col = adjacency_coo.col
    sel = np.where(row != col)[0]  # ignore trivial diagonals
    adjacency = csr_array((np.ones_like(sel), (row[sel], col[sel])), (n_x, n_x))
    adjacency.sort_indices()
    return adjacency


class CellDivision(TaskDivisionCustom):
    """Divide cells and communicate ghosts between processes."""

    def __init__(
        self, mesh: Mesh, counts: np.ndarray, group: dist.ProcessGroup
    ) -> None:
        super().__init__(n_mine=counts[group.rank()], group=group)
        self.group = group
        adj_vert = mesh.cell_adjacency_vertex
        adj_edge = mesh.cell_adjacency_edge

        def ring(cells: np.ndarray, adj: csr_array) -> np.ndarray:
            """1-ring vertex closure of a set of cells (cells + their neighbours)."""
            v = np.zeros(self.n_tot)
            v[cells] = 1
            v += adj @ v
            return np.where(v)[0]

        owned = np.arange(self.i_start, self.i_stop)
        ring1 = ring(owned, adj_edge)  # cells to reconstruct (+edge-neighbors)
        ring2 = ring(ring1, adj_vert)  # cells accessed overall (+vertex-neighbors)
        self.recon_rows = ring1
        ghosts = ring2[~self.belongs(self.i_proc, ring2)]

        # Plan communications:
        recv: list[np.ndarray] = []
        send: list[np.ndarray] = []
        for j_proc in range(self.n_procs):
            if j_proc == self.i_proc:
                recv.append(np.empty(0, dtype=int))
                send.append(np.empty(0, dtype=int))
            else:
                # Receive each ghost from its owning rank:
                recv.append(ghosts[self.belongs(j_proc, ghosts)])
                # Send the owned cells that other ranks need:
                owned_q = np.arange(self.n_prev[j_proc], self.n_prev[j_proc + 1])
                ring2_q = ring(ring(owned_q, adj_edge), adj_vert)
                ghost_q = ring2_q[~self.belongs(j_proc, ring2_q)]
                send.append(owned[np.isin(owned, ghost_q)])
        self.recv_index = torch.from_numpy(np.concatenate(recv)).to(rc.device)
        self.send_index = torch.from_numpy(np.concatenate(send)).to(rc.device)
        self.recv_counts = [len(idx) for idx in recv]
        self.send_counts = [len(idx) for idx in send]

    def exchange(self, u: torch.Tensor) -> None:
        """Fill this rank's ghost rows of ``u`` (K, Nk) with their owners' values."""
        send_buf = u[self.send_index].contiguous()
        recv_buf = torch.empty(
            (len(self.recv_index), u.shape[1]), dtype=u.dtype, device=u.device
        )
        dist.all_to_all_single(
            recv_buf, send_buf, self.recv_counts, self.send_counts, group=self.group
        )
        u[self.recv_index] = recv_buf
