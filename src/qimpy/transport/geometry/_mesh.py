from __future__ import annotations

from scipy.sparse import csr_array
import numpy as np
import torch
import torch.distributed as dist
import pymetis

from qimpy import rc, TreeNode
from qimpy.io import CheckpointPath, CheckpointContext
from qimpy.mpi import TaskDivisionCustom
from . import mesh_source


class Mesh(TreeNode):
    """Definition of mesh geometry (dimension, d = 2 or 1)"""

    source: mesh_source.Options  #: Method by which mesh is constructed
    vertices: np.ndarray  #: vertex coordinates (Nv, 2)
    cells: np.ndarray  #: vertex indices and region ID in cells (Nc, d+2)
    region_names: list[str]  #: names of interior regions (for cell IDs >= 0)
    boundary_faces: np.ndarray  #: sorted vertex indices, ID of boundary faces (Nb, d+1)
    boundary_names: list[str]  #: names of boundary regions (for face IDs >= 0)
    cell_permutation: np.ndarray | None  #: Reordering of cells relative to the source
    fields: dict[str, torch.Tensor]  #: Spatial variation / sweep of material parameters

    # Calculated members:
    cell_division: CellDivision  #: split of cells over processes
    face_division: FaceDivision  #: split of faces over processes
    faces: np.ndarray  #: Vertex indices by cell and face (Nc, d+1, d)
    face_map: np.ndarray  #: Pair of cell and face index within cell by face (Nf, 2, 2)

    def __init__(
        self,
        *,
        group: dist.ProcessGroup | None = None,
        file: mesh_source.File | dict | None = None,
        line: mesh_source.Line | dict | None = None,
        rect: mesh_source.Rect | dict | None = None,
        triangulate: mesh_source.Triangulate | dict | None = None,
        parameter_grid: mesh_source.ParameterGrid | dict | None = None,
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
        parameter_grid
            :yaml:`Material dynamics batched over parameters (no spatial transport).`
            Only one source for the mesh must be specified.
        """
        super().__init__()
        self.cell_permutation = None
        self.fields = {}
        self.add_child_one_of(
            "source",
            checkpoint_in,
            TreeNode.ChildOptions("file", mesh_source.File, file, mesh=self),
            TreeNode.ChildOptions("line", mesh_source.Line, line, mesh=self),
            TreeNode.ChildOptions("rect", mesh_source.Rect, rect, mesh=self),
            TreeNode.ChildOptions(
                "triangulate", mesh_source.Triangulate, triangulate, mesh=self
            ),
            TreeNode.ChildOptions(
                "parameter_grid", mesh_source.ParameterGrid, parameter_grid, mesh=self
            ),
            have_default=False,
        )
        if checkpoint_in:
            self.load(checkpoint_in)

        self._initialize_face_maps()
        self.cell_division = self._partition_cells(group)
        self.face_division = FaceDivision(self.face_map, self.cell_division)

    def load(self, cp: CheckpointPath) -> None:
        self.vertices = cp.read_np("vertices")
        self.cells = cp.read_np("cells")
        self.region_names = cp.read_str_list("region_names")
        self.boundary_faces = cp.read_np("boundary_faces")
        self.boundary_names = cp.read_str_list("boundary_names")
        self.cell_permutation = cp.read_optional_np("cell_permutation")
        if cp_fields := cp.member("fields"):
            checkpoint, path = cp_fields
            assert checkpoint is not None
            for field_name in checkpoint[path].keys():
                self.fields[field_name] = cp_fields.read(field_name)

    def save(self, cp: CheckpointPath) -> list[str]:
        """Save within h5 file and return names of saved variables."""
        saved_list = [
            cp.write("vertices", self.vertices),
            cp.write("cells", self.cells),
            cp.write_str("region_names", ",".join(self.region_names)),
            cp.write("boundary_faces", self.boundary_faces),
            cp.write_str("boundary_names", ",".join(self.boundary_names)),
        ]
        if self.cell_permutation is not None:
            saved_list.append(cp.write("cell_permutation", self.cell_permutation))
        for field_name, field_value in self.fields.items():
            saved_list.append(cp.write(f"fields/{field_name}", field_value))
        return saved_list

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        return self.save(cp_path)

    @property
    def dimension(self) -> int:
        return self.cells.shape[1] - 2  # cells coontain d + 1 vertices, 1 region index

    def _partition_cells(self, group: dist.ProcessGroup | None) -> CellDivision:
        """Optimally partition the mesh cells for communication over `group`.
        This reorders `cells` such that contiguous ranges of cells can be owned by
        each process within `group`, and returns a `CellDivision` instance to
        track the division and communicate ghost cells across processes.
        Correspondingly, `cell_permutation` tracks any reordering of the cells."""
        size = 1 if group is None else group.size()
        if size == 1:
            counts = np.array([len(self.cells)])  # no partitioning required
            return CellDivision(self, counts, group)

        # Compute partition on overall head process:
        # (Partition must be stable across process grid, not just the "r" dimension.)
        partition = np.empty(len(self.cells), dtype=int)
        if rc.is_head:
            adj = self.get_cell_adjacency_vertex()
            csr_adj = pymetis.CSRAdjacency(adj.indptr, adj.indices)
            partition = np.asarray(pymetis.part_graph(size, csr_adj)[1], dtype=int)
            # Sort process assignments to minimize permutation:
            mean_index = [np.where(partition == rank)[0].mean() for rank in range(size)]
            rank_remap = np.empty(size, dtype=int)
            rank_remap[np.argsort(mean_index)] = np.arange(size)
            partition = rank_remap[partition]

        # Synchronize partition across all processes:
        partition_t = torch.from_numpy(partition).to(rc.device)
        dist.broadcast(partition_t, src=0)  # on world, not just on group
        partition = partition_t.to(rc.cpu).numpy()

        # Determine permutation and bounds:
        permutation = np.argsort(partition, kind="stable")  # group cells by rank
        counts = np.bincount(partition, minlength=size)

        # Apply and track permutation on quantities stored by cells:
        self.cells = self.cells[permutation]
        self.faces = self.faces[permutation]
        if self.cell_permutation is None:
            self.cell_permutation = permutation
        else:
            self.cell_permutation = self.cell_permutation[permutation]

        # Remap cell indices in face_map, and keep it sorted by cell indices:
        inv_permutation = np.full(len(permutation) + 1, -1)  # last entry maps -1 to -1
        inv_permutation[permutation] = np.arange(len(permutation))
        self.face_map[..., 0] = inv_permutation[self.face_map[..., 0]]
        self.face_map = sort_face_map(self.face_map)
        return CellDivision(self, counts, group)

    def get_cell_adjacency_vertex(self) -> csr_array:
        """Adjacency matrix of cells that share a vertex."""
        cells = self.cells[:, :-1]
        cell_to_vertex = csr_array(
            (
                np.ones_like(cells).flatten(),
                (np.repeat(np.arange(len(cells)), cells.shape[1]), cells.flatten()),
            ),
        )
        return make_adjacency(cell_to_vertex)

    def get_cell_adjacency_face(self) -> csr_array:
        """Adjacency matrix of cells that share an face."""
        cell_pairs = self.face_map[self.face_map[:, 1, 0] >= 0, :, 0]  # interior faces
        i_cell, j_cell = np.vstack((cell_pairs, cell_pairs[:, ::-1])).T  # both orders
        n_cells = len(self.cells)
        return csr_array((np.ones_like(i_cell), (i_cell, j_cell)), (n_cells, n_cells))

    def _initialize_face_maps(self) -> None:
        """Initialize interior and boundary faces, mapped to cells."""
        n_faces_per_cell = self.cells.shape[1] - 1  # for each vertex dropped from cell
        faces = np.stack(
            [
                np.roll(self.cells[:, :-1], -i_drop, axis=1)[:, 1:]
                for i_drop in range(n_faces_per_cell)
            ],
            axis=1,
        )
        stride = len(self.vertices) ** np.arange(faces.shape[-1])[::-1]
        face_ids = np.sort(faces, axis=-1) @ stride  # equal for all vertex orders
        boundary_face_ids = np.sort(self.boundary_faces[:, :-1], axis=-1) @ stride
        self.faces = faces

        # Find unique face IDs and counts:
        unique_ids, unique_indices, unique_inverse, counts = np.unique(
            face_ids, return_index=True, return_inverse=True, return_counts=True
        )
        assert counts.max() <= 2
        unique_rank = np.ones_like(unique_inverse)  # rank within equivalence class
        unique_rank[unique_indices] = 0  # and others are 1 because count <= 2
        face_map = np.full((len(unique_ids), 2, 2), -1)
        face_map[unique_inverse, unique_rank] = np.stack(
            np.unravel_index(np.arange(len(unique_inverse)), face_ids.shape), axis=-1
        )  # cell and face within cell, for each copy of each unique face

        # Boundary faces have only one cell set in face_map, with the other being -1s.
        # Use the second cell index being -1 as the marker for boundary faces.
        # Set the face within cell index of the second cell to be the region ID:
        i_unique_boundary = np.searchsorted(unique_ids, boundary_face_ids)
        assert (face_map[i_unique_boundary, 1, 0] == -1).all()
        face_map[i_unique_boundary, 1, 1] = self.boundary_faces[:, -1]
        self.face_map = sort_face_map(face_map)

    def get_face_reconstruction_op(self) -> tuple[np.ndarray, np.ndarray]:
        """Construct vertex neighbor list (n_reconstruct x max_neighbors) for local
        cells requiring face reconstruction, and corresponding cell center to face
        center reconstruction operator (n_reconstruct x n_faces x max_neighbors)."""
        # Get neighbor list:
        cell_div = self.cell_division
        adjacency = self.get_cell_adjacency_vertex()
        neighbor_indices = cell_div.local_index[adjacency.indices]  # local index
        index_ranges = adjacency.indptr  # but index ranges by global index
        max_neighbors = np.diff(index_ranges).max()
        min_neighbors = self.cells.shape[1] - 2  # needed for valid reconstruction

        # Create inverse-distance-weighted least-squares gradient operator:
        n_reconstruct = cell_div.n_reconstruct
        neighbors = np.tile(
            np.arange(n_reconstruct)[:, None], max_neighbors
        )  # padded with self when neighbors < max_neighbors
        grad_op = np.zeros((n_reconstruct, 2, max_neighbors))
        cells = self.cells[cell_div.global_index, :-1]
        cell_centers = self.vertices[cells].mean(axis=1)
        for i, i_global in enumerate(cell_div.global_index[:n_reconstruct]):
            js = neighbor_indices[index_ranges[i_global] : index_ranges[i_global + 1]]
            if len(js) < min_neighbors:
                continue
            neighbors[i, : len(js)] = js
            D = cell_centers[js] - cell_centers[i]
            w = 1.0 / np.linalg.norm(D, axis=1)
            grad_op[i, :, : len(js)] = np.linalg.pinv(w[:, None] * D) * w[None, :]

        # Combine to reconstruction operator:
        faces = self.faces[cell_div.global_index[:n_reconstruct]]
        face_centers = self.vertices[faces].mean(axis=2)
        face_offsets = face_centers - cell_centers[:n_reconstruct, None]
        reconstruct = np.einsum("cfi, cin -> cfn", face_offsets, grad_op)
        return neighbors, reconstruct

    def get_face_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculate directed face weights (Nf x 2), i.e. normal * face area,
        only for local faces (boundary and interior). Also, return the local version
        of `face_map` of local faces indexed to local cells."""
        # Calculate for all local cells:
        face_vertices = self.vertices[self.faces[self.cell_division.global_index]]
        if self.dimension == 1:
            # Unit vectors along/opposite 1D domain (unit transverse area)
            face_centers = face_vertices.mean(axis=2)
            normals = face_centers - face_centers.mean(axis=1, keepdims=True)
            face_weights = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
        else:
            # Same length as vector along face (edge), but rotated by 90 degrees:
            face_vectors = (
                face_vertices[:, :, 1] - face_vertices[:, :, 0]
            )  # Nc x Nf x 2
            rot90 = np.array([[0.0, -1.0], [1.0, 0.0]])
            face_weights = face_vectors @ rot90

        # Index to unique faces:
        face_map = np.copy(self.face_map[self.face_division.global_index])
        face_map[..., 0] = np.where(
            face_map[..., 0] == -1, -1, self.cell_division.local_index[face_map[..., 0]]
        )  # map cell indices to local ones (taking care of -1s for the boundary faces)
        i_cell, i_face = face_map[:, 0].T
        return face_weights[i_cell, i_face], face_map

    def get_cell_weights(self) -> np.ndarray:
        """Calculate local cell weights (volumes with unit transverse length/area)."""
        cells = self.cells[self.cell_division.global_index, :-1]  # local, ignore region
        cell_vertices = self.vertices[cells]
        if self.dimension == 1:
            return np.linalg.norm(cell_vertices[:, 1] - cell_vertices[:, 0], axis=-1)
        else:
            return 0.5 * abs(np.linalg.det(cell_vertices[:, 1:] - cell_vertices[:, :1]))


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


def sort_face_map(face_map: np.ndarray) -> np.ndarray:
    cell1, cell2 = face_map[..., 0].T
    should_flip = (cell2 >= 0) & (cell2 < cell1)
    face_map = np.copy(face_map)
    face_map[should_flip] = face_map[should_flip, ::-1]  # cell1 < cell2 now
    return face_map[np.lexsort(face_map[:, ::-1, 0].T)]  # sort by (cell1, cell2)


class CellDivision(TaskDivisionCustom):
    """Divide cells and communicate ghosts between processes."""

    group: dist.ProcessGroup | None
    n_reconstruct: int  #: number of local cells requiring face reconstruction
    global_index: np.ndarray  #: indices of all local cells in global array of cells
    local_index: np.ndarray  #: inverse of global_index (-1 for locally unused cells)

    def __init__(
        self, mesh: Mesh, counts: np.ndarray, group: dist.ProcessGroup | None
    ) -> None:
        i_proc = 0 if group is None else group.rank()
        super().__init__(n_mine=counts[i_proc], group=group)
        self.group = group
        adj_vert = mesh.get_cell_adjacency_vertex()
        adj_face = mesh.get_cell_adjacency_face()

        def ring(cells: np.ndarray, adj: csr_array) -> np.ndarray:
            """1-ring vertex closure of a set of cells (cells + their neighbours)."""
            v = np.zeros(self.n_tot)
            v[cells] = 1
            v += adj @ v
            return np.where(v)[0]

        owned = np.arange(self.i_start, self.i_stop)
        ring1 = ring(owned, adj_face)  # cells to reconstruct (+face-neighbors)
        ring2 = ring(ring1, adj_vert)  # cells accessed overall (+vertex-neighbors)
        ghosts = ring2[~self.belongs(self.i_proc, ring2)]  # global indices from others
        self.n_reconstruct = len(ring1)  # number of local cells to reconstruct

        # Collect global cell indices of owned, ring1 and ring2 on current process
        ring1_only = ring1[~np.isin(ring1, owned)]
        ring2_only = ring2[~np.isin(ring2, ring1)]
        self.global_index = np.concatenate((owned, ring1_only, ring2_only))

        # Inverse map from all global cell indices to local ones (-1 for untouched ones)
        self.local_index = np.full(self.n_tot, -1)
        self.local_index[self.global_index] = np.arange(len(ring2))

        # Plan communications:
        recv: list[np.ndarray] = []
        send: list[np.ndarray] = []
        for j_proc in range(self.n_procs):
            if j_proc == self.i_proc:
                recv.append(np.empty(0, dtype=int))
                send.append(np.empty(0, dtype=int))
            else:
                # Receive each ghost from its owning rank:
                recv.append(self.local_index[ghosts[self.belongs(j_proc, ghosts)]])
                # Send the owned cells that other ranks need:
                owned_q = np.arange(self.n_prev[j_proc], self.n_prev[j_proc + 1])
                ring2_q = ring(ring(owned_q, adj_face), adj_vert)
                ghost_q = ring2_q[~self.belongs(j_proc, ring2_q)]
                send.append(self.local_index[owned[np.isin(owned, ghost_q)]])
        self._recv_index = torch.from_numpy(np.concatenate(recv)).to(rc.device)
        self._send_index = torch.from_numpy(np.concatenate(send)).to(rc.device)
        self._recv_counts = [len(idx) for idx in recv]
        self._send_counts = [len(idx) for idx in send]

    def communicate_ghosts(self, rho: torch.Tensor) -> None:
        """Communicate ghost cells from owned processes to wherever needed."""
        if self.n_procs == 1:
            return
        send_buf = rho[self._send_index].contiguous()
        recv_buf = torch.empty(
            (len(self._recv_index), rho.shape[1]), dtype=rho.dtype, device=rho.device
        )
        dist.all_to_all_single(
            recv_buf, send_buf, self._recv_counts, self._send_counts, group=self.group
        )
        rho[self._recv_index] = recv_buf


class FaceDivision(TaskDivisionCustom):
    """Track index mapping and process ownership of faces in Mesh."""

    n_boundary_mine: int  #: number of owned boundary faces
    n_interior_mine: int  #: number of owned interior faces
    global_index: np.ndarray  #: indices of all local faces in global face map
    local_index: np.ndarray  #: inverse of global_index (-1 for locally unused faces)

    def __init__(self, face_map: np.ndarray, cell_division: CellDivision) -> None:
        cell1, cell2 = face_map[:, :, 0].T

        # Define face ownership by first cell:
        is_face_mine = (cell1 >= cell_division.i_start) & (cell1 < cell_division.i_stop)
        n_faces_mine = np.count_nonzero(is_face_mine)
        super().__init__(n_mine=n_faces_mine, group=cell_division.group)

        # Sort by boundary, interior (owned) and interior (not owned, but local):
        local = np.where(
            is_face_mine
            | ((cell2 >= cell_division.i_start) & (cell2 < cell_division.i_stop))
        )[0]
        owned = np.arange(self.i_start, self.i_stop)
        boundary = np.where(is_face_mine & (cell2 == -1))[0]
        interior_mine = owned[~np.isin(owned, boundary)]
        interior_not_mine = local[~np.isin(local, owned)]
        self.n_boundary_mine = len(boundary)
        self.n_interior_mine = len(interior_mine)
        self.global_index = np.concatenate((boundary, interior_mine, interior_not_mine))

        # Inverse map from all global face indices to local ones (-1 for untouched ones)
        self.local_index = np.full(self.n_tot, -1)
        self.local_index[self.global_index] = np.arange(len(local))
