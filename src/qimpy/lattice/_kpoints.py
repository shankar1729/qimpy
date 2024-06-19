from __future__ import annotations
from typing import Union, Sequence

import numpy as np
import torch

import qimpy
from qimpy import log, rc, TreeNode, MPI
from qimpy.io import CheckpointPath, CheckpointContext
from qimpy.mpi import ProcessGrid, TaskDivision
from . import Lattice


class Kpoints(TreeNode):
    """Set of k-points in Brillouin zone."""

    comm: MPI.Comm  #: Communicator for k-point division
    k: torch.Tensor  #: Array of k-points (N x 3)
    wk: torch.Tensor  #: Integration weights for each k (adds to 1)
    division: TaskDivision  #: Division of k-points across `comm`

    def __init__(
        self,
        *,
        process_grid: ProcessGrid,
        k: torch.Tensor,
        wk: torch.Tensor,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        """Initialize from list of k-points and weights. Typically, this should
        be used only by derived classes :class:`Kmesh` or :class:`Kpath`.
        """
        super().__init__()
        self.k = k
        self.wk = wk
        assert abs(wk.sum() - 1.0) < 1e-14

        # Initialize process grid dimension (if -1) and split k-points:
        process_grid.provide_n_tasks("k", k.shape[0])
        self.comm = process_grid.get_comm("k")
        self.division = TaskDivision(
            n_tot=k.shape[0],
            n_procs=self.comm.size,
            i_proc=self.comm.rank,
            name="k-point",
        )

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        return [cp_path.write("k", self.k), cp_path.write("wk", self.wk)]


class Kmesh(Kpoints):
    """Uniform k-mesh sampling of Brillouin zone"""

    size: tuple[int, ...]  #: Dimensions of k-mesh
    i_reduced: torch.Tensor  #: Reduced index of each k-point in mesh
    i_sym: torch.Tensor  #: Symmetry index that maps mesh points to reduced set
    invert: torch.Tensor  #: Inversion factor (1, -1) in reduction of each k

    def __init__(
        self,
        *,
        process_grid: ProcessGrid,
        symmetries: qimpy.symmetries.Symmetries,
        lattice: Lattice,
        checkpoint_in: CheckpointPath = CheckpointPath(),
        offset: Union[Sequence[float], np.ndarray] = (0.0, 0.0, 0.0),
        size: Union[float, Sequence[int], np.ndarray] = (1, 1, 1),
        use_inversion: bool = True,
    ) -> None:
        """Construct k-mesh of specified `size` and `offset`.

        Parameters
        ----------
        symmetries
            Symmetry group used to reduce k-points to irreducible set.
        lattice
            Lattice specification used for automatic size determination.
        offset
            :yaml:`Offset of k-point mesh in k-mesh coordinates.`
            (That is, by offset / size in fractional reciprocal coordinates.)
            For example, use [0.5, 0.5, 0.5] for the Monkhorst-Pack scheme.
            Default: [0., 0., 0.] selects Gamma-centered mesh.
        size
            :yaml:`Number of k per dimension, or minimum supercell size.`
            If given as a list of 3 integers, number of k-points along each
            reciprocal lattice direction. Instead, a single float specifies
            the minimum real-space size of the k-point sampled supercell
            i.e. pick number of k-points along dimension i = ceil(size / L_i),
            where L_i is the length of lattice vector i (in bohrs).
            Default: [1, 1, 1] selects a single k-point = offset.
        use_inversion
            :yaml:`Whether to use inversion in k-space to reduce k-points.`
            This corresponds to complex conjugation in real space, and only
            matters for systems without inversion symmetry in real space.
            Default: True; should only need to disable this when interfacing
            with codes that do not support this symmetry eg. BerkeleyGW.
        """

        # Select size from real-space dimension if needed:
        if isinstance(size, float) or isinstance(size, int):
            sup_length = float(size)
            L_i = torch.linalg.norm(lattice.Rbasis, dim=0)  # lattice lengths
            size = torch.ceil(sup_length / L_i).to(torch.int).tolist()
            log.info(
                f"Selecting {size[0]} x {size[1]} x {size[2]} k-mesh"
                f" for supercell size >= {sup_length:g} bohrs"
            )

        # Check types and sizes:
        offset = np.array(offset)
        size = np.array(size)
        assert (offset.shape == (3,)) and (offset.dtype == float)
        assert (size.shape == (3,)) and (size.dtype == int)
        kmesh_method_str = (
            "centered at Gamma"
            if (np.linalg.norm(offset) == 0.0)
            else ("offset by " + np.array2string(offset, separator=", "))
        )
        log.info(
            f"Creating {size[0]} x {size[1]} x {size[2]} uniform"
            f" k-mesh {kmesh_method_str}"
        )

        # Check that offset is resolvable:
        min_offset = symmetries.tolerance  # detectable at that threshold
        if np.any(np.logical_and(offset != 0, np.abs(offset) < min_offset)):
            raise ValueError(f"Nonzero offset < {min_offset:g} symmetry tolerance")

        # Create full mesh:
        grids1d = [
            (offset[i] + torch.arange(size[i], device=rc.device)) / size[i]
            for i in range(3)
        ]
        mesh = torch.stack(torch.meshgrid(*tuple(grids1d), indexing="ij")).view(3, -1).T
        mesh -= torch.floor(0.5 + mesh)  # wrap to [-0.5,0.5)

        # Compute mapping of arbitrary k-points to mesh:
        def mesh_map(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # Sizes and dimensions on torch:
            assert isinstance(size, np.ndarray)
            size_i = torch.tensor(size, dtype=torch.int, device=rc.device)
            size_f = size_i.to(torch.double)  # need as both int and float
            offset_f = torch.tensor(offset, device=rc.device)
            stride_i = torch.tensor(
                [size[1] * size[2], size[2], 1], dtype=torch.int, device=rc.device
            )
            not_found_index = size.prod()
            # Compute mesh coordinates:
            mesh_coord = k * size_f - offset_f
            int_coord = torch.round(mesh_coord)
            on_mesh = ((mesh_coord - int_coord).abs() < min_offset).all(dim=-1)
            mesh_index = ((int_coord.to(torch.int) % size_i) * stride_i).sum(dim=-1)
            return on_mesh, torch.where(on_mesh, mesh_index, not_found_index)

        # Check whether to add explicit inversion:
        if use_inversion and not symmetries.i_inv:
            rot = torch.cat((symmetries.rot, -symmetries.rot))
        else:
            rot = symmetries.rot

        # Transform every k-point under every symmetry:
        # --- k-points transform by rot.T, so no transpose on right-multiply
        on_mesh, mesh_index = mesh_map(mesh @ rot)
        if not on_mesh.all():
            log.info(
                "WARNING: k-mesh symmetries are a subgroup of size "
                + str(on_mesh.all(dim=-1).count_nonzero().item())
            )
        first_equiv, i_sym = mesh_index.min(dim=0)  # first equiv k and sym
        reduced_index, i_reduced, reduced_counts = first_equiv.unique(
            return_inverse=True, return_counts=True
        )
        k = mesh[reduced_index]  # k in irreducible wedge
        wk = reduced_counts / size.prod()  # corresponding weights
        log.info(
            f"Reduced {size.prod()} points on k-mesh to" f" {len(k)} under symmetries"
        )
        # --- store mapping from full k-mesh to reduced set:
        size = tuple(size)
        self.size = size
        self.i_reduced = i_reduced.reshape(size)  # index into k
        self.i_sym = i_sym.reshape(size)  # symmetry number to get to k
        # --- seperate combined symmetry index into symmetry and inversion:
        self.invert = torch.where(self.i_sym > symmetries.n_sym, -1, +1)
        self.i_sym = self.i_sym % symmetries.n_sym
        if self.invert.min() < 0:
            log.info("Note: used k-inversion (conjugation) symmetry")

        # Initialize base class:
        super().__init__(
            process_grid=process_grid, k=k, wk=wk, checkpoint_in=checkpoint_in
        )

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        saved_list = super()._save_checkpoint(cp_path, context)
        cp_path.attrs["size"] = self.size
        saved_list.extend(
            [
                "size",
                cp_path.write("i_reduced", self.i_reduced),
                cp_path.write("i_sym", self.i_sym),
                cp_path.write("invert", self.invert),
            ]
        )
        return saved_list


class Kpath(Kpoints):
    """Path of k-points traversing Brillouin zone.
    Typically used only for band structure calculations."""

    labels: dict[int, str]  #: Special k-point indices and corresponding labels
    k_length: torch.Tensor  #: Cumulative k-path length till each k

    def __init__(
        self,
        *,
        process_grid: ProcessGrid,
        lattice: Lattice,
        dk: float,
        points: list,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        """Initialize k-path with spacing `dk` connecting `points`.

        Parameters
        ----------
        lattice
            Lattice specification for converting k-points from
            reciprocal fractional coordinates (input) to Cartesian
            for determining path lengths.
        dk
            :yaml:`Maximum distance between adjacent points on k-path.`
            (Units: :math:`a_0^{-1}`.)
        points
            :yaml:`List of special k-points [kx, ky, kz, label] along path.`
            Each point should contain three fractional coordinates (float)
            and optionally a string label for this point for use in
            band structure plots.
        """

        # Check types, sizes and separate labels from points:
        dk = float(dk)
        labels = [(point[3] if (len(point) > 3) else "") for point in points]
        kverts = torch.tensor(
            [point[:3] for point in points], dtype=torch.double, device=rc.device
        )
        log.info(
            f"Creating k-path with dk = {dk:g} connecting"
            f" {kverts.shape[0]} special points"
        )

        # Create path one segment at a time:
        k_list = [kverts[:1]]
        self.labels = {0: labels[0]}
        k_length = [torch.zeros(1, device=rc.device)]
        nk_tot = 1
        distance_tot = 0.0
        dkverts = kverts.diff(dim=0)
        distances = torch.sqrt(((dkverts @ lattice.Gbasis.T) ** 2).sum(dim=1))
        for i, distance in enumerate(distances):
            nk = int(torch.ceil(distance / dk).item())  # for this segment
            t = torch.arange(1, nk + 1, device=rc.device) / nk
            k_list.append(kverts[i] + t[:, None] * dkverts[i])
            nk_tot += nk
            self.labels[nk_tot - 1] = labels[i + 1]  # label at end of segment
            k_length.append(distance_tot + distance * t)
            distance_tot += distance
        k = torch.cat(k_list)
        wk = torch.full((nk_tot,), 1.0 / nk_tot, device=rc.device)
        self.k_length = torch.cat(k_length)
        log.info(f"Created {nk_tot} k-points on k-path of" f" length {distance_tot:g}")

        # Initialize base class:
        super().__init__(
            process_grid=process_grid, k=k, wk=wk, checkpoint_in=checkpoint_in
        )

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        saved_list = super()._save_checkpoint(cp_path, context)
        cp_path.attrs["labels"] = str(self.labels)
        saved_list.extend(["labels", cp_path.write("k_length", self.k_length)])
        return saved_list
