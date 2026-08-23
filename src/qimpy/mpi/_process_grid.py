from typing import Optional, Sequence
import functools

import numpy as np
import torch.distributed as dist

import qimpy
from qimpy import rc, log, MPI

IMBALANCE_THRESHOLD = 20.0  #: max cpu time% waste tolerated in process grid dimension


class ProcessGrid:
    """Process grid of `shape` dimensions over the global process group.
    Any -1 entries in `shape` are undetermined and will be resolved after the
    number of tasks split along that dimension are set using `provide_n_tasks`.
    Subsequently, use `get_group` to get arbitrary hyperplane process groups that
    connect processes whose index only varies along specified subsets of dimensions.
    """

    dim_names: str  #: Each character (must be unique) names a dimension.
    shape: np.ndarray  #: Grid dimensions. Unresolved dimensions are -1.
    is_resolved: bool  #: Whether all dimensions have been resolved
    device_mesh: dist.DeviceMesh  #: Corresponding device mesh

    def __init__(self, dim_names: str, shape: Optional[Sequence[int]] = None) -> None:
        self.dim_names = dim_names
        assert len(set(dim_names)) == len(dim_names)  # characters must be unique
        if shape:
            assert len(shape) == len(dim_names)
            self.shape = np.array(shape, dtype=int)
        else:
            self.shape = np.full(len(dim_names), -1)  # all dimensions undetermined
        self._check_report()

    def provide_n_tasks(self, dim_name: str, n_tasks: int) -> None:
        """Provide task count for a process grid dimension named `dim_name`.
        If that dimension is undetermined (-1), set it to a suitable value that is
        compatible with the total processes, any other known dimensions, and with
        splitting n_tasks tasks with reasonable load balancing over this dimension.

        Parameters
        ----------
        dim_name
            Name of dimension (single charcater) to provide n_tasks for.
        n_tasks
            Number of tasks available to split on this dimension of the process grid,
            used for setting dimension to ensure reasonable load balancing.
        """

        # Identify dimension:
        dim = self.dim_names.find(dim_name)
        assert dim >= 0
        if self.shape[dim] != -1:
            return  # Shape already known for this dimension

        # Dimension undetermined: set it based on n_tasks
        prod_known = self.shape[self.shape != -1].prod()
        prod_unknown = rc.n_procs // prod_known
        n_procs_dim = np.arange(1, prod_unknown + 1, dtype=int)  # shape[dim] candidates
        n_procs_dim = n_procs_dim[rc.n_procs % n_procs_dim == 0]  # must be a factor
        # --- filter by imbalance:
        n_tasks_each = qimpy.math.ceildiv(n_tasks, n_procs_dim)  # for each candidate
        imbalance = 100.0 * (1.0 - n_tasks / (n_tasks_each * n_procs_dim))
        n_procs_dim = n_procs_dim[imbalance < IMBALANCE_THRESHOLD]
        # --- pick largest candidate
        self.shape[dim] = n_procs_dim[-1]
        self._check_report()

    @functools.cache
    def get_submesh(self, dim_names: str) -> dist.DeviceMesh:
        """Get device mesh with a subset of dimensions."""
        return self.device_mesh[tuple(dim_names)]

    @functools.cache
    def get_group(self, dim_names: str) -> dist.ProcessGroup:
        """Get communicator for a hyper-plane spanning `dim_names`.
        The resulting communicator will connect processes whose index in
        the process grid only varies along dimensions within `dim_names`.
        All dimensions should be fully resolved before getting any groups."""
        assert self.is_resolved
        if len(dim_names) == 1:
            return self.device_mesh.get_group(dim_names[0])
        else:
            return self.get_submesh(dim_names)._flatten().get_group(0)

    def _check_report(self) -> None:
        """Check known dimensions and report current state.
        If fully resolved, initialize the device mesh."""
        self.shape, n_unknown = self._fill_unkwown(self.shape)
        dims_str = " x ".join(
            f"{dim} {name}" for dim, name in zip(self.shape, self.dim_names)
        )
        unknown_str = " (-1's determined later)" if n_unknown else ""
        log.info(f"Process grid: {dims_str}{unknown_str}")

        # Initialize the device mesh if all dimensions are resolved:
        self.is_resolved = n_unknown == 0
        if self.is_resolved:
            self.device_mesh = dist.init_device_mesh(
                rc.device.type,
                self.shape.tolist(),
                mesh_dim_names=tuple(self.dim_names),
            )

    def _fill_unkwown(self, shape: np.ndarray) -> tuple[np.ndarray, int]:
        """Fill in unknown dimensions in special cases where possible.
        Returns modified shape and number of dimensions that remain unknown."""

        # Check compatibility of known dimensions with total:
        prod_known = shape[shape != -1].prod()
        if rc.n_procs % prod_known:
            raise ValueError(
                f"Cannot distribute {rc.n_procs} processes to"
                f" {' x '.join(map(str, shape))} grid"
            )

        # Compute a single unknown dimension if present:
        n_unknown = int(np.count_nonzero(shape == -1))
        if n_unknown == 1:
            shape[shape == -1] = rc.n_procs // prod_known
            n_unknown = 0

        # Set unknown dimensions to 1 if no factor left:
        if n_unknown and (prod_known == rc.n_procs):
            shape[shape == -1] = 1
            n_unknown = 0

        return shape, n_unknown


@functools.cache
def get_comm(group: dist.ProcessGroup) -> MPI.Comm:
    """Get MPI communicator corresponding to process group."""
    proc_list = dist.get_process_group_ranks(group)
    return rc.comm.Create_group(rc.comm.Get_group().Incl(proc_list))
