import qimpy as qp
import numpy as np
from functools import lru_cache
from typing import Optional, Sequence, Tuple


IMBALANCE_THRESHOLD = 20.0  #: max cpu time% waste tolerated in process grid dimension


class ProcessGrid:
    """Process grid of `shape` dimensions over communicator `comm`.
    Any -1 entries in `shape` are undetermined and will be resolved after the
    number of tasks split along that dimension are set using `provide_n_tasks`.
    Subsequently, use `get_comm` to get arbitrary hyperplane communicators that
    connect processes whose index only varies along specified subsets of dimensions.
    """

    comm: qp.MPI.Comm  #: Overall communicator within which grid is set-up.
    n_procs: int  #: Total number of processes in grid.
    i_proc: int  #: Overall rank of current process in grid.
    dim_names: str  #: Each character (must be unique) names a dimension.
    shape: np.ndarray  #: Grid dimensions. Unresolved dimensions are -1.

    def __init__(
        self, comm: qp.MPI.Comm, dim_names: str, shape: Optional[Sequence[int]] = None
    ) -> None:
        self.comm = comm
        self.n_procs = comm.size
        self.i_proc = comm.rank
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
        prod_unknown = self.n_procs // prod_known
        n_procs_dim = np.arange(1, prod_unknown + 1, dtype=int)  # shape[dim] candidates
        n_procs_dim = n_procs_dim[self.n_procs % n_procs_dim == 0]  # must be a factor
        # --- filter by imbalance:
        n_tasks_each = qp.utils.ceildiv(n_tasks, n_procs_dim)  # for each candidate
        imbalance = 100.0 * (1.0 - n_tasks / (n_tasks_each * n_procs_dim))
        n_procs_dim = n_procs_dim[imbalance < IMBALANCE_THRESHOLD]
        # --- pick largest candidate
        self.shape[dim] = n_procs_dim[-1]
        self._check_report()

    @lru_cache
    def get_comm(self, dim_names: str) -> qp.MPI.Comm:
        """Get communicator for a hyper-plane spanning `dim_names`.
        The resulting communicator will connect processes whose index in
        the process grid only varies along dimensions within `dim_names`.
        Dimensions before and including those in `dim_names` must be known,
        except when `dim_names` is a contiguous block of dimensions whose
        product can be determined now based on other dimensions."""

        # Check input:
        if not dim_names:
            return qp.MPI.COMM_SELF  # no varying dimensions => self only
        dim_names_uniq = set(dim_names)
        assert len(dim_names_uniq) == len(dim_names)  # no repetitions
        assert dim_names_uniq.issubset(self.dim_names)  # each dim valid
        if len(dim_names) == len(self.dim_names):
            return self.comm  # all dimensions varying => original communicator

        # Create mask of dimensions to be indexed:
        shape = list(self.shape)
        mask = [(dim_name in dim_names_uniq) for dim_name in self.dim_names]

        # Coalesce contiguous indexed / not-indexed dimensions:
        i_dim = 0
        while i_dim + 1 < len(shape):
            if mask[i_dim] == mask[i_dim + 1]:
                mask.pop(i_dim + 1)
                # Correspondingly merge shape:
                shape_next = shape.pop(i_dim + 1)
                shape_cur = shape[i_dim]
                shape_unknown = (shape_next == -1) or (shape_cur == -1)
                shape[i_dim] = -1 if shape_unknown else (shape_next * shape_cur)
            i_dim += 1
        shape_arr, n_unknown = self._fill_unkwown(np.array(shape))
        shape = list(shape_arr)
        assert n_unknown == 0  # need to know full shape (after coalescing) to proceed

        # Find processes that only vary along selected dimensions:
        index_cur = np.unravel_index(self.i_proc, shape)
        index = tuple(
            (slice(None) if mask_i else index_i)
            for index_i, mask_i in zip(index_cur, mask)
        )
        proc_list = np.arange(self.n_procs).reshape(shape)[index].flatten()
        return self.comm.Create_group(self.comm.Get_group().Incl(proc_list))

    def _check_report(self) -> None:
        """Check known dimensions and report current state."""
        self.shape, n_unknown = self._fill_unkwown(self.shape)
        dims_str = " x ".join(
            f"{dim} {name}" for dim, name in zip(self.shape, self.dim_names)
        )
        unknown_str = " (-1's determined later)" if n_unknown else ""
        qp.log.info(f"Process grid: {dims_str}{unknown_str}")

    def _fill_unkwown(self, shape: np.ndarray) -> Tuple[np.ndarray, int]:
        """Fill in unknown dimensions in special cases where possible.
        Returns modified shape and number of dimensions that remain unknown."""

        # Check compatibility of known dimensions with total:
        prod_known = shape[shape != -1].prod()
        if self.n_procs % prod_known:
            raise ValueError(
                f"Cannot distribute {self.n_procs} processes to"
                f" {' x '.join(map(str, shape))} grid"
            )

        # Compute a single unknown dimension if present:
        n_unknown = np.count_nonzero(shape == -1)
        if n_unknown == 1:
            shape[shape == -1] = self.n_procs // prod_known
            n_unknown = 0

        # Set unknown dimensions to 1 if no factor left:
        if n_unknown and (prod_known == self.n_procs):
            shape[shape == -1] = 1
            n_unknown = 0

        return shape, n_unknown
