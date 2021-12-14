import qimpy as qp
import numpy as np
from typing import Optional, Sequence


class ProcessGrid:
    """Process grid of `shape` dimensions over communicator `comm`.
    Any -1 entries in `shape` are undetermined and will be resolved after the
    number of tasks split along that dimension are set using `provide_n_tasks`.
    Subsequently, use `get_comm` to get arbitrary hyperplane communicators that
    connect processes with equal index along specified subsets of dimensions.
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
        def get_imbalance():
            """Compute cpu time% wasted in splitting n_tasks over n_procs_dim"""
            n_tasks_each = qp.utils.ceildiv(n_tasks, n_procs_dim)
            return 100.0 * (1.0 - n_tasks / (n_tasks_each * n_procs_dim))

        imbalance_threshold = 20.0  # max cpu time% waste to tolerate
        prod_known = self.shape[self.shape != -1].prod()
        n_procs_dim = self.n_procs // prod_known  # max possible value
        imbalance = get_imbalance()
        if imbalance > imbalance_threshold:
            # Drop primes factors starting from smallest till balanced:
            factors = qp.utils.prime_factorization(n_procs_dim)
            for factor in factors:
                n_procs_dim //= factor
                imbalance = get_imbalance()
                if imbalance <= imbalance_threshold:
                    break
        assert imbalance <= imbalance_threshold
        self.shape[dim] = n_procs_dim
        self._check_report()

    def _check_report(self) -> None:
        """Check known dimensions and report current state."""

        # Check compatibility of known dimensions with total:
        prod_known = self.shape[self.shape != -1].prod()
        if self.n_procs % prod_known:
            raise ValueError(
                f"Cannot distribute {self.n_procs} processes to"
                f" {' x '.join(self.shape)} grid"
            )

        # Compute a single unknown dimension if present:
        n_unknown = np.count_nonzero(self.shape == -1)
        if n_unknown == 1:
            self.shape[self.shape == -1] = self.n_procs // prod_known
            n_unknown = 0

        # Set unknown dimensions to 1 if no factor left:
        if n_unknown and (prod_known == self.n_procs):
            self.shape[self.shape == -1] = 1
            n_unknown = 0

        # Report grid as it is now:
        dims_str = " x ".join(
            f"{dim} {name}" for dim, name in zip(self.shape, self.dim_names)
        )
        unknown_str = " (-1's determined later)" if n_unknown else ""
        qp.log.info(f"Process grid: {dims_str}{unknown_str}")
