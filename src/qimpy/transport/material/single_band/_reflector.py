import torch

from qimpy import MPI
from qimpy.mpi import TaskDivision


N_TOL = 1.0e-12  #: tolerance for comparing normal / velocity directions


class Reflector:
    """Reflect single-band model states with energy and transverse k conservation."""

    def __init__(
        self,
        n: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        E: torch.Tensor,
        comm: MPI.Comm,
        k_division: TaskDivision,
        specularity: float,
    ) -> None:
        pass


def get_reflection_map(
    n: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    E: torch.Tensor,
) -> None:
    """Calculate reflection map for a singe normal direction."""
    v_dot_n = v @ n
    is_parallel = v_dot_n.abs() < N_TOL
    i_rows, i_cols, values = [], [], []  # sparse map in coordinate format

    # Don't transform parallel velocities:
    i_parallel = torch.where(is_parallel)[0]
    if len(i_parallel):
        i_rows.append(i_parallel)
        i_cols.append(i_parallel)
        values.append(torch.ones_like(i_parallel))
