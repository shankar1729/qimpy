"""QimPy: Quantum-Integrated Multi-PhYsics"""
# List exported symbols for doc generation
__all__ = (
    "log",
    "set_gpu_visibility",
    "MPI",
    "rc",
    "profiler",
    "io",
    "mpi",
    "utils",
    "math",
    "TreeNode",
    "Energy",
    "algorithms",
    "lattice",
    "symmetries",
    "grid",
    "dft",
    "transport",
)

# Module import definition
from .pre_init import log, set_gpu_visibility
from mpi4py import MPI  #: Must initialize MPI after pre_init for correct GPU behavior.
from . import rc, profiler, io, mpi, math
from ._tree import TreeNode
from ._energy import Energy
from . import algorithms, lattice, symmetries, grid, dft, transport

# Automatic versioning added by versioneer
from ._version import get_versions

__version__: str = get_versions()["version"]
del get_versions
