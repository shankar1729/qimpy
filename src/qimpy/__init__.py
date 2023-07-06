"""QimPy: Quantum-Integrated Multi-PhYsics"""
# List exported symbols for doc generation
__all__ = (
    "log",
    "set_gpu_visibility",
    "rc",
    "utils",
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
from . import rc, utils
from .tree import TreeNode
from .energy import Energy
from . import algorithms, lattice, symmetries, grid, dft, transport

# Automatic versioning added by versioneer
from ._version import get_versions

__version__: str = get_versions()["version"]
del get_versions
