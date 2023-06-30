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
    "ions",
    "lattice",
    "symmetries",
    "grid",
    "electrons",
    "geometry",
    "export",
    "System",
    "transport",
)

# Module import definition
from .pre_init import log, set_gpu_visibility
from . import rc, utils
from .tree import TreeNode
from .energy import Energy
from . import algorithms
from . import ions
from . import lattice
from . import symmetries
from . import grid
from . import electrons
from . import geometry
from . import export
from ._system import System
from . import transport

# Automatic versioning added by versioneer
from ._version import get_versions

__version__: str = get_versions()["version"]
del get_versions
