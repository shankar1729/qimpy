"""QimPy: Quantum-Integrated Multi-PhYsics"""
# List exported symbols for doc generation
__all__ = [
    "rc",
    "TreeNode",
    "Energy",
    "utils",
    "ions",
    "lattice",
    "symmetries",
    "grid",
    "electrons",
    "geometry",
    "export",
    "System",
    "transport",
    "log",
]

# Module import definition
from . import rc
from ._tree import TreeNode
from ._energy import Energy
from . import utils
from . import ions
from . import lattice
from . import symmetries
from . import grid
from . import electrons
from . import geometry
from . import export
from ._system import System
from . import transport
import logging

# Automatic versioning added by versioneer
from ._version import get_versions

__version__: str = get_versions()["version"]
del get_versions

# Module-level attributes
log: logging.Logger = logging.getLogger("qimpy")
"Log for the qimpy module, configurable using :func:`qimpy.utils.log_config`"
