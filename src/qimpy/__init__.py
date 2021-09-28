"""QimPy: Quantum-Integrated Multi-PhYsics"""
# List exported symbols for doc generation
__all__ = [
    'MPI', 'System', 'TreeNode', 'TreeNodeOptions', 'Energy',
    'utils', 'ions', 'lattice', 'symmetries', 'grid', 'electrons', 'log']

# Module import definition
from mpi4py import MPI
from ._tree import TreeNode, TreeNodeOptions
from ._energy import Energy
from . import utils
from . import ions
from . import lattice
from . import symmetries
from . import grid
from . import electrons
from ._system import System
import logging

# Automatic versioning added by versioneer
from ._version import get_versions
__version__: str = get_versions()['version']
del get_versions

# Module-level attributes
log: logging.Logger = logging.getLogger('qimpy')
'Log for the qimpy module, configurable using :func:`qimpy.utils.log_config`'
