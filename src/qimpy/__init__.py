# List exported symbols for doc generation
__all__ = [
    'MPI', 'System', 'fmt',
    'utils', 'ions', 'lattice',
    'log']

# Module import definition
from mpi4py import MPI
from ._system import System, fmt
from . import utils
from . import ions
from . import lattice
from . import symmetries
import logging

# Automatic versioning added by versioneer
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

# Module-level attributes
log = logging.getLogger('qimpy')
'''logging.Logger : Logging by the qimpy module,
which can be configured using :func:`~qimpy.log_config`'''
