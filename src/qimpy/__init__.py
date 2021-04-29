# List exported symbols for doc generation
__all__ = [
    'MPI', 'System', 'fmt',
    'Lattice', 'ions',
    'log', 'log_config']

# Module import definition
from mpi4py import MPI
from ._system import System, fmt
from ._log import log_config
from ._runconfig import RunConfig
from . import ions
from . import lattice
import logging

# Automatic versioning added by versioneer
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

# Module-level attributes
log = logging.getLogger('qimpy')
'''logging.Logger : Logging by the qimpy module,
which can be configured using :func:`~qimpy.log_config`'''
