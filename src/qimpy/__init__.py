# List exported symbols for doc generation
__all__ = [
    'MPI', 'System', 'construct', 'dict_input_cleanup',
    'utils', 'ions', 'lattice', 'symmetries', 'grid', 'electrons',
    'log']

# Module import definition
from mpi4py import MPI
from ._system import System, construct, dict_input_cleanup
from . import utils
from . import ions
from . import lattice
from . import symmetries
from . import grid
from . import electrons
import logging

# Automatic versioning added by versioneer
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

# Module-level attributes
log = logging.getLogger('qimpy')
'''logging.Logger : Logging by the qimpy module,
which can be configured using :func:`~qimpy.log_config`'''
