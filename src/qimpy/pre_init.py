"""
Initialization that must occur before all other imports:
- Create log
- Import torch before any other libraries (e.g. MPI) that could cause version issues
- Get package version
"""

import logging
from importlib.metadata import version, PackageNotFoundError

import torch

log: logging.Logger = logging.getLogger("qimpy")
"Log for the qimpy module, configurable using :func:`qimpy.io.log_config`"

del torch

try:
    __version__ = version("qimpy")
except PackageNotFoundError:
    __version__ = "unknown"
