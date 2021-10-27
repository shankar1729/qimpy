"""Exchange-correlation functional."""
# List exported symbols for doc generation
__all__ = ["XC", "functional", "lda", "gga"]

from ._xc import XC
from . import functional
from . import lda
from . import gga
