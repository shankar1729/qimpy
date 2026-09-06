"""Exact ballistic transport by the method of characteristics.

An independent check on the finite-volume solver, on the SAME mesh files: no
cells, no k-grid, no time step, no limiter, no wall closure.  Run it directly
on any qimpy transport input::

    python -m qimpy.transport.ballistic -i input.yaml
"""
__all__ = ("Ballistic", "Polygon")

from ._polygon import Polygon
from ._solver import Ballistic
