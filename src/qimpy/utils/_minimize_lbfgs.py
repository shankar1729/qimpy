import qimpy as qp
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import Minimize, ConvergenceCheck
    from .. import Energy


def _lbfgs(self: 'Minimize') -> 'Energy':
    """L-BFGS implementation for `Minimize.minimize`"""
    assert self.method == 'l-bfgs'
    return qp.Energy()  # TODO
