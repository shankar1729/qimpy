from typing import Generic, TypeVar

GradientType = TypeVar("GradientType")


class Gradable(Generic[GradientType]):
    """Interface to store gradient w.r.t current object, analogous to pytorch."""

    grad: GradientType  #: optional gradient (of energy) with respect to this object.

    @property
    def requires_grad(self) -> bool:
        """Return whether gradient with respect to this object is needed."""
        return self._requires_grad

    def requires_grad_(self, requires_grad):
        """Set whether gradient with respect to this object is needed."""
        self._requires_grad = requires_grad

    def __init__(self):
        self._requires_grad = False
