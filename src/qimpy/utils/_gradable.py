from typing import Generic, TypeVar
from abc import ABC

GradientType = TypeVar("GradientType")


class Gradable(ABC, Generic[GradientType]):
    """Interface to store gradient w.r.t current object, analogous to pytorch."""

    grad: GradientType  #: optional gradient (of energy) with respect to this object.

    @property
    def requires_grad(self) -> bool:
        """Return whether gradient with respect to this object is needed."""
        return self._requires_grad

    def requires_grad_(self, requires_grad: bool = True) -> None:
        """Set whether gradient with respect to this object is needed."""
        self._requires_grad = requires_grad
        self.__dict__.pop("grad", None)  # remove previous gradient (if any)
        if requires_grad:
            self.grad = self.zeros_like()  # prepare new zero'd gradient

    def __init__(self) -> None:
        self._requires_grad = False
