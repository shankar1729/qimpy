from typing import Generic, TypeVar
from abc import ABC, abstractmethod


GradientType = TypeVar("GradientType")


class Gradable(ABC, Generic[GradientType]):
    """Interface to store gradient w.r.t current object, analogous to pytorch."""

    grad: GradientType  #: optional gradient (of energy) with respect to this object.

    @property
    def requires_grad(self) -> bool:
        """Return whether gradient with respect to this object is needed."""
        return self._requires_grad

    def requires_grad_(self, requires_grad: bool = True, clear: bool = False) -> None:
        """Set whether gradient with respect to this object is needed.
        If `clear`, also clear previous gradient / set to zero as needed.
        """
        self._requires_grad = requires_grad
        if clear:
            if requires_grad:
                self.grad = self.zeros_like()  # prepare new zero'd gradient
            else:
                self.__dict__.pop("grad", None)  # remove previous gradient (if any)

    def __init__(self) -> None:
        self._requires_grad = False

    @abstractmethod
    def zeros_like(self: GradientType) -> GradientType:
        ...
