import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class Gradient:
    """Geometry gradient used for relaxation / dynamics."""

    ions: torch.Tensor  #: ionic gradient (forces)
    lattice: Optional[torch.Tensor]  #: lattice gradient (stress)

    def clone(self) -> "Gradient":
        return Gradient(
            ions=self.ions.clone().detach(),
            lattice=(None if (self.lattice is None) else self.lattice.clone().detach()),
        )

    def __add__(self, other: "Gradient") -> "Gradient":
        return Gradient(
            ions=(self.ions + other.ions),
            lattice=(None if (self.lattice is None) else self.lattice + other.lattice),
        )

    def __iadd__(self, other: "Gradient") -> "Gradient":
        self.ions += other.ions
        if self.lattice is not None:
            self.lattice += other.lattice
        return self

    def __sub__(self, other: "Gradient") -> "Gradient":
        return Gradient(
            ions=(self.ions - other.ions),
            lattice=(None if (self.lattice is None) else self.lattice - other.lattice),
        )

    def __isub__(self, other: "Gradient") -> "Gradient":
        self.ions -= other.ions
        if self.lattice is not None:
            self.lattice -= other.lattice
        return self

    def __mul__(self, other: float) -> "Gradient":
        return Gradient(
            ions=(self.ions * other),
            lattice=(None if (self.lattice is None) else self.lattice * other),
        )

    __rmul__ = __mul__

    def __imul__(self, other: float) -> "Gradient":
        self.ions *= other
        if self.lattice is not None:
            self.lattice *= other
        return self

    def vdot(self, other: "Gradient") -> float:
        result = self.ions.flatten() @ other.ions.flatten()
        if self.lattice is not None:
            assert other.lattice is not None
            result += self.lattice.flatten() @ other.lattice.flatten()
        return float(result.item())
