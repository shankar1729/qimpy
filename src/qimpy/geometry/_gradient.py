import torch
from dataclasses import dataclass
from typing import Optional, ClassVar


@dataclass
class Gradient:
    """Geometry gradient used for relaxation / dynamics."""

    ions: torch.Tensor  #: ionic gradient (forces)
    lattice: Optional[torch.Tensor] = None  #: lattice gradient (stress)
    thermostat: Optional[torch.Tensor] = None  #: thermostat gradient (e.g. Nose-Hoover)
    barostat: Optional[torch.Tensor] = None  #: barostat gradient (e.g. Nose-Hoover)
    OPTIONAL_ATTRIBUTE_NAMES: ClassVar[set[str]] = {"lattice", "thermostat", "barostat"}

    def clone(self) -> "Gradient":
        result = Gradient(ions=self.ions.clone().detach())
        for attribute_name in Gradient.OPTIONAL_ATTRIBUTE_NAMES:
            if (self_attribute := getattr(self, attribute_name)) is not None:
                setattr(result, attribute_name, self_attribute.clone().detach())
        return result

    def __add__(self, other: "Gradient") -> "Gradient":
        result = Gradient(ions=(self.ions + other.ions))
        for attribute_name in Gradient.OPTIONAL_ATTRIBUTE_NAMES:
            if (self_attribute := getattr(self, attribute_name)) is not None:
                other_attribute = getattr(other, attribute_name)
                assert other_attribute is not None
                setattr(result, attribute_name, self_attribute + other_attribute)
        return result

    def __iadd__(self, other: "Gradient") -> "Gradient":
        self.ions += other.ions
        for attribute_name in Gradient.OPTIONAL_ATTRIBUTE_NAMES:
            if (self_attribute := getattr(self, attribute_name)) is not None:
                other_attribute = getattr(other, attribute_name)
                assert other_attribute is not None
                self_attribute += other_attribute
        return self

    def __sub__(self, other: "Gradient") -> "Gradient":
        result = Gradient(ions=(self.ions - other.ions))
        for attribute_name in Gradient.OPTIONAL_ATTRIBUTE_NAMES:
            if (self_attribute := getattr(self, attribute_name)) is not None:
                other_attribute = getattr(other, attribute_name)
                assert other_attribute is not None
                setattr(result, attribute_name, self_attribute - other_attribute)
        return result

    def __isub__(self, other: "Gradient") -> "Gradient":
        self.ions -= other.ions
        for attribute_name in Gradient.OPTIONAL_ATTRIBUTE_NAMES:
            if (self_attribute := getattr(self, attribute_name)) is not None:
                other_attribute = getattr(other, attribute_name)
                assert other_attribute is not None
                self_attribute -= other_attribute
        return self

    def __mul__(self, other: float) -> "Gradient":
        result = Gradient(ions=(self.ions * other))
        for attribute_name in Gradient.OPTIONAL_ATTRIBUTE_NAMES:
            if (self_attribute := getattr(self, attribute_name)) is not None:
                setattr(result, attribute_name, self_attribute * other)
        return result

    __rmul__ = __mul__

    def __imul__(self, other: float) -> "Gradient":
        self.ions *= other
        for attribute_name in Gradient.OPTIONAL_ATTRIBUTE_NAMES:
            if (self_attribute := getattr(self, attribute_name)) is not None:
                self_attribute *= other
        return self

    def vdot(self, other: "Gradient") -> float:
        result = self.ions.flatten() @ other.ions.flatten()
        for attribute_name in Gradient.OPTIONAL_ATTRIBUTE_NAMES:
            if (self_attribute := getattr(self, attribute_name)) is not None:
                other_attribute = getattr(other, attribute_name)
                assert other_attribute is not None
                result += self_attribute.flatten() @ other_attribute.flatten()
        return float(result.item())
