from __future__ import annotations
from dataclasses import dataclass
from typing import Union, ClassVar
import yaml
import re


@dataclass
class Unit:
    """Represent value and unit combination."""

    value: float
    unit: str

    def __repr__(self) -> str:
        return f"{self.value} * {self.unit}"

    def __float__(self) -> float:
        return self.value * Unit.MAP[self.unit]

    @staticmethod
    def parse(text: str) -> Unit:
        value_text, unit_text = text.split("*")  # ensures no extra / missing tokens
        value = float(value_text.rstrip())  # make sure value converts
        return Unit(value, unit_text.lstrip())  # unit matched when needed

    MAP: ClassVar[dict[str, float]] = {
        "Angstrom": 1.0 / 0.5291772
    }  #: Mapping from unit names to values


UnitOrFloat = Union[Unit, float]


def unit_representer(
    dumper: Union[yaml.Dumper, yaml.SafeDumper], unit: Unit
) -> yaml.ScalarNode:
    return dumper.represent_scalar("!unit", repr(unit))


def unit_constructor(loader, node) -> Unit:
    value = loader.construct_scalar(node)
    assert isinstance(value, str)
    return Unit.parse(value)


# Add reresenter (for dumping units in yaml):
yaml.add_representer(Unit, unit_representer)
yaml.SafeDumper.add_representer(Unit, unit_representer)

# Add constructor (for loading units in yaml):
yaml.add_constructor("!unit", unit_constructor)
yaml.SafeLoader.add_constructor("!unit", unit_constructor)

# Add implicit resolver (so that !unit is not needed):
unit_pattern = re.compile(r"-?(0|[1-9][0-9]*)(\.[0-9]*)?([eE][-+]?[0-9]+)?\s*\*\s*")
yaml.add_implicit_resolver("!unit", unit_pattern)
yaml.SafeLoader.add_implicit_resolver("!unit", unit_pattern, None)
