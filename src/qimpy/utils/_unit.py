from __future__ import annotations
from dataclasses import dataclass
from typing import Union, ClassVar
import math
import yaml
import re


@dataclass
class Unit:
    """Represent value and unit combination."""

    value: float
    unit: str

    def __repr__(self) -> str:
        return f"{self.value} {self.unit}"

    def __float__(self) -> float:
        return self.value * Unit.MAP[self.unit]

    @staticmethod
    def parse(text: str) -> Unit:
        value_text, unit_text = text.split()  # ensures no extra / missing tokens
        value = float(value_text)  # make sure value converts
        return Unit(value, unit_text)  # unit matched when needed

    MAP: ClassVar[dict[str, float]] = {}  #: Mapping from unit names to values


UnitOrFloat = Union[Unit, float]


def unit_representer(
    dumper: Union[yaml.Dumper, yaml.SafeDumper], unit: Unit
) -> yaml.ScalarNode:
    return dumper.represent_scalar("!unit", repr(unit))


def unit_constructor(loader, node) -> Unit:
    value = loader.construct_scalar(node)
    assert isinstance(value, str)
    return Unit.parse(value)


# Add representer (for dumping units in yaml):
yaml.add_representer(Unit, unit_representer)
yaml.SafeDumper.add_representer(Unit, unit_representer)

# Add constructor (for loading units in yaml):
yaml.add_constructor("!unit", unit_constructor)
yaml.SafeLoader.add_constructor("!unit", unit_constructor)

# Add implicit resolver (so that !unit is not needed):
unit_pattern = re.compile(r"[-+]?(0|[1-9][0-9]*)(\.[0-9]*)?([eE][-+]?[0-9]+)?\s+\S+")
yaml.add_implicit_resolver("!unit", unit_pattern)
yaml.SafeLoader.add_implicit_resolver("!unit", unit_pattern, None)


def _initialize_unit_map():
    """Initialize values of non-atomic units in terms of atomic units."""

    # Fundamental constants:
    e = 1.0
    m_e = 1.0
    hbar = 1.0
    h = 2.0 * math.pi * hbar
    a_0 = 1.0
    E_h = 1.0
    Unit.MAP = dict(e=e, m_e=m_e, hbar=hbar, h=h, a_0=a_0, E_h=E_h)

    # SI defining units [NIST values from https://physics.nist.gov/cuu/Constants]:
    m = 1.0 / 5.29177210903e-11  # using Bohr radius reference value
    kg = 1.0 / 9.1093837015e-31  # using electron mass reference value
    C = 1.0 / 1.602176634e-19  # using electron charge reference value
    s = 1.0 / 2.4188843265857e-17  # using Hz-Hartree reference relation (/2pi)
    A = C / s
    J = kg * (m / s) ** 2
    c = 299792458.0 * m / s  # using SI definition of light speed
    K = J * 1.380649e-23  # using SI definition of Boltzmann constant
    mol = 6.02214076e23  # using SI definition of Avogadro number
    amu = 1.66053906660e-27 * kg  # using atomic mass constant reference value
    Unit.MAP.update(m=m, kg=kg, C=C, s=s, A=A, J=J, c=c, K=K, mol=mol, amu=amu)

    # Derived units:
    Angstrom = 1e-10 * m
    V = J / C
    Ohm = V / A
    mu_B = e * hbar / (2.0 * m_e)
    Pa = J / (m**3)
    Unit.MAP.update(
        Angstrom=Angstrom,
        nm=1e-9 * m,
        L=(0.1 * m) ** 3,
        Ha=E_h,
        Ry=0.5 * E_h,
        eV=e * V,
        Hz=1.0 / s,
        ps=1e-12 * s,
        fs=1e-15 * s,
        N=J / m,
        Pa=Pa,
        kPa=1e3 * Pa,
        bar=1e5 * Pa,
        MPa=1e6 * Pa,
        GPa=1e9 * Pa,
        mmHg=133.322387415 * Pa,
        V=V,
        Ohm=Ohm,
        T=kg / (A * (s**2)),
        mu_B=mu_B,
        g_e=2.00231930436256,
    )

    # Derived units with non-Python-identifier/unicode names:
    Unit.MAP["kJ/mol"] = 1e3 * J / mol
    Unit.MAP["kcal/mol"] = 4.184e3 * J / mol  # thermochemical calorie definition
    Unit.MAP["cm^-1"] = h * c / (1e-2 * m)
    Unit.MAP["Å"] = Angstrom
    Unit.MAP["Ω"] = Ohm
    Unit.MAP["μ_B"] = mu_B


_initialize_unit_map()
