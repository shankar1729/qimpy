from typing import NamedTuple, Optional

from qimpy.io import Unit, InvalidInputException


class DielectricProperty(NamedTuple):
    """Solvent dielectric properties used in Linear and Nonlinear fluid models."""

    epsilon_0: float  #: Static dielectric constant
    epsilon_inf: float  #: Optical dielectric constant
    p_mol: float  #: Dipole moment per molecule
    N_bulk: float  #: Bulk number density of molecules


DIELECTRIC_PROPERTIES: dict[str, DielectricProperty] = {
    "H2O": DielectricProperty(
        epsilon_0=78.4, epsilon_inf=1.77, p_mol=0.92466, N_bulk=4.9383e-3
    ),
}


class IonProperty(NamedTuple):
    """Ion properties used in Linear and Nonlinear fluid models."""

    Z: float  #: Charge on ion
    Rhs: float  #: Hard-sphere radius of ion (to set packing limit)


ION_PROPERTIES: dict[str, IonProperty] = {
    "Na+": IonProperty(Z=+1.0, Rhs=1.16 * Unit.MAP["Angstrom"]),
    "Cl-": IonProperty(Z=-1.0, Rhs=1.67 * Unit.MAP["Angstrom"]),
}


def set_solvent_properties(
    solvent: str,
    property_map: dict[str, NamedTuple],
    specified_values: dict[str, Optional[float]],
    model,
) -> None:
    """Merge default and specified solvent properties/parameters.
    Get default properties from `property_map` for `solvent`.
    Replace them by any explicitly specified values from `specified_values`.
    Ensure that every property has been set (if solvent is not specified)
    and then set all of them as attrs of `model`."""
    if solvent:
        if solvent not in property_map:
            raise InvalidInputException(
                f"{solvent = } not parameterized for {model.__class__.__name__}."
                f" Available options: {', '.join(property_map.keys())}"
            )
        defaults = property_map[solvent]
        for key, value in specified_values.items():
            if value is None:
                specified_values[key] = getattr(defaults, key)

    for key, value in specified_values.items():
        if value is None:
            raise InvalidInputException(
                f"{key} must be specified explicitly for {model.__class__.__name__}"
                " when solvent is not specified."
            )
        setattr(model, key, value)
