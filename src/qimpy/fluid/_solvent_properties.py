from typing import NamedTuple


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
