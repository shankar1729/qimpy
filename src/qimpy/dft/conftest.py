import pytest

from qimpy.dft import System
from qimpy.dft.ions import Pseudopotential


@pytest.fixture
def system_Si() -> System:
    return make_system_Si()


@pytest.fixture
def system_Si_perturbed() -> System:
    return make_system_Si(perturbed=True)


def make_system_Si(perturbed: bool = False) -> System:
    try:
        psp_filename = Pseudopotential.get_filename("Si", ["GBRV/$ID_pbe_v1.uspp"])
    except ValueError:
        pytest.skip("GBRV pseudopotentials not found")
    delta = 0.01 if perturbed else 0.0  # lower symmetries by displacing an atom
    return System(
        lattice=dict(system=dict(name="Cubic", modification="face-centered", a=10.35)),
        ions=dict(
            pseudopotentials=psp_filename,
            coordinates=[["Si", 0.0, 0.0, 0.0], ["Si", 0.25, 0.25, 0.25 + delta]],
        ),
        electrons=dict(basis=dict(ke_cutoff=10.0), k_mesh=dict(size=[2, 2, 2])),
    )
