import os

import pytest
import torch

from qimpy import rc
from qimpy.math import dagger
from qimpy.dft import System
from qimpy.dft.ions import Ions


@pytest.fixture
def Si_system() -> System:
    prefixes = [""]
    if "QIMPY_PSEUDO_DIR" in os.environ:
        prefixes.extend(os.environ["QIMPY_PSEUDO_DIR"].split(":"))
    try:
        psp_filename = Ions.get_pseudopotential_filename(
            "Si", ["GBRV/$ID_pbe_v1.uspp"], prefixes
        )
    except ValueError:
        pytest.skip("GBRV pseudopotentials not found")
    return System(
        lattice=dict(system=dict(name="Cubic", modification="face-centered", a=10.35)),
        ions=dict(
            pseudopotentials=psp_filename,
            coordinates=[["Si", 0.0, 0.0, 0.0], ["Si", 0.25, 0.25, 0.25]],
        ),
        electrons=dict(
            basis=dict(ke_cutoff=10.0), k_mesh=dict(size=[2, 2, 2]), lcao=False
        ),
    )


def test_overlap(Si_system) -> None:
    """Check Hermitian, positive definite, dot_O agrees with applying O,
    and that orthonormalize works with an augmented overlap."""
    el = Si_system.electrons
    assert el.basis.ions.q_all is not None
    C = el.C.clone()
    C.randomize()
    C1, C2 = C[:, :, :3], C[:, :, 3:]
    O12 = C1.dot_O(C2).wait()
    assert torch.allclose(O12, dagger(C2.dot_O(C1).wait()))
    assert torch.allclose(O12, (C1 ^ C2.overlap()).wait())
    assert torch.linalg.eigvalsh(C.dot_O(C).wait()).min() > 0.0
    # Orthonormalization with augmented overlap:
    C_ortho = C.orthonormalize()
    eye = torch.eye(C.n_bands(), device=rc.device, dtype=O12.dtype)
    assert torch.allclose(C_ortho.dot_O(C_ortho).wait(), eye)
