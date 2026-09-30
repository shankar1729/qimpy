"""Overlap operator O for ultrasoft pseudopotentials, on silicon with the GBRV
pseudopotential when available (via QIMPY_PSEUDO_DIR, e.g. the JDFTx library)."""

import os
from functools import cache

import pytest
import torch

from qimpy import rc
from qimpy.math import dagger
from qimpy.dft import System

PSEUDOPOTENTIAL = "GBRV/$ID_pbe.uspp"


def find_pseudopotential() -> bool:
    """Whether the ultrasoft silicon pseudopotential can be located."""
    prefixes = [""] + os.environ.get("QIMPY_PSEUDO_DIR", "").split(":")
    filename = PSEUDOPOTENTIAL.replace("$ID", "si")
    return any(os.path.isfile(os.path.join(prefix, filename)) for prefix in prefixes)


@cache
def get_system() -> System:
    a = 5.175323929535814  # half the cubic lattice constant (bohr)
    return System(
        lattice=dict(Rbasis=[[0.0, a, a], [a, 0.0, a], [a, a, 0.0]]),
        ions=dict(
            pseudopotentials=PSEUDOPOTENTIAL,
            coordinates=[["Si", 0.0, 0.0, 0.0], ["Si", 0.25, 0.25, 0.25]],
        ),
        electrons=dict(
            basis=dict(ke_cutoff=10.0), k_mesh=dict(size=[2, 2, 2]), lcao=False
        ),
        grid=dict(ke_cutoff=40.0),
        checkpoint=None,
        checkpoint_out=None,
    )


@pytest.mark.mpi_skip
@pytest.mark.skipif(not find_pseudopotential(), reason="needs GBRV pseudopotential")
def test_overlap() -> None:
    """O is Hermitian and positive definite, dot_O agrees with applying O,
    orthonormalize works in the O metric, and cached projections are reused."""
    el = get_system().electrons
    assert el.basis.ions.q_all is not None
    C = el.C.clone()
    C.randomize()
    C1, C2 = C[:, :, :3], C[:, :, 3:]
    O_12 = C1.dot_O(C2).wait()
    assert torch.allclose(O_12, dagger(C2.dot_O(C1).wait()))
    assert torch.allclose(O_12, (C1 ^ C2.overlap()).wait())
    assert torch.linalg.eigvalsh(C.dot_O(C).wait()).min() > 0.0
    # Cached projections are used, not recomputed:
    proj = C.proj
    C.dot_O(C).wait()
    C.overlap()
    assert C._proj is proj
    # The ultrasoft part matters (the test is not trivially norm-conserving):
    assert not torch.allclose(C.dot_O(C).wait(), (C ^ C).wait())
    # Orthonormalization in the O metric:
    C_ortho = C.orthonormalize()
    eye = torch.eye(C.n_bands(), device=rc.device, dtype=torch.cdouble)
    assert torch.allclose(C_ortho.dot_O(C_ortho).wait(), eye)  # broadcast over s, k
