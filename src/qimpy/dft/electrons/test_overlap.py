import torch

from qimpy import rc
from qimpy.math import dagger


def test_overlap(system_Si) -> None:
    """Check Hermitian, positive definite, dot_O agrees with applying O,
    and that orthonormalize works with an augmented overlap."""
    el = system_Si.electrons
    assert el.basis.ions.augmentation is not None
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
