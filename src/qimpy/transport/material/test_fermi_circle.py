from __future__ import annotations

import numpy as np
import torch
import pytest

from qimpy import log, rc
from qimpy.io import log_config
from qimpy.mpi import get_process_grid
from qimpy.transport.material import FermiCircle

pytestmark = pytest.mark.mpi_skip  # All tests in this module are serial


def get_fermi_circle(**kwargs) -> FermiCircle:
    """Wrap FermiCircle construction, handling process grid and adding more defaults."""
    kwargs.setdefault("n_theta", 32)
    kwargs.setdefault("kF", 1.5)
    kwargs.setdefault("vF", 0.8)
    kwargs.setdefault("dk_r", 0.1)
    kwargs.setdefault("T", kwargs["vF"] * kwargs["dk_r"])
    return FermiCircle(**kwargs, process_grid=get_process_grid("rk", (1, 1)))


def test_contact_n_r_equivalence() -> None:
    """Verify that contact distribution at large n_r > 1 matches n_r = 1."""
    t = 0.0
    dmu = 1.0e-6
    n_hat = torch.tensor([[1.0, 0.0]], device=rc.device)
    fc0 = get_fermi_circle(n_r=1)
    n_ref = fc0.measure_observables(fc0.get_contactor(n_hat, dmu=dmu)(t), t)
    for n_r in [2, 4, 8, 16, 32]:
        fc = get_fermi_circle(n_r=n_r)
        n_check = fc.measure_observables(fc.get_contactor(n_hat, dmu=dmu)(t), t)[:, 0]
        relative_error = (n_check / n_ref - 1.0).item()
        log.info(f"{n_r=}: {relative_error=:.3e}")
        assert abs(relative_error) < 2 * np.exp(-0.5 * n_ref / dmu)


def main():
    log_config()
    pytest.main([__file__, "-vs"])


if __name__ == "__main__":
    main()
