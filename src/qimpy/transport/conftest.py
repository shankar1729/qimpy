import pytest
import torch

from .. import rc


@pytest.fixture(scope="package", autouse=True)
def with_default_device():
    # Transport tests build bare tensors without an explicit device; place them on the
    # run device so the suite is correct on GPU. No-op on CPU (rc.device == cpu).
    with torch.device(rc.device):
        yield
