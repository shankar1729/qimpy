import pytest
import torch

from qimpy import rc


@pytest.fixture(scope="package", autouse=True)
def with_default_device():
    # Disable default device for e-e tests which assume certain tensors on CPU.
    with torch.device(rc.cpu):
        yield
