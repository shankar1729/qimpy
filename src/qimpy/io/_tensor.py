from __future__ import annotations
from typing import Union, Sequence

import torch
import numpy as np

from qimpy import rc
from . import InvalidInputException


TensorCompatible = Union[torch.Tensor, np.ndarray, float, Sequence[float]]


def cast_tensor(t: TensorCompatible) -> torch.Tensor:
    """Convert `t` to a torch tensor on current device (if not already so).
    Useful to handle input from yaml, checkpoint or python code on an equal footing."""
    if isinstance(t, torch.Tensor):
        return t.to(rc.device)
    if isinstance(t, np.ndarray):
        return torch.from_numpy(t).to(rc.device)
    try:
        return torch.tensor(t, device=rc.device)
    except ValueError:
        raise InvalidInputException(f"Could not convert {t} to a tensor")
