"""Fast random numbers for reproducible initialization."""
# List exported symbols for doc generation
__all__ = ["initialize_state", "rand", "randn"]

import torch
import numpy as np


def initialize_state(state: torch.Tensor) -> None:
    """Improve initial state randomness using split-mix-64 method."""
    assert state.dtype is torch.int64
    state += -7046029254386353131
    state ^= state >> 30
    state *= -4658895280553007687
    state ^= state >> 27
    state *= -7723592293110705685
    state ^= state >> 31


def rand(state: torch.Tensor) -> torch.Tensor:
    """Generate real uniform random Tensor using integer tensor `state`.
    Uses a 64-bit xor-shift generator in parallel for fast generation."""
    state ^= state << 13
    state ^= state >> 7
    state ^= state << 17
    return 0.5 + state * (0.5**64)


def randn(state: torch.Tensor) -> torch.Tensor:
    """Generate complex standard-normal random Tensor using integer tensor `state`."""
    magnitude = torch.sqrt(-2.0 * torch.log(rand(state)))
    phase = (2 * np.pi) * rand(state)
    return torch.polar(magnitude, phase)
