from typing import Optional

import torch
import numpy as np


def get_Rbasis(*, name: str, **kwargs) -> torch.Tensor:
    """Create lattice vectors from lattice system and modification"""
    return {
        "cubic": _get_Rbasis_cubic,
        "tetragonal": _get_Rbasis_tetragonal,
        "orthorhombic": _get_Rbasis_orthorhombic,
        "hexagonal": _get_Rbasis_hexagonal,
        "rhombohedral": _get_Rbasis_rhombohedral,
        "monoclinic": _get_Rbasis_monoclinic,
        "triclinic": _get_Rbasis_triclinic,
    }[name.lower()](
        **kwargs
    )  # type: ignore


def _get_Rbasis_cubic(*, a: float, modification: Optional[str] = None) -> torch.Tensor:
    Rbasis = _get_Rbasis_lengths_angles(a, a, a)
    if modification is None:
        return Rbasis
    else:
        M = {
            "body-centered": 0.5 * torch.tensor([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]),
            "face-centered": 0.5 * torch.tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
        }[modification.lower()]
        return Rbasis @ M


def _get_Rbasis_tetragonal(
    *, a: float, c: float, modification: Optional[str] = None
) -> torch.Tensor:
    Rbasis = _get_Rbasis_lengths_angles(a, a, c)
    if modification is None:
        return Rbasis
    else:
        assert modification.lower() == "body-centered"
        M = 0.5 * torch.tensor([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
        return Rbasis @ M


def _get_Rbasis_orthorhombic(
    *, a: float, b: float, c: float, modification: Optional[str] = None
) -> torch.Tensor:
    Rbasis = _get_Rbasis_lengths_angles(a, b, c)
    if modification is None:
        return Rbasis
    else:
        M = {
            "body-centered": 0.5 * torch.tensor([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]),
            "base-centered": 0.5 * torch.tensor([[1, -1, 0], [1, 1, 0], [0, 0, 2]]),
            "face-centered": 0.5 * torch.tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
        }[modification.lower()]
        return Rbasis @ M


def _get_Rbasis_hexagonal(*, a: float, c: float) -> torch.Tensor:
    return _get_Rbasis_lengths_angles(a, a, c, gamma=(2 / 3 * np.pi))


def _get_Rbasis_rhombohedral(*, a: float, alpha: float) -> torch.Tensor:
    return _get_Rbasis_lengths_angles(a, a, a, alpha, alpha, alpha)


def _get_Rbasis_monoclinic(
    *, a: float, b: float, c: float, beta: float, modification: Optional[str] = None
) -> torch.Tensor:
    Rbasis = _get_Rbasis_lengths_angles(a, b, c, beta=beta)
    if modification is None:
        return Rbasis
    else:
        assert modification.lower() == "base-centered"
        M = 0.5 * torch.tensor([[1, -1, 0], [1, 1, 0], [0, 0, 2]])
        return Rbasis @ M


def _get_Rbasis_triclinic(
    *, a: float, b: float, c: float, alpha: float, beta: float, gamma: float
) -> torch.Tensor:
    return _get_Rbasis_lengths_angles(a, b, c, alpha, beta, gamma)


def _get_Rbasis_lengths_angles(
    a: float,
    b: float,
    c: float,
    alpha: float = 0.5 * np.pi,
    beta: float = 0.5 * np.pi,
    gamma: float = 0.5 * np.pi,
) -> torch.Tensor:
    cos_alpha = np.cos(alpha)
    cos_beta = np.cos(beta)
    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)
    v0 = np.array((1.0, 0, 0))
    v1 = np.array((cos_gamma, sin_gamma, 0))
    v2 = np.array((cos_beta, (cos_alpha - cos_beta * cos_gamma) / sin_gamma, 0))
    v2[2] = np.sqrt(1 - (v2**2).sum())
    return torch.tensor(np.array([a * v0, b * v1, c * v2]).T)
