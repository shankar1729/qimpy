import torch
import numbers
import numpy as np
from typing import Optional


def get_Rbasis(
    system: str,
    modification: Optional[str],
    a: Optional[float],
    b: Optional[float],
    c: Optional[float],
    alpha: Optional[float],
    beta: Optional[float],
    gamma: Optional[float],
) -> torch.Tensor:
    """Create lattice vectors from lattice system and modification"""

    def check_needed(**kwargs):
        """Check if all needed arguments are provided"""
        for key, value in kwargs.items():
            if value is None:
                raise KeyError(system + " lattice system requires parameter " + key)
            if not isinstance(value, numbers.Number):
                raise TypeError("Lattice paramater " + key + " must be numeric")
            if value <= 0.0:
                raise ValueError("Lattice paramater " + key + " must be > 0")

    def check_spurious(**kwargs):
        """Check if any spurious arguments are provided"""
        for key, value in kwargs.items():
            if value is not None:
                raise KeyError(
                    system + " lattice system does not require" " parameter " + key
                )

    def check_modification(allowed_systems):
        """Check compatibility of modification with lattice system"""
        if system not in allowed_systems:
            raise KeyError(
                modification + " modification not allowed for " + system + " lattices"
            )

    # Check inputs and get a, b, c, alpha, beta, gamma for all cases:
    assert isinstance(system, str)
    system = system.lower()
    if system == "triclinic":
        check_needed(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
    elif system == "monoclinic":
        check_needed(a=a, b=b, c=c, beta=beta)
        check_spurious(alpha=alpha, gamma=gamma)
        alpha = gamma = 90.0
    elif system == "orthorhombic":
        check_needed(a=a, b=b, c=c)
        check_spurious(alpha=alpha, beta=beta, gamma=gamma)
        alpha = beta = gamma = 90.0
    elif system == "tetragonal":
        check_needed(a=a, c=c)
        check_spurious(b=b, alpha=alpha, beta=beta, gamma=gamma)
        b = a
        alpha = beta = gamma = 90.0
    elif system == "rhombohedral":
        check_needed(a=a, alpha=alpha)
        check_spurious(b=b, c=c, beta=beta, gamma=gamma)
        b = c = a
        beta = gamma = alpha
    elif system == "hexagonal":
        check_needed(a=a, c=c)
        check_spurious(b=b, alpha=alpha, beta=beta, gamma=gamma)
        b = a
        alpha = beta = 90.0
        gamma = 120.0
    elif system == "cubic":
        check_needed(a=a)
        check_spurious(b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
        b = c = a
        alpha = beta = gamma = 90.0
    else:
        raise KeyError("Unknown lattice system: " + system)

    # Confirm that all geometry parameters are now available:
    assert isinstance(a, numbers.Number)
    assert isinstance(b, numbers.Number)
    assert isinstance(c, numbers.Number)
    assert isinstance(alpha, numbers.Number)
    assert isinstance(beta, numbers.Number)
    assert isinstance(gamma, numbers.Number)

    # Compute base lattice vectors:
    cos_alpha = np.cos(np.deg2rad(alpha))
    cos_beta = np.cos(np.deg2rad(beta))
    cos_gamma = np.cos(np.deg2rad(gamma))
    sin_gamma = np.sin(np.deg2rad(gamma))
    v0 = np.array((1.0, 0, 0))
    v1 = np.array((cos_gamma, sin_gamma, 0))
    v2 = np.array((cos_beta, (cos_alpha - cos_beta * cos_gamma) / sin_gamma, 0))
    v2[2] = np.sqrt(1 - (v2**2).sum())
    Rbasis = torch.tensor(np.array([a * v0, b * v1, c * v2]).T)

    # Apply modifications if any:
    if modification is None:
        M = torch.eye(3)  # transformation from base lattice
    else:
        assert isinstance(modification, str)
        modification = modification.lower()
        if modification == "body-centered":
            check_modification(["orthorhombic", "tetragonal", "cubic"])
            M = 0.5 * torch.tensor([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
        elif modification == "base-centered":
            check_modification(["monoclinic"])
            M = 0.5 * torch.tensor([[1, -1, 0], [1, 1, 0], [0, 0, 2]])
        elif modification == "face-centered":
            check_modification(["orthorhombic", "cubic"])
            M = 0.5 * torch.tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        else:
            raise KeyError("Unknown lattice modification: " + modification)

    return Rbasis @ M
