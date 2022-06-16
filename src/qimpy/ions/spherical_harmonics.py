"""Calculate spherical harmonics and their product expansions."""
from __future__ import annotations
import qimpy.ions._spherical_harmonics_data as shdata
import torch
from typing import List, Tuple, Dict

# List exported symbols for doc generation
__all__ = ["L_MAX", "L_MAX_HLF", "get_harmonics"]


# Versions of shdata converted to torch.Tensors on appropriate device
_YLM_RECUR: List[torch.Tensor] = []
_YLM_PROD: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
L_MAX: int = shdata.L_MAX  #: Maximum l for calculation of harmonics
L_MAX_HLF: int = shdata.L_MAX_HLF  #: Maximum l of harmonics in products


def _initialize_device(device: torch.device) -> None:
    """Initialize spherical harmonic data as torch tensors on device"""
    global _YLM_RECUR, _YLM_PROD
    # Recurrence coefficients:
    _YLM_RECUR.clear()
    for l, (i1, i2, coeff) in enumerate(shdata.YLM_RECUR):
        if l < 2:
            _YLM_RECUR.append(torch.tensor(coeff, device=device))
        else:
            indices = torch.tensor((i1, i2), device=device)
            _YLM_RECUR.append(
                torch.sparse_coo_tensor(
                    indices, coeff, size=(2 * l + 1, 3 * (2 * l - 1)), device=device
                )
            )
    # Product coefficients:
    _YLM_PROD.clear()
    for ilm_pair, (ilm, coeff) in shdata.YLM_PROD.items():
        _YLM_PROD[ilm_pair] = (
            torch.tensor(ilm, device=device),
            torch.tensor(coeff, device=device),
        )


def get_harmonics(l_max: int, r: torch.Tensor) -> torch.Tensor:
    """Compute real solid harmonics :math:`r^l Y_{lm}(r)` for each l <= l_max.
    Contains l=0, followed by all m for l=1, and so on till l_max."""
    if not _YLM_PROD:
        _initialize_device(r.device)
    assert l_max <= shdata.L_MAX
    result = torch.empty(
        ((l_max + 1) ** 2,) + r.shape[:-1], dtype=r.dtype, device=r.device
    )
    if l_max >= 0:
        # l = 0: constant
        result[0] = _YLM_RECUR[0]
    if l_max >= 1:
        # l = 1: proportional to (y, z, x) for m = (-1, 0, +1):
        Y1 = (_YLM_RECUR[1] * r.flatten(0, -2).T[(1, 2, 0), :]).view(
            (3,) + r.shape[:-1]
        )
        result[1:4] = Y1
        Yprev = Y1
    for l in range(2, l_max + 1):
        # l > 1: compute from product of harmonics at l = 1 and l - 1:
        Yl = (
            _YLM_RECUR[l]
            @ (Yprev[:, None, :] * Y1[None, :, :]).view(3 * (2 * l - 1), -1)
        ).view((2 * l + 1,) + r.shape[:-1])
        result[l ** 2 : (l + 1) ** 2] = Yl
        Yprev = Yl
    return result


def get_harmonics_tilde(l_max: int, G: torch.Tensor) -> torch.Tensor:
    """Same as :func:`get_harmonics`, but in reciprocal space. The result
    is a complex tensor containing :math:`(iG)^l Y_{lm}(G)`, where the extra
    phase factor is from the Fourier transform of spherical harmonics. This is
    required for the corresponding real-space version to be real."""
    return get_harmonics(l_max, G) * _reciprocal_phase(l_max, G)


def _reciprocal_phase(l_max: int, G: torch.Tensor) -> torch.Tensor:
    """Return (-1)^l reciprocal-space phases shaped appropriately."""
    phase = []
    phase_cur = 1.0 + 0.0j
    for l in range(l_max + 1):
        phase.extend([phase_cur] * (2 * l + 1))  # repeated for m
        phase_cur *= 1.0j
    return torch.tensor(phase, device=G.device).view((-1,) + (1,) * (len(G.shape) - 1))
