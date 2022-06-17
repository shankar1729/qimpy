"""Calculate spherical harmonics and their product expansions."""
from __future__ import annotations
import qimpy.ions._spherical_harmonics_data as shdata
import torch
from typing import List, Tuple, Dict

# List exported symbols for doc generation
__all__ = [
    "L_MAX",
    "L_MAX_HLF",
    "get_harmonics",
    "get_harmonics_and_prime",
    "get_harmonics_tilde",
    "get_harmonics_tilde_and_prime",
]


# Versions of shdata converted to torch.Tensors on appropriate device
_YLM_RECUR: List[torch.Tensor] = []
_YLM_PROD: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
L_MAX: int = shdata.L_MAX  #: Maximum l for calculation of harmonics
L_MAX_HLF: int = shdata.L_MAX_HLF  #: Maximum l of harmonics in products


def _initialize_device(device: torch.device) -> None:
    """Initialize spherical harmonic data as torch tensors on device."""
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
    Contains l=0, followed by all m for l=1, and so on till l_max along first dimension.
    Remaining dimensions are same as input `r`."""
    return get_harmonics_and_prime(l_max, r, compute_prime=False)[0]


def get_harmonics_and_prime(
    l_max: int, r: torch.Tensor, compute_prime=True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute real solid harmonics :math:`r^l Y_{lm}(r)` for each l <= l_max.
    First return value is harmonics with l=0, followed by all m for l=1, and so on
    till l_max along first dimension. Remaining dimensions are same as input `r`.
    Second return value is the derivative with respect to `r`, with the direction
    as a first additional dimension of length 3. (Empty if `compute_prime` is False.)
    """
    if not _YLM_PROD:
        _initialize_device(r.device)
    assert l_max <= shdata.L_MAX
    shape = r.shape[:-1]  # shape of r except for Cartesian direction
    Y = torch.empty(((l_max + 1) ** 2,) + shape, dtype=r.dtype, device=r.device)
    Y_prime = (
        torch.empty((3,) + Y.shape, dtype=r.dtype, device=r.device)
        if compute_prime
        else torch.zeros((1,), dtype=r.dtype, device=r.device)
    )
    if l_max >= 0:
        # l = 0: constant
        Y[0] = _YLM_RECUR[0]
        if compute_prime:
            Y_prime[:, 0] = 0.0
    if l_max >= 1:
        # l = 1: proportional to (y, z, x) for m = (-1, 0, +1):
        lm_slice = slice(1, 4)
        Y1 = _YLM_RECUR[1] * r.flatten(0, -2).T[(1, 2, 0), :]
        Y[lm_slice] = Y1.unflatten(1, shape)
        if compute_prime:
            eye3 = torch.eye(3, dtype=r.dtype, device=r.device)
            Y1_prime = _YLM_RECUR[1] * eye3[:, (1, 2, 0), None]
            Y_prime[:, lm_slice] = Y1_prime.unflatten(2, (1,) * len(shape))
            Yprev_prime = Y1_prime
        Yprev = Y1
    for l in range(2, l_max + 1):
        # l > 1: compute from product of harmonics at l = 1 and l - 1:
        lm_slice = slice(l ** 2, (l + 1) ** 2)
        Yprod = Yprev[:, None] * Y1  # outer product of l = 1 and l - 1
        Yl = _YLM_RECUR[l] @ Yprod.flatten(0, 1)
        Y[lm_slice] = Yl.unflatten(1, shape)
        if compute_prime:
            Yprod_prime = (
                Yprev_prime[:, :, None] * Y1 + Yprev[:, None] * Y1_prime[:, None]
            )
            Yl_prime = (
                (_YLM_RECUR[l] @ Yprod_prime.flatten(1, 2).transpose(0, 1).flatten(1))
                .unflatten(1, (3, -1))
                .transpose(0, 1)
            )
            Y_prime[:, lm_slice] = Yl_prime.unflatten(2, shape)
            Yprev_prime = Yl_prime
        Yprev = Yl
    return Y, Y_prime


def get_harmonics_tilde(l_max: int, G: torch.Tensor) -> torch.Tensor:
    """Same as :func:`get_harmonics`, but in reciprocal space. The result
    is a complex tensor containing :math:`(iG)^l Y_{lm}(G)`, where the extra
    phase factor is from the Fourier transform of spherical harmonics. This is
    required for the corresponding real-space version to be real."""
    return get_harmonics(l_max, G) * _reciprocal_phase(l_max, G)


def get_harmonics_tilde_and_prime(
    l_max: int, G: torch.Tensor, compute_prime=True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same as :func:`get_harmonics_and_prime`, but in reciprocal space.
    The result contains an extra phase of :math:`(iG)^l`, as discussed
    in `get_harmonics_tilde`."""
    Y, Y_prime = get_harmonics_and_prime(l_max, G, compute_prime)
    phase = _reciprocal_phase(l_max, G)
    return Y * phase, Y_prime * phase


def _reciprocal_phase(l_max: int, G: torch.Tensor) -> torch.Tensor:
    """Return (-1)^l reciprocal-space phases shaped appropriately."""
    phase = []
    phase_cur = 1.0 + 0.0j
    for l in range(l_max + 1):
        phase.extend([phase_cur] * (2 * l + 1))  # repeated for m
        phase_cur *= 1.0j
    return torch.tensor(phase, device=G.device).view((-1,) + (1,) * (len(G.shape) - 1))
