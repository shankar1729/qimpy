"""Calculate spherical harmonics and their product expansions."""

from functools import cache

import torch

from qimpy import rc
from . import _spherical_harmonics_data as shdata

# List exported symbols for doc generation
__all__ = [
    "L_MAX",
    "L_MAX_HLF",
    "get_harmonics",
    "get_harmonics_and_prime",
    "get_harmonics_tilde",
    "get_harmonics_tilde_and_prime",
    "get_product_coefficients",
]


L_MAX: int = shdata.L_MAX  #: Maximum l for calculation of harmonics
L_MAX_HLF: int = shdata.L_MAX_HLF  #: Maximum l of harmonics in products


@cache
def _get_YLM_RECUR() -> list[torch.Tensor]:
    """Get spherical harmonic coefficients as tensors on target device."""
    result = []
    for l, (i1, i2, coeff) in enumerate(shdata.YLM_RECUR):
        if l < 2:
            result.append(torch.tensor(coeff, device=rc.device))
        else:
            indices = torch.tensor((i1, i2), device=rc.device)
            result.append(
                torch.sparse_coo_tensor(
                    indices, coeff, size=(2 * l + 1, 3 * (2 * l - 1)), device=rc.device
                )
            )
    return result


def get_harmonics(l_max: int, r: torch.Tensor) -> torch.Tensor:
    """Compute real solid harmonics :math:`r^l Y_{lm}(r)` for each l <= l_max.
    Contains l=0, followed by all m for l=1, and so on till l_max along first dimension.
    Remaining dimensions are same as input `r`."""
    return get_harmonics_and_prime(l_max, r, compute_prime=False)[0]


def get_harmonics_and_prime(
    l_max: int, r: torch.Tensor, compute_prime=True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute real solid harmonics :math:`r^l Y_{lm}(r)` for each l <= l_max.
    First return value is harmonics with l=0, followed by all m for l=1, and so on
    till l_max along first dimension. Remaining dimensions are same as input `r`.
    Second return value is the derivative with respect to `r`, with the direction
    as a first additional dimension of length 3. (Empty if `compute_prime` is False.)
    """
    YLM_RECUR = _get_YLM_RECUR()
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
        Y[0] = YLM_RECUR[0]
        if compute_prime:
            Y_prime[:, 0] = 0.0
    if l_max >= 1:
        # l = 1: proportional to (y, z, x) for m = (-1, 0, +1):
        lm_slice = slice(1, 4)
        Y1 = YLM_RECUR[1] * r.flatten(0, -2).T[(1, 2, 0), :]
        Y[lm_slice] = Y1.unflatten(1, shape)
        if compute_prime:
            eye3 = torch.eye(3, dtype=r.dtype, device=r.device)
            Y1_prime = YLM_RECUR[1] * eye3[:, (1, 2, 0), None]
            Y_prime[:, lm_slice] = Y1_prime.unflatten(2, (1,) * len(shape))
            Yprev_prime = Y1_prime
        Yprev = Y1
    for l in range(2, l_max + 1):
        # l > 1: compute from product of harmonics at l = 1 and l - 1:
        lm_slice = slice(l**2, (l + 1) ** 2)
        Yprod = Yprev[:, None] * Y1  # outer product of l = 1 and l - 1
        Yl = YLM_RECUR[l] @ Yprod.flatten(0, 1)
        Y[lm_slice] = Yl.unflatten(1, shape)
        if compute_prime:
            Yprod_prime = (
                Yprev_prime[:, :, None] * Y1 + Yprev[:, None] * Y1_prime[:, None]
            )
            Yl_prime = (
                (YLM_RECUR[l] @ Yprod_prime.flatten(1, 2).transpose(0, 1).flatten(1))
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same as :func:`get_harmonics_and_prime`, but in reciprocal space.
    The result contains an extra phase of :math:`(iG)^l`, as discussed
    in `get_harmonics_tilde`."""
    Y, Y_prime = get_harmonics_and_prime(l_max, G, compute_prime)
    phase = _reciprocal_phase(l_max, G)
    return Y * phase, Y_prime * phase


@cache
def get_product_coefficients(l_max: int) -> torch.Tensor:
    """Expansion coefficients of products of real harmonics (real Gaunt coefficients)
    for l <= `l_max`, as a tensor of shape ((l_max+1)^2, (l_max+1)^2, (2 l_max+1)^2),
    with each (l, m) indexed by l(l+1)+m, as for the harmonics above."""
    assert l_max <= shdata.L_MAX_HLF
    n_lm = (l_max + 1) ** 2
    result = torch.zeros((n_lm, n_lm, (2 * l_max + 1) ** 2), device=rc.device)
    for (i_lm1, i_lm2), (i_LM, coeff) in shdata.YLM_PROD.items():
        if i_lm1 < n_lm:  # YLM_PROD stores only i_lm1 >= i_lm2; result is symmetric
            result[i_lm1, i_lm2, i_LM] = torch.tensor(coeff, device=rc.device)
            result[i_lm2, i_lm1, i_LM] = result[i_lm1, i_lm2, i_LM]
    return result


def _reciprocal_phase(l_max: int, G: torch.Tensor) -> torch.Tensor:
    """Return (-1)^l reciprocal-space phases shaped appropriately."""
    phase = []
    phase_cur = 1.0 + 0.0j
    for l in range(l_max + 1):
        phase.extend([phase_cur] * (2 * l + 1))  # repeated for m
        phase_cur *= 1.0j
    return torch.tensor(phase, device=G.device).view((-1,) + (1,) * (len(G.shape) - 1))
