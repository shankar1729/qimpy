"""Calculate spherical harmonics and their product expansions."""
from __future__ import annotations
import qimpy.ions._spherical_harmonics_data as shdata
import numpy as np
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


def get_harmonics_prime(l_max: int, r: torch.Tensor) -> torch.Tensor:
    """Compute derivative of `get_harmonics` result with respect to `r`.
    The derivative dimension of length 3 is first, followed by those
    in the result of `get_harmonics`."""
    Ylm = get_harmonics(l_max - 1, r)
    n_lm = (l_max + 1) ** 2  # total number of Ylm up to l_max + 1
    Ylm_prime = torch.zeros((3, n_lm) + r.shape[:-1], dtype=r.dtype, device=r.device)
    bcast_shape = (-1,) + (1,) * (len(r.shape) - 1)

    # Use recurrence relations to set non-zero Ylm derivatives:
    for l in range(1, l_max + 1):
        i0 = l * (l + 1)  # index of m = 0 component at l (indexing Ylm_prime output)
        i0_in = (l - 1) * l  # index of m = 0 component at l - 1  (indexing Ylm input)
        norm_fac = (2 * l + 1) / (2 * l - 1)  # common factor in C-G coefficients below

        # Change m = 0 norm at input to simplify factors:
        Ylm[i0_in] *= np.sqrt(2.0)

        # z-component (same form for all m):
        m = torch.arange(-l + 1, l, dtype=torch.long, device=r.device)
        alpha = ((l * l - m.square()) * norm_fac).sqrt().view(bcast_shape)
        Ylm_prime[2, i0 + m] = alpha * Ylm[i0_in + m]

        # x, y components (m-dependent formulae because of real/imag in real harmonics):
        m = torch.arange(-l, l - 1, dtype=torch.long, device=r.device)
        alpha = 0.5 * ((l - m) * (l - m - 1) * norm_fac).sqrt().view(bcast_shape)
        Ylm_prime[0, i0 + m] -= Ylm[i0_in + m + 1] * alpha
        Ylm_prime[0, i0 - m] += Ylm[i0_in - (m + 1)] * alpha
        Ylm_prime[1, i0 + m] -= Ylm[i0_in - (m + 1)] * alpha
        Ylm_prime[1, i0 - m] -= Ylm[i0_in + m + 1] * alpha

        # Correct exceptions near m = 0 to above formulae:
        if l > 1:
            Ylm_1, Ylm0, Ylm1 = Ylm[i0_in - 1 : i0_in + 2]  # m = -1, 0, +1
            alpha_1, alpha0 = alpha[l - 1 : l + 1]  # m = -1, 0
            Ylm_prime[0, i0] -= (Ylm1 + Ylm_1) * alpha0
            Ylm_prime[1, i0] += (Ylm1 - Ylm_1) * alpha0
            Ylm_prime[0, i0 - 1] += Ylm0 * alpha_1
            Ylm_prime[1, i0 + 1] += Ylm0 * alpha_1
        else:
            Ylm0 = Ylm[i0_in]  # m = 0
            alpha_1 = alpha[0]  # m = -1
            Ylm_prime[0, i0 - 1] += Ylm0 * alpha_1
            Ylm_prime[1, i0 + 1] += Ylm0 * alpha_1

        # Restore m = 0 norm of output and flip m < 0 sign to simplify factors above:
        Ylm_prime[:, i0] *= np.sqrt(0.5)
        Ylm_prime[:2, i0 - l : i0] *= -1.0
    return Ylm_prime


def get_harmonics_tilde_prime(l_max: int, G: torch.Tensor) -> torch.Tensor:
    """Same as :func:`get_harmonics_prime`, but in reciprocal space.
    The result contains an extra phase of :math:`(iG)^l`, as discussed
    in `get_harmonics_tilde`."""
    return get_harmonics_prime(l_max, G) * _reciprocal_phase(l_max, G)


def _reciprocal_phase(l_max: int, G: torch.Tensor) -> torch.Tensor:
    """Return (-1)^l reciprocal-space phases shaped appropriately."""
    phase = []
    phase_cur = 1.0 + 0.0j
    for l in range(l_max + 1):
        phase.extend([phase_cur] * (2 * l + 1))  # repeated for m
        phase_cur *= 1.0j
    return torch.tensor(phase, device=G.device).view((-1,) + (1,) * (len(G.shape) - 1))
