"""Generate and evaluate quintic (fifth-order) splines."""
# List exported symbols for doc generation
__all__ = ["get_coeff", "Interpolator"]

import numpy as np
import torch
from scipy.linalg import solve_banded
from typing import Optional, Tuple


def get_coeff(samples: torch.Tensor) -> torch.Tensor:
    """Compute spline coefficients from uniformly-spaced `samples`.
    This 1D spline assumes a mirror-symmetry boundary condition at the left
    end, since this is used primarily for radial functions defined with this
    symmetry.  At the right end, the coefficients are generated using natural
    boundary conditions i.e. zero third and fourth derivatives.

    The output are coefficients to localized 5th order 'blip' functions,
    and will have two extra coefficients on each end since the blip functions
    have a range of two intervals. Any preceding dimensions in `samples`
    are batched over. Specifically, a (..., N) input will result in a
    coefficient output with dimensions (N+4, ...) suitable for use with
    :class:`Interpolator`.
    """
    N = samples.shape[-1]
    # Set up pentadiagonal system for quintic blip coefficients:
    off1 = 13.0 / 33  # value of quintic blip function one grid point away
    off2 = 1.0 / 66  # value of quintic blip function two grid points away
    band = np.tile(np.array([[off2, off1, 1.0, off1, off2]]).T, (1, N))
    # --- Mirror boundary at left end:
    band[2, 1] += off2
    band[1, 1] += off1
    band[0, 2] += off2
    # --- Natural boundary conditions at right end:
    extrap = np.array(((1.0, -3.0, 3.0), (3.0, -8.0, 6.0)))
    band[(3, 2, 1), (-3, -2, -1)] += off2 * extrap[0]
    band[(4, 3, 2), (-3, -2, -1)] += (off1, off2) @ extrap

    # Solve system (flatten any batch dims here):
    samples_np = samples.view(-1, N).to(torch.device("cpu")).numpy()
    coeff_np = solve_banded((2, 2), band, samples_np.T)

    # Pad boundaries:
    coeff_np = np.vstack(
        (
            coeff_np[2:0:-1],  # mirror B.C.
            coeff_np,
            extrap @ coeff_np[-3:],  # extrapolation with natural B.C.
        )
    ).reshape(
        (N + 4,) + samples.shape[:-1]
    )  # un-flatten any batch dims
    return torch.tensor(coeff_np).to(samples.device)


class Interpolator:
    """Interpolate by evaluating quintic blip functions.
    Create Interpolator object for a set of points to calculate for,
    and then use it with any blip coefficients generated with same spacing.
    Typical usage:

    .. code-block:: python

        y_coeff = get_coeff(y_samples)  # convert samples to blip coefficients
        interp = Interpolator(x, dx)  # create interpolator for points x
        y = interp(y_coeff)  # interpolate y_samples at locations x
    """

    _shape: Tuple[int, ...]  #: Dimensions of x to reproduced at output
    _mat: torch.Tensor  #: internal sparse matrix used for interpolation
    _BLIP_TO_POLY: Optional[torch.Tensor] = None  #: blip to poly transform

    def __init__(self, x: torch.Tensor, dx: float, deriv: int = 0):
        """Initialize interpolator for evaluating at points `x`. Here `dx` is
        the spacing between coefficients that this will be used with later.
        Optionally, initialize to calculate derivative of order `deriv`
        (must be <= 4, since quintic splines are :math:`C^4` continuous).
        """
        self._shape = x.shape  # remember shape before flattening below
        t = x.view(-1, 1) / dx  # dimensionless coordinate (flattened)
        i = torch.floor(t).to(torch.int)  # interval index
        t -= i  # convert to fractional coordinate within interval
        # Initialize blip matrix if not done so:
        if Interpolator._BLIP_TO_POLY is None:
            Interpolator._BLIP_TO_POLY = (1.0 / 66) * torch.tensor(
                [
                    [+1.0, 26.0, 66.0, 26.0, 1.0, 0.0],
                    [-5.0, -50.0, 0.0, 50.0, 5.0, 0.0],
                    [+10.0, 20.0, -60.0, 20.0, 10.0, 0.0],
                    [-10.0, 20.0, 0.0, -20.0, 10.0, 0.0],
                    [+5.0, -20.0, 30.0, -20.0, 5.0, 0.0],
                    [-1.0, 5.0, -10.0, 10.0, -5.0, 1.0],
                ],
                device=x.device,
            )

        # Compute blip polynomials (or derivative to order deriv) for each t:
        powers = torch.arange(6, device=x.device, dtype=torch.int)[None, :]
        if deriv:
            assert 0 < deriv < 5
            # Take deriv'th order derivative of (t ** powers):
            prefac = powers[:, deriv:] * ((1.0 / dx) ** deriv)
            for order in range(1, deriv):
                prefac *= powers[:, deriv - order : -order]
            f = ((t ** powers[:, :-deriv]) * prefac) @ Interpolator._BLIP_TO_POLY[
                deriv:
            ]
        else:
            f = (t**powers) @ Interpolator._BLIP_TO_POLY
        # Construct the sparse marix interpolator:
        n_x = t.shape[0]
        indices = torch.empty((2, n_x, 6))
        indices[0] = torch.arange(n_x, device=x.device)[:, None]
        indices[1] = i + powers  # fetch 6 adjacent coefficients for each
        self._mat = torch.sparse_coo_tensor(
            indices.flatten(1), f.flatten(), device=x.device
        )

    def __call__(self, coeff: torch.Tensor) -> torch.Tensor:
        """Apply interpolation to specified coefficients."""
        n_coeff_needed = self._mat.shape[1]
        assert n_coeff_needed <= coeff.shape[0]
        if len(coeff.shape) == 1:
            return (self._mat @ coeff[:n_coeff_needed]).view(self._shape)
        else:
            result = (self._mat @ coeff[:n_coeff_needed].flatten(1)).T
            return result.view(coeff.shape[1:] + self._shape)
