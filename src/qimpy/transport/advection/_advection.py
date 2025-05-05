from __future__ import annotations

import torch
from qimpy import rc


class Vprime(torch.nn.Module):

    N_GHOST: int = 2  #: currently a constant, but could depend on slope method later

    # Initialize slices for accessing ghost regions in padded version:
    # These are also the slices for the boundary region in non-padded version
    NON_GHOST: slice = slice(N_GHOST, -N_GHOST)
    GHOST_L: slice = slice(0, N_GHOST)  #: ghost indices on left/bottom
    GHOST_R: slice = slice(-N_GHOST, None)  #: ghost indices on right/top side

    def __init__(self, slope_lim_theta: float = 2.0, cent_diff_deriv: bool = False):
        # Initialize convolution that computes slopes using 3 difference formulae.
        # Here, `slope_lim_theta` controls the scaling of the forward/backward
        # difference formulae relative to the central difference one.
        # This convolution takes input Nbatch x 1 x N and produces output with
        # dimensions Nbatch x 3 x (N-2), containing backward, central and forward
        # difference computations of the slope.
        super().__init__()
        weight_data = torch.tensor(
            [
                [-slope_lim_theta, slope_lim_theta, 0.0],
                [-0.5, 0.0, 0.5],
                [0.0, -slope_lim_theta, slope_lim_theta],
            ],
            device=rc.device,
        )
        if cent_diff_deriv:
            self.slope_conv = torch.nn.Conv1d(1, 1, 1, bias=False)
            weight_data = weight_data[1]
        else:
            self.slope_conv = torch.nn.Conv1d(1, 3, 3, bias=False)
        self.slope_conv.weight.data = weight_data.view(-1, 1, 3)  # add in_channels dim
        self.slope_conv.weight.requires_grad = False

    def slope_minmod(self, f: torch.Tensor) -> torch.Tensor:
        """Compute slope of `f` along its last axis with a minmod limiter."""
        # Flatten all but last axis into a single batch dimension:
        batch_shape = f.shape[:-1]
        f = f.flatten(0, -2)[:, None]  # n_batch x 1 x n_axis
        # Compute slopes by convolution and apply minmod filter:
        slope = minmod(self.slope_conv(f), axis=1)  # n_batch x n_axis
        return slope.unflatten(0, batch_shape)  # restore dimensions

    def forward(self, rho: torch.Tensor, v: torch.Tensor, axis: int) -> torch.Tensor:
        """Compute v * d`rho`/dx, with velocity `v` along `axis`."""
        # Bring active axis to end
        rho = rho.swapaxes(axis, -1)
        v = v.swapaxes(axis, -1)

        # Reconstruction
        half_slope = 0.5 * self.slope_minmod(rho)

        # Central difference from half points & Riemann selection based on velocity:
        rho_diff = rho[..., 1:-1].diff(dim=-1)
        half_slope_diff = half_slope.diff(dim=-1)
        result_minus = (rho_diff - half_slope_diff)[..., 1:]
        result_plus = (rho_diff + half_slope_diff)[..., :-1]
        delta_rho = torch.where(v < 0.0, result_minus, result_plus)
        return (v * delta_rho).swapaxes(axis, -1)  # original axis order


def minmod(f: torch.Tensor, axis: int) -> torch.Tensor:
    """Return min|`f`| along `axis` when all same sign, and 0 otherwise."""
    fmin, fmax = torch.aminmax(f, dim=axis)
    return torch.where(
        fmin < 0.0,
        torch.clamp(fmax, max=0.0),  # fmin < 0, so fmax if also < 0, else 0.
        fmin,  # fmin >= 0, so this is the min mod
    )
