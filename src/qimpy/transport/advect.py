from __future__ import annotations

import torch
from qimpy import rc


N_GHOST: int = 2  #: currently a constant, but could depend on slope method later

# Initialize slices for accessing ghost regions in padded version:
# These are also the slices for the boundary region in non-padded version
NON_GHOST: slice = slice(N_GHOST, -N_GHOST)
GHOST_L: slice = slice(0, N_GHOST)  #: ghost indices on left/bottom
GHOST_R: slice = slice(-N_GHOST, None)  #: ghost indices on right/top side


class Advect(torch.nn.Module):
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

    def forward(
        self, rho: torch.Tensor, v: torch.Tensor, axis: int, retain_padding: bool
    ) -> list[torch.Tensor]:
        """Compute advection -d`v``rho`/dx of `rho` with velocity `v` along `axis`.
        Along `axis`, for a domain of actual length N, `rho` and `v` are ghost-padded
        with length N + 4. The result contains the contribution within the domain, along
        with optional left and right edge contributions if `retain_padding` is True.
        All tensors have equal/broadcastable dimensions along all other axes.
        """
        # Bring active axis to end
        flux = (rho * v).swapaxes(axis, -1)
        v = v.swapaxes(axis, -1)[..., 1:-1]  # now same dimension as result

        # Reconstruction
        half_slope = 0.5 * self.slope_minmod(flux)

        # Central difference from half points & Riemann selection based on velocity:
        flux_minus = flux[..., 2:-1] - half_slope[..., 1:]  # - flux from next cell
        flux_plus = flux[..., 1:-2] + half_slope[..., :-1]  # + flux from prev cell
        flux_minus[v[..., 1:] >= 0.0] = 0.0  # select v < 0 contributions
        flux_plus[v[..., :-1] <= 0.0] = 0.0  # select v > 0 contributions
        flux_minus[..., -1] = 0.0
        flux_plus[..., 0] = 0.0
        flux_net = flux_minus + flux_plus
        out = [-flux_net.diff(dim=-1).swapaxes(axis, -1)]
        if retain_padding:
            out.append(-flux_net[..., 0].swapaxes(axis, -1))  # left edge
            out.append(flux_net[..., -1].swapaxes(axis, -1))  # right edge
        return out


def minmod(f: torch.Tensor, axis: int) -> torch.Tensor:
    """Return min|`f`| along `axis` when all same sign, and 0 otherwise."""
    fmin, fmax = torch.aminmax(f, dim=axis)
    return torch.where(
        fmin < 0.0,
        torch.clamp(fmax, max=0.0),  # fmin < 0, so fmax if also < 0, else 0.
        fmin,  # fmin >= 0, so this is the min mod
    )
