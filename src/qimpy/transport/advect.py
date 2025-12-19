from __future__ import annotations
from typing import Any

import torch
from qimpy import rc

N_GHOST: int = 1  #: Implementation optimized heavily for single ghost layer
NON_GHOST: slice = slice(N_GHOST, -N_GHOST)


class Advect(torch.nn.Module):
    def __init__(
        self,
        *,
        pad: bool,
        slope_lim_theta: float = 2.0,
        cent_diff_deriv: bool = False,
    ):
        # Initialize convolution that computes slopes using 3 difference formulae.
        # Here, `slope_lim_theta` controls the scaling of the forward/backward
        # difference formulae relative to the central difference one.
        # Optionally, override this with central difference only if `cent_diff_deriv`.
        # This convolution takes input Nbatch x 1 x N and produces output with
        # dimensions Nbatch x 3 x Nout, containing backward, central and forward
        # difference computations of the slope, where Nout = N if pad and N-2 otherwise.
        super().__init__()
        weight_data = torch.tensor(
            [
                [-slope_lim_theta, slope_lim_theta, 0.0],
                [-0.5, 0.0, 0.5],
                [0.0, -slope_lim_theta, slope_lim_theta],
            ],
            device=rc.device,
        )
        conv_kwargs = dict[str, Any](bias=False)
        if pad:
            conv_kwargs.update(padding=1, padding_mode="replicate")
        if cent_diff_deriv:
            self.slope_conv = torch.nn.Conv1d(1, 1, 1, **conv_kwargs)
            weight_data = weight_data[1]
        else:
            self.slope_conv = torch.nn.Conv1d(1, 3, 3, **conv_kwargs)
        self.slope_conv.weight.data = weight_data.view(-1, 1, 3)  # add in_channels dim
        self.slope_conv.weight.requires_grad = False
        self.pad = pad

    def slope_minmod(self, f: torch.Tensor) -> torch.Tensor:
        """Compute slope of `f` along its last axis with a minmod limiter."""
        # Flatten all but last axis into a single batch dimension:
        batch_shape = f.shape[:-1]
        f = f.flatten(0, -2)[:, None]  # n_batch x 1 x n_axis
        # Compute slopes by convolution and apply minmod filter:
        slope = minmod(self.slope_conv(f), axis=1)  # n_batch x n_axis
        return slope.unflatten(0, batch_shape)  # restore dimensions

    def forward(
        self, rho: torch.Tensor, v: torch.Tensor, axis: int
    ) -> list[torch.Tensor]:
        """Compute advection -d`v``rho`/dx of `rho` with velocity `v` along `axis`.
        Along `axis`, for a domain of actual length N, `rho` and `v` are ghost-padded
        with length N + 2 if `pad`, and N + 4 otherwise. The result contains the domain
        contribution, along with left and right edge contributions if `pad` is True.
        All tensors have equal/broadcastable dimensions along all other axes.
        """
        # Bring active axis to end
        flux = (rho * v).swapaxes(axis, -1)
        v = v.swapaxes(axis, -1)

        # Reconstruction
        half_slope = 0.5 * self.slope_minmod(flux)
        if not self.pad:
            flux = flux[..., 1:-1]
            v = v[..., 1:-1]
            # v, flux and half_slope now have the same dimension regardless

        # Central difference from half points & Riemann selection based on velocity:
        flux_minus = (flux - half_slope)[..., 1:]  # - flux from next cell
        flux_plus = (flux + half_slope)[..., :-1]  # + flux from prev cell
        flux_minus[v[..., 1:] >= 0.0] = 0.0  # select v < 0 contributions
        flux_plus[v[..., :-1] <= 0.0] = 0.0  # select v > 0 contributions
        flux_minus[..., -1] = 0.0
        flux_plus[..., 0] = 0.0
        flux_net = flux_minus + flux_plus
        out = [-flux_net.diff(dim=-1).swapaxes(axis, -1)]
        if self.pad:
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
