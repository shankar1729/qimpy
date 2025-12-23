from __future__ import annotations
from typing import Union

import torch
from qimpy import rc

N_GHOST: int = 2
NON_GHOST: slice = slice(N_GHOST, -N_GHOST)
Mask = Union[bool, torch.Tensor]


class Advect(torch.nn.Module):
    def __init__(
        self,
        *,
        slope_lim_theta: float = 2.0,
        cent_diff_deriv: bool = False,
    ):
        # Initialize convolution that computes slopes using 3 difference formulae.
        # Here, `slope_lim_theta` controls the scaling of the forward/backward
        # difference formulae relative to the central difference one.
        # Optionally, override this with central difference only if `cent_diff_deriv`.
        # This convolution takes input Nbatch x 1 x N and produces output with
        # dimensions Nbatch x 3 x N-2, containing backward, central and forward
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
        self,
        rho: torch.Tensor,
        v: torch.Tensor,
        axis: int,
        retain_padding: bool = False,
        edge_masks: tuple[Mask, Mask] = (False, False),
    ) -> list[torch.Tensor]:
        """Compute advection -d`v``rho`/dx of `rho` with velocity `v` along `axis`.
        Along `axis`, for a domain of actual length N, `rho` and `v` are ghost-padded
        with length N + 4. The result contains the domain contribution, along with
        additional left and right edge contributions if `retain_padding` is True.
        All tensors have equal/broadcastable dimensions along all other axes.

        The `edge_masks` control whether and where the incoming flux is zeroed out on
        the left and right boundaries. For the typical case with `retain_padding` set to
        True, this mask should zero out the incoming flux, except in contact regions.
        The flux is zeroed out completely on an edge if its mask is True, and left
        unchanged if mask is False. If the mask is a tensor, then it corresponds to
        the indices of points on the edge that must be zeroed; this is supported only
        for two-dimensional domains (i.e rho has three dimensions overall), and the
        indexing happens on axis 0 for transport along axis 1 and vice versa.
        Note that with `retain_paddig` enabled, leaving the masks at the default False
        (not zerong the edge flux) will typically lead to double counting the edge.
        """
        # Bring active axis to end
        flux = (rho * v).swapaxes(axis, -1)
        v = v.swapaxes(axis, -1)

        # Reconstruction
        half_slope = 0.5 * self.slope_minmod(flux)  # length N + 2
        flux = flux[..., 1:-1]  # make same length as half_slope
        v = v[..., 1:-1]  # make same length as half_slope

        # Central difference from half points & Riemann selection based on velocity:
        flux_minus = (flux - half_slope)[..., 1:]  # - flux from next cell
        flux_plus = (flux + half_slope)[..., :-1]  # + flux from prev cell
        flux_minus[v[..., 1:] >= 0.0] = 0.0  # select v < 0 contributions
        flux_plus[v[..., :-1] <= 0.0] = 0.0  # select v > 0 contributions
        for flux_sign, edge_mask, index in zip(
            (flux_plus, flux_minus), edge_masks, (0, -1)
        ):
            if edge_mask is False:
                pass
            elif edge_mask is True:
                flux_sign[..., index] = 0.0
            elif isinstance(edge_mask, torch.Tensor):
                flux_sign[..., index].index_fill_(1 - axis, edge_mask, 0.0)
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
