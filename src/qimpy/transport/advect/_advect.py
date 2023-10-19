from __future__ import annotations
from typing import Callable

import numpy as np
import torch

from qimpy import rc
from qimpy.profiler import stopwatch


class Advect:
    def __init__(
        self,
        *,
        transformation: Callable[[torch.Tensor], torch.Tensor],
        v: torch.Tensor,
        N: tuple[int, ...],
        N_ghost: int = 2,
        dk: float = 1.0,
    ) -> None:
        self.N = N
        self.N_ghost = N_ghost

        # Initialize mesh:
        grids1d = [(torch.arange(Ni, device=rc.device) + 0.5) for Ni in N]
        self.Q = torch.stack(torch.meshgrid(*grids1d, indexing="ij"), dim=-1)

        # Initialize transformed coordinates and jacobian using auto-grad
        grad_q = torch.tile(
            torch.eye(2, device=rc.device)[:, None, None], (1,) + N + (1,)
        )
        self.Q.requires_grad = True
        Qfrac = self.Q / torch.tensor(N, device=rc.device)
        self.q = transformation(Qfrac)
        jacobian = torch.autograd.grad(
            self.q,
            self.Q,
            grad_outputs=grad_q,
            is_grads_batched=True,
            retain_graph=False,
        )[0]
        jacobian = torch.permute(jacobian, (1, 2, 0, 3))
        self.Q.requires_grad = False

        # Initialize metric:
        metric = torch.einsum("...aB, ...aC -> ...BC", jacobian, jacobian)
        self.g = torch.linalg.det(metric).sqrt()[:, :, None]

        # Initialize velocities:
        self.v = v
        self.V = torch.einsum("ta, ...Ba -> ...tB", v, torch.linalg.inv(jacobian))
        self.dt = 0.5 / self.V.abs().max().item()
        self.dk = dk

        # Initialize distribution function:
        Nk = v.shape[0]
        self.rho_shape = (N[0], N[1], Nk)
        self.rho_padded_shape = (N[0] + 2 * N_ghost, N[1] + 2 * N_ghost, Nk)
        self.rho = torch.zeros(self.rho_shape, device=rc.device)

        # Initialize slices for accessing ghost regions in padded version:
        # These are also the slices for the boundary region in non-padded version
        self.non_ghost = slice(N_ghost, -N_ghost)
        self.ghost_l = slice(0, N_ghost)  # ghost indices on left/bottom
        self.ghost_r = slice(-N_ghost, None)  # ghost indices on right/top side

        # Initialize v*drho/dx calculator:
        self.v_prime = torch.jit.script(Vprime())

    @stopwatch(name="apply_boundaries")
    def apply_boundaries(self, rho: torch.Tensor) -> torch.Tensor:
        """Apply all boundary conditions to `rho` and produce ghost-padded version."""
        non_ghost = self.non_ghost
        ghost_l = self.ghost_l
        ghost_r = self.ghost_r
        out = torch.zeros(self.rho_padded_shape, device=rc.device)
        out[non_ghost, non_ghost] = rho

        # Periodic boundary conditions
        out[ghost_l, non_ghost] = rho[ghost_r]
        out[ghost_r, non_ghost] = rho[ghost_l]
        out[non_ghost, self.ghost_l] = rho[:, ghost_r]
        out[non_ghost, self.ghost_r] = rho[:, ghost_l]
        return out

    @stopwatch(name="drho")
    def drho(self, dt: float, rho: torch.Tensor) -> torch.Tensor:
        """Compute drho for time step dt, given current rho."""
        non_ghost = self.non_ghost
        return (-dt) * (
            self.v_prime(rho[:, non_ghost], self.V[..., 0], axis=0)
            + self.v_prime(rho[non_ghost, :], self.V[..., 1], axis=1)
        )

    def time_step(self):
        rho_half = self.rho + self.drho(0.5 * self.dt, self.apply_boundaries(self.rho))
        self.rho += self.drho(self.dt, self.apply_boundaries(rho_half))

    @property
    def density(self):
        """Density at each point (integrate over momenta)."""
        return self.rho.sum(dim=2) * self.dk

    @property
    def velocity(self):
        """Average velocity at each point (integrate over momenta)."""
        return (self.rho @ self.v) * self.dk

    @stopwatch(name="plot_streamlines")
    def plot_streamlines(self, plt, contour_kwargs, stream_kwargs):
        contour_kwargs.setdefault("levels", 100)
        contour_kwargs.setdefault("cmap", "bwr")
        stream_kwargs.setdefault("density", 2.0)
        stream_kwargs.setdefault("linewidth", 1.0)
        stream_kwargs.setdefault("color", "k")
        stream_kwargs.setdefault("arrowsize", 1.0)
        q = to_numpy(self.q[self.non_ghost, self.non_ghost])
        x = q[:, :, 0]
        y = q[:, :, 1]
        # v = to_numpy(self.velocity)
        rho = to_numpy(self.density)
        plt.contourf(x, y, np.clip(rho, 1e-3, None), **contour_kwargs)
        plt.gca().set_aspect("equal")
        # plt.streamplot(x, y, v[..., 0].T, v[..., 1].T, **stream_kwargs)


def to_numpy(f: torch.Tensor) -> np.ndarray:
    """Move torch.Tensor to numpy array, regardless of input device etc."""
    return f.detach().cpu().numpy()


def minmod(f: torch.Tensor, axis: int) -> torch.Tensor:
    """Return min|`f`| along `axis` when all same sign, and 0 otherwise."""
    fmin, fmax = torch.aminmax(f, dim=axis)
    return torch.where(
        fmin < 0.0,
        torch.clamp(fmax, max=0.0),  # fmin < 0, so fmax if also < 0, else 0.
        fmin,  # fmin >= 0, so this is the min mod
    )


class Vprime(torch.nn.Module):
    def __init__(self, slope_lim_theta: float = 2.0):
        # Initialize convolution that computes slopes using 3 difference formulae.
        # Here, `slope_lim_theta` controls the scaling of the forward/backward
        # difference formulae relative to the central difference one.
        # This convolution takes input Nbatch x 1 x N and produces output with
        # dimensions Nbatch x 3 x (N-2), containing backward, central and forward
        # difference computations of the slope.
        super().__init__()
        self.slope_conv = torch.nn.Conv1d(1, 3, 3, bias=False)
        self.slope_conv.weight.data = torch.tensor(
            [
                [-slope_lim_theta, slope_lim_theta, 0.0],
                [-0.5, 0.0, 0.5],
                [0.0, -slope_lim_theta, slope_lim_theta],
            ],
            device=rc.device,
        ).view(
            3, 1, 3
        )  # add singleton in_channels dim
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
