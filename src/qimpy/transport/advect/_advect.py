import functools

import numpy as np
import torch

from qimpy import rc
from qimpy.profiler import stopwatch
from qimpy.transport import Geometry
from qimpy.transport._geometry import sqrt_det_g, jacobian_inv


class Advect(Geometry):
    def __init__(
        self,
        *,
        Lx: float = 1.0,
        Ly: float = 1.25,
        v_F: float = 1.0,
        Nx: int = 64,
        Ny: int = 80,
        N_theta: int = 256,
        N_ghost: int = 2,
        contact_width: float = 0.25,
        reflect_boundaries: bool = True,
        init_angle: float = 0.0,
    ) -> None:
        self.Lx = Lx
        self.Ly = Ly
        self.v_F = v_F
        self.Nx = Nx
        self.Ny = Ny
        self.N1 = Nx
        self.N2 = Ny
        self.N_theta = N_theta
        self.N_ghost = N_ghost
        self.contact_width = contact_width
        self.reflect_boundaries = reflect_boundaries

        self.dx = Lx / Nx
        self.dy = Ly / Ny
        self.dtheta = 2 * np.pi / self.N_theta
        self.dt = 0.5 * self.dx / v_F
        self.drift_velocity_fraction = 1e-3  # as a fraction of v_F

        # self.x = centered_grid(-N_ghost, Nx + N_ghost) * self.dx
        # self.y = centered_grid(-N_ghost, Ny + N_ghost) * self.dy

        X = torch.arange(0, self.N1, 1, dtype=torch.float, device=rc.device)
        Y = torch.arange(0, self.N2, 1, dtype=torch.float, device=rc.device)

        self.dX = X[1] - X[0]
        self.dY = Y[1] - Y[0]
        X, Y = torch.meshgrid(X, Y, indexing="ij")
        self.X, self.Y = torch.nn.functional.pad(
            X, [self.N_ghost] * 4
        ), torch.nn.functional.pad(Y, [self.N_ghost] * 4)
        self.Q = torch.stack([self.X, self.Y], dim=-1)

        # self.x, self.y, jacobian = affine(self.X, self.Y, x_y_corners)
        self.q, jacobian = self.custom_transformation(self.Q)

        jac_inv = jacobian_inv(self.Q, self.custom_transformation)
        dX_dx = jac_inv[0][0]
        dX_dy = jac_inv[0][1]
        dY_dx = jac_inv[1][0]
        dY_dy = jac_inv[1][1]

        self.g = sqrt_det_g(self.Q, self.custom_transformation).detach()[:, :, None]
        # self.g = torch.nn.functional.pad(self.g, [self.N_ghost] * 4, value=1.0)

        #self.theta = centered_grid(0, N_theta) * self.dtheta - init_angle
        self.theta = torch.arange(0, N_theta, device=rc.device)*self.dtheta - init_angle

        # self.v_x = v_F * self.theta.cos()
        # self.v_y = v_F * self.theta.sin()
        # self.v = torch.stack((self.v_x, self.v_y)).T

        # Initialize distribution function:
        self.rho_shape = (self.q.shape[0], self.q.shape[1], N_theta)
        self.rho = torch.zeros(self.rho_shape, device=rc.device)

        self.v_x = torch.zeros(self.rho_shape, device=rc.device)
        self.v_y = torch.zeros(self.rho_shape, device=rc.device)

        self.v_x[:, :, :] = v_F * self.theta.cos()
        self.v_y[:, :, :] = v_F * self.theta.sin()

        self.v = torch.stack([self.v_x, self.v_y], dim=-1)

        self.v_X = self.v_x * dX_dx[:, :, None] + self.v_y * dX_dy[:, :, None]
        self.v_Y = self.v_x * dY_dx[:, :, None] + self.v_y * dY_dy[:, :, None]

        self.V = torch.stack([self.v_X, self.v_Y], dim=-1)

        # Initialize slices for contact and ghost/non-ghost regions:
        if N_ghost == 0:
            self.non_ghost = slice(0, -1)
        else:
            self.non_ghost = slice(N_ghost, -N_ghost)
        self.ghost_l = slice(0, N_ghost)  # ghost indices on left/bottom side
        self.ghost_r = slice(-N_ghost, None)  # ghost indices on right/top side

        # Slices to access boundary region adjacent to ghost:
        self.boundary_l = slice(N_ghost, 2 * N_ghost)
        self.boundary_r = slice(-2 * N_ghost, -N_ghost)

    def apply_dirichlet_boundary(self, rho: torch.Tensor) -> None:
        """Apply Dirichlet boundary conditions in-place."""
        rho_contact = self.drift_velocity_fraction * self.v_x
        rho[self.ghost_l, self.y_contact] = rho_contact
        rho[self.ghost_r, self.y_contact] = rho_contact

    @stopwatch(name="apply_boundaries")
    def apply_boundaries(self, rho: torch.Tensor) -> None:
        """Apply all boundary conditions in-place in `rho`."""
        if self.reflect_boundaries:
            rho[self.ghost_l] = reflect_x(rho[self.boundary_l])
            rho[self.ghost_r] = reflect_x(rho[self.boundary_r])
            rho[:, self.ghost_l] = reflect_y(rho[:, self.boundary_l])
            rho[:, self.ghost_r] = reflect_y(rho[:, self.boundary_r])
        else:
            # Periodic boundary conditions
            rho[self.ghost_l] = rho[self.boundary_r]
            rho[self.ghost_r] = rho[self.boundary_l]
            rho[:, self.ghost_l] = rho[:, self.boundary_r]
            rho[:, self.ghost_r] = rho[:, self.boundary_l]
        # self.apply_dirichlet_boundary(rho)

    @stopwatch(name="drho")
    def drho(self, dt: float, rho: torch.Tensor) -> torch.Tensor:
        """Compute drho for time step dt, given current rho."""
        return (-dt / (self.g * self.dX)) * v_prime(
            rho, self.g * self.V[:, :, :, 0], axis=0
        ) + (-dt / (self.g * self.dY)) * v_prime(
            rho, self.g * self.V[:, :, :, 1], axis=1
        )

    def time_step(self):
        # Half step:
        self.apply_boundaries(self.rho)
        rho_half = self.rho + self.drho(0.5 * self.dt, self.rho)
        # Full step:
        self.apply_boundaries(rho_half)
        self.rho += self.drho(self.dt, rho_half)

    @property
    def density(self):
        """Density at each point (integrate over momenta)."""
        return self.rho[self.non_ghost, self.non_ghost].sum(dim=2) * self.dtheta

    @property
    def velocity(self):
        """Average velocity at each point (integrate over momenta)."""
        return (self.rho[self.non_ghost, self.non_ghost] @ self.v) * self.dtheta

    @stopwatch(name="plot_streamlines")
    def plot_streamlines(self, plt, contour_kwargs, stream_kwargs):
        contour_kwargs.setdefault("levels", 100)
        contour_kwargs.setdefault("cmap", "bwr")
        stream_kwargs.setdefault("density", 2.0)
        stream_kwargs.setdefault("linewidth", 1.0)
        stream_kwargs.setdefault("color", "k")
        stream_kwargs.setdefault("arrowsize", 1.0)
        x = to_numpy(self.q[:, :, 0][self.non_ghost, self.non_ghost])
        y = to_numpy(self.q[:, :, 1][self.non_ghost, self.non_ghost])
        # v = to_numpy(self.velocity)
        rho = to_numpy(self.density)
        plt.contourf(x, y, np.clip(rho, 1e-3, None), **contour_kwargs)
        plt.gca().set_aspect("equal")
        # plt.streamplot(x, y, v[..., 0].T, v[..., 1].T, **stream_kwargs)

    def custom_transformation(self, Q, kx=1, ky=1, amp=-0.05):
        L = torch.tensor([self.Lx, self.Ly], device=rc.device)
        k = torch.tensor([kx, ky], device=rc.device)
        N = torch.tensor(
            [self.Nx + 2 * self.N_ghost, self.Ny + 2 * self.N_ghost],
            device=rc.device,
        )
        Nx, Ny, _ = Q.shape
        grad_q = torch.tile(
            torch.eye(2, device=rc.device)[:, None, None], (1, Nx, Ny, 1)
        )
        Q.requires_grad = True
        Q_by_N = Q / N
        q = L * (Q_by_N)  + amp * torch.sin(2 * np.pi * k * torch.roll(Q_by_N, 1))

        jacobian = torch.autograd.grad(
            q, Q, grad_outputs=grad_q, is_grads_batched=True, retain_graph=False
        )[0]
        jacobian = torch.permute(jacobian, (1, 2, 0, 3))

        Q.requires_grad = False
        jacobian = [
            [jacobian[:, :, 0, 0], jacobian[:, :, 0, 1]],
            [jacobian[:, :, 1, 0], jacobian[:, :, 1, 1]],
        ]

        return q, jacobian


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


@functools.cache
def get_slope_conv(slope_lim_theta: float = 2.0) -> torch.nn.Conv1d:
    """Get a Conv1D object that computes slopes using 3 difference formulae.
    Here, `slope_lim_theta` controls the scaling of the forward/backward
    difference formulae relative to the central difference one.
    The inputs to the convolution should be rank-3, with the axis for slope
    computation brought to the end, a singleton dimension in the middle,
    and all remaining dimensions flattened into the first 'batch' dimension.
    On output, the singleton dimension will be replaced by length 3,
    containing backward, central and forward differences (in that order)."""
    conv = torch.nn.Conv1d(1, 3, 3, padding=1, bias=False)
    conv.weight.data = torch.tensor(
        [
            [-slope_lim_theta, slope_lim_theta, 0.0],
            [-0.5, 0.0, 0.5],
            [0.0, -slope_lim_theta, slope_lim_theta],
        ],
        device=rc.device,
    ).view(
        3, 1, 3
    )  # add singleton in_channels dim
    conv.weight.requires_grad = False
    return conv


def slope_minmod(f: torch.Tensor) -> torch.Tensor:
    """Compute slope of `f` along its last axis with a minmod limiter."""
    # Flatten all but last axis into a single batch dimension:
    batch_shape = f.shape[:-1]
    f = f.flatten(0, -2)[:, None]  # n_batch x 1 x n_axis
    # Compute slopes by convolution and apply minmod filter:
    slope_conv = get_slope_conv(slope_lim_theta=2.0)
    slope = minmod(slope_conv(f), axis=1)  # n_batch x n_axis
    return slope.unflatten(0, batch_shape)  # restore dimensions


def v_prime(rho: torch.Tensor, v: torch.Tensor, axis: int) -> torch.Tensor:
    """Compute v * d`rho`/dx, with velocity `v` along `axis`."""
    # Axis permutations to bring velocity to front and active axis to end
    if axis == 0:
        permute_forward = (2, 1, 0)
        permute_inverse = (2, 1, 0)
    else:
        assert axis == 1
        permute_forward = (2, 0, 1)
        permute_inverse = (1, 2, 0)

    rho = rho.permute(permute_forward)
    v = v.permute(permute_forward)

    # Reconstruction
    half_slope = 0.5 * slope_minmod(rho)

    # Riemann selection based on velocity:
    result_minus = rho - half_slope
    result_plus = (rho + half_slope).roll(+1, dims=-1)
    rho_minus_half = torch.where(v < 0.0, result_minus, result_plus)

    # Final central difference derivative from plus and minus half points:
    delta_rho = rho_minus_half.diff(dim=-1, append=rho_minus_half[..., :1])
    return (v * delta_rho).permute(permute_inverse)  # original axis order


def centered_grid(start: int, stop: int) -> torch.Tensor:
    """Create a grid centered on the intervals of [start, stop]."""
    return torch.arange(start + 0.5, stop, device=rc.device)


def reflect_x(rho: torch.Tensor) -> torch.Tensor:
    """Reflect a distribution function along x."""
    # p_x -> -p_x, p_y -> p_y
    # => cos(theta) -> -cos(theta), sin(theta) -> sin(theta)
    # => theta -> pi - theta
    # can achieve this by flipping each half of the theta dimension
    rho = rho.unflatten(-1, (2, -1))  # split N_theta into 2 x N_theta/2
    rho = rho.flip(dims=(0, -1))  # flip x and N_theta/2
    return rho.flatten(-2)  # remerge 2 x N_theta/2 into N_theta


def reflect_y(rho: torch.Tensor) -> torch.Tensor:
    """Reflect a distribution function along y."""
    # p_x -> p_x, p_y -> -p_y
    # => cos(theta) -> cos(theta), sin(theta) -> -sin(theta)
    # => theta -> -theta
    return rho.flip(dims=(1, -1))
