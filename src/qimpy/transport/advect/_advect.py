import qimpy as qp
import numpy as np
import torch
import functools
from qimpy.transport import Geometry
from qimpy.transport._geometry import affine, sqrt_det_g
from torch.autograd.functional import jacobian


class Advect(Geometry):
    def __init__(
        self,
        x_y_corners,
        *,
        Lx: float = 1.0,
        Ly: float = 1.25,
        v_F: float = 1.0,
        Nx: int = 80,
        Ny: int = 64,
        N_theta: int = 256,
        N_ghost: int = 2,
        contact_width: float = 0.25,
        reflect_boundaries: bool = True,
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

        X = np.arange(0, self.N1, 1)
        Y = np.arange(0, self.N2, 1)

        self.dX = X[1] - X[0]
        self.dY = Y[1] - Y[0]
        X, Y = np.meshgrid(X, Y, indexing="ij")
        self.X, self.Y = (
            torch.nn.functional.pad(torch.tensor(X), [self.N_ghost] * 4).float(),
            torch.nn.functional.pad(torch.tensor(Y), [self.N_ghost] * 4).float(),
        )

        # self.x, self.y, jacobian = affine(self.X, self.Y, x_y_corners)
        self.x, self.y, jacobian = self.custom_transformation(
            self.sinewave_metric, self.X, self.Y
        )

        self.g = torch.tensor(
            sqrt_det_g(X, Y, affine, x_y_corners), device=qp.rc.device
        )
        self.g = torch.nn.functional.pad(self.g, [self.N_ghost] * 4, value=1.0)

        self.theta = centered_grid(0, N_theta) * self.dtheta - np.pi / 4

        # self.v_x = v_F * self.theta.cos()
        # self.v_y = v_F * self.theta.sin()
        # self.v = torch.stack((self.v_x, self.v_y)).T

        # Initialize distribution function:
        self.rho_shape = (self.x.shape[0], self.y.shape[1], N_theta)
        self.rho = torch.zeros(self.rho_shape, device=qp.rc.device)

        self.v_x = torch.zeros(self.rho_shape, device=qp.rc.device)
        self.v_y = torch.zeros(self.rho_shape, device=qp.rc.device)

        self.v_x[:, :, :] = v_F * self.theta.cos()
        self.v_y[:, :, :] = v_F * self.theta.sin()

        # Initialize slices for contact and ghost/non-ghost regions:
        self.y_contact = slice(0, len(torch.where(self.y < contact_width)[0]))
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

    @qp.utils.stopwatch(name="apply_boundaries")
    def apply_boundaries(self, rho: torch.Tensor) -> None:
        """Apply all boundary conditions in-place in `rho`."""
        if self.reflect_boundaries:
            rho[self.ghost_l] = reflect_x(rho[self.boundary_l])
            rho[self.ghost_r] = reflect_x(rho[self.boundary_r])
            rho[:, self.ghost_l] = reflect_y(rho[:, self.boundary_l])
            rho[:, self.ghost_r] = reflect_y(rho[:, self.boundary_r])
        # self.apply_dirichlet_boundary(rho)

    @qp.utils.stopwatch(name="drho")
    def drho(self, dt: float, rho: torch.Tensor) -> torch.Tensor:
        """Compute drho for time step dt, given current rho."""
        return (-dt / self.dx) * v_prime(rho, self.v_x, axis=0) + (
            -dt / self.dy
        ) * v_prime(rho, self.v_y, axis=1)

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

    @qp.utils.stopwatch(name="plot_streamlines")
    def plot_streamlines(self, plt, contour_kwargs, stream_kwargs):
        contour_kwargs.setdefault("levels", 100)
        contour_kwargs.setdefault("cmap", "bwr")
        stream_kwargs.setdefault("density", 2.0)
        stream_kwargs.setdefault("linewidth", 1.0)
        stream_kwargs.setdefault("color", "k")
        stream_kwargs.setdefault("arrowsize", 1.0)
        x = to_numpy(self.x[self.non_ghost, self.non_ghost])
        y = to_numpy(self.y[self.non_ghost, self.non_ghost])
        # v = to_numpy(self.velocity)
        rho = to_numpy(torch.clamp(self.density, min=1e-4))
        plt.contourf(x, y, rho, **contour_kwargs)
        # plt.streamplot(x, y, v[..., 0].T, v[..., 1].T, **stream_kwargs)

    def sinewave_metric(self, X, Y, kx=3, ky=4, amp=0.02):
        return self.Lx * X / self.N1 + amp * torch.sin(
            2 * np.pi * ky * Y / self.N2
        ), self.Ly * Y / self.N2 + amp * torch.sin(2 * np.pi * kx * X / self.N1)

    def custom_transformation(self, func, X, Y):
        x, y = func(X, Y)
        jac = jacobian(func, (X, Y))

        return (x, y, jac)


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
        device=qp.rc.device,
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
    return torch.arange(start + 0.5, stop, device=qp.rc.device)


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
