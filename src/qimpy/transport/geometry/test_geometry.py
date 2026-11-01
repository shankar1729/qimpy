import numpy as np
import pytest
import torch
import torch.distributed as dist

from qimpy import rc, log
from qimpy.io import log_config
from qimpy.mpi import globalreduce, get_process_grid
from .. import Transport
from ..material import FermiCircle
from . import Geometry, Mesh


def make_geometry(geometry_kwargs: dict, fermi_circle_kwargs: dict) -> Geometry:
    """Create Geometry and FermiSurface (without Transport), for single step tests."""
    process_grid = get_process_grid("rk", (-1, 1))
    material = FermiCircle(**fermi_circle_kwargs, process_grid=process_grid)
    return Geometry(**geometry_kwargs, material=material, process_grid=process_grid)


def random_polygon(N: int) -> tuple[np.ndarray, np.ndarray]:
    """Create a random N-sided convex polygon, return vertices and segment indices."""
    np.random.seed(0)
    theta = np.sort(np.random.rand(N)) * (2 * np.pi)
    vertices = np.stack((np.cos(theta), np.sin(theta))).T
    segments = np.stack((np.arange(N), np.arange(1, N + 1) % N)).T
    return vertices, segments


@pytest.mark.parametrize("N", [3, 5, 7])
def test_linear_field(N: int) -> None:
    """Verify face reconstruction and v . gradient operator on a linear field."""
    vertices, segments = random_polygon(N)
    poly_kwargs = dict(vertices=vertices, segments=segments, contacts={}, max_area=0.01)
    geometry = make_geometry(
        dict(mesh=dict(triangulate=poly_kwargs), contacts={}, epsilon_sq=0.0),
        dict(kF=1.5, vF=0.8, n_theta=6, n_r=1),
    )
    v = geometry.material.transport_velocity

    # Construct linear gradient field:
    torch.random.manual_seed(0)
    gradient = torch.randn_like(v)  # independent gradient for each k
    gradient *= 1 / gradient.norm(dim=-1, keepdim=True)  # unit gradient
    if geometry.group.size() > 1:
        dist.broadcast(gradient, src=0)
    mesh = geometry.mesh
    cell_div = mesh.cell_division
    cells = mesh.cells[cell_div.global_index, :-1]
    cell_centers = rc.from_np(mesh.vertices[cells].mean(axis=1))
    geometry.rho = cell_centers @ gradient.T

    # Calculate reference result:
    faces = mesh.faces[cell_div.global_index[: cell_div.n_reconstruct]]
    face_centers = rc.from_np(mesh.vertices[faces].mean(axis=2))
    rho_faces_ref = (face_centers @ gradient.T).flatten(0, 1)
    d_max = globalreduce.max(
        (face_centers - cell_centers[: cell_div.n_reconstruct, None]).norm(dim=-1),
        geometry.group,
    )

    # Find interior cell-face indices (since boundaries are modified in to_faces):
    face_div = mesh.face_division
    _, face_map = mesh.get_face_weights()
    interior_map = face_map[face_div.n_boundary_mine :, 0]
    n_faces = mesh.faces.shape[1]
    interior_cf = interior_map[..., 0] * n_faces + interior_map[..., 1]

    # Compare for several limiter values:
    for epsilon_sq in [0.0, 0.1, 1.0, 10.0]:
        geometry.epsilon_sq = epsilon_sq
        rho_faces = geometry.to_faces(geometry.rho, 0.0)
        err = (rho_faces - rho_faces_ref)[interior_cf].abs()
        max_err = globalreduce.max(err, geometry.group)
        err_limit = d_max / (1 + 0.5 * epsilon_sq / d_max**2)
        log.info(f"{epsilon_sq=:.2g}: {max_err=:.2g} {err_limit=:.2g}")
        assert max_err < err_limit

    # Check divergence operator:
    rho_dot = (geometry.minus_v_dot_del @ rho_faces_ref.flatten()).view(-1, len(v))
    rho_dot_ref = -torch.einsum("ki, ki -> k", v, gradient)
    max_err = globalreduce.max(
        (rho_dot[: cell_div.n_mine] - rho_dot_ref).abs(), geometry.group
    )
    log.info(f"Maximum error in v . del = {max_err:.2g}")
    assert max_err < 1.0e-12


@pytest.mark.parametrize("n_r", [1])
@pytest.mark.parametrize("specularity", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("contacted", [False, True])
def test_charge_conservation(n_r: int, specularity: float, contacted: bool) -> None:
    vertices, segments = random_polygon(5)
    poly_kwargs = dict(vertices=vertices, segments=segments, contacts={}, max_area=0.01)
    contacts = {}
    if contacted:
        length = np.linalg.norm(np.diff(vertices[segments[0]], axis=0))
        poly_kwargs["contacts"] = dict(
            source=(*vertices[segments[0, 0]], 0.5 * length),
            drain=(*vertices[segments[0, 1]], 0.5 * length),
        )
        contacts = dict(source=dict(dmu=0.1), drain=dict(dmu=-0.1))
    geometry = make_geometry(
        dict(mesh=dict(triangulate=poly_kwargs), contacts=contacts),
        dict(kF=1.5, vF=0.8, n_theta=6, n_r=n_r, specularity=specularity),
    )
    material = geometry.material
    dV = rc.from_np(geometry.mesh.get_cell_weights())
    torch.manual_seed(0)
    geometry.rho = torch.rand_like(geometry.rho)
    drho_dt = geometry.rho_dot(geometry.rho, 0.0)
    dq_dt = dV @ material.measure_observables(drho_dt, 0.0)[:, 0]
    if geometry.group.size() > 1:
        dist.all_reduce(dq_dt, group=geometry.group)
    if contacted:
        geometry.update_probes(0, 0.0)
        contact_currents = geometry.stash_probe.flux[0][:, 0]
        dq_dt += contact_currents.sum()  # current entering the contacts
    assert dq_dt.abs().item() < 1.0e-9


@pytest.mark.parametrize("style", ["1D", "2D"])
def test_wire_equilibration(style: str) -> None:
    """Check density and current equilibration in a wire against analytical solution."""
    Lx = 2.0
    dmu = 0.1

    # Create mesh with selected style (1d or 2d)
    if style == "1D":
        h = 0.025
        Ly = 1.0
        Nx = int(np.round(Lx / h))
        n_theta = 22
        mesh_kwargs = dict(line=dict(L=Lx, N=Nx))
    else:
        assert style == "2D"
        h = 0.05
        radius = 2 * h  # contact radius - half-width of wire
        Ly = 2 * radius
        Nx = int(np.round(Lx / h))
        Ny = int(np.round(Ly / h))
        n_theta = 10
        contacts = dict(source=(0.0, radius, radius), drain=(Lx, radius, radius))
        mesh_kwargs = dict(rect=dict(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, contacts=contacts))

    # Run calculation with density and contact current collection:
    transport = Transport(
        fermi_circle=dict(kF=1.5, vF=1.2, n_theta=n_theta),
        geometry=dict(
            contacts=dict(source=dict(dmu=dmu), drain=dict(dmu=-dmu)), mesh=mesh_kwargs
        ),
        time_evolution=dict(t_max=6.01, dt_probe=1.0, dt_save=1.0, n_collate=100),
    )
    transport.run()

    # Analyze equilibration of density profile and contact currents:
    geometry = transport.geometry
    mesh = geometry.mesh
    cell_div = mesh.cell_division
    cells_mine = mesh.cells[cell_div.i_start : cell_div.i_stop, :-1]
    x = rc.from_np(mesh.vertices[cells_mine, 0].mean(axis=1))
    v_x = geometry.material.transport_velocity[:, 0]
    v_x = v_x[v_x > 0]  # calculate for positive velocities; complete by symmetry

    prefactor = Ly * dmu / n_theta
    for t, probe_flux in zip(geometry.stash_probe.t, geometry.stash_probe.flux):
        # Check current against analytical ray-trace prediction:
        I = probe_flux[..., 0]
        I_ref = [
            prefactor * (torch.where(v_x * t_check > Lx, 2, 1) * v_x).sum().item()
            for t_check in t * np.array([0.9, 1.1])
        ]
        I_tol = 1.0e-3 * np.mean(I_ref)
        I_min = min(I_ref) - I_tol
        I_max = max(I_ref) + I_tol
        I_obs = I.abs().mean().item()
        log.info(f"Measured current: {I_obs:.3g} (bounds: {I_min:.3g}, {I_max:.3g})")
        assert abs(I.sum().item()) < I_tol  # Symmetry
        assert I_min < I_obs < I_max

    prefactor = dmu / n_theta
    for t, observables in zip(geometry.stash.t, geometry.stash.observables):
        # Check density profile against analytical ray-trace prediction:
        n = observables[..., 0]
        n_ref = prefactor * (
            torch.where(v_x[:, None] * t > x, 1, 0).sum(dim=0)
            - torch.where(v_x[:, None] * t > Lx - x, 1, 0).sum(dim=0)
        )
        n_mae = globalreduce.sum((n - n_ref).abs(), geometry.group) / len(mesh.cells)
        log.info(f"Density MAE: {n_mae:.3g} (typical n: {dmu:.3g})")
        assert n_mae < dmu * 0.02


@pytest.mark.parametrize(
    "h, n_theta, rotation_deg, rel_err_rate_bound",
    (
        [
            (0.025, 8, 0.0, 2.0e-2),  # baseline
            (0.018, 8, 0.0, 1.0e-2),  # quadratic improvement with h
            (0.025, 16, 0.0, 2.0e-2),  # no effect of n_theta when aligned
            (0.025, 8, 17.9, 4.0e-1),  # reflections worsen when not aligned,
            (0.025, 32, 17.9, 6.0e-2),  # but improve with n_theta
        ]
    ),
)
def test_gaussian_pong(
    h: float, n_theta: int, rotation_deg: float, rel_err_rate_bound: float
) -> None:
    """Bounce a k-weighted gaussian blob with net charge and drift velocity
    within a rectangular domain, with and without rotations, and compare the
    full transient to an analytical (forward ray-trace) solution.
    This test confirms quadratic error scaling with grid resolution, as well as
    linear error scaling with angular resolution for the non-aligned (rotated) case."""
    Lx = 1.1
    Ly = 0.9
    vF = 0.8

    # Construct mesh with vertex rotation:
    Nx = int(np.round(Lx / h))
    Ny = int(np.round(Ly / h))
    mesh = Mesh(
        rect=dict(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, contacts={}), group=dist.group.WORLD
    )
    cos = np.cos(np.deg2rad(rotation_deg))
    sin = np.sin(np.deg2rad(rotation_deg))
    rot = np.array([[cos, -sin], [sin, cos]])
    vertices = np.copy(mesh.vertices)  # un-rotated copy
    mesh.vertices[:] = vertices @ rot  # mesh has rotated version
    transport = Transport(
        fermi_circle=dict(kF=1.5, vF=vF, n_theta=n_theta),
        geometry=dict(contacts={}, mesh=mesh),
        time_evolution=dict(t_max=2.01, dt_probe=0.5, dt_save=10.0),
    )

    # Create test distribution in unrotated frame:
    geometry = transport.geometry
    cell_div = mesh.cell_division
    cells_mine = mesh.cells[cell_div.i_start : cell_div.i_stop, :-1]
    r = rc.from_np(vertices[cells_mine].mean(axis=1))  # unrotated frame
    r0 = torch.tensor([0.5, 0.4], device=rc.device)
    sigma = 0.1
    v = transport.material.transport_velocity @ rc.from_np(rot.T)  # unrotated frame
    wk = (v @ torch.tensor([0.8, 0.6], device=rc.device)) / vF + 0.5  # charge + drift
    geometry.rho[: cell_div.n_mine] = wk * torch.exp(
        -0.5 * ((r - r0) / sigma).square().sum(dim=1, keepdim=True)
    )
    L = torch.tensor([Lx, Ly], device=rc.device)

    # Pre-calculate nearest boundary reflections of r for rho prediction below:
    x, y = r.T
    x_reflected = torch.where(x < 0.5 * Lx, -x, 2 * Lx - x)
    y_reflected = torch.where(y < 0.5 * Ly, -y, 2 * Ly - y)
    r_reflected = torch.stack(
        (
            torch.stack((x, y), dim=-1),
            torch.stack((x, y_reflected), dim=-1),
            torch.stack((x_reflected, y), dim=-1),
            torch.stack((x_reflected, y_reflected), dim=-1),
        )
    )
    rel_err_rates = []

    def check_results(i_step: int, t: float) -> None:
        """Intercept Geometry.update_probes to check results"""
        if not i_step:
            return  # don't test the initial distribution
        # Predict where the blob should be at each k:
        frac = (r0 + v * t) / L
        i_cell = frac.floor().to(torch.int)
        frac -= i_cell
        frac_reflected = torch.where(i_cell % 2 == 0, frac, 1 - frac)
        r_t = (frac_reflected * L)[:, None, None]
        # Compute corresponding density and error:
        kernel = torch.exp(-0.5 * ((r_reflected - r_t) / sigma).square().sum(dim=-1))
        n_pred = wk @ kernel.sum(dim=1)
        n = geometry.rho[: cell_div.n_mine].sum(dim=-1)
        rel_err = (
            globalreduce.sum((n - n_pred).abs(), group=geometry.group)
            / globalreduce.sum(n_pred.abs(), group=geometry.group)
        ).item()
        rel_err_rate = rel_err / t
        rel_err_rates.append(rel_err_rate)
        assert rel_err_rate < rel_err_rate_bound

    transport.geometry.update_probes = check_results
    transport.run()
    errors = np.array(rel_err_rates)
    log.info(f"Relative error rates: {errors} (max: {errors.max():.2g})")


def main():
    log_config()
    pytest.main([__file__, "-vs"])


if __name__ == "__main__":
    main()
