"""End-to-end tests for the modal (angular-harmonic) Fermi-circle material.

These exercise the full pipeline: the coupled-system DG spatial advection
(shared kernel; discrete-ordinates is its diagonal special case), the modal
collision and reflector operators, and conservation. Run directly (the
transport suite needs rc.init / --with-mpi under pytest)."""
from __future__ import annotations
import tempfile, os

import numpy as np
import torch

from qimpy import rc
from qimpy.mpi import ProcessGrid
from ..material import FermiCircleModes
from ..material._angular_modes import AngularModes
from ._dg_mesh import load_mesh
from ._dg_mpi import build_distributed
from . import test_advect as TA


def _setup(M, *, tau_p=np.inf, tau_ee=np.inf, contacts=None, order=3, vF=1.5):
    """Build a modal material on the rect mesh, plus a smooth band-limited modal
    test field f (low harmonics populated)."""
    tmp = tempfile.mkdtemp()
    mesh = load_mesh(TA._make_rect_mesh(12.0, os.path.join(tmp, "rect.npz")))
    am = AngularModes(M, vF)
    pg = ProcessGrid(rc.comm, "rk", (1, 1))
    mm = FermiCircleModes(kF=1.0, vF=vF, M=M, tau_p=tau_p, tau_ee=tau_ee,
                          r_c=np.inf, specularity=1.0, process_grid=pg)
    sp_m, da_m = build_distributed(mesh, order, mm, contacts or {}, rc.comm)
    dg = sp_m.dg
    x = torch.as_tensor(dg.x); y = torch.as_tensor(dg.y)
    Lx, Ly = dg.x.max() - dg.x.min(), dg.y.max() - dg.y.min()
    f = torch.zeros(dg.Np, dg.K, am.dim)
    f[..., 0] = 0.5 + 0.2 * torch.sin(2 * np.pi * x / Lx) * torch.cos(2 * np.pi * y / Ly)
    f[..., 1] = 0.3 * torch.cos(2 * np.pi * x / Lx)
    f[..., 2] = 0.25 * torch.sin(2 * np.pi * y / Ly)
    f[..., 3] = 0.1 * torch.cos(2 * np.pi * x / Lx) * torch.sin(2 * np.pi * y / Ly)
    return am, mm, da_m, dg, f


def test_collisions_and_observables_are_closed_form() -> None:
    """The modal collision operator is diagonal in the harmonics -- m=0 (density)
    conserved, m=1 relaxes at 1/tau_p, m>=2 at 1/tau_p + 1/tau_ee -- and the
    observables are the closed-form moments n=a0, jx=(vF/2)a1, jy=(vF/2)b1."""
    torch.set_default_dtype(torch.float64)
    am, mm, da_m, dg, f = _setup(M=6, tau_p=30.0, tau_ee=12.0)
    col = mm.rho_dot(f, 0.3, 0)                       # r_c = inf -> collisions only
    rp, ree = 1.0 / 30.0, 1.0 / 12.0
    rate = torch.zeros(am.dim)
    for m in range(1, am.M + 1):
        rate[2 * m - 1] = rate[2 * m] = rp if m == 1 else (rp + ree)
    assert (col + rate * f).abs().max() < 1e-12       # exactly diagonal at the rates
    assert col[..., 0].abs().max() < 1e-14            # density (m=0) conserved
    obs = mm.measure_observables(f, 0.3)
    assert (obs[..., 0] - mm.wk * f[..., 0]).abs().max() < 1e-12
    assert (obs[..., 1] - mm.wk * 0.5 * mm.vF * f[..., 1]).abs().max() < 1e-12
    assert (obs[..., 2] - mm.wk * 0.5 * mm.vF * f[..., 2]).abs().max() < 1e-12


def test_specular_reflector_mirrors_normal_current() -> None:
    """Specular reflection (s=1) preserves density (m=0) and reflects the m=1
    current vector across the wall, d -> d - 2(d.n)n, at arbitrary wall angles --
    the modal reflector reproduces the exact vector reflection with no on-grid
    constraint (the discrete-ordinate snapping it replaces)."""
    torch.set_default_dtype(torch.float64)
    am = AngularModes(6, 1.5)
    pg = ProcessGrid(rc.comm, "rk", (1, 1))
    mm = FermiCircleModes(kF=1.0, vF=1.5, M=6, tau_p=np.inf, tau_ee=np.inf,
                          r_c=np.inf, specularity=1.0, process_grid=pg)
    rng = np.random.default_rng(1)
    phis = torch.tensor(rng.uniform(0, 2 * np.pi, 5))
    normals = torch.stack([torch.cos(phis), torch.sin(phis)], dim=-1)
    a0 = torch.tensor(rng.standard_normal(5))
    d = torch.tensor(rng.standard_normal((5, 2)))      # incident m=1 current vector
    fm = torch.zeros(1, 5, am.dim)
    fm[0, :, 0] = a0; fm[0, :, 1] = d[:, 0]; fm[0, :, 2] = d[:, 1]
    uP = mm.get_reflector(normals)(fm)[0]
    d_ref = d - 2.0 * (d * normals).sum(-1, keepdim=True) * normals
    assert (uP[:, 0] - a0).abs().max() < 1e-12          # density preserved
    assert (uP[:, 1] - d_ref[:, 0]).abs().max() < 1e-12  # current reflected across wall
    assert (uP[:, 2] - d_ref[:, 1]).abs().max() < 1e-12


def test_spatial_rhs_converges_in_M() -> None:
    """The coupled-flux spatial RHS for a fixed band-limited field converges as M
    grows: the change on the shared low harmonics relative to the M=16 result
    decreases monotonically (Cauchy), confirming a consistent discretization."""
    torch.set_default_dtype(torch.float64)
    rl = {}
    for M in (4, 8, 12, 16):
        am, mm, da_m, dg, f = _setup(M)
        rl[M] = da_m.adv.rhs(f, da_m.vx, da_m.vy, 0.0,
                             coupling=da_m.coupling)[..., :5].clone()
    ref = rl[16]
    e = {M: float((rl[M] - ref).abs().max()) for M in (4, 8, 12)}
    assert e[12] < e[8] < e[4]                          # monotone convergence
    assert e[12] / float(ref.abs().max()) < 1e-3        # well-converged by M=12


def test_mass_conservation_specular_walls() -> None:
    """Specular walls reflect with zero net mass flux: total density (m=0) is
    conserved to ~machine precision under time stepping."""
    torch.set_default_dtype(torch.float64)
    am, mm, da_m, dg, f = _setup(M=8)
    x = torch.as_tensor(dg.x); y = torch.as_tensor(dg.y)
    u0 = torch.zeros(dg.Np, dg.K, am.dim)
    u0[..., 0] = torch.exp(-(((x - 55) ** 2 + (y - 30) ** 2)) / (2 * 8.0 ** 2))
    u0[..., 1] = 0.2 * u0[..., 0]; u0[..., 2] = 0.1 * u0[..., 0]
    w = da_m.adv.apply_mass(u0)
    dt = 0.4 * float(dg.dt_scale) / mm.vF

    def rhs_w(w):
        u = da_m.adv.apply_mass_inv(w)
        return da_m.local_rhs(w, 0.0) + da_m.adv.apply_mass(mm.rho_dot(u, 0.0, 0))

    m0 = float(w[..., 0].sum())
    for _ in range(40):
        k1 = rhs_w(w); k2 = rhs_w(w + 0.5 * dt * k1)
        k3 = rhs_w(w + 0.5 * dt * k2); k4 = rhs_w(w + dt * k3)
        w = w + (dt / 6.0) * (k1 + 2 * (k2 + k3) + k4)
    assert torch.isfinite(w).all()
    assert abs(float(w[..., 0].sum()) - m0) / abs(m0) < 1e-10


def test_diffuse_reflection_conserves_mass_any_angle() -> None:
    """Diffuse (specularity < 1) reflection re-emits isotropically with a flux-
    balanced amplitude, so the net normal mass current at the wall vanishes
    pointwise -- at arbitrary (non-grid-aligned) wall angles. s=1 is specular."""
    torch.set_default_dtype(torch.float64)
    am = AngularModes(8, 1.3); pg = ProcessGrid(rc.comm, "rk", (1, 1))
    rng = np.random.default_rng(0)
    phis = torch.tensor(rng.uniform(0, 2 * np.pi, 7))
    normals = torch.stack([torch.cos(phis), torch.sin(phis)], dim=-1)
    uM = torch.zeros(1, 7, am.dim)
    for c in range(am.dim):
        uM[0, :, c] = torch.tensor(rng.standard_normal(7))
    uM[..., 2 * am.M - 1] = uM[..., 2 * am.M] = 0.0
    nx, ny = torch.cos(phis), torch.sin(phis)
    for s in (1.0, 0.7, 0.3, 0.0):
        refl = FermiCircleModes(kF=1.0, vF=1.3, M=8, tau_p=np.inf, tau_ee=np.inf,
                                r_c=np.inf, specularity=s,
                                process_grid=pg).get_reflector(normals)
        uP = refl(uM)
        fstar = (0.5 * (nx[:, None] * ((uM + uP) @ am.Ax.T)
                        + ny[:, None] * ((uM + uP) @ am.Ay.T))
                 + 0.5 * am.abs_flux(uM - uP, phis))
        assert fstar[..., 0].abs().max() < 1e-12   # zero net mass flux at the wall


def _beam_density_error(M, kappa, *, L=40.0, n=14, order=2, sigma=4.0,
                        theta0=np.deg2rad(30.0), vF=1.0, tfrac=0.2):
    """Ballistically propagate a von Mises angular beam exp(kappa cos(theta-theta0))
    on a periodic domain and return the relative density error vs the exact
    transport integral n(x,t) = <b(theta) g(x - vF t vhat(theta))>_theta."""
    from .. import Transport
    tmp = tempfile.mkdtemp()
    mesh = TA._make_periodic_rect(n, L, os.path.join(tmp, "per.npz"))
    t = Transport(
        fermi_circle_modes=dict(kF=1.0, vF=vF, M=M, tau_p=np.inf, tau_ee=np.inf,
                                r_c=np.inf, specularity=1.0),
        tri_set=dict(mesh_file=mesh, contacts={}, order=order),
        time_evolution=dict(t_max=1.0, dt_save=2.0, n_collate=10))
    g = t.geometry; dg = g.dg; da = g.dist
    am = AngularModes(M, vF)
    x = np.asarray(dg.x); y = np.asarray(dg.y)
    th = np.asarray(am.theta)
    b = np.exp(kappa * np.cos(th - theta0)); b /= b.mean()      # a0 (mean) = 1
    b_modes = np.asarray(torch.einsum('cq,q->c', am.Tinv, torch.as_tensor(b)))
    blob = np.exp(-(((x - L / 2) ** 2 + (y - L / 2) ** 2)) / (2 * sigma ** 2))
    u0 = torch.zeros(dg.Np, dg.K, da.Nk)
    for c in range(da.Nk):
        u0[..., c] = torch.as_tensor(blob * b_modes[c])
    w = da.adv.apply_mass(u0)
    dt = 0.4 * float(dg.dt_scale) / vF
    T = tfrac * L / vF
    nsteps = max(1, int(round(T / dt))); dt = T / nsteps
    for _ in range(nsteps):
        k1 = da.local_rhs(w, 0.0); k2 = da.local_rhs(w + 0.5 * dt * k1, 0.0)
        k3 = da.local_rhs(w + 0.5 * dt * k2, 0.0); k4 = da.local_rhs(w + dt * k3, 0.0)
        w = w + (dt / 6.0) * (k1 + 2 * (k2 + k3) + k4)
    n_modal = np.asarray(da.adv.apply_mass_inv(w)[..., 0])
    # exact reference: fine angular quadrature, periodic Gaussian images
    NF = 512; thf = np.linspace(0, 2 * np.pi, NF, endpoint=False)
    bf = np.exp(kappa * np.cos(thf - theta0)); bf /= bf.mean()
    n_exact = np.zeros_like(n_modal)
    for a in range(NF):
        dxx = x - (L / 2 + vF * T * np.cos(thf[a]))
        dyy = y - (L / 2 + vF * T * np.sin(thf[a]))
        acc = 0.0
        for ix in (-1, 0, 1):
            for iy in (-1, 0, 1):
                acc = acc + np.exp(-(((dxx - ix * L) ** 2 + (dyy - iy * L) ** 2))
                                   / (2 * sigma ** 2))
        n_exact += bf[a] * acc
    n_exact /= NF
    return (float(dg.integrate(np.abs(n_modal - n_exact)))
            / float(dg.integrate(np.abs(n_exact))))


def test_beam_at_angle_converges_in_M() -> None:
    """A collimated angular beam at a non-grid angle (30 deg, FWHM ~38 deg)
    converges as the harmonic count M grows: a spectral basis resolves a sharp
    angular feature only with enough harmonics (empirically M ~ 2*pi/FWHM). The
    density error vs the exact transport integral falls monotonically and reaches
    the spatial-discretization floor; an under-resolved beam (small M) is visibly
    wrong (Gibbs ringing, even negative density). This quantifies the modal cost
    for beams -- the regime that previously motivated discrete ordinates."""
    torch.set_default_dtype(torch.float64)
    e4 = _beam_density_error(4, 12.0)
    e8 = _beam_density_error(8, 12.0)
    e12 = _beam_density_error(12, 12.0)
    assert e8 < e4 and e12 <= e8 * 1.05       # monotone convergence in M
    assert e4 > 0.02                           # M=4 beam is visibly under-resolved
    assert e12 < 0.01                          # converged to ~spatial floor by M=12


def test_checkpoint_roundtrip(tmp_path) -> None:
    """Write a modal run (finite r_c cyclotron + diffuse walls + contacts) to a
    checkpoint, then reconstruct purely from the file: the FermiCircleModes
    variant, its parameters and the modal density are all restored."""
    import glob
    from .. import Transport
    torch.set_default_dtype(torch.float64)
    mesh = TA._make_rect_mesh(14.0, str(tmp_path / "rect.npz"))
    out = str(tmp_path / "ck_{:04d}.h5")
    cdict = {"source": {"dmu": 0.1}, "drain": {"dmu": -0.1}}
    t1 = Transport(
        fermi_circle_modes=dict(kF=1.0, vF=1.5, M=5, tau_p=40.0, tau_ee=15.0,
                                r_c=8.0, specularity=0.6),
        tri_set=dict(mesh_file=mesh, contacts=cdict, order=2),
        time_evolution=dict(t_max=2.0, dt_save=1.0, n_collate=5),
        checkpoint_out=out)
    g = t1.geometry
    if t1.material.comm.rank == 0:
        x = torch.as_tensor(g.dg.x); y = torch.as_tensor(g.dg.y)
        d = torch.zeros_like(g.density)
        d[:, :, 0] = torch.exp(-(((x - 55) ** 2 + (y - 30) ** 2)) / (2 * 8.0 ** 2))
        g.density = d
    t1.run()
    rho1 = g.density.detach().cpu().numpy().copy()
    last = sorted(glob.glob(str(tmp_path / "ck_*.h5")))[-1]
    t2 = Transport(checkpoint=last,
                   tri_set=dict(mesh_file=mesh, contacts=cdict, order=2),
                   time_evolution=dict(t_max=2.0, dt_save=1.0, n_collate=5))
    m2 = t2.material
    assert type(m2).__name__ == "FermiCircleModes"
    assert m2.M == 5 and abs(m2.r_c - 8.0) < 1e-12 and abs(m2.specularity - 0.6) < 1e-12
    rho2 = t2.geometry.density.detach().cpu().numpy()
    assert rho1.shape == rho2.shape and np.abs(rho1 - rho2).max() < 1e-10


if __name__ == "__main__":
    import tempfile, pathlib
    rc.init()
    test_collisions_and_observables_are_closed_form(); print("collisions_and_observables: PASS")
    test_specular_reflector_mirrors_normal_current(); print("specular_reflector: PASS")
    test_diffuse_reflection_conserves_mass_any_angle(); print("diffuse_reflection: PASS")
    test_spatial_rhs_converges_in_M(); print("spatial_converges: PASS")
    test_beam_at_angle_converges_in_M(); print("beam_converges_in_M: PASS")
    test_mass_conservation_specular_walls(); print("mass_conservation: PASS")
    test_checkpoint_roundtrip(pathlib.Path(tempfile.mkdtemp())); print("checkpoint_roundtrip: PASS")
    print("ALL PASS")
