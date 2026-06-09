"""Tests for the cell-centered finite-volume solver (:class:`TriSet`).

Covers the geometry operators (least-squares gradient, periodic edge pairing),
conservation (closed-domain mass, conservative contact-current readout) and the
full contact parity with the DG solver (fixed-voltage, floating probe, current
source). Run directly (serial) or under pytest.
"""
from __future__ import annotations
import os
import tempfile
from collections import Counter

import numpy as np
import torch

from qimpy import rc
from qimpy.mpi import ProcessGrid
from ..material import FermiSurface
from ._mesh import load_mesh, save_mesh
from ._tri_set import TriSet, build_fv_geom


# --------------------------------------------------------------------------- #
#  mesh generators (self-contained; qimpy does not mesh -- `triangle` is used
#  here only to produce small fixtures for the tests)
# --------------------------------------------------------------------------- #
def _make_rect_mesh(grid_spacing, path, all_walls=False):
    """rect-domain [5,105]x[5,55] with source/drain contact faces (mirrors
    examples/.../rect-domain.svg); all_walls=True closes it into a cavity."""
    import triangle as tr
    pts = np.array([[5, 5], [105, 5], [105, 55], [5, 55]], float)
    seg = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    m = tr.triangulate({"vertices": pts, "segments": seg},
                       f"pq30a{grid_spacing ** 2:g}")
    V, T = m["vertices"], m["triangles"]
    ec: Counter = Counter()
    for t in T:
        for x, y in [(0, 1), (1, 2), (2, 0)]:
            ec[tuple(sorted((int(t[x]), int(t[y]))))] += 1
    be = [e for e, c in ec.items() if c == 1]
    SRC, DRN = (10.0, 55.0, 5.0), (10.0, 5.0, 5.0)
    if all_walls:
        bm = ["wall"] * len(be)
    else:
        bm = []
        for a, b in be:
            mx, my = 0.5 * (V[a] + V[b])
            if (mx - SRC[0]) ** 2 + (my - SRC[1]) ** 2 <= SRC[2] ** 2:
                bm.append("source")
            elif (mx - DRN[0]) ** 2 + (my - DRN[1]) ** 2 <= DRN[2] ** 2:
                bm.append("drain")
            else:
                bm.append("wall")
    save_mesh(path, V, T, np.array(be), bm)
    return path


def _make_periodic_rect(n, L, path):
    """Structured n x n triangulation of [0,L]^2 with periodic lattice vectors."""
    xs = np.linspace(0.0, L, n + 1)
    V = np.array([[x, y] for y in xs for x in xs], float)

    def idx(i, j):
        return j * (n + 1) + i

    T, be = [], []
    for j in range(n):
        for i in range(n):
            a, b = idx(i, j), idx(i + 1, j)
            c, d = idx(i + 1, j + 1), idx(i, j + 1)
            T += [[a, b, c], [a, c, d]]
    for i in range(n):
        be += [[idx(i, 0), idx(i + 1, 0)], [idx(i, n), idx(i + 1, n)]]
    for j in range(n):
        be += [[idx(0, j), idx(0, j + 1)], [idx(n, j), idx(n, j + 1)]]
    save_mesh(path, V, np.array(T), np.array(be), ["periodic"] * len(be),
              lattice=[[L, 0.0], [0.0, L]])
    return path


def _make_strip_mesh(nx, ny, Lx, Ly, alpha_deg, path):
    """Tilted strip: [0,Lx]x[0,Ly] rotated by alpha so the top/bottom walls are
    OBLIQUE and the periodic lattice vector is (Lx cos a, Lx sin a) along the
    slant. At axis-aligned walls the discrete reflection coincides with a
    Galerkin operator by symmetry, masking the finite-N_theta tangential-
    quadrature artifact; at an oblique angle it does not, so this geometry
    discriminates the reflector's (D, T) tangential-momentum correction."""
    a = np.deg2rad(alpha_deg)
    c, s = np.cos(a), np.sin(a)
    xs = np.linspace(0.0, Lx, nx + 1)
    ys = np.linspace(0.0, Ly, ny + 1)
    V = np.array([[x * c - y * s, x * s + y * c] for y in ys for x in xs], float)

    def idx(i, j):
        return j * (nx + 1) + i

    T = []
    for j in range(ny):
        for i in range(nx):
            a_, b_ = idx(i, j), idx(i + 1, j)
            c_, d_ = idx(i + 1, j + 1), idx(i, j + 1)
            T += [[a_, b_, c_], [a_, c_, d_]]
    be, bm = [], []
    for i in range(nx):
        be.append([idx(i, 0), idx(i + 1, 0)]); bm.append("wall")
        be.append([idx(i, ny), idx(i + 1, ny)]); bm.append("wall")
    for j in range(ny):
        be.append([idx(0, j), idx(0, j + 1)]); bm.append("periodic")
        be.append([idx(nx, j), idx(nx, j + 1)]); bm.append("periodic")
    save_mesh(path, V, np.array(T), np.array(be), bm, lattice=[[Lx * c, Lx * s]])
    return path


def _make_disk_mesh(R, n_seg, max_area, path, center=(50.0, 30.0)):
    """Triangulated disk: a circular boundary approximated by ``n_seg`` straight
    segments, every boundary edge a reflective wall. The boundary normals span
    all orientations, so it exercises the wall reflector at arbitrary angles."""
    import triangle as tr
    th = np.linspace(0.0, 2 * np.pi, n_seg, endpoint=False)
    pts = np.column_stack([center[0] + R * np.cos(th), center[1] + R * np.sin(th)])
    seg = np.column_stack([np.arange(n_seg), (np.arange(n_seg) + 1) % n_seg])
    m = tr.triangulate({"vertices": pts, "segments": seg}, f"pq30a{max_area:g}")
    V, T = m["vertices"], m["triangles"]
    ec: Counter = Counter()
    for t in T:
        for x, y in [(0, 1), (1, 2), (2, 0)]:
            ec[tuple(sorted((int(t[x]), int(t[y]))))] += 1
    be = [e for e, c in ec.items() if c == 1]
    save_mesh(path, V, T, np.array(be), ["wall"] * len(be))
    return path


# --------------------------------------------------------------------------- #
#  builders
# --------------------------------------------------------------------------- #
def _make_line_mesh(nx, path, Lx=1.0, ends=("source", "drain")):
    """1D line mesh: nx interval cells on [0, Lx] (y=0); the two ends are tagged
    ``ends`` (default source/drain)."""
    x = np.linspace(0.0, Lx, nx + 1)
    V = np.column_stack([x, np.zeros(nx + 1)])
    cells = np.column_stack([np.arange(nx), np.arange(1, nx + 1)])
    be = np.array([[0, 0], [nx, nx]], int)                # ends as degenerate (v,v)
    save_mesh(path, V, cells, be, list(ends))
    return path


def _build_fv(contacts, *, mesh_path=None, gs=12.0, vF=1.5, M=8, **mat_kw):
    """FermiSurface(Nr=1) device on a triangle mesh, wrapped in a TriSet geometry."""
    tmp = tempfile.mkdtemp()
    path = mesh_path or _make_rect_mesh(gs, os.path.join(tmp, "rect.npz"))
    pg = ProcessGrid(rc.comm, "rk", (1, 1))
    kw = dict(kF=1.0, vF=vF, M_theta=M, Nr=1, T=1.0,
              tau_p=np.inf, tau_ee=np.inf, r_c=np.inf, specularity=1.0)
    kw.update(mat_kw)
    material = FermiSurface(process_grid=pg, **kw)
    geom = TriSet(material=material, mesh_file=path, contacts=contacts,
                 process_grid=pg)
    return geom, material


def _step(geom, nsteps):
    """RK4 advance (advection + collisions, both inside rho_dot)."""
    dt = 0.5 * geom.dt_max
    for _ in range(nsteps):
        r0 = geom.rho
        k1 = geom.rho_dot(r0, 0.0)
        k2 = geom.rho_dot(r0 + (0.5 * dt) * k1, 0.0)
        k3 = geom.rho_dot(r0 + (0.5 * dt) * k2, 0.0)
        k4 = geom.rho_dot(r0 + dt * k3, 0.0)
        geom.rho = r0 + (dt / 6.0) * (k1 + 2 * (k2 + k3) + k4)


def _mass_rate(geom):
    """d/dt of total particle number = sum_k area_k * sum_c ncoef_c (du/dt)_kc."""
    dudt = geom.rho_dot(geom.rho, 0.0)[0]
    return float((geom.geom.area[:, None] * geom._ncoef[None, :] * dudt).sum())


def _integral(geom, material, obs_idx, t=0.0):
    """Domain integral of observable `obs_idx` (0=n, 1=jx, 2=jy): sum_k area_k o_k."""
    obs = torch.einsum("oc,kc->ko", material.get_observables(t), geom._u)  # (K,3)
    return float((geom.geom.area * obs[:, obs_idx]).sum())


# --------------------------------------------------------------------------- #
#  geometry operators
# --------------------------------------------------------------------------- #
def test_lsq_gradient_is_exact_on_linear_fields() -> None:
    """The fused reconstruction operator reproduces any linear field's face
    increments to machine precision -- so the scheme is exactly 2nd-order in the
    unlimited (smooth) regime, on a distorted/irregular mesh."""
    torch.set_default_dtype(torch.float64)
    tmp = tempfile.mkdtemp()
    g = build_fv_geom(load_mesh(_make_rect_mesh(9.0, os.path.join(tmp, "r.npz"))))
    dev = g.recon.device
    grad = torch.tensor([0.37, -1.21], dtype=torch.float64, device=dev)
    cen = torch.from_numpy(g.centroid_np).to(dev)
    u = (cen @ grad)[:, None]                            # (K, 1) linear field
    d = torch.einsum("kfg,kgc->kfc", g.recon, u[g.nbr] - u[:, None])  # (K,3,1)
    d_exact = torch.einsum("kfx,x->kf", _face_offsets(g).to(dev), grad)
    assert float((d[..., 0] - d_exact).abs().max()) < 1e-11


def _face_offsets(g):
    """(centroid -> face-midpoint) offset per cell/face, from the stored mesh."""
    p = torch.from_numpy(g.vertices_np)[torch.from_numpy(g.triangles_np)]  # (K,3,2)
    fmid = 0.5 * (p[:, [0, 1, 2]] + p[:, [1, 2, 0]])
    return fmid - p.mean(1)[:, None]


def test_periodic_lattice_promotes_all_boundary_edges() -> None:
    """On a fully periodic square every boundary edge pairs through the lattice
    and becomes interior, so there are no boundary edges and an arbitrary state
    conserves mass exactly (no faces can leak)."""
    torch.set_default_dtype(torch.float64)
    geom, _ = _build_fv({}, mesh_path=_periodic_mesh(6, 10.0))
    assert geom.geom.bcell.numel() == 0
    geom._u = torch.randn(geom.K, geom.Nk, device=rc.device)
    assert abs(_mass_rate(geom)) < 1e-12


def _periodic_mesh(n, L):
    tmp = tempfile.mkdtemp()
    return _make_periodic_rect(n, L, os.path.join(tmp, "per.npz"))


# --------------------------------------------------------------------------- #
#  conservation
# --------------------------------------------------------------------------- #
def test_closed_domain_conserves_mass() -> None:
    """A fully reflective (all-walls) cavity neither gains nor loses particles:
    the mass-conserving reflector drives the total mass rate to ~0."""
    torch.set_default_dtype(torch.float64)
    tmp = tempfile.mkdtemp()
    path = _make_rect_mesh(12.0, os.path.join(tmp, "rect.npz"), all_walls=True)
    geom, _ = _build_fv({}, mesh_path=path)
    geom._u = torch.randn(geom.K, geom.Nk, device=rc.device)   # arbitrary state
    assert abs(_mass_rate(geom)) < 1e-9


def test_contact_current_readout_is_conservative() -> None:
    """Sum of contact currents equals minus the total mass rate (walls carry no
    current), so the readout exactly accounts for the device's charge balance."""
    torch.set_default_dtype(torch.float64)
    geom, _ = _build_fv({"source": {"dmu": 0.1}, "drain": {"dmu": -0.1}})
    _step(geom, 40)
    I = geom.contact_currents(0.0)
    assert abs(sum(I.values()) + _mass_rate(geom)) < 1e-9


# --------------------------------------------------------------------------- #
#  contact parity
# --------------------------------------------------------------------------- #
def test_floating_contact_carries_no_current() -> None:
    """A voltage probe's level floats to zero its own current, exactly."""
    torch.set_default_dtype(torch.float64)
    geom, _ = _build_fv({"source": {"dmu": 0.1}, "drain": {"floating": True}})
    _step(geom, 150)
    assert abs(geom.contact_currents(0.0)["drain"]) < 1e-12


def test_floating_contact_reads_uniform_potential() -> None:
    """In a device at a uniform isotropic level V0, a floating probe reads V0."""
    torch.set_default_dtype(torch.float64)
    geom, _ = _build_fv({"source": {"dmu": 0.1}, "drain": {"floating": True}})
    for V0 in (0.05, -0.1, 0.2):
        geom._u = torch.full((geom.K, geom.Nk), float(V0), device=rc.device)
        geom.contact_currents(0.0)                       # solves the feedback level
        assert abs(geom.contact_potentials()["drain"] - V0) < 1e-12


def test_current_source_zero_equals_floating() -> None:
    """A current source with I_set = 0 reproduces a floating probe exactly."""
    torch.set_default_dtype(torch.float64)
    geom, _ = _build_fv({"source": {"dmu": 0.1}, "drain": {"I_set": 0.0}})
    _step(geom, 150)
    assert abs(geom.contact_currents(0.0)["drain"]) < 1e-12


def test_current_source_delivers_prescribed_current() -> None:
    """Each evaluation a current source self-adjusts its level so the net outward
    flux equals I_set; the same-flux readout then agrees to roundoff every step.
    Two sources drive current through a resistive device."""
    torch.set_default_dtype(torch.float64)
    I_target = 0.05
    geom, _ = _build_fv(
        {"source": {"I_set": -I_target}, "drain": {"I_set": +I_target}},
        tau_p=15.0, tau_ee=8.0)
    _step(geom, 40)
    I = geom.contact_currents(0.0)
    assert abs(I["source"] + I_target) < 1e-10, I["source"]
    assert abs(I["drain"] - I_target) < 1e-10, I["drain"]
    V = geom.contact_potentials()
    assert V["source"] > V["drain"]                      # injector sits at higher mu


def test_current_source_polarity_reverses_with_sign() -> None:
    """Flipping the sign of I_set swaps the device potentials."""
    torch.set_default_dtype(torch.float64)
    gp, _ = _build_fv({"source": {"I_set": -0.02}, "drain": {"I_set": +0.02}})
    gn, _ = _build_fv({"source": {"I_set": +0.02}, "drain": {"I_set": -0.02}})
    _step(gp, 40); _step(gn, 40)
    Ip, In = gp.contact_currents(0.0), gn.contact_currents(0.0)
    Vp, Vn = gp.contact_potentials(), gn.contact_potentials()
    assert abs(Ip["source"] + In["source"]) < 1e-10
    assert (Vp["source"] - Vp["drain"]) * (Vn["source"] - Vn["drain"]) < 0


# --------------------------------------------------------------------------- #
#  long-time wall physics (specular reflection conserves mass + tangential mom.)
# --------------------------------------------------------------------------- #
def _steps_for(geom, t_end):
    return int(t_end / (0.5 * geom.dt_max))


def test_reflective_walls_conserve_mass_long_time() -> None:
    """A closed (all-wall) cavity conserves total particle number to ~machine
    precision as a density blob traverses the domain and actively reflects --
    the time-integrated companion to the instantaneous-rate test above."""
    torch.set_default_dtype(torch.float64)
    tmp = tempfile.mkdtemp()
    path = _make_rect_mesh(10.0, os.path.join(tmp, "rect.npz"), all_walls=True)
    geom, mat = _build_fv({}, mesh_path=path)
    cen = torch.from_numpy(geom.geom.centroid_np).to(rc.device)
    q0 = torch.tensor([55.0, 30.0], dtype=torch.float64, device=rc.device)
    blob = torch.exp(-((cen - q0) ** 2).sum(-1) / (2 * 6.0 ** 2))   # (K,)
    geom._u = blob[:, None].repeat(1, geom.Nk)                      # isotropic = density
    m0 = _integral(geom, mat, 0)
    _step(geom, _steps_for(geom, 30.0))
    assert abs(_integral(geom, mat, 0) - m0) / abs(m0) < 1e-10


def test_oblique_wall_conserves_tangential_momentum() -> None:
    """On a tilted strip (oblique specular walls + periodic along the slant) the
    wall-tangent current J_tang = cos(a) jx + sin(a) jy is a global invariant of
    specular reflection. Holds to round-off with the (D,T) reflector; the older
    single-D scheme drifts ~1e-4 here -- this is the discriminating test."""
    torch.set_default_dtype(torch.float64)
    alpha = 23.7                                       # oblique: not 0/45/90 deg
    a = np.deg2rad(alpha); ca, sa = float(np.cos(a)), float(np.sin(a))
    Lx, Ly = 40.0, 20.0
    tmp = tempfile.mkdtemp()
    mesh = _make_strip_mesh(8, 4, Lx, Ly, alpha, os.path.join(tmp, "strip.npz"))
    geom, mat = _build_fv({}, mesh_path=mesh)
    cen = geom.geom.centroid_np
    d_perp = -sa * cen[:, 0] + ca * cen[:, 1] - 0.5 * Ly
    blob = np.exp(-(d_perp ** 2) / (2 * 2.0 ** 2))                  # (K,)
    theta = mat.angular.theta.detach().cpu().numpy()               # (Nk,)
    u0 = 1.0 * blob[:, None] + 0.3 * np.cos(theta - a)[None, :]     # density + drift
    geom._u = torch.as_tensor(u0, device=rc.device, dtype=torch.float64)
    n0 = _integral(geom, mat, 0)
    J0 = ca * _integral(geom, mat, 1) + sa * _integral(geom, mat, 2)
    _step(geom, _steps_for(geom, 20.0))
    n1 = _integral(geom, mat, 0)
    J1 = ca * _integral(geom, mat, 1) + sa * _integral(geom, mat, 2)
    assert abs(n1 - n0) / abs(n0) < 1e-10, f"mass drift {(n1 - n0) / n0:.2e}"
    assert abs(J1 - J0) / abs(J0) < 1e-10, f"J_tang drift {(J1 - J0) / J0:.2e}"


# --------------------------------------------------------------------------- #
#  contact-driven steady state
# --------------------------------------------------------------------------- #
def test_contact_driven_state_is_bounded() -> None:
    """Source/drain contacts (dmu = +/-0.1) drive a finite, bounded solution
    (no blow-up; interior density stays within the contact range)."""
    torch.set_default_dtype(torch.float64)
    geom, mat = _build_fv({"source": {"dmu": 0.1}, "drain": {"dmu": -0.1}})
    _step(geom, _steps_for(geom, 50.0))
    n = torch.einsum("oc,kc->ko", mat.get_observables(0.0), geom._u)[:, 0]
    assert torch.isfinite(n).all(), "contact-driven solution diverged"
    assert float(n.abs().max()) < 0.15, "interior density exceeds contact range"


def test_biased_contacts_balance_at_steady_state() -> None:
    """In a resistive device (finite tau) the source and drain currents relax to
    equal and opposite as the device approaches DC steady state, with a real
    current flowing. The exact-balance residual is -d/dt(mass), which decays on
    the device's (slow) charging time, so the resolution-independent invariant
    checked here is the *relative* imbalance |I_s + I_d| / |I_s|. Stepped with
    collisions, since a ballistic cavity rings rather than settling."""
    torch.set_default_dtype(torch.float64)
    geom, _ = _build_fv({"source": {"dmu": 0.1}, "drain": {"dmu": -0.1}},
                        tau_p=15.0, tau_ee=8.0)
    _step(geom, _steps_for(geom, 600.0))
    I = geom.contact_currents(0.0)
    assert abs(I["source"]) > 1e-3, "no current flowing"
    assert abs(I["source"] + I["drain"]) / abs(I["source"]) < 1e-2, I  # equal & opp.


def test_curved_mass_conservation() -> None:
    """A closed disk -- a curved (circular) boundary approximated by straight wall
    edges whose normals span all orientations -- conserves total mass to ~machine
    precision as a blob expands and reflects. The FV flux form is conservative and
    the reflector zeroes net mass flux on every edge regardless of its angle, so
    cell-centered FV needs no isoparametric/arc projectors (unlike the high-order
    DG version this replaces, which was skipped for that reason)."""
    torch.set_default_dtype(torch.float64)
    tmp = tempfile.mkdtemp()
    path = _make_disk_mesh(20.0, 64, 4.0, os.path.join(tmp, "disk.npz"))
    geom, mat = _build_fv({}, mesh_path=path)
    cen = torch.from_numpy(geom.geom.centroid_np).to(rc.device)
    q0 = torch.tensor([50.0, 30.0], dtype=torch.float64, device=rc.device)
    blob = torch.exp(-((cen - q0) ** 2).sum(-1) / (2 * 5.0 ** 2))
    geom._u = blob[:, None].repeat(1, geom.Nk)
    m0 = _integral(geom, mat, 0)
    _step(geom, _steps_for(geom, 30.0))
    assert abs(_integral(geom, mat, 0) - m0) / abs(m0) < 1e-10


# --------------------------------------------------------------------------- #
#  spatial decomposition (bit-for-bit vs serial). Runs as two subprocesses of
#  this module in "worker" mode (FV_MPI_OUT set) -- one serial, one mpirun -n 2.
# --------------------------------------------------------------------------- #
def _decomp_worker() -> None:
    """Step the rect problem and save the final state in input (un-permuted)
    cell order; invoked as a subprocess by test_decomp_matches_serial. Builds an
    auto-sized process grid (r split over ranks, k=1) so it works at any rank
    count, unlike the fixed (1,1) grid the serial-test builder uses."""
    rc.init()
    torch.set_default_dtype(torch.float64)
    pg = ProcessGrid(rc.comm, "rk", None)
    pg.provide_n_tasks("k", 1)
    mat = FermiSurface(kF=1.0, vF=1.5, M_theta=8, Nr=1, T=1.0,
                       tau_p=15.0, tau_ee=8.0, r_c=np.inf, specularity=1.0,
                       process_grid=pg)
    geom = TriSet(material=mat, mesh_file=os.environ["FV_MPI_MESH"],
                  contacts={"source": {"dmu": 0.1}, "drain": {"floating": True}},
                  process_grid=pg)
    cen = torch.from_numpy(geom.geom.centroid_np).to(rc.device)
    geom._u = torch.zeros(geom.K, geom.Nk, device=rc.device)
    geom._u[:, 0] = 0.01 * (1.0 + cen[:, 0] / 100.0 + cen[:, 1] / 50.0)  # partition-invariant
    dt = 0.5 * geom.dt_max
    for _ in range(30):
        r0 = geom.rho
        geom.rho = r0 + dt * geom.rho_dot(r0 + 0.5 * dt * geom.rho_dot(r0, 0.0), 0.0)
    owned = geom._u[geom._own_start:geom._own_stop].detach().cpu().numpy()
    parts = rc.comm.gather(owned, root=0)
    if rc.comm.rank == 0:
        full = np.concatenate(parts, axis=0)            # renumbered order
        u = np.empty_like(full)
        if geom._perm is not None:
            u[geom._perm] = full                        # back to input order
        else:
            u = full
        np.save(os.environ["FV_MPI_OUT"], u)


def test_1d_line_mesh_ballistic_is_antisymmetric() -> None:
    """A 1D wire (interval cells, 2 faces/cell) with source/drain dmu=+/-0.1 runs
    stably through the TriSet 1D geometry path.  Its ballistic steady state is
    antisymmetric, n(L-x) = -n(x), with a spatially uniform current: the +/-x
    populations cancel in the density and carry the current straight through."""
    torch.set_default_dtype(torch.float64)
    tmp = tempfile.mkdtemp()
    mesh = _make_line_mesh(40, os.path.join(tmp, "line.npz"))
    geom, mat = _build_fv({"source": {"dmu": 0.1}, "drain": {"dmu": -0.1}},
                          mesh_path=mesh)
    assert geom._nf == 2                                   # interval cells -> 2 faces
    _step(geom, _steps_for(geom, 20.0))
    x = geom.geom.centroid_np[:, 0]
    obs = torch.einsum("oc,kc->ko", mat.get_observables(0.0), geom._u)
    n = obs[:, 0].cpu().numpy()
    jx = obs[:, 1].cpu().numpy()
    assert np.isfinite(n).all(), "1D solution diverged"
    mirror = np.array([int(np.argmin(np.abs(x - (1.0 - xi)))) for xi in x])
    assert np.linalg.norm(n + n[mirror]) / (np.linalg.norm(n) + 1e-30) < 1e-9
    assert abs(jx.mean()) > 1e-3, "no ballistic current"
    assert jx.std() / abs(jx.mean()) < 1e-2, "ballistic current not uniform"


def test_decomp_matches_serial() -> None:
    """The METIS spatial decomposition reproduces the serial solve bit-for-bit:
    a 2-rank run (partition + 2-ring halo exchange) equals the 1-rank run on the
    same problem to round-off. Spawns this module in worker mode (serial, then
    mpirun -n 2) and compares; needs mpirun + pymetis."""
    import subprocess
    import sys
    tmp = tempfile.mkdtemp()
    mesh = _make_rect_mesh(12.0, os.path.join(tmp, "rect.npz"))
    mod = "qimpy.transport.geometry.test_tri_set"
    f1, f2 = os.path.join(tmp, "u1.npy"), os.path.join(tmp, "u2.npy")
    env = dict(os.environ, FV_MPI_MESH=mesh)
    subprocess.run([sys.executable, "-m", mod], check=True, env=dict(env, FV_MPI_OUT=f1))
    subprocess.run(["mpirun", "-n", "2", sys.executable, "-m", mod], check=True,
                   env=dict(env, FV_MPI_OUT=f2))
    u1, u2 = np.load(f1), np.load(f2)
    assert np.allclose(u1, u2, atol=1e-12, rtol=0), float(np.abs(u1 - u2).max())


if __name__ == "__main__":
    if os.environ.get("FV_MPI_OUT"):           # subprocess worker for the test above
        _decomp_worker()
        raise SystemExit
    rc.init()
    test_lsq_gradient_is_exact_on_linear_fields(); print("lsq_gradient_exact: PASS")
    test_periodic_lattice_promotes_all_boundary_edges(); print("periodic_promote: PASS")
    test_closed_domain_conserves_mass(); print("closed_domain_mass: PASS")
    test_contact_current_readout_is_conservative(); print("contact_readout_conservative: PASS")
    test_floating_contact_carries_no_current(); print("floating_zero_current: PASS")
    test_floating_contact_reads_uniform_potential(); print("floating_reads_potential: PASS")
    test_current_source_zero_equals_floating(); print("current_source_zero: PASS")
    test_current_source_delivers_prescribed_current(); print("current_source_delivers: PASS")
    test_current_source_polarity_reverses_with_sign(); print("current_source_polarity: PASS")
    test_reflective_walls_conserve_mass_long_time(); print("reflective_mass_long_time: PASS")
    test_oblique_wall_conserves_tangential_momentum(); print("oblique_wall_tang_momentum: PASS")
    test_contact_driven_state_is_bounded(); print("contact_driven_bounded: PASS")
    test_biased_contacts_balance_at_steady_state(); print("biased_balance: PASS")
    test_curved_mass_conservation(); print("curved_mass_conservation: PASS")
    test_decomp_matches_serial(); print("decomp_matches_serial: PASS")
    print("ALL PASS")
