"""Tests for the cell-centered finite-volume solver (:class:`FiniteVolume`).

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
from ._finite_volume import FiniteVolume, build_fv_geom


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
    """FermiSurface(Nr=1) device on a triangle mesh, wrapped in a FiniteVolume geometry."""
    tmp = tempfile.mkdtemp()
    path = mesh_path or _make_rect_mesh(gs, os.path.join(tmp, "rect.npz"))
    pg = ProcessGrid(rc.comm, "rk", (1, 1))
    kw = dict(kF=1.0, vF=vF, M_theta=M, Nr=1, T=1.0,
              tau_p=np.inf, tau_ee=np.inf, r_c=np.inf, specularity=1.0)
    kw.update(mat_kw)
    material = FermiSurface(process_grid=pg, **kw)
    geom = FiniteVolume(material=material, mesh_file=path, contacts=contacts,
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


def _obs_weights(material, t=0.0):
    """(3, Nk) weights [n, jx, jy]: the staggered-output refactor reduced
    get_observables to the density row; rebuild the current weights from it and
    the transport velocity (j = int f v)."""
    nw = material.get_observables(t)[0]                       # (Nk,)
    v = material.transport_velocity                           # (Nk, 2)
    return torch.stack([nw, nw * v[:, 0], nw * v[:, 1]])


def _integral(geom, material, obs_idx, t=0.0):
    """Domain integral of observable `obs_idx` (0=n, 1=jx, 2=jy): sum_k area_k o_k."""
    obs = torch.einsum("oc,kc->ko", _obs_weights(material, t), geom._u)  # (K,3)
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


def test_cartesian_reflective_walls_conserve_mass() -> None:
    """The same closed cavity on the CARTESIAN k-representation.

    ⛔ THIS IS NOT REDUNDANT WITH THE TEST ABOVE.  Every conservation test in
    this file uses the MODAL FermiSurface, where the mirror image of a
    quadrature node IS a node, so specular reflection is an exact permutation
    and the wall conserves flux for free.  On the Cartesian grid the mirrored
    point generally lands between nodes and is bilinearly interpolated: the
    weights sum to 1, so OCCUPANCY is conserved, but the four corners carry
    different |v.n| so the particle FLUX moment is not -- and corners outside
    the active set are dropped outright.  Measured before the rank-1 closure was
    added to _CartesianReflector: the wall was ~0.4% absorbing per bounce and a
    closed cavity lost 5.9e-3 of its deviation mass over 3000 steps, LINEARLY in
    step count, while all four modal conservation tests passed.

    The bound below is 1e-12 over 400 steps; the uncorrected code gives ~8e-4.
    """
    torch.set_default_dtype(torch.float64)
    tmp = tempfile.mkdtemp()
    path = _make_rect_mesh(10.0, os.path.join(tmp, "rect.npz"), all_walls=True)
    pg = ProcessGrid(rc.comm, "rk", (1, 1))
    material = FermiSurface(
        process_grid=pg, kF=1.0, vF=1.5, M_theta=8, Nr=1, T=1.0,
        tau_p=np.inf, tau_ee=np.inf, specularity=1.0,
        cartesian=dict(annulus_xi=0.0, te_fac_max=2.0, n_k=32))
    geom = FiniteVolume(material=material, mesh_file=path, contacts={},
                        process_grid=pg)
    cen = torch.from_numpy(geom.geom.centroid_np).to(rc.device)
    q0 = torch.tensor([55.0, 30.0], dtype=torch.float64, device=rc.device)
    blob = torch.exp(-((cen - q0) ** 2).sum(-1) / (2 * 6.0 ** 2))
    geom._u = blob[:, None].repeat(1, geom.Nk)
    w = material.representation.get_density_weight()
    mass = lambda: float((geom.geom.area[:, None] * geom._u * w[None]).sum())
    m0 = mass()
    _step(geom, 400)
    assert abs(mass() - m0) / abs(m0) < 1e-12, abs(mass() - m0) / abs(m0)


def test_cartesian_wall_energy_and_pressure() -> None:
    """A specular wall is elastic and reverses only the normal velocity, so it
    conserves FOUR flux moments, not two:

        particle   sum_I |v.n| g        ==  sum_O |v.n| u
        tangential sum_I |v.n|(v.t) g   ==  sum_O |v.n|(v.t) u
        energy     sum_I |v.n| eps g    ==  sum_O |v.n| eps u
        pressure   sum_I |v.n||v.n| g   ==  sum_O |v.n||v.n| u

    The first two are pinned by the diffuse-refill closure; the last two were
    left free and carried the whole bilinear-interpolation error -- 1.76e-4 and
    9.19e-5 at production dk, converging only at order 2. The stage-1 rank-4
    correction removes them.

    ⛔ The energy moment must be CENTRED on mu before it is used as either a
    constraint row or a test weight: eps ~ mu across the active shell, so the
    raw moment is dominated by a constant and both the Gram and this assertion
    lose their resolving power."""
    torch.set_default_dtype(torch.float64)
    pg = ProcessGrid(rc.comm, "rk", (1, 1))
    material = FermiSurface(
        process_grid=pg, kF=7.5e-3, vF=0.11194, M_theta=32, Nr=6, T=1.3301e-5,
        xi_max=6.0, tau_p=np.inf, tau_ee=np.inf, specularity=1.0,
        cartesian=dict(annulus_xi=0.0, te_fac_max=2.0))
    rep = material.representation
    th = torch.linspace(0.0, 2 * np.pi, 17, device=rc.device)[:-1]
    n = torch.stack([th.cos(), th.sin()], -1)
    refl = material.get_reflector(n)
    v = rep.k / rep.m_star
    vmax = float(v.norm(dim=1).max())
    vn = (v[None] * n[:, None]).sum(-1)
    t_hat = torch.stack([-n[:, 1], n[:, 0]], -1)
    vt = (v[None] * t_hat[:, None]).sum(-1) / vmax
    e_c = (((rep.k ** 2).sum(-1) / (2 * rep.m_star) - rep.mu)
           / (rep.xi_max * rep.T_temp))[None].expand_as(vt)
    w_in, w_out = vn.abs() * (vn < 0), vn.abs() * (vn > 0)
    kD = 0.03 * torch.tensor([1.0, 0.3], device=rc.device)
    eps = ((rep.k - kD) ** 2).sum(-1) / (2 * rep.m_star)
    u = (torch.special.expit(-(eps - rep.mu) / rep.T_temp)
         - rep._f0_lab)[None].repeat(n.shape[0], 1)
    out = refl(u[None])[0]
    for name, wgt in (("particle", torch.ones_like(vt)), ("tangential", vt),
                      ("energy", e_c), ("pressure", vn.abs() / vmax)):
        lhs = (w_in * wgt * out).sum(-1)
        rhs = (w_out * wgt * u).sum(-1)
        scale = (w_out * wgt.abs() * u.abs()).sum(-1).clamp(min=1e-300)
        err = float(((lhs - rhs).abs() / scale).max())
        assert err < 1e-12, (name, err)


def test_cartesian_wall_specularity() -> None:
    """The Cartesian wall at arbitrary specularity s.

    Two invariants, flux-weighted over the respective half-spaces:
        particle:  sum_{v.n<0} |v.n| ghost       ==     sum_{v.n>0} |v.n| u
        shear:     sum_{v.n<0} |v.n|(v.t) ghost  ==  s* sum_{v.n>0} |v.n|(v.t) u
    The first is s-independent (a wall passes zero net current whatever it does
    to momentum); the second carries the whole s dependence, so the DRAG ratio
    (returned tangential flux)/(incident) must come out exactly s.

    ⛔ Both must be checked PER FACE. Summing across faces mixes incompatible
    tangent directions and lets the denominator cross zero -- the same trap that
    made the old antisymmetry gate unusable.

    At s = 0 the refill is the whole ghost and must be the Maxwell law, i.e.
    CONSTANT over each face's inflow set. That is what forces the correction
    basis to be the inflow indicator rather than the flux weight |v.n|."""
    torch.set_default_dtype(torch.float64)
    pg = ProcessGrid(rc.comm, "rk", (1, 1))
    th = torch.linspace(0.0, 2 * np.pi, 17, device=rc.device)[:-1]
    n = torch.stack([th.cos(), th.sin()], -1)
    for s in (0.0, 0.5, 1.0):
        material = FermiSurface(
            process_grid=pg, kF=1.0, vF=1.5, M_theta=8, Nr=1, T=1.0,
            tau_p=np.inf, tau_ee=np.inf, specularity=s,
            cartesian=dict(annulus_xi=0.0, te_fac_max=2.0, n_k=32))
        rep = material.representation
        refl = material.get_reflector(n)
        v = material.transport_velocity
        vn = (v[None] * n[:, None]).sum(-1)
        t_hat = torch.stack([-n[:, 1], n[:, 0]], -1)
        vt = (v[None] * t_hat[:, None]).sum(-1)
        w_in, w_out = vn.abs() * (vn < 0), vn.abs() * (vn > 0)
        kD = 0.03 * torch.tensor([1.0, 0.3], device=rc.device)
        eps = ((rep.k - kD) ** 2).sum(-1) / (2 * rep.m_star)
        u = (torch.special.expit(-(eps - rep.mu) / rep.T_temp)
             - rep._f0_lab)[None].repeat(n.shape[0], 1)
        out = refl(u[None])[0]
        for wgt, tgt in ((torch.ones_like(vt), 1.0), (vt, s)):
            lhs = (w_in * wgt * out).sum(-1)
            rhs = tgt * (w_out * wgt * u).sum(-1)
            scale = (w_out * wgt.abs() * u.abs()).sum(-1).clamp(min=1e-300)
            err = float(((lhs - rhs).abs() / scale).max())
            assert err < 1e-12, (s, tgt, err)
        # drag ratio, per face, on the faces that carry real tangential momentum
        s2 = (w_out * vt * u).sum(-1)
        r2 = (w_in * vt * out).sum(-1)
        sc = (w_out * vt.abs() * u.abs()).sum(-1)
        good = s2.abs() > 1e-6 * sc
        if bool(good.any()):
            ratio = r2[good] / s2[good]
            assert float((ratio - s).abs().max()) < 1e-10, (s, float(ratio.max()))
        if s == 0.0:                       # the refill must be the Maxwell law
            # ⛔ The Maxwell law for a DEGENERATE gas is not "flat in k".
            # A diffuse wall re-emits electrons thermalised to the wall, at a
            # mu_w fixed by particle-flux balance, so what it returns is
            #     delta-f = FD(eps; mu_w, T) - FD(eps; mu, T)
            #             = (mu_w - mu) f0 (1 - f0) / T + O(dmu^2),
            # proportional to the Fermi shell envelope -- the same weight
            # qimpy's own linear contactor already uses
            # (df_contact = (dmu - vD k.n) f0 (1 - f0) / T).
            #
            # This assertion used to demand that `out` ITSELF be constant over
            # the inflow set, which is the classical non-degenerate law and
            # implies the wall re-emits at |k| >> kF with the same weight as at
            # the Fermi surface.  That is unphysical, and it was also the
            # mechanism driving the ballistic run's occupancies negative: the
            # closure deposited a Fermi-scale correction on an empty tail,
            # reaching f = -2.04e-6 where the physical occupancy is 1.5e-17.
            # What is constant is the RATIO to the envelope.
            fw = torch.special.expit(
                -((rep.k ** 2).sum(-1) / (2 * rep.m_star) - rep.mu)
                / rep.T_temp)
            envelope = fw * (1.0 - fw)
            envelope = envelope / envelope.max()
            for e in range(n.shape[0]):
                live = (w_in[e] > 0) & (envelope > 1e-8)
                sel = out[e][live] / envelope[live]
                m = sel.mean()
                dev = float((sel - m).abs().max() / m.abs().clamp(min=1e-300))
                assert dev < 1e-10, dev
            # and nothing may survive outside the shell at all
            for e in range(n.shape[0]):
                far = (w_in[e] > 0) & (envelope < 1e-20)
                if bool(far.any()):
                    assert float(out[e][far].abs().max()) < 1e-20, \
                        float(out[e][far].abs().max())


def test_cartesian_wall_conserves_flux_and_shear() -> None:
    """The instantaneous form: a specular wall returns exactly the flux it
    receives, and exerts no tangential force, for ANY trace:

        particle:  sum_{v.n<0} |v.n| ghost       == sum_{v.n>0} |v.n| u
        shear:     sum_{v.n<0} |v.n| (v.t) ghost == sum_{v.n>0} |v.n| (v.t) u

    Both are enforced by the rank-2 closure in _CartesianReflector.  Without it
    the residuals are 4.5e-3 and 4.0e-3 on a drifted Fermi-Dirac trace, and up
    to 7.8e-2 on the worst face of an all-angles sweep."""
    torch.set_default_dtype(torch.float64)
    pg = ProcessGrid(rc.comm, "rk", (1, 1))
    material = FermiSurface(
        process_grid=pg, kF=1.0, vF=1.5, M_theta=8, Nr=1, T=1.0,
        tau_p=np.inf, tau_ee=np.inf, specularity=1.0,
        cartesian=dict(annulus_xi=0.0, te_fac_max=2.0, n_k=32))
    rep = material.representation
    th = torch.linspace(0.0, 2 * np.pi, 17, device=rc.device)[:-1]  # all angles
    n = torch.stack([th.cos(), th.sin()], -1)
    refl = material.get_reflector(n)
    v = material.transport_velocity
    vn = (v[None] * n[:, None]).sum(-1)
    t_hat = torch.stack([-n[:, 1], n[:, 0]], -1)
    vt = (v[None] * t_hat[:, None]).sum(-1)
    w_in, w_out = vn.abs() * (vn < 0), vn.abs() * (vn > 0)
    torch.manual_seed(0)
    # a random trace and a physically shaped one (a drifted Fermi-Dirac, which
    # is the only one with a non-degenerate tangential moment)
    kD = 0.03 * torch.tensor([1.0, 0.3], device=rc.device)
    eps = ((rep.k - kD) ** 2).sum(-1) / (2 * rep.m_star)
    drift = (torch.special.expit(-(eps - rep.mu) / rep.T_temp) - rep._f0_lab)
    for u in (torch.rand(n.shape[0], rep.k.shape[0], device=rc.device),
              torch.ones(n.shape[0], rep.k.shape[0], device=rc.device),
              drift[None].repeat(n.shape[0], 1)):
        out = refl(u[None])[0]
        for wgt in (torch.ones_like(vt), vt):           # particle, then shear
            lhs = (w_in * wgt * out).sum(-1)
            rhs = (w_out * wgt * u).sum(-1)
            # Normalise by the ABSOLUTE-value integral, not by |rhs|: for an
            # isotropic trace the signed tangential moment cancels to ~1e-13, so
            # dividing by it compares roundoff with roundoff (this test read 1.5
            # for a residual of 1.7e-13 before the scale was fixed).
            scale = (w_out * wgt.abs() * u.abs()).sum(-1).max().clamp(min=1e-300)
            assert float((lhs - rhs).abs().max() / scale) < 1e-12, \
                float((lhs - rhs).abs().max() / scale)
    # and it must stay LINEAR: _setup_boundary caches it as a dense matrix by
    # pushing the Nk basis vectors through, which assumes additivity.
    a = torch.rand(n.shape[0], rep.k.shape[0], device=rc.device)
    b = torch.rand(n.shape[0], rep.k.shape[0], device=rc.device)
    d = (refl((a + b)[None]) - refl(a[None]) - refl(b[None])).abs().max()
    assert float(d) < 1e-12 * float(refl((a + b)[None]).abs().max()), float(d)


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
    geom = FiniteVolume(material=mat, mesh_file=os.environ["FV_MPI_MESH"],
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
    stably through the FiniteVolume 1D geometry path.  Its ballistic steady state is
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
    obs = torch.einsum("oc,kc->ko", _obs_weights(mat), geom._u)
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
    mod = "qimpy.transport.geometry.test_finite_volume"
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
    test_cartesian_reflective_walls_conserve_mass(); print("cartesian_wall_mass: PASS")
    test_cartesian_wall_conserves_flux_and_shear(); print("cartesian_wall_flux_shear: PASS")
    test_cartesian_wall_specularity(); print("cartesian_wall_specularity: PASS")
    test_cartesian_wall_energy_and_pressure(); print("cartesian_wall_energy_pressure: PASS")
    test_oblique_wall_conserves_tangential_momentum(); print("oblique_wall_tang_momentum: PASS")
    test_contact_driven_state_is_bounded(); print("contact_driven_bounded: PASS")
    test_biased_contacts_balance_at_steady_state(); print("biased_balance: PASS")
    test_curved_mass_conservation(); print("curved_mass_conservation: PASS")
    test_decomp_matches_serial(); print("decomp_matches_serial: PASS")
    print("ALL PASS")


def test_cartesian_wall_preserves_occupancy_bounds() -> None:
    """The wall may not put electrons where there are none.

    The four moment tests above pin what the wall CONSERVES; none of them
    constrains WHERE in k the closure deposits its correction.  Built on the
    bare inflow indicator, the biorthogonal vectors c_a are O(1) across the
    whole inflow set, so the correction c_a * (T_a - got) -- whose size is set
    by the O(1) trace at the Fermi surface -- lands with equal weight on the
    tail at |k| >> kF, where the occupancy is e^-69.  Measured before the fix:
    the reflected f reached -2.04e-6 in a region physically holding 1.5e-17,
    and that is where the ballistic run's negative occupancies came from.

    Weighting the closure basis with the Fermi shell envelope f0(1-f0) confines
    the correction to the shell.  This test is the one that fails without it;
    every moment test above passes either way, which is exactly why this was
    missed.

    ⛔ Checks f = f0 + delta-f, not delta-f: the bound being asserted is Pauli
    occupancy in [0, 1], and delta-f is legitimately negative on its own.
    ⛔ Includes a HOT trace.  The envelope has width T while te_fac_max lets the
    physical distribution be several T wide, so the correction is concentrated
    on a shell thinner than the data -- safe in principle (it decays faster
    than f) but it has to be measured, not argued.
    """
    torch.set_default_dtype(torch.float64)
    pg = ProcessGrid(rc.comm, "rk", (1, 1))
    material = FermiSurface(
        process_grid=pg, kF=7.5e-3, vF=0.11194, M_theta=32, Nr=6, T=1.3301e-5,
        xi_max=6.0, tau_p=np.inf, tau_ee=np.inf, specularity=1.0,
        cartesian=dict(annulus_xi=0.0, te_fac_max=6.0))
    rep = material.representation
    f0 = rep._f0_lab
    th = torch.linspace(0.0, 2 * np.pi, 17, device=rc.device)[:-1]
    n = torch.stack([th.cos(), th.sin()], -1)
    kD = torch.tensor([6.0e-5, 0.0], device=rc.device)
    eps = ((rep.k - kD) ** 2).sum(-1) / (2 * rep.m_star)
    # ⛔ specularity must be set on EVERY iteration.  Setting it only in the
    # s != 1 branch left it at 0.0 from the previous temperature, so the "s = 1"
    # arms after the first silently ran fully diffuse and reported a specular
    # failure (max f = 1.0095) that did not exist.
    for te_fac in (1.0, 2.0, 5.6):
        f_tr = torch.special.expit(-(eps - rep.mu) / (te_fac * rep.T_temp))
        u = (f_tr - f0)[None].repeat(n.shape[0], 1)
        for s in (1.0, 0.5, 0.0):
            rep.fs.specularity = s
            refl = rep.get_reflector(n)
            f = f0[None] + refl(u[None])[0]
            assert float(f.min()) > -1e-14, (te_fac, s, float(f.min()))
            if s == 1.0:
                assert float(f.max()) < 1.0 + 1e-14, (te_fac, s, float(f.max()))
    rep.fs.specularity = 1.0
