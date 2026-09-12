"""Tests for the exact ballistic solver, and FV measured against it.

⛔ WHY THIS FILE EXISTS.  Every other transport test asks whether the code is
SELF-CONSISTENT: mass conserved, currents balanced, walls conserving their
moments, states bounded.  Not one of them asks whether the answer is RIGHT.
That is how a 20% error sat undetected in the C[f] reference for months, and
how the finite-volume solver ran for a year with a wall that absorbed 0.4% per
bounce while all four shipped conservation tests passed.

The method of characteristics shares no machinery with the FV solver -- no
cells, no k-grid, no time step, no reconstruction, no limiter, no wall closure
-- so agreement between them is real evidence, not a tautology.
"""
from __future__ import annotations

import os
import tempfile
from collections import Counter

import numpy as np
import pytest
import torch

from qimpy import rc
from ..geometry._mesh import save_mesh
from . import Ballistic, Polygon
from ._polygon import AU

# ⛔ Keep the fixture SMALL.  The production mixer takes 2238 s per bias point
# at n_ang=4096 / max_bounce=12800; a channel this size runs in seconds, and a
# test nobody can afford to run is a test that does not run.
CHANNEL = dict(kF=7.5e-3, vF=0.11194, T=1.3301e-5)
DMU = 5.37e-5


def _channel_mesh(path: str, length: float = 40.0, width: float = 10.0,
                  h: float = 3.0, all_walls: bool = False) -> str:
    """Straight rectangular channel: source at x=0, drain at x=length.

    A channel is the one geometry whose ballistic conductance is analytic, so
    it can be checked against a formula and not only against another code.
    """
    tr = pytest.importorskip("triangle")
    pts = np.array([[0, 0], [length, 0], [length, width], [0, width]], float)
    seg = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    m = tr.triangulate({"vertices": pts, "segments": seg}, f"pq30a{h * h:g}")
    V, T = m["vertices"], m["triangles"]
    ec: Counter = Counter()
    for t in T:
        for x, y in ((0, 1), (1, 2), (2, 0)):
            ec[tuple(sorted((int(t[x]), int(t[y]))))] += 1
    be = [e for e, c in ec.items() if c == 1]
    bm = []
    for a, b in be:
        mx = 0.5 * (V[a][0] + V[b][0])
        if all_walls:
            bm.append("wall")
        elif mx < 1e-9:
            bm.append("source")
        elif mx > length - 1e-9:
            bm.append("drain")
        else:
            bm.append("wall")
    save_mesh(path, V * AU, T, np.array(be), bm)
    return path


def _multi_contact_mesh(path: str, size: float = 30.0, h: float = 4.0) -> str:
    """Square with FOUR distinct contact names, one per side."""
    tr = pytest.importorskip("triangle")
    pts = np.array([[0, 0], [size, 0], [size, size], [0, size]], float)
    seg = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    m = tr.triangulate({"vertices": pts, "segments": seg}, f"pq30a{h * h:g}")
    V, T = m["vertices"], m["triangles"]
    ec: Counter = Counter()
    for t in T:
        for x, y in ((0, 1), (1, 2), (2, 0)):
            ec[tuple(sorted((int(t[x]), int(t[y]))))] += 1
    be = [e for e, c in ec.items() if c == 1]
    bm = []
    for a, b in be:
        mx, my = 0.5 * (V[a] + V[b])
        if my < 1e-9:
            bm.append("south")
        elif my > size - 1e-9:
            bm.append("north")
        elif mx < 1e-9:
            bm.append("west")
        else:
            bm.append("east")
    save_mesh(path, V * AU, T, np.array(be), bm)
    return path


def test_polygon_normals_point_outward() -> None:
    """Every boundary normal must leave the domain.

    ⛔ An inverted normal silently inverts specular reflection, which turns a
    wall into a lens and is invisible in any conservation check -- reflection
    conserves flux either way.
    """
    torch.set_default_dtype(torch.float64)
    with tempfile.TemporaryDirectory() as td:
        poly = Polygon(_channel_mesh(os.path.join(td, "c.npz")), rc.device)
        cen = 0.5 * (poly.a + poly.b)
        eps = 1e-3 * float((poly.b - poly.a).norm(dim=1).min())
        out = (cen + eps * poly.n_hat).cpu().numpy()
        ins = (cen - eps * poly.n_hat).cpu().numpy()
        d = np.load(_channel_mesh(os.path.join(td, "c2.npz")), allow_pickle=True)
        V, E = d["vertices"] / AU, np.asarray(d["boundary_edges"])
        assert not Polygon._inside(V, E, out).any(), "a normal points INWARD"
        assert Polygon._inside(V, E, ins).all(), "the inward probe fell outside"


def test_marker_names_are_not_hardcoded() -> None:
    """Arbitrary contact names, not just source/drain.

    ⛔ The scratch version hardcoded {"wall":0,"source":1,"drain":2} and raised
    KeyError otherwise, so it could not run the seven- or nine-terminal devices
    at all -- the geometries where an independent check matters most.
    """
    torch.set_default_dtype(torch.float64)
    with tempfile.TemporaryDirectory() as td:
        mesh = _multi_contact_mesh(os.path.join(td, "quad.npz"))
        b = Ballistic(mesh, contacts={"north": DMU, "south": -DMU}, **CHANNEL)
        assert b.poly.contact_names == ["east", "north", "south", "west"]
        assert b.poly.code["wall"] == 0
        assert set(b.dmu) == set(b.poly.contact_names)
        assert b.dmu["east"] == 0.0 and b.dmu["north"] == DMU


def test_unknown_contact_name_is_rejected() -> None:
    """A typo'd contact must fail loudly, not drive nothing in silence."""
    torch.set_default_dtype(torch.float64)
    with tempfile.TemporaryDirectory() as td:
        mesh = _channel_mesh(os.path.join(td, "c.npz"))
        with pytest.raises(KeyError):
            Ballistic(mesh, contacts={"sauce": DMU}, **CHANNEL)


def test_equilibrium_carries_no_current() -> None:
    """Zero bias everywhere => zero current, to roundoff."""
    torch.set_default_dtype(torch.float64)
    with tempfile.TemporaryDirectory() as td:
        mesh = _channel_mesh(os.path.join(td, "c.npz"))
        b = Ballistic(mesh, contacts={"source": 0.0, "drain": 0.0}, **CHANNEL)
        I, _ = b.contact_current("source", n_ang=256, n_edge=8, max_bounce=200)
        assert abs(I) < 1e-20, I


def test_current_is_antisymmetric_and_conserved() -> None:
    """I(-dmu) = -I(dmu), and the contacts balance.

    Charge conservation across contacts is a property of the exact solution,
    so it holds here without any discrete flux bookkeeping -- unlike the FV
    solver, where it is a consequence of the scheme.
    """
    torch.set_default_dtype(torch.float64)
    with tempfile.TemporaryDirectory() as td:
        mesh = _channel_mesh(os.path.join(td, "c.npz"))
        kw = dict(n_ang=512, n_edge=12, max_bounce=400)
        fwd = Ballistic(mesh, contacts={"source": DMU, "drain": -DMU}, **CHANNEL)
        rev = Ballistic(mesh, contacts={"source": -DMU, "drain": DMU}, **CHANNEL)
        Is, ts = fwd.contact_current("source", **kw)
        Id, _ = fwd.contact_current("drain", **kw)
        Ir, _ = rev.contact_current("source", **kw)
        assert abs(Is + Id) < 2e-2 * abs(Is), (Is, Id)      # balance
        assert abs(Is + Ir) < 2e-2 * abs(Is), (Is, Ir)      # antisymmetry
        assert ts < 0.5, f"unresolved fraction {ts} too high to conclude"


def test_current_grows_with_channel_width() -> None:
    """Sharvin scaling: a ballistic channel's conductance is set by its width.

    ⛔ This is deliberately a MONOTONICITY test, not a fit to
    G = (2e^2/h) k_F W / pi.  The prefactor depends on the contact model and on
    how the reservoir is projected onto the injected half-space, so pinning an
    absolute number here would be encoding this implementation's conventions
    and calling it physics.  What must hold regardless is that twice the width
    carries appreciably more current, and roughly in proportion.
    """
    torch.set_default_dtype(torch.float64)
    with tempfile.TemporaryDirectory() as td:
        kw = dict(n_ang=512, n_edge=12, max_bounce=400)
        Is = []
        for w in (8.0, 16.0):
            mesh = _channel_mesh(os.path.join(td, f"w{w}.npz"), width=w, h=2.5)
            b = Ballistic(mesh, contacts={"source": DMU, "drain": -DMU},
                          **CHANNEL)
            Is.append(abs(b.contact_current("source", **kw)[0]))
        ratio = Is[1] / Is[0]
        assert 1.5 < ratio < 2.6, (Is, ratio)


def test_closed_cavity_is_entirely_unresolved() -> None:
    """With no contact, every ray is trapped -- and it is REPORTED as such.

    The value of this solver is that it declares what it cannot determine
    instead of letting numerical diffusion invent it.
    """
    torch.set_default_dtype(torch.float64)
    with tempfile.TemporaryDirectory() as td:
        mesh = _channel_mesh(os.path.join(td, "cav.npz"), all_walls=True)
        b = Ballistic(mesh, contacts={}, **CHANNEL)
        assert b.poly.contact_names == []
        p = 0.5 * (b.poly.a + b.poly.b) - 1e-6 * b.poly.n_hat
        th = torch.linspace(0.0, 2 * np.pi, 33, device=rc.device)[:-1]
        v = torch.stack([th.cos(), th.sin()], -1)
        n = p.shape[0]
        cid, _ = b.trace_back(
            p[:, None, :].expand(n, len(th), 2).reshape(-1, 2).contiguous(),
            v[None].expand(n, len(th), 2).reshape(-1, 2).contiguous(),
            max_bounce=64)
        assert int((cid != 0).sum()) == 0, "a ray escaped a closed cavity"


@pytest.mark.timeout(1800)
def test_finite_volume_matches_exact_ballistic() -> None:
    """★ THE CROSS-CHECK: FV must reproduce the exact ballistic current.

    This is the only test in the suite that compares qimpy against an
    independently derived answer rather than against itself.

    ⛔ TOLERANCE.  The exact solver's own uncertainty is the unresolved-orbit
    fraction, which is reported and asserted small; the FV solver carries
    discretisation error at this coarse test resolution.  On the production
    mixer the two agree to 2.1% (9.9780 vs 9.77499 uA) at 5.0% unresolved, so
    10% here is a real constraint that a broken wall or flux assembly cannot
    slip through -- the wall bug this suite previously missed was worth 4x.
    """
    torch.set_default_dtype(torch.float64)
    from .. import Transport
    from ..geometry import TensorList

    with tempfile.TemporaryDirectory() as td:
        mesh = _channel_mesh(os.path.join(td, "c.npz"), length=30.0,
                             width=12.0, h=3.0)
        exact = Ballistic(mesh, contacts={"source": DMU, "drain": -DMU},
                          **CHANNEL)
        # ⛔ n_ang must be high enough to be converged: on the production
        # mixer 1024 reads 2.8% low and is not even monotone (9.6908 / 9.5130 /
        # 9.7649 / 9.7863 at 512/1024/2048/4096).  A cross-check against an
        # unconverged reference is worse than no cross-check.
        I_exact, trap = exact.contact_current("source", n_ang=2048, n_edge=24,
                                              max_bounce=3200)
        assert trap < 0.25, f"unresolved {trap}: reference too weak to test FV"

        t = Transport(
            fermi_surface=dict(
                kF=CHANNEL["kF"], vF=CHANNEL["vF"], M_theta=32, Nr=4,
                T=CHANNEL["T"], xi_max=6.0, tau_p=np.inf, specularity=1.0,
                residual_damping=False,
                cartesian=dict(annulus_xi=0.0, te_fac_max=2.0, n_k=48)),
            spatial_transport=dict(
                mesh_file=mesh, compile=False, save_rho=True,
                contacts={"source": {"dmu": DMU, "nonlinear": True},
                          "drain": {"dmu": -DMU, "nonlinear": True}}),
            time_evolution=dict(t_max=1e30, dt_save=1e30, n_collate=1))
        g = t.geometry
        u = g.rho[0].clone()
        dt = float(g.dt_max)
        # ⛔ Run to STEADY STATE, not a fixed step count: the comparison is
        # against a steady solution, and a transient would disagree for
        # reasons that have nothing to do with correctness.
        prev = None
        for block in range(60):
            for _ in range(50):
                uh = u + (0.5 * dt) * g.rho_dot(TensorList([u]), 0.0)[0]
                u = u + dt * g.rho_dot(TensorList([uh]), 0.5 * dt)[0]
            g._u.copy_(u)
            I_fv = g.contact_currents(0.0)["source"]
            if prev is not None and abs(I_fv - prev) < 1e-4 * abs(I_fv):
                break
            prev = I_fv
        rel = abs(I_fv - I_exact) / abs(I_exact)
        assert rel < 0.10, (
            f"FV {I_fv:.6e} vs exact {I_exact:.6e} = {100 * rel:.2f}% apart "
            f"(unresolved {trap:.3f}, {block + 1} blocks)")
