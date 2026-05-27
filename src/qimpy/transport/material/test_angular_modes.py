"""Tests for the angular-harmonic (modal) k-space representation.

Run directly (not via pytest, which needs --with-mpi here)."""
from __future__ import annotations

import numpy as np
import torch

from qimpy import rc
from ._angular_modes import AngularModes


def test_transforms_and_ladder() -> None:
    """mode<->node round-trip is exact, and the ladder operator equals nodal
    multiplication by vF cos(theta) (with correct harmonic closure)."""
    torch.set_default_dtype(torch.float64)
    am = AngularModes(M=8, vF=1.0)
    T, Tinv, Ax = am.T.numpy(), am.Tinv.numpy(), am.Ax.numpy()
    assert np.abs(Tinv @ T - np.eye(am.dim)).max() < 1e-12
    rng = np.random.default_rng(0)
    f = rng.standard_normal(am.dim)
    f[2 * am.M - 1] = f[2 * am.M] = 0.0  # no content at M (no closure leakage)
    nodal_prod = am.vF * np.cos(am.theta) * (T @ f)
    assert np.abs(Ax @ f - Tinv @ nodal_prod).max() < 1e-12


def test_cyclotron_exact() -> None:
    """Momentum-space (cyclotron) advection is an exact rotation in modes: one
    full cyclotron period is the identity, with no CFL or truncation error."""
    from scipy.linalg import expm
    torch.set_default_dtype(torch.float64)
    am = AngularModes(M=10, vF=1.0)
    G = am.G.numpy()
    assert np.abs(expm(2 * np.pi * G) - np.eye(am.dim)).max() < 1e-10
    # harmonic m rotates at frequency m: check m=2 after some time
    wc, t = 0.37, 1.3
    f0 = np.zeros(am.dim); f0[3] = 1.0  # a_2
    ang = 2 * wc * t
    expect = np.zeros(am.dim); expect[3] = np.cos(ang); expect[4] = np.sin(ang)
    assert np.abs(expm(wc * G * t) @ f0 - expect).max() < 1e-12


def test_collision_rates() -> None:
    """Collisions are diagonal per harmonic: m=0 conserved, m=1 relaxes via
    tau_p only, m>=2 via tau_p^-1 + tau_ee^-1."""
    am = AngularModes(M=5, vF=1.0)
    r = am.collision_rates(tau_p=2.0, tau_ee=0.5).numpy()
    assert r[0] == 0.0
    assert abs(r[1] - 0.5) < 1e-12 and abs(r[2] - 0.5) < 1e-12   # 1/tau_p
    assert abs(r[3] - 2.5) < 1e-12                               # 1/tau_p + 1/tau_ee


def test_dispersion_bounded_and_stable() -> None:
    """Collisionless transport is purely propagating with speeds bounded by vF;
    collisions make every mode decay (stable)."""
    from scipy.linalg import eigvals
    am = AngularModes(M=12, vF=1.0)
    Ax, Ay = am.Ax.numpy(), am.Ay.numpy()
    q = (0.8, 0.6); qmag = np.hypot(*q)
    w = eigvals(-1j * (q[0] * Ax + q[1] * Ay))            # collisionless
    assert np.abs(w.real).max() < 1e-9                    # no damping
    assert np.abs(w.imag).max() / qmag <= am.vF + 1e-9    # bounded by vF
    wc = eigvals(-1j * (q[0] * Ax + q[1] * Ay)
                 - np.diag(am.collision_rates(2.0, 0.5).numpy()))
    assert wc.real.max() <= 1e-9                          # stable


def test_equivalence_to_discrete_ordinates() -> None:
    """Modal and discrete-ordinates are the same operator in two bases: a full
    single-spatial-mode evolution agrees to round-off once M resolves the
    (smooth) angular structure."""
    from scipy.linalg import expm
    torch.set_default_dtype(torch.float64)
    vF, q, wc, tau_p, tau_ee, t = 1.0, (0.8, 0.6), 0.3, 3.0, 0.8, 2.0

    def modal(M):
        am = AngularModes(M, vF)
        L = (-1j * (q[0] * am.Ax.numpy() + q[1] * am.Ay.numpy())
             - np.diag(am.collision_rates(tau_p, tau_ee).numpy())
             + wc * am.G.numpy())
        f0 = np.zeros(am.dim); f0[0] = 1.0; f0[1] = 0.5; f0[3] = 0.3
        return (expm(L * t) @ f0)[:4]

    Nth = 384
    th = 2 * np.pi * np.arange(Nth) / Nth
    vx, vy = vF * np.cos(th), vF * np.sin(th)
    Dth = np.zeros((Nth, Nth))
    for a in range(Nth):
        d = np.arange(Nth) - a; nz = d != 0
        Dth[a, nz] = 0.5 * ((-1.0) ** d[nz]) / np.tan(np.pi * d[nz] / Nth)
    rp, ree = 1 / tau_p, 1 / tau_ee
    vv = np.stack([vx, vy]); P1 = vv.T @ np.linalg.inv(vv @ vv.T) @ vv
    C = (rp + ree) * (np.eye(Nth) - np.ones((Nth, Nth)) / Nth) - ree * P1
    Ldo = -1j * (q[0] * np.diag(vx) + q[1] * np.diag(vy)) - C + wc * Dth
    ut = expm(Ldo * t) @ (1.0 + 0.5 * np.cos(th) + 0.3 * np.cos(2 * th))
    ref = np.array([ut.mean(), 2 * np.mean(np.cos(th) * ut),
                    2 * np.mean(np.sin(th) * ut), 2 * np.mean(np.cos(2 * th) * ut)])
    assert np.abs(modal(10) - ref).max() < 1e-10


if __name__ == "__main__":
    rc.init()
    test_transforms_and_ladder(); print("transforms_and_ladder: PASS")
    test_cyclotron_exact(); print("cyclotron_exact: PASS")
    test_collision_rates(); print("collision_rates: PASS")
    test_dispersion_bounded_and_stable(); print("dispersion_bounded_and_stable: PASS")
    test_equivalence_to_discrete_ordinates(); print("equivalence_to_discrete_ordinates: PASS")
