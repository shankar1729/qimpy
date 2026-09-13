"""Seconds-scale e-e checks that run in the DEFAULT suite, on a laptop.

⛔ WHY THIS FILE EXISTS.  `test_ee.py` is the real e-e validation battery and it
is marked `validate`, so `make test` does not run it: every test in there builds
and applies the collision operator, which is a GPU-scale object (measured 30-60x
slower on CPU -- 17 s -> >17 min for one of them).  Deselecting it wholesale
would mean a commit could break the operator outright and still pass
`make test`, which is not an acceptable trade for speed.

So this file keeps the cheapest possible version of each guarantee -- the
operator CONSTRUCTS, it APPLIES, and it CONSERVES what it must -- at a size
where all of it is seconds on a CPU.  It is deliberately NOT a numerical
accuracy check: nothing here would catch a wrong prefactor or a bad vertex, and
it must not be mistaken for the thing that does.  That is `make test-validate`.
"""
from __future__ import annotations

import numpy as np
import torch

from qimpy import rc
from qimpy.mpi import ProcessGrid
from .. import FermiSurface

KF = 7.5e-3
M_STAR = 0.067
T0 = 1.3301e-5
EPS_B = 12.9
KAPPA = 0.0

# The smallest size that still exercises every code path: M_theta = 3 gives a
# non-trivial angular structure (m = 0, 1, 2, 3), Nr = 1 the surface mode, and
# n_phi = 64 is the coarsest angular quadrature the kinematics accepts.
SMALL = dict(epsilon_bg=EPS_B, nonlinear=True, n_xi=8, n_phi=64, n_xi_proj=6)


def _fs(**kw):
    return FermiSurface(
        kF=KF, vF=KF / M_STAR, M_theta=3, Nr=1, T=T0,
        tau_p=float("inf"), process_grid=ProcessGrid("rk", (1, 1)), **kw)


def test_ee_operator_constructs_and_applies() -> None:
    """Construction + a_dot run, and produce finite numbers of the right shape.

    A NaN or a shape error here is the failure mode a wholesale deselect of
    test_ee.py would otherwise let through.
    """
    torch.set_default_dtype(torch.float64)
    fs = _fs(ee_scattering=dict(**SMALL))
    dim = 2 * fs.M_theta + 1
    torch.manual_seed(0)
    a = 1e-2 * torch.randn(4, fs.Nr * dim, dtype=torch.float64,
                           device=rc.device)
    out = fs.ee_scattering.a_dot(a)
    assert out.shape == a.shape, (out.shape, a.shape)
    assert torch.isfinite(out).all(), "a_dot produced non-finite values"
    assert float(out.abs().max()) > 0.0, "a_dot returned identically zero"


def test_ee_conserves_particles_and_momentum() -> None:
    """The e-e operator's null space: it moves neither particles nor momentum.

    These are structural (they follow from the collision kinematics, not from
    any quadrature being converged), so they hold even at this tiny size --
    which is exactly what makes them worth keeping in the fast suite.
    """
    torch.set_default_dtype(torch.float64)
    fs = _fs(ee_scattering=dict(**SMALL))
    dim = 2 * fs.M_theta + 1
    torch.manual_seed(1)
    a = 1e-2 * torch.randn(4, fs.Nr * dim, dtype=torch.float64,
                           device=rc.device)
    out = fs.ee_scattering.a_dot(a).reshape(4, fs.Nr, dim)
    scale = float(out.abs().max().clamp(min=1e-300))
    # m = 0 is the particle-number channel; m = 1 (cos, sin) is momentum.
    assert float(out[:, :, 0].abs().max()) / scale < 1e-9, "particle leak"
    assert float(out[:, :, 1:3].abs().max()) / scale < 1e-9, "momentum leak"


def test_ee_linear_is_linear_and_nonlinear_is_not() -> None:
    """`nonlinear=False` scales exactly; `nonlinear=True` does not.

    Two structural facts, both certain at any resolution, and together they
    pin down which operator was actually assembled: the linear block must
    satisfy a_dot(2a) = 2 a_dot(a) to roundoff, and if the cubic is switched on
    it must break that -- which is also the check that the nonlinear path is
    live rather than silently falling back to the linear one.

    ⛔ I FIRST WROTE A gamma ~ T^2 CHECK HERE AND IT WAS WRONG.  Measured 1.17x
    for a doubling of T, not 4x: at fixed MODAL amplitude a different T is a
    different physical delta-f (the basis carries a w_eq weight), and with the
    cubic active the response is not a single power of T anyway.  The T^2 law
    is real but belongs where it is already tested properly, in test_ee.py.
    """
    torch.set_default_dtype(torch.float64)
    dim = 2 * 3 + 1
    torch.manual_seed(2)
    a = 1e-2 * torch.randn(1, dim, dtype=torch.float64, device=rc.device)

    lin = _fs(ee_scattering=dict(**{**SMALL, "nonlinear": False}))
    o1 = lin.ee_scattering.a_dot(a)
    o2 = lin.ee_scattering.a_dot(2.0 * a)
    err = float((o2 - 2.0 * o1).abs().max()
                / o1.abs().max().clamp(min=1e-300))
    assert err < 1e-12, f"linear operator is not linear: {err:.2e}"

    nl = _fs(ee_scattering=dict(**SMALL))
    n1 = nl.ee_scattering.a_dot(a)
    n2 = nl.ee_scattering.a_dot(2.0 * a)
    dev = float((n2 - 2.0 * n1).abs().max()
                / n1.abs().max().clamp(min=1e-300))
    assert dev > 1e-9, (
        f"nonlinear=True behaves linearly ({dev:.2e}): the cubic is not live")
