"""Tests for contact boundary conditions: biased (ammeter) and floating
(voltmeter) contacts, and the conservative contact-current readout.

Run directly (rc.init / serial)."""
from __future__ import annotations
import tempfile, os

import numpy as np
import torch

from qimpy import rc
from qimpy.mpi import ProcessGrid
from ..material import FermiCircleModes
from ._dg_mesh import load_mesh
from ._dg_mpi import build_distributed
from . import test_advect as TA


def _build(contacts, order=3, vF=1.5, M=8):
    """Modal (angular-harmonic) FermiCircleModes device on the rect mesh; contacts
    exercise the coupled-flux path. Modal requires Pk=1 (serial)."""
    tmp = tempfile.mkdtemp()
    mesh = load_mesh(TA._make_rect_mesh(12.0, os.path.join(tmp, "rect.npz")))
    pg = ProcessGrid(rc.comm, "rk", (1, 1))
    mat = FermiCircleModes(kF=1.0, vF=vF, M=M, tau_p=np.inf, tau_ee=np.inf,
                           r_c=np.inf, specularity=1.0, process_grid=pg)
    sp, da = build_distributed(mesh, order, mat, contacts, rc.comm)
    return sp, da, vF


def _step(da, dg, nsteps, vF):
    w = da.adv.apply_mass(torch.zeros(dg.Np, dg.K, da.Nk))
    dt = 0.5 * float(dg.dt_scale) / vF
    for _ in range(nsteps):
        k1 = da.local_rhs(w); k2 = da.local_rhs(w + 0.5 * dt * k1)
        k3 = da.local_rhs(w + 0.5 * dt * k2); k4 = da.local_rhs(w + dt * k3)
        w = w + (dt / 6.0) * (k1 + 2 * (k2 + k3) + k4)
    return w


def test_contact_current_readout_is_conservative() -> None:
    """Sum of contact currents equals minus the total mass rate (walls carry no
    current), so the readout exactly accounts for the device's charge balance."""
    torch.set_default_dtype(torch.float64)
    sp, da, vF = _build({"source": {"dmu": 0.1}, "drain": {"dmu": -0.1}})
    w = _step(da, sp.dg, 50, vF)
    I = da.contact_currents(w)
    dmdt = float((da.local_rhs(w) * da.mass_proj).sum())
    assert abs(sum(I.values()) + dmdt) < 1e-10


def test_biased_contacts_balance_at_steady_state() -> None:
    """In a resistive device (finite tau_p), source and drain currents relax to
    equal and opposite at the DC steady state. Stepped with collisions included,
    since a collisionless ballistic cavity rings rather than settling."""
    torch.set_default_dtype(torch.float64)
    tmp = tempfile.mkdtemp()
    mesh = load_mesh(TA._make_rect_mesh(12.0, os.path.join(tmp, "rect.npz")))
    pg = ProcessGrid(rc.comm, "rk", (1, 1))
    mat = FermiCircleModes(kF=1.0, vF=1.5, M=8, tau_p=15.0, tau_ee=8.0,
                           r_c=np.inf, specularity=1.0, process_grid=pg)
    sp, da = build_distributed(
        mesh, 3, mat, {"source": {"dmu": 0.1}, "drain": {"dmu": -0.1}}, rc.comm)
    w = da.adv.apply_mass(torch.zeros(sp.dg.Np, sp.dg.K, da.Nk))
    dt = 0.5 * float(sp.dg.dt_scale) / mat.vF

    def rhs(w):                               # spatial advection + collisions
        return da.local_rhs(w) + da.adv.apply_mass(
            mat.rho_dot(da.adv.apply_mass_inv(w), 0.0, 0))

    for _ in range(600):
        k1 = rhs(w); k2 = rhs(w + 0.5 * dt * k1)
        k3 = rhs(w + 0.5 * dt * k2); k4 = rhs(w + dt * k3)
        w = w + (dt / 6.0) * (k1 + 2 * (k2 + k3) + k4)
    I = da.contact_currents(w)
    assert abs(I["source"] + I["drain"]) < 1e-4
    assert abs(I["source"]) > 1e-3            # a real current is flowing


def test_floating_contact_carries_no_current() -> None:
    """A voltage probe's level floats to zero its own current, exactly."""
    torch.set_default_dtype(torch.float64)
    sp, da, vF = _build({"source": {"dmu": 0.1}, "drain": {"floating": True}})
    w = _step(da, sp.dg, 200, vF)
    assert abs(da.contact_currents(w)["drain"]) < 1e-12


def test_floating_contact_reads_uniform_potential() -> None:
    """In a device at a uniform isotropic level V0, a floating probe reads V0."""
    torch.set_default_dtype(torch.float64)
    sp, da, vF = _build({"source": {"dmu": 0.1}, "drain": {"floating": True}})
    for V0 in (0.05, -0.1, 0.2):
        u = torch.zeros(sp.dg.Np, sp.dg.K, da.Nk)
        u[..., 0] = float(V0)                 # isotropic m=0 coefficient = level V0
        da._update_floating(u)
        assert abs(da.contact_potentials()["drain"] - V0) < 1e-12


if __name__ == "__main__":
    rc.init()
    test_contact_current_readout_is_conservative(); print("current_readout_conservative: PASS")
    test_biased_contacts_balance_at_steady_state(); print("biased_balance: PASS")
    test_floating_contact_carries_no_current(); print("floating_zero_current: PASS")
    test_floating_contact_reads_uniform_potential(); print("floating_reads_potential: PASS")
    print("ALL PASS")
