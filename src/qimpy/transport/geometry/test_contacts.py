"""Tests for contact boundary conditions: biased (ammeter) and floating
(voltmeter) contacts, and the conservative contact-current readout.

Run directly (rc.init / serial)."""
from __future__ import annotations
import tempfile, os

import numpy as np
import torch

from qimpy import rc
from qimpy.mpi import ProcessGrid
from ..material import FermiSurface
from ._dg_mesh import load_mesh
from ._dg_mpi import build_distributed
from . import test_advect as TA


def _build(contacts, order=3, vF=1.5, M=8):
    """FermiSurface(Nr=1, M_theta=M) device on the rect mesh; contacts exercise
    the scalar (per-collocation) advection path."""
    tmp = tempfile.mkdtemp()
    mesh = load_mesh(TA._make_rect_mesh(12.0, os.path.join(tmp, "rect.npz")))
    pg = ProcessGrid(rc.comm, "rk", (1, 1))
    mat = FermiSurface(kF=1.0, vF=vF, M_theta=M, Nr=1, T=1.0,
                       tau_p=np.inf, tau_ee=np.inf, r_c=np.inf,
                       specularity=1.0, process_grid=pg)
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
    mat = FermiSurface(kF=1.0, vF=1.5, M_theta=8, Nr=1, T=1.0,
                       tau_p=15.0, tau_ee=8.0, r_c=np.inf, specularity=1.0,
                       process_grid=pg)
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
        u[..., :] = float(V0)                # constant in theta = isotropic level V0
        da._update_feedback(u)
        assert abs(da.contact_potentials()["drain"] - V0) < 1e-12


def test_current_source_zero_equals_floating() -> None:
    """A current source with I_set = 0 is mathematically identical to a floating
    probe: the feedback formulas differ only by a constant.  At I_set=0 the
    contact carries zero current to the same precision as a floating probe."""
    torch.set_default_dtype(torch.float64)
    sp, da, vF = _build({"source": {"dmu": 0.1}, "drain": {"I_set": 0.0}})
    w = _step(da, sp.dg, 200, vF)
    assert abs(da.contact_currents(w)["drain"]) < 1e-12


def test_current_source_delivers_prescribed_current() -> None:
    """At every RHS evaluation a current source contact self-adjusts its level
    so the net outward face-integrated number-flux equals I_set.  When the
    readout (contact_currents) uses the same numerical flux, agreement is to
    roundoff at every step -- no convergence required.  Test with two current
    sources driving current through a resistive device."""
    torch.set_default_dtype(torch.float64)
    tmp = tempfile.mkdtemp()
    mesh = load_mesh(TA._make_rect_mesh(12.0, os.path.join(tmp, "rect.npz")))
    pg = ProcessGrid(rc.comm, "rk", (1, 1))
    mat = FermiSurface(kF=1.0, vF=1.5, M_theta=8, Nr=1, T=1.0,
                       tau_p=15.0, tau_ee=8.0, r_c=np.inf, specularity=1.0,
                       process_grid=pg)
    I_target = 0.05
    # source INJECTS current (negative outward), drain EXTRACTS (positive outward)
    sp, da = build_distributed(
        mesh, 3, mat,
        {"source": {"I_set": -I_target}, "drain": {"I_set": +I_target}},
        rc.comm)
    w = da.adv.apply_mass(torch.zeros(sp.dg.Np, sp.dg.K, da.Nk))
    dt = 0.5 * float(sp.dg.dt_scale) / mat.vF

    def rhs(w):
        return da.local_rhs(w) + da.adv.apply_mass(
            mat.rho_dot(da.adv.apply_mass_inv(w), 0.0, 0))

    # A modest number of steps -- the assertion is per-step exact, not asymptotic;
    # we step at all to confirm it holds under non-trivial bulk state.
    for _ in range(50):
        k1 = rhs(w); k2 = rhs(w + 0.5 * dt * k1)
        k3 = rhs(w + 0.5 * dt * k2); k4 = rhs(w + dt * k3)
        w = w + (dt / 6.0) * (k1 + 2 * (k2 + k3) + k4)
    I = da.contact_currents(w)
    assert abs(I["source"] + I_target) < 1e-10, \
        f"source current = {I['source']:.6e}, expected {-I_target:.6e}"
    assert abs(I["drain"] - I_target) < 1e-10, \
        f"drain current = {I['drain']:.6e}, expected {I_target:.6e}"
    # The voltages emerge as the device response.  Source injects current
    # (higher mu pushes particles in), drain extracts (lower mu pulls them out).
    V = da.contact_potentials()
    assert V["source"] > V["drain"], \
        f"current direction wrong: V_source={V['source']:.4f}, V_drain={V['drain']:.4f}"


def test_current_source_polarity_reverses_with_sign() -> None:
    """Flipping the sign of I_set swaps the device potentials: the contact's
    response voltage tracks the sign of the prescribed current."""
    torch.set_default_dtype(torch.float64)
    sp_p, da_p, vF = _build({"source": {"I_set": -0.02}, "drain": {"I_set": +0.02}})
    sp_n, da_n, _  = _build({"source": {"I_set": +0.02}, "drain": {"I_set": -0.02}})
    w_p = _step(da_p, sp_p.dg, 50, vF)
    w_n = _step(da_n, sp_n.dg, 50, vF)
    Ip = da_p.contact_currents(w_p); In = da_n.contact_currents(w_n)
    Vp = da_p.contact_potentials(); Vn = da_n.contact_potentials()
    # currents flip exactly
    assert abs(Ip["source"] + In["source"]) < 1e-10
    assert abs(Ip["drain"]  + In["drain"])  < 1e-10
    # the potential of each contact follows the sign of its own I_set
    assert (Vp["source"] - Vp["drain"]) * (Vn["source"] - Vn["drain"]) < 0


if __name__ == "__main__":
    rc.init()
    test_contact_current_readout_is_conservative(); print("current_readout_conservative: PASS")
    test_biased_contacts_balance_at_steady_state(); print("biased_balance: PASS")
    test_floating_contact_carries_no_current(); print("floating_zero_current: PASS")
    test_floating_contact_reads_uniform_potential(); print("floating_reads_potential: PASS")
    test_current_source_zero_equals_floating(); print("current_source_zero_equals_floating: PASS")
    test_current_source_delivers_prescribed_current(); print("current_source_delivers_current: PASS")
    test_current_source_polarity_reverses_with_sign(); print("current_source_polarity: PASS")
    print("ALL PASS")
