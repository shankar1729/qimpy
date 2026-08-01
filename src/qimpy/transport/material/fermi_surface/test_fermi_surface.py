"""Unit tests for FermiSurface (unified Fermi-surface / Fermi-circle material).

Covers:
- AngularBasis: exact round-trip and orthonormal projection of constants.
- RadialBasis: discrete orthonormality (T_to_modes @ T_from_modes = I) at Nr>1.
- FermiSurface transforms: tensor-product round-trip across Nr and M_theta.
- rho_dot in modes: collision is diagonal in (l, n), cyclotron is omega_c * G
  where G is the block-skew Fourier generator; compare against a hand-built
  reference (no dependency on legacy materials).
- _DeltaKContactor: voltage + drift modal structure round-trips.
- _DeltaKReflector: specular at axis-aligned wall; mass conservation at
  arbitrary normals and specularities (the discrete-quadrature leak is folded
  into D so the net mass flux at the wall is zero to roundoff).
- realizability_floor: defaults to None (no-op limiter for delta-f).
"""
from __future__ import annotations
import numpy as np
import torch
import pytest

from qimpy import rc
from qimpy.mpi import ProcessGrid
from qimpy.transport.material import FermiSurface
from ._fermi_surface import AngularBasis, RadialBasis
from ._representation import _DeltaKReflector


def _pg() -> ProcessGrid:
    return ProcessGrid(rc.comm, "rk", (1, 1))


def _make(M_theta: int, Nr: int = 1, *, T_temp: float = 1.0,
          tau_p: float = np.inf, tau_ee: float = np.inf,
          r_c: float = np.inf, specularity: float = 1.0) -> FermiSurface:
    return FermiSurface(
        kF=1.0, vF=1.5, M_theta=M_theta, Nr=Nr, T=T_temp,
        tau_p=tau_p, tau_ee=tau_ee, r_c=r_c, specularity=specularity,
        process_grid=_pg(),
    )


# ----------------------------------------------------------------------------
# Bases
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("M", [4, 8, 16, 32])
def test_angular_basis_roundtrip(M: int) -> None:
    """T_to_modes @ T_from_modes = I for the Fourier basis at N_q = 2M+1."""
    torch.set_default_dtype(torch.float64)
    ab = AngularBasis(M, n_quad=2 * M + 1)
    eye = ab.T_to_modes @ ab.T_from_modes
    assert torch.allclose(eye, torch.eye(2 * M + 1, dtype=torch.float64),
                          atol=1e-12)


@pytest.mark.parametrize("Nr", [2, 4, 8])
@pytest.mark.parametrize("T_temp", [0.1, 1.0, 5.0])
def test_radial_basis_orthonormal(Nr: int, T_temp: float) -> None:
    """T_to_modes @ T_from_modes = I_Nr under the discrete measure."""
    torch.set_default_dtype(torch.float64)
    rb = RadialBasis(Nr, T_temp=T_temp)
    eye = rb.T_to_modes @ rb.T_from_modes
    assert torch.allclose(eye, torch.eye(Nr, dtype=torch.float64), atol=1e-10)


@pytest.mark.parametrize("Nr", [1, 2, 4])
@pytest.mark.parametrize("M", [4, 16])
def test_fermisurface_transform_roundtrip(Nr: int, M: int) -> None:
    """Transforms round-trip to roundoff.

    The angular nodes oversample the 2M+1 modes (the symmetric quadrature uses an
    even N_theta >= 2M+1), so the invariant is modal -> nodal -> modal = identity.
    Equivalently, a *physical* (band-limited) nodal state f = from_modes(a) also
    satisfies from_modes(to_modes(f)) = f; only out-of-band nodal noise is
    projected away (that content carries no represented harmonic, n, or current).
    """
    torch.set_default_dtype(torch.float64)
    fs = _make(M_theta=M, Nr=Nr)
    n_modes = Nr * fs.angular.dim
    a = torch.randn(4, n_modes, dtype=torch.float64, device=rc.device)
    assert torch.allclose(a, fs.to_modes(fs.from_modes(a)), atol=1e-12)
    f = fs.from_modes(a)                                   # band-limited nodal state
    assert torch.allclose(f, fs.from_modes(fs.to_modes(f)), atol=1e-12)


def test_fermisurface_constant_projects_to_n0m0() -> None:
    """A constant nodal field projects to exactly the (n=0, m=0) mode."""
    torch.set_default_dtype(torch.float64)
    fs = _make(M_theta=8, Nr=1)
    f = torch.full((1, fs.angular.N_theta), 0.7, dtype=torch.float64,
                    device=rc.device)
    a = fs.to_modes(f).reshape(fs.Nr, fs.angular.dim)
    assert a[0, 0].abs() > 1e-6
    others = a.clone(); others[0, 0] = 0.0
    assert others.abs().max() < 1e-12


# ----------------------------------------------------------------------------
# rho_dot: cyclotron + collision in modes, identity in delta-k
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("tau_p, tau_ee, r_c", [
    (np.inf, np.inf, np.inf),           # ballistic, no field
    (0.7,    np.inf, np.inf),           # tau_p only
    (0.7,    1.3,    np.inf),           # tau_p + tau_ee
    (np.inf, np.inf, 2.0),              # cyclotron only
    (0.7,    1.3,    2.0),              # full
])
def test_rho_dot_matches_hand_built_modal(
    tau_p: float, tau_ee: float, r_c: float,
) -> None:
    """At Nr=1 the rho_dot is the diagonal collision plus omega_c * G acting in
    modal space.  Compare FermiSurface.rho_dot against the hand-built modal
    operator applied to the same input.
    """
    torch.set_default_dtype(torch.float64)
    M, vF = 8, 1.5
    fs = _make(M, Nr=1, tau_p=tau_p, tau_ee=tau_ee, r_c=r_c)
    # Hand-built modal rates: m=0 conserved, m=1 -> 1/tau_p, m>=2 -> 1/tau_p + 1/tau_ee
    rates = np.zeros(2 * M + 1)
    rp  = 0.0 if not np.isfinite(tau_p)  else 1.0 / tau_p
    ree = 0.0 if not np.isfinite(tau_ee) else 1.0 / tau_ee
    for m in range(1, M + 1):
        rates[2 * m - 1] = rates[2 * m] = rp if m == 1 else (rp + ree)
    rates_t = torch.as_tensor(rates, dtype=torch.float64, device=rc.device)
    omega_c = (vF / r_c) if np.isfinite(r_c) else 0.0
    a = torch.randn(4, fs.angular.dim, dtype=torch.float64, device=rc.device)
    rho = fs.from_modes(a)
    rd_fs = fs.rho_dot(rho, 0.0, 0)
    ref_modal = -rates_t * a + omega_c * torch.einsum("dc,...c->...d",
                                                       fs.angular.G, a)
    rd_ref = fs.from_modes(ref_modal)
    nrm = max(float(rd_ref.abs().max()), 1e-30)
    assert float((rd_fs - rd_ref).abs().max()) / nrm < 1e-12


# ----------------------------------------------------------------------------
# Contactor: voltage + drift in modal -> delta-k
# ----------------------------------------------------------------------------
def test_contactor_voltage_drift_at_Nr1() -> None:
    """Linear contactor: dmu -> m=0; PHYSICAL inward drift vD -> m=1 with
    coefficient -kF*vD (shell momentum shift), per unit Phi convention."""
    torch.set_default_dtype(torch.float64)
    M = 8; kF = 1.0
    fs = _make(M, Nr=1)
    # Wall normals at three angles
    phi = torch.tensor([0.3, 1.1, -2.4], dtype=torch.float64)
    n = torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1)
    contactor = fs.get_contactor(n, dmu=0.07, vD=0.13)
    rho_dk = contactor(0.0)                               # (3, N_theta)
    a = fs.to_modes(rho_dk)
    # Expected: a_0 = dmu; a_1 = -kF vD cos phi; b_1 = -kF vD sin phi
    assert torch.allclose(a[:, 0], torch.full_like(a[:, 0], 0.07), atol=1e-12)
    assert torch.allclose(a[:, 1], -(kF * 0.13) * torch.cos(phi), atol=1e-12)
    assert torch.allclose(a[:, 2], -(kF * 0.13) * torch.sin(phi), atol=1e-12)


def test_contactor_nonlinear_linear_limit() -> None:
    """The nonlinear (exact shifted-FD) ghost linearizes to the linear ghost:
    same convention, different fidelity."""
    torch.set_default_dtype(torch.float64)
    fs = _make(8, Nr=1)
    phi = torch.tensor([0.4, -1.9], dtype=torch.float64)
    n = torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1)
    eps = 1e-6 * fs.T_temp
    lin = fs.get_contactor(n, dmu=eps, vD=eps)(0.0)
    nl = fs.get_contactor(n, dmu=eps, vD=eps, nonlinear=True)(0.0)
    scale = lin.abs().max()
    assert (nl - lin).abs().max() < 1e-6 * scale


def test_contactor_nonlinear_saturates() -> None:
    """Nr=1 nonlinear ghost = 2T tanh(s/2): saturating (Pauli-bounded), below
    the linear ghost at strong bias."""
    torch.set_default_dtype(torch.float64)
    T = 0.02
    fs = _make(8, Nr=1, T_temp=T)
    n = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    dmu = 5.0 * T
    nl = fs.get_contactor(n, dmu=dmu, nonlinear=True)(0.0)
    expect = 2.0 * T * np.tanh(2.5)                       # s = dmu/T = 5
    assert torch.allclose(nl, torch.full_like(nl, expect), rtol=1e-10)
    lin = fs.get_contactor(n, dmu=dmu)(0.0)
    assert nl.abs().max() < lin.abs().max()               # saturation


# ----------------------------------------------------------------------------
# Reflector: specular axis-aligned, mass conservation everywhere
# ----------------------------------------------------------------------------
def test_reflector_specular_axis_aligned() -> None:
    """At n=(0,1) specular maps b_m -> -b_m, a_m unchanged."""
    torch.set_default_dtype(torch.float64)
    fs = _make(M_theta=8, Nr=1)
    n = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    refl = _DeltaKReflector(fs, n, specularity=1.0)
    a_in = torch.randn(1, fs.angular.dim, dtype=torch.float64)
    u_in = fs.from_modes(a_in)
    a_out = fs.to_modes(refl(u_in))
    expected = a_in.clone()
    for m in range(1, fs.M_theta + 1):
        expected[..., 2 * m] = -expected[..., 2 * m]
    assert torch.allclose(a_out, expected, atol=1e-12)


@pytest.mark.parametrize("phi_deg", [0.0, 23.7, 45.0, 90.0, 137.0])
@pytest.mark.parametrize("s", [0.0, 0.3, 0.7, 1.0])
def test_reflector_mass_conservation(phi_deg: float, s: float) -> None:
    """Net normal mass flux at the wall is zero for any specularity, any wall
    angle.  The (D, T) formulation absorbs the discrete-quadrature artifacts
    of the kinked (v.n)_+/_- weights, so mass conservation holds exactly."""
    torch.set_default_dtype(torch.float64)
    fs = _make(M_theta=8, Nr=1, specularity=s)
    phi = np.deg2rad(phi_deg)
    n = torch.tensor([[np.cos(phi), np.sin(phi)]], dtype=torch.float64)
    refl = _DeltaKReflector(fs, n, specularity=s)
    rng = torch.Generator(device=rc.device).manual_seed(0)
    u_in = torch.randn(1, 1, fs.angular.N_theta, dtype=torch.float64,
                       generator=rng)
    u_out = refl(u_in)
    v_dot_n = fs.vF * (
        n[0, 0] * torch.cos(fs.angular.theta) + n[0, 1] * torch.sin(fs.angular.theta)
    )
    out_pos = v_dot_n.clamp(min=0); out_neg = v_dot_n.clamp(max=0)
    F_out = (out_pos * u_in[0, 0]).sum()
    F_in  = (out_neg * u_out[0, 0]).sum()
    net = float((F_out + F_in).abs())
    scale = max(float(F_out.abs()), 1e-30)
    assert net / scale < 1e-12


@pytest.mark.parametrize("phi_deg", [0.0, 23.7, 45.0, 90.0, 137.0])
@pytest.mark.parametrize("s", [0.0, 0.3, 0.7, 1.0])
def test_reflector_tang_momentum_conservation(phi_deg: float, s: float) -> None:
    """At a partially-specular wall the discrete tangential-momentum flux into
    the wall equals the continuum value  (1 - s) * F_out_tang^M  to roundoff
    (the gas keeps the specular fraction, the diffuse fraction is absorbed).
    At s=1 (pure specular) the flux is identically zero.  This is what the
    (D, T) coefficients in the diffuse correction enforce."""
    torch.set_default_dtype(torch.float64)
    fs = _make(M_theta=8, Nr=1, specularity=s)
    phi = np.deg2rad(phi_deg)
    n = torch.tensor([[np.cos(phi), np.sin(phi)]], dtype=torch.float64)
    refl = _DeltaKReflector(fs, n, specularity=s)
    rng = torch.Generator(device=rc.device).manual_seed(0)
    u_in = torch.randn(1, 1, fs.angular.N_theta, dtype=torch.float64,
                       generator=rng)
    u_out = refl(u_in)
    theta = fs.angular.theta
    v_dot_n = fs.vF * (n[0, 0] * torch.cos(theta) + n[0, 1] * torch.sin(theta))
    v_tang  = fs.vF * (-n[0, 1] * torch.cos(theta) + n[0, 0] * torch.sin(theta))
    out_pos = v_dot_n.clamp(min=0); out_neg = v_dot_n.clamp(max=0)
    F_out_tang = (out_pos * v_tang * u_in[0, 0]).sum()
    F_in_tang  = (out_neg * v_tang * u_out[0, 0]).sum()
    F_total = F_out_tang + F_in_tang
    expected = (1.0 - s) * F_out_tang                       # continuum identity
    scale = max(float(F_out_tang.abs()), 1e-30)
    err = float((F_total - expected).abs()) / scale
    assert err < 1e-12


@pytest.mark.parametrize("phi_deg", [0.0, 23.7, 45.0])
@pytest.mark.parametrize("s", [0.0, 1.0])
def test_reflector_energy_conservation(phi_deg: float, s: float) -> None:
    """On the Fermi circle |v| = vF is constant, so the kinetic-energy flux is
    proportional to the mass flux.  Wall conservation of energy is therefore
    inherited from mass conservation at every (phi, s)."""
    torch.set_default_dtype(torch.float64)
    fs = _make(M_theta=8, Nr=1, specularity=s)
    phi = np.deg2rad(phi_deg)
    n = torch.tensor([[np.cos(phi), np.sin(phi)]], dtype=torch.float64)
    refl = _DeltaKReflector(fs, n, specularity=s)
    rng = torch.Generator(device=rc.device).manual_seed(0)
    u_in = torch.randn(1, 1, fs.angular.N_theta, dtype=torch.float64,
                       generator=rng)
    u_out = refl(u_in)
    theta = fs.angular.theta
    v_dot_n = fs.vF * (n[0, 0] * torch.cos(theta) + n[0, 1] * torch.sin(theta))
    out_pos = v_dot_n.clamp(min=0); out_neg = v_dot_n.clamp(max=0)
    energy = 0.5 * fs.vF ** 2                                # constant on Fermi circle
    F_out = (out_pos * energy * u_in[0, 0]).sum()
    F_in  = (out_neg * energy * u_out[0, 0]).sum()
    net = float((F_out + F_in).abs())
    scale = max(float(F_out.abs()), 1e-30)
    assert net / scale < 1e-12


# ----------------------------------------------------------------------------
# Realizability floor: default None -> limiter is a no-op (delta-f safe)
# ----------------------------------------------------------------------------
def test_realizability_floor_default_is_none() -> None:
    fs = _make(M_theta=4, Nr=1)
    assert fs.realizability_floor() is None


# ----------------------------------------------------------------------------
# Cartesian representation: projection contract, round-trip, local-T_e rates
# (the 2026-07 audit found the projection returned G_band@c/T instead of the
# operator contract's c, and the reconstruction was a factor T too large --
# invisible at Nr=1 linear where the two nearly cancel, catastrophic elsewhere)
# ----------------------------------------------------------------------------
def _make_cart(Nr: int, T: float = 0.05, *, local_te_rates: bool = True,
               ee=None) -> FermiSurface:
    return FermiSurface(
        kF=1.0, vF=1.5, M_theta=6, Nr=Nr, T=T, xi_max=6.0,
        cartesian=dict(dk=T / (3.0 * 1.5), local_te_rates=local_te_rates),
        ee_scattering=ee, process_grid=_pg())


def _cart_state(fs, radial_mode: int, amp: float):
    """delta-f = w_eq_RB(xi_lab) * amp * psi_n(xi) * cos(2 theta): a pure
    (n, m=2 cos) mode of Phi about the lab frame (no N/J/E content)."""
    rep = fs.representation
    xi = (rep.eps_k - fs.mu) / fs.T_temp
    th = torch.atan2(rep.k[:, 1], rep.k[:, 0])
    f0 = torch.special.expit(-xi)
    psi = rep._psi(xi)[:, radial_mode]
    return (f0 * (1 - f0) / fs.T_temp) * amp * psi * torch.cos(2 * th)


@pytest.mark.parametrize("Nr,mode", [(1, 0), (2, 0), (2, 1)])
def test_cartesian_projection_contract(Nr: int, mode: int) -> None:
    """The coefficients handed to the modal operator ARE the Phi coefficients
    (Gram-corrected, T-normalized): a pure (n, m=2) input of amplitude amp
    projects to amp on that channel and ~0 elsewhere."""
    torch.set_default_dtype(torch.float64)
    fs = _make_cart(Nr)
    amp = 1e-8
    rho = _cart_state(fs, mode, amp)[None]        # one spatial cell
    seen = {}

    def capture(a, te=None):
        seen["a"] = a.detach().clone()
        return torch.zeros_like(a)

    fs.representation.apply_collision(rho, capture)
    a = seen["a"].reshape(Nr, fs.angular.dim)
    idx = (mode, 3)                               # (n, m=2 cos)
    err_chan = abs(float(a[idx]) - amp) / amp
    others = a.clone(); others[idx] = 0.0
    # was 2e-2 with the fine-rule normalization; the iterative solve
    # against the Gram the k-sum actually realizes makes this exact.
    assert err_chan < 1e-4, f"channel amplitude off by {err_chan:.1e}"
    assert float(others.abs().max()) < 2e-2 * amp, "cross-channel leakage"


@pytest.mark.parametrize("Nr,mode", [(1, 0), (2, 1)])
def test_cartesian_projection_roundtrip(Nr: int, mode: int) -> None:
    """apply_collision with the identity modal operator returns the input
    (P then R is the identity on non-conserved modes)."""
    torch.set_default_dtype(torch.float64)
    fs = _make_cart(Nr)
    rho = _cart_state(fs, mode, 1e-8)[None]
    out = fs.representation.apply_collision(rho, lambda a, te=None: a)
    rel = float((out - rho).abs().max() / rho.abs().max())
    assert rel < 1e-4, f"P.R != identity: rel={rel:.1e}"



# ---- residual (unresolved-mode) closure ------------------------------------

def _make_cart_closure(Nr=4, T=0.05, tau_ee=200.0):
    """Closure harness on the PHENOMENOLOGICAL rate: it isolates the closure
    mechanics from the microscopic e-e build (whose K_table hits a
    pre-existing CUDA-default-device .numpy() bug in this environment), and
    gives a known, exactly-flat gamma at every m >= 2."""
    return FermiSurface(
        kF=1.0, vF=1.5, M_theta=6, Nr=Nr, T=T, xi_max=6.0, tau_ee=tau_ee,
        cartesian=dict(dk=T / (3.0 * 1.5), local_te_rates=False),
        residual_damping=True, process_grid=_pg())


def _drifted_heated(rep, fs, kD, Te, dmu=0.0):
    """Exact local drifted-heated FD as a deviation about f0_lab."""
    kp = rep.k - torch.as_tensor(kD, device=rep.k.device, dtype=rep.k.dtype)
    eps_p = kp.square().sum(-1) / (2 * fs.m_star)
    f_le = torch.special.expit(-(eps_p - (fs.mu + dmu)) / Te)
    return f_le - rep._f0_lab


def test_residual_closure_vanishes_on_local_equilibrium() -> None:
    """C[f_le] = 0 must survive the closure.  Checked at a NON-half-cell drift:
    at kD = 0 the parity reflection is an exact grid permutation, so a zero-
    drift test cannot see an interpolation defect."""
    fs_on = _make_cart_closure()
    fs_off = FermiSurface(
        kF=1.0, vF=1.5, M_theta=6, Nr=4, T=0.05, xi_max=6.0, tau_ee=200.0,
        cartesian=dict(dk=0.05 / 4.5, local_te_rates=False),
        process_grid=_pg())
    rep = fs_on.representation
    kD = 0.37 * rep._dk_grid * np.array([1.0, 0.61])
    df = _drifted_heated(rep, fs_on, kD, 1.3 * fs_on.T_temp)[None]
    # ON vs OFF isolates the CLOSURE.  Comparing against zero instead would
    # measure the pre-existing frame-recovery / projection quadrature error,
    # which is finite on any grid and has nothing to do with this patch.
    a = rep.apply_collision(df.clone(), fs_on._modal_collision)
    b = fs_off.representation.apply_collision(df.clone(), fs_off._modal_collision)
    # Normalize by the ACTUAL residual, not by |df|.  The discrete state is
    # not exactly the local equilibrium -- frame recovery returns (kD,Te,mu)
    # from grid moments, so df_loc is small but nonzero -- and damping that
    # genuine residual is precisely the closure's job.  The invariant that
    # must hold is that the closure never exceeds gamma_res * |df_loc|, i.e.
    # it damps what is there and does not invent a source of its own.
    f = rep._f0_lab[None] + df
    kD_r, Te_r, mu_r = rep._recover_frame(f)
    kp = rep.k[None] - kD_r[:, None]
    xi = (kp.square().sum(-1) / (2 * fs_on.m_star) - mu_r[:, None]) / Te_r[:, None]
    df_loc_max = float((f - torch.special.expit(-xi)).abs().max())
    rel = float((a - b).abs().max()) / (fs_on.gamma_residual() * df_loc_max)
    assert df_loc_max < 0.05 * float(df.abs().max()), "frame recovery is off"
    assert rel < 1.0, (
        f"closure output is {rel:.3f} x gamma_res*|df_loc| -- it is producing "
        "more than a relaxation of the residual actually present")


def test_residual_closure_spares_odd_harmonics() -> None:
    """Odd angular harmonics are gated to zero at leading order; the parity
    split must leave them essentially untouched."""
    fs = _make_cart_closure()
    rep = fs.representation
    kD = 0.37 * rep._dk_grid * np.array([1.0, 0.61])
    kp = rep.k - torch.as_tensor(kD, device=rep.k.device, dtype=rep.k.dtype)
    xi = (kp.square().sum(-1) / (2 * fs.m_star) - fs.mu) / fs.T_temp
    th = torch.atan2(kp[:, 1], kp[:, 0])
    f0 = torch.special.expit(-xi)
    # a HIGH odd harmonic, above M_theta = 6, so nothing else acts on it:
    df = ((f0 * (1 - f0) / fs.T_temp) * torch.cos(9 * th))[None]
    out = rep.apply_collision(df.clone(), lambda a, te=None: torch.zeros_like(a))
    leak = float(out.abs().max()) / (fs.gamma_residual() * float(df.abs().max()))
    # MEASURED 0.057 at dk = T/(3 vF).  This is the bilinear interpolation
    # error of the drift reflection, not a parity-logic error: without the
    # split the same mode would be damped at 1.0 x gamma_res, so the gate buys
    # ~20x.  The physical odd-m rate is ~0.01 gamma_2, so the artifact is still
    # ~20x the physics it protects -- a bicubic (16-corner, O(h^4)) gather
    # should reach ~0.5%.  Threshold here is a regression guard.
    assert leak < 8e-2, f"odd harmonic damped at {leak:.3f} of gamma_res"


def test_residual_closure_damps_even_harmonics() -> None:
    """The even counterpart of the previous test MUST be damped, at ~gamma_res
    and with the sign of a decay."""
    fs = _make_cart_closure()
    rep = fs.representation
    kD = 0.37 * rep._dk_grid * np.array([1.0, 0.61])
    kp = rep.k - torch.as_tensor(kD, device=rep.k.device, dtype=rep.k.dtype)
    xi = (kp.square().sum(-1) / (2 * fs.m_star) - fs.mu) / fs.T_temp
    th = torch.atan2(kp[:, 1], kp[:, 0])
    f0 = torch.special.expit(-xi)
    df = ((f0 * (1 - f0) / fs.T_temp) * torch.cos(10 * th))[None]
    out = rep.apply_collision(df.clone(), lambda a, te=None: torch.zeros_like(a))
    ov = float((out[0] * df[0]).sum() / (df[0] * df[0]).sum())
    assert ov < 0.0, "even residual is not damped"
    assert 0.5 < abs(ov) / fs.gamma_residual() < 1.5, (
        f"even residual damped at {abs(ov)/fs.gamma_residual():.2f} x gamma_res")


def test_residual_closure_off_is_bit_identical() -> None:
    """residual_damping=False must reproduce the pre-closure path exactly."""
    fs_off = FermiSurface(
        kF=1.0, vF=1.5, M_theta=6, Nr=4, T=0.05, xi_max=6.0, tau_ee=200.0,
        cartesian=dict(dk=0.05 / 4.5, local_te_rates=False),
        process_grid=_pg())
    rep = fs_off.representation
    torch.manual_seed(0)
    df = (0.01 * torch.randn(3, rep.Nk, device=rep.k.device,
                             dtype=rep.k.dtype))
    a = rep.apply_collision(df.clone(), fs_off._modal_collision)
    b = rep.apply_collision(df.clone(), fs_off._modal_collision)
    assert torch.equal(a, b), "collision apply is not deterministic"
    assert fs_off.gamma_residual() is None
