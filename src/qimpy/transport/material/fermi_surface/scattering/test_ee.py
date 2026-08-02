"""Tests for the microscopic 2D Fermi-liquid e-e collision operator.

Reference values: GaAs 2DEG of the derivation notes (atomic units):
m* = 0.067, kF = 7.5e-3 (E_F = 4.198e-4), T = 1.33e-5 (4.2 K),
eps_b = 12.9, kappa = 2 m*/eps_b.  Doc targets: K2 = 5.108e3,
K4 = 7.930e3, K6 = 9.811e3, gamma_2/T^2 = 2173.6 (closed form, spin degeneracy g_s = 2).
The exact thermal-shell (Galerkin) rates at T/E_F = 0.032 are larger:
gamma_2 = 1.299x, gamma_4 = 1.156x closed form (verified independently
against the unreduced collision integral by Monte-Carlo quadratic form).
"""
import numpy as np
import torch
import pytest

from qimpy import rc
from qimpy.mpi import ProcessGrid
from . import _kernels

M_STAR, EPS_B, KF, T0 = 0.067, 12.9, 7.5e-3, 1.33e-5
E_F = 0.5 * KF**2 / M_STAR
KAPPA = 2 * M_STAR / EPS_B


@pytest.fixture(autouse=True)
def _no_default_device_mode():
    """qimpy's conftest installs torch.set_default_device(rc.device), whose
    TorchFunctionMode wrapper (a) breaks the kernels' numpy interop on GPU
    nodes and (b) injects a device= kwarg into torch.vander, which does not
    accept one -- silently failing this ENTIRE module on GPU nodes (the
    long-standing 'torch.vander env bug').  Disable the mode for this module;
    code that explicitly asks for rc.device is unaffected."""
    try:
        prev = torch.get_default_device()
    except (AttributeError, RuntimeError):
        prev = None
    torch.set_default_device(None)                # remove the wrapper entirely
    yield
    if prev is not None:
        torch.set_default_device(prev)


def make_fs(M_theta=8, Nr=1, ee=None, **kwargs):
    from qimpy.transport.material import FermiSurface

    process_grid = ProcessGrid(rc.comm, "rk", (-1, 1))
    return FermiSurface(
        kF=KF, vF=KF / M_STAR, M_theta=M_theta, Nr=Nr, T=T0,
        process_grid=process_grid, ee_scattering=ee, **kwargs,
    )


def test_K_table():
    K = _kernels.K_table(8, kF=KF, epsilon_bg=EPS_B, kappa=KAPPA, n_alpha=4096)
    assert torch.allclose(
        K[torch.tensor([2, 4, 6])],
        torch.tensor([5.1084e3, 7.9297e3, 9.8115e3], dtype=K.dtype),
        rtol=1e-3,
    )
    assert K[0] == 0.0 and K[1] == 0.0 and K[3] == 0.0  # parity gate
    K2 = _kernels.K_table(8, kF=KF, epsilon_bg=EPS_B, kappa=KAPPA, n_alpha=8192)
    assert (K2[2:] - K[2:]).abs().max() / K2[2:].abs().max() < 1e-10


def test_closed_form_rate():
    K = _kernels.K_table(2, kF=KF, epsilon_bg=EPS_B, kappa=KAPPA)
    gam = _kernels.gamma_linear(K, m_star=M_STAR, T=T0, E_F=E_F)
    # g_s = 2 (unpolarized 2DEG): the doc's 1086.8 is the g_s = 1 value.
    # gamma scales LINEARLY in g_s -- every RATIO in this file is
    # invariant, so this is the only expectation that moves.
    assert abs(gam[2] / T0**2 - 2 * 1086.8) < 2.0


def test_on_shell_rates():
    """on_shell=True applies the closed-form surface rate (Nr=1 only); the exact
    default (on_shell=False) is the larger finite-T/E_F thermal-shell rate."""
    K = _kernels.K_table(4, kF=KF, epsilon_bg=EPS_B, kappa=KAPPA)
    gam = _kernels.gamma_linear(K, m_star=M_STAR, T=T0, E_F=E_F)
    fs = make_fs(M_theta=4,
                 ee=dict(epsilon_bg=EPS_B, on_shell=True, nonlinear=False))
    # the m=2 (cos) block, l=0 diagonal, equals the closed-form gamma_2 exactly:
    assert abs(fs.ee_scattering.L_coeff[3, 0, 0].item() - gam[2].item()) \
        < 1e-12 * gam[2].item()
    # the exact default is ~30% larger for the shear mode at T/E_F ~ 0.03:
    fs_x = make_fs(M_theta=4, ee=dict(epsilon_bg=EPS_B, nonlinear=False,
                                      n_xi=24, n_phi=512, n_xi_proj=12))
    assert fs_x.ee_scattering.L_coeff[3, 0, 0].item() > 1.2 * gam[2].item()
    # on_shell=True with the radial (energy) tower Nr>1 is rejected:
    with pytest.raises(Exception):
        make_fs(M_theta=4, Nr=2, ee=dict(epsilon_bg=EPS_B, on_shell=True))


def test_form_factor():
    q = torch.tensor([1e-9, 0.03, 0.3, 1.0, 3.0, 8.5, 30.0, 200.0])
    F = _kernels.well_form_factor(q)
    assert abs(F[0] - 1.0) < 1e-6  # F(0) = 1
    assert abs(F[-1] - 3.0 / 200.0) / (3.0 / 200.0) < 0.05  # F -> 3/(qW)
    # against direct double integral:
    z = torch.linspace(0, 1, 801, dtype=torch.float64)[1:-1]
    rho = 2 * torch.sin(np.pi * z) ** 2
    dz = float(z[1] - z[0])
    for x in (0.3, 3.0, 8.5):
        kern = torch.exp(-x * (z[:, None] - z[None, :]).abs())
        Fn = (rho[:, None] * rho[None, :] * kern).sum() * dz**2
        Fc = _kernels.well_form_factor(torch.tensor([x], dtype=torch.float64))[0]
        assert abs(Fc - Fn) / Fn < 1e-3


# ---------------------------------------------------------------------------
# Helpers for the exact nonlinear-vertex acceptance tests
# ---------------------------------------------------------------------------
def _w_occ(x):
    return 0.25 / torch.cosh(x / 2) ** 2


def _real_coeffs(c_vec, M):
    """Real cos/sin harmonic coefficients (a_0, a_1, b_1, ...) -> complex
    harmonics ghat[m], m = -M..M (offset M)."""
    nh = 2 * M + 1
    ghat = torch.zeros(nh, dtype=torch.complex128)
    ghat[M] = c_vec[0]
    for m in range(1, M + 1):
        ghat[M + m] = 0.5 * (c_vec[2 * m - 1] - 1j * c_vec[2 * m])
        ghat[M - m] = 0.5 * (c_vec[2 * m - 1] + 1j * c_vec[2 * m])
    return ghat


def _reference_eps_terms(df_of_phi, x1, phi1, n_phi, order,
                         n_xi=24, xi_cut=10.0):
    """Amplitude-isolate the eps^k term of exact_collision_reference at the
    given (x1, phi1) via the 4-point stencil of proto_cubic_nr1.

    B - F is EXACTLY cubic in the amplitude and its order-0 part vanishes on
    shell (detailed balance), so with o_s = (f(s) - f(-s))/2 = s L1 + s^3 C3:

    order=1 -> L (odd, eps^1):   (8 o1 - o2)/6
    order=2 -> Q (even, eps^2):  (f(1) + f(-1))/2
    order=3 -> C (odd, eps^3):   (o2 - 2 o1)/6
    """
    common = dict(kF=KF, m_star=M_STAR, T=T0, epsilon_bg=EPS_B, kappa=KAPPA)
    refkw = dict(n_xi=n_xi, xi_cut=xi_cut, n_phi=n_phi, chunk=2, **common)
    fd = {
        s: _kernels.exact_collision_reference(
            (lambda s_: (lambda x, phi: df_of_phi(x, phi) * s_))(s),
            x1, phi1, linearize=False, **refkw,
        )
        for s in (1.0, 2.0, -1.0, -2.0)
    }
    if order == 2:
        return 0.5 * (fd[1.0] + fd[-1.0])
    o1 = 0.5 * (fd[1.0] - fd[-1.0])
    o2 = 0.5 * (fd[2.0] - fd[-2.0])
    if order == 1:
        return (8.0 * o1 - o2) / 6.0
    return (o2 - 2.0 * o1) / 6.0


def test_cubic_vertex_vs_reference():
    """Criterion 2: the exact cubic vertex reproduces the eps^3 term of the
    brute-force reference for a SURFACE (l=0) field at phi1=0 (rotationally
    consistent: identical quadrature node placement) to the quadrature floor,
    and the residual at rotated phi1 is the shared van-Hove edge error
    decreasing with n_phi.  This is the l=0 special case of the full vertex
    (Nr=1, constant psi_0)."""
    # Even-only surface field with a single base harmonic m=2, so the full
    # cubic output (harmonics 2 +/- 2 +/- 2 in {2, 6}) lies within the
    # retained band |m| <= M = 6 -- the tensor and the (un-truncated)
    # reference then describe the SAME quantity at phi1 = 0.  This exercises
    # both the m=2 self-interaction and the m=6 (2+2+2) pumping channel.
    M = 6
    g_pos = {2: 1.0}
    nh = 2 * M + 1
    c = torch.zeros(1, nh, dtype=torch.float64)  # (Nr=1, nh) modal field
    for m, gm in g_pos.items():
        c[0, 2 * m - 1] = 2 * gm  # g(phi) = sum_m 2 gm cos(m phi)

    def g_of(phi):
        out = torch.zeros_like(phi)
        for m, gm in g_pos.items():
            out = out + 2 * gm * torch.cos(m * phi)
        return out

    def df(x, phi):
        return _w_occ(x) * g_of(phi) / T0  # delta_f = w_eq Phi, Phi = g

    common = dict(kF=KF, m_star=M_STAR, T=T0, epsilon_bg=EPS_B, kappa=KAPPA)
    phi1 = torch.tensor(np.linspace(0, 2 * np.pi, 16, endpoint=False))
    x1 = torch.zeros_like(phi1)

    rels0, rels_grid = [], []
    for n_phi in (256, 512):
        # Full-radial cubic at Nr=1 with constant psi_0 = 1 (the surface mode):
        # V[node, co, la, a, lb, b, lc, c].  -Phi_dot real coeffs at x1=0
        # (conv=4): reconstruct the output harmonics, then Phi_dot(phi1).
        V = _kernels.cubic_vertex(
            x_nodes=torch.zeros(1), psi_coeff=torch.ones(1, 1), M=M,
            n_xi=24, xi_cut=10.0, n_phi=n_phi, **common,
        )
        coeff = torch.einsum(
            "oxaybzd,xa,yb,zd->o", V[0], c, c, c
        )  # -Phi_dot^co
        # Vc returns -Phi_dot coeffs (decay/conv convention); recover f_dot
        # real coeffs: f_dot_coeffs = -coeff / conv, conv = 4 T0 at x1 = 0:
        fdot_re = -coeff / (4.0 * T0)
        # reconstruct f_dot(phi1):
        cols = [torch.ones_like(phi1)]
        for m in range(1, M + 1):
            cols.append(torch.cos(m * phi1))
            cols.append(torch.sin(m * phi1))
        basis = torch.stack(cols, dim=-1)  # (n_phi1, nh)
        fdot_tensor = basis @ fdot_re
        C_ref = _reference_eps_terms(df, x1, phi1, n_phi, order=3)
        rels0.append(
            abs(fdot_tensor[0] - C_ref[0]).item() / abs(C_ref[0]).item()
        )
        rels_grid.append(
            (fdot_tensor - C_ref).abs().max().item()
            / C_ref.abs().max().item()
        )
    # The vertex (built at phi1 = 0) and the reference now share the SAME
    # edge-clustered azimuth grid relative to phi1 (the reference grades phi3 =
    # phi1 + beta around the beta = pi van-Hove edge, _kernels._beta_grid), so
    # they sample identical relative kinematics at EVERY phi1 -- the rotated-phi1
    # residual that used to be O(1e-1) (the unresolved backscattering edge) is
    # now at float64 roundoff, not just at phi1 = 0:
    assert rels0[-1] < 1e-9, f"phi1=0 cubic vs reference: {rels0}"
    assert max(rels0) < 1e-9
    assert max(rels_grid) < 1e-9, f"rotated-phi1 cubic vs reference: {rels_grid}"


def test_cubic_vertex_energy_structured_vs_reference():
    """Criterion 1 (DISCRIMINATING): the FULL-radial cubic reproduces the eps^3
    term of the brute-force reference for an ENERGY-STRUCTURED field (Nr>=2 with
    l>=1 content, e.g. an energy-weighted shear coefficient a_{2,1}) at phi1=0,
    while the l=0-restricted cubic (ignoring the l>0 inputs) MISMATCHES the same
    reference.  This proves the l>0 input legs are actually included and exact.

    All base harmonics are kept to |m| <= 2 so every cubic output harmonic
    (|m| <= 6) fits the retained band M = 6; the tensor and the un-truncated
    reference then describe the SAME quantity at phi1 = 0."""
    from qimpy.transport.material.fermi_surface import RadialBasis

    M, Nr = 6, 2
    rb = RadialBasis(Nr, T_temp=T0, xi_max=6.0)
    xi_c = rb.xi.to(torch.float64).cpu()
    Tfm = rb.T_from_modes.to(torch.float64).cpu()
    _fc = [torch.ones_like(xi_c), xi_c]
    _pe, _po = 2, 1
    while len(_fc) < Nr:
        _fc.append(torch.tanh(0.5 * xi_c) ** _pe); _pe += 2
        if len(_fc) < Nr:
            _fc.append(torch.tanh(0.5 * xi_c) ** _po); _po += 2
    psi_coeff = torch.linalg.solve(torch.stack(_fc[:Nr], dim=-1), Tfm)
    nh = 2 * M + 1
    common = dict(kF=KF, m_star=M_STAR, T=T0, epsilon_bg=EPS_B, kappa=KAPPA)

    # Energy-structured field: l=0 and l=1 content, harmonics up to m=2.  The
    # l=1 cos(2) coefficient is the energy-weighted shear that the l=0-only
    # cubic cannot see.
    amod = torch.zeros(Nr, nh, dtype=torch.float64)
    amod[0, 3] = 0.8    # l=0, cos(2 phi)
    amod[1, 3] = 0.6    # l=1, cos(2 phi)  <- energy-weighted shear discriminator
    amod[1, 1] = 0.4    # l=1, cos(1 phi)
    amod[0, 4] = -0.3   # l=0, sin(2 phi)

    def psi_eval(x):
        v = np.tanh(0.5 * x)
        cols = [np.ones_like(x), x]
        pe, po = 2, 1
        while len(cols) < Nr:
            cols.append(v ** pe); pe += 2
            if len(cols) < Nr:
                cols.append(v ** po); po += 2
        return np.stack(cols[:Nr], axis=-1) @ psi_coeff.numpy()

    def ec_real(phi):
        cols = [np.ones_like(phi)]
        for m in range(1, M + 1):
            cols.append(np.cos(m * phi))
            cols.append(np.sin(m * phi))
        return np.stack(cols, axis=-1)

    def df_struct(x, phi):
        xpsi = psi_eval(x.numpy())
        eph = ec_real(phi.numpy())
        Phi = np.einsum("...l,...c,lc->...", xpsi, eph, amod.numpy())
        return torch.as_tensor(_w_occ(x).numpy() * Phi / T0)

    phi1 = torch.zeros(1)
    x1 = torch.zeros(1)
    rels_full, rels_l0 = [], []
    for n_phi in (256, 512):
        Vc = _kernels.cubic_vertex(
            x_nodes=torch.zeros(1), psi_coeff=psi_coeff, M=M,
            n_xi=24, xi_cut=10.0, n_phi=n_phi, **common,
        )

        def fdot0_from(field):
            # Vc[node, co, la, a, lb, b, lc, c]; node x1=0 (conv=4 T0) ->
            # -4 T0 Phi_dot coeffs.  Recover f_dot real coeffs and evaluate at
            # phi1 = 0 (only cos channels contribute, e_co(0) = 1):
            coeff = torch.einsum(
                "oxaybzd,xa,yb,zd->o", Vc[0], field, field, field
            )
            fdot_re = -coeff / (4.0 * T0)
            val = fdot_re[0]
            for m in range(1, M + 1):
                val = val + fdot_re[2 * m - 1]
            return val.item()

        full = fdot0_from(amod)
        # l=0-restricted input: zero the l>0 radial modes (what a cubic that
        # only sees the Fermi-surface deformation would compute):
        amod_l0 = amod.clone()
        amod_l0[1:, :] = 0.0
        l0_only = fdot0_from(amod_l0)

        C_ref = _reference_eps_terms(df_struct, x1, phi1, n_phi, order=3)[0]
        rels_full.append(abs(full - C_ref.item()) / abs(C_ref.item()))
        rels_l0.append(abs(l0_only - C_ref.item()) / abs(C_ref.item()))

    # The full cubic matches the reference to the quadrature floor at phi1=0
    # (identical node placement); the l=0-only restriction is off by O(10%) --
    # the l>0 inputs carry a real, distinct contribution that is now exact.
    assert rels_full[-1] < 1e-6, (
        f"full cubic vs reference (energy-structured): {rels_full}"
    )
    assert max(rels_full) < 1e-6
    # the l=0-only cubic FAILS the same test by a wide, resolution-independent
    # margin (NOT a quadrature artifact) -- proof that l>0 matters:
    assert min(rels_l0) > 1e-3, (
        f"l=0-only should mismatch but matched: {rels_l0}"
    )
    assert min(rels_l0) > 1e3 * max(rels_full)  # full is orders better


def test_quadratic_vertex_vs_reference():
    """Criterion 2: the exact quadratic (thermoelectric) vertex reproduces the
    eps^2 term of the reference for an energy-structured (Nr>=2) field at
    phi1=0, and VANISHES for a pure on-Fermi-surface (l=0) field."""
    from qimpy.transport.material.fermi_surface import RadialBasis

    M, Nr = 4, 2
    rb = RadialBasis(Nr, T_temp=T0, xi_max=6.0)
    xi_c = rb.xi.to(torch.float64).cpu()
    Tfm = rb.T_from_modes.to(torch.float64).cpu()
    _fc = [torch.ones_like(xi_c), xi_c]
    _pe, _po = 2, 1
    while len(_fc) < Nr:
        _fc.append(torch.tanh(0.5 * xi_c) ** _pe); _pe += 2
        if len(_fc) < Nr:
            _fc.append(torch.tanh(0.5 * xi_c) ** _po); _po += 2
    psi_coeff = torch.linalg.solve(torch.stack(_fc[:Nr], dim=-1), Tfm)
    nh = 2 * M + 1
    common = dict(kF=KF, m_star=M_STAR, T=T0, epsilon_bg=EPS_B, kappa=KAPPA)

    # energy-structured field: mixes l=0 and l=1, harmonics m=1,2:
    amod = torch.zeros(Nr, nh, dtype=torch.float64)
    amod[0, 3] = 0.8   # l=0, cos(2 phi)
    amod[1, 1] = 0.5   # l=1, cos(1 phi)
    amod[1, 4] = -0.3  # l=1, sin(2 phi)
    amod[0, 2] = 0.4   # l=0, sin(1 phi)

    def psi_eval(x):
        v = np.tanh(0.5 * x)
        cols = [np.ones_like(x), x]
        pe, po = 2, 1
        while len(cols) < Nr:
            cols.append(v ** pe); pe += 2
            if len(cols) < Nr:
                cols.append(v ** po); po += 2
        return np.stack(cols[:Nr], axis=-1) @ psi_coeff.numpy()

    def ec_real(phi):
        cols = [np.ones_like(phi)]
        for m in range(1, M + 1):
            cols.append(np.cos(m * phi))
            cols.append(np.sin(m * phi))
        return np.stack(cols, axis=-1)

    def df_struct(x, phi):
        xpsi = psi_eval(x.numpy())
        eph = ec_real(phi.numpy())
        Phi = np.einsum("...l,...c,lc->...", xpsi, eph, amod.numpy())
        return torch.as_tensor(_w_occ(x).numpy() * Phi / T0)

    phi1 = torch.zeros(1)
    x1 = torch.zeros(1)
    U = _kernels._real_to_complex(M)
    ahat = torch.einsum("lc,mc->lm", amod.to(torch.complex128), U)
    rels = []
    for n_phi in (256, 512):
        Vq = _kernels.quadratic_vertex(
            x_nodes=torch.zeros(1), psi_coeff=psi_coeff, M=M,
            n_xi=24, xi_cut=10.0, n_phi=n_phi, **common,
        )
        # Vq[node, co, la, a, lb, b]; node x1=0 (conv=4 T0): -4 T0*Phi_dot coeffs.
        coeff = torch.einsum("oxayb,xa,yb->o", Vq[0], amod, amod)
        # Vq returns -Phi_dot coeffs (decay/conv convention); recover f_dot:
        # f_dot_coeffs = -coeff / conv, conv = 4 T0 at x1 = 0; then evaluate at
        # phi1 = 0 (only cos channels contribute, e_co(0) = 1):
        fdot_re = -coeff / (4.0 * T0)
        fdot0 = fdot_re[0]
        for m in range(1, M + 1):
            fdot0 = fdot0 + fdot_re[2 * m - 1]
        Q_ref = _reference_eps_terms(df_struct, x1, phi1, n_phi, order=2)[0]
        rels.append(abs(fdot0.item() - Q_ref.item()) / abs(Q_ref.item()))
    assert rels[-1] < 1e-6, f"quadratic vs reference (eps^2): {rels}"

    # Pure surface field (l=0, on the Fermi surface): Q2 is strongly
    # suppressed.  Its particle-hole-odd coefficients (proportional to
    # f0 - 1/2, odd about x = 0) cancel at the Fermi surface; for a surface
    # field the residual is O(T/E_F) and the Q2 surface output is ~1e-3 of the
    # cubic at the same (unit-order) amplitude -- three orders below the
    # relaxing channels.  (T/E_F = 0.032 here; the ~1e-3 ratio is amplitude
    # scaled and matches the expected smallness.)
    a_surf = torch.zeros(Nr, nh, dtype=torch.float64)
    a_surf[0, 3] = 0.8  # l=0 cos(2 phi)
    Vc = _kernels.cubic_vertex(
        x_nodes=torch.zeros(1), psi_coeff=psi_coeff, M=M,
        n_xi=24, xi_cut=10.0, n_phi=512, **common,
    )
    Vq2 = _kernels.quadratic_vertex(
        x_nodes=torch.zeros(1), psi_coeff=psi_coeff, M=M,
        n_xi=24, xi_cut=10.0, n_phi=512, **common,
    )
    q_out = torch.einsum("oxayb,xa,yb->o", Vq2[0], a_surf, a_surf).abs().max()
    c_out = torch.einsum(
        "oxaybzd,xa,yb,zd->o", Vc[0], a_surf, a_surf, a_surf
    ).abs().max()
    ratio = (q_out / c_out).item()
    # Finite-T particle-hole residual ~ T/E_F.  The numeric value scales with
    # 1/psi_0, i.e. with the basis band-mass normalization (hybrid basis:
    # exact band mass -> psi_0 smaller x1.68 vs legacy at Nr=2 -> ratio x1.68,
    # 1.355e-3 -> 2.274e-3).
    assert ratio < 4e-3, f"Q2 surface/cubic = {ratio:.2e} (expected ~1e-3)"


def test_nonlinear_conservation():
    """Criterion 3: cubic and quadratic outputs annihilate number, momentum,
    energy nulls (post null-space projection) at Nr=1 and Nr>=2."""
    for Nr in (1, 3):
        M = 2
        fs = make_fs(
            M_theta=M, Nr=Nr,
            ee=dict(
                epsilon_bg=EPS_B,
                nonlinear=True, n_xi=16, n_phi=256, n_xi_proj=8,
            ),
        )
        dim = fs.angular.dim
        Ttm = fs.radial.T_to_modes.to(torch.float64).cpu()
        ones_c = Ttm @ torch.ones(Nr, dtype=torch.float64)
        t_ratio = T0 / E_F
        k_c = Ttm @ torch.sqrt(
            1.0 + t_ratio * fs.radial.xi.to(torch.float64).cpu()
        )
        torch.manual_seed(3)
        a = 1e-2 * torch.randn(4, Nr * dim, dtype=fs.v.dtype, device=rc.device)
        a_lin = -torch.einsum(  # isolate the nonlinear part of a_dot
            "cij,...jc->...ic", fs.ee_scattering.L_coeff,
            a.reshape(4, Nr, dim),
        ).reshape(4, Nr * dim)
        nl = (fs.ee_scattering.a_dot(a) - a_lin).reshape(4, Nr, dim)
        scale = nl.abs().max().item()
        # number (m=0) and energy (m=0, l-weighted) at output mode co=0:
        num = torch.einsum("bl,l->b", nl[..., 0], ones_c.to(nl.device, nl.dtype))
        assert num.abs().max().item() < 1e-10 * scale, "number"
        if Nr > 1:
            x_c = Ttm @ fs.radial.xi.to(torch.float64).cpu()
            ene = torch.einsum("bl,l->b", nl[..., 0], x_c.to(nl.device, nl.dtype))
            assert ene.abs().max().item() < 1e-10 * scale, "energy"
        # momentum (m=1) at output cos/sin modes co=1,2:
        mc = k_c.to(nl.device, nl.dtype)
        momx = torch.einsum("bl,l->b", nl[..., 1], mc)
        momy = torch.einsum("bl,l->b", nl[..., 2], mc)
        assert momx.abs().max().item() < 1e-10 * scale, "momentum x"
        assert momy.abs().max().item() < 1e-10 * scale, "momentum y"


def test_even_m_selection():
    """Criterion 4: the EXACT cubic carries the angular parity selection by
    itself (no gate).  A purely even input field produces purely even output
    (even+even+even=even) to machine precision; but a field carrying odd
    angular content genuinely populates odd output channels (1+1+1=3,
    3+3-2=4, ...) -- odd modes DO relax through the cubic when present.  This
    is the correct physics confirmed against the brute-force reference, not
    the artifact of forcing odd outputs to zero."""
    fs = make_fs(
        M_theta=6,
        ee=dict(
            epsilon_bg=EPS_B, nonlinear=True,
            n_xi=16, n_phi=256, n_xi_proj=8,
        ),
    )
    dim = fs.angular.dim
    amp = 0.3 * 4 * T0

    def cubic_only(a):  # cubic term of a_dot, isolated by amplitude parity
        def nl(s):
            aa = s * a
            a4 = aa.reshape(*aa.shape[:-1], fs.Nr, dim)
            lin = -torch.einsum("cij,...jc->...ic", fs.ee_scattering.L_coeff, a4)
            return fs.ee_scattering.a_dot(aa) - lin.reshape(aa.shape)
        o1 = 0.5 * (nl(1.0) - nl(-1.0))
        o2 = 0.5 * (nl(2.0) - nl(-2.0))  # odd-parity stencil -> cubic
        cub = (o2 - 2.0 * o1) / 6.0
        return cub.reshape(*a.shape[:-1], fs.Nr, dim)[..., 0, :]

    # pure even field cos(2 phi) -> purely even cubic output (m=2 self, m=6
    # pumped); odd output channels are EXACTLY zero by the natural selection:
    a_even = torch.zeros(1, dim, dtype=fs.v.dtype, device=rc.device)
    a_even[0, 3] = amp
    cub_even = cubic_only(a_even)[0]
    assert cub_even[3].abs() > 0  # m=2 self-interaction
    assert cub_even[11].abs() > 0  # m=6 pumped (2+2+2)
    for m in (1, 3, 5):  # odd channels vanish (to roundoff) for an even field
        assert cub_even[2 * m - 1].abs() < 1e-12 * cub_even.abs().max()
        assert cub_even[2 * m].abs() < 1e-12 * cub_even.abs().max()
    # pure odd deformation cos(3 phi): the cubic output is NONZERO -- odd modes
    # relax through the cubic (3+3-3=3 channel populated):
    a_odd = torch.zeros(1, dim, dtype=fs.v.dtype, device=rc.device)
    a_odd[0, 5] = amp  # cos(3 phi)
    cub_odd = cubic_only(a_odd)[0]
    assert cub_odd[5].abs() > 0  # cos(3 phi) output: pure-odd field relaxes
    # mixed odd+even: pumps both an even channel (3+3-2 = 4) AND the odd cos(3)
    # channel (genuine, not gated):
    a_mix = torch.zeros(1, dim, dtype=fs.v.dtype, device=rc.device)
    a_mix[0, 5] = amp  # cos(3 phi)
    a_mix[0, 3] = 0.7 * amp  # cos(2 phi)
    cub_mix = cubic_only(a_mix)[0]
    assert cub_mix[7].abs() > 0  # cos(4 phi) output present
    assert cub_mix[5].abs() > 0  # cos(3 phi) output present (odd relaxes)


def test_no_free_parameter():
    """Criterion 5: cubic_scale is gone everywhere (no tunable nonlinear
    scale); the constructor rejects it."""
    import inspect
    from ._ee import EEScattering

    sig = inspect.signature(EEScattering.__init__)
    assert "cubic_scale" not in sig.parameters
    with pytest.raises(TypeError):
        make_fs(ee=dict(epsilon_bg=EPS_B, cubic_scale=0.5))


def test_exact_rates_nr1():
    """Exact thermal-shell rates: corner ratios vs verified reference."""
    fs = make_fs(
        M_theta=4,
        ee=dict(epsilon_bg=EPS_B, nonlinear=False,
                n_xi=24, n_phi=512, n_xi_proj=12),
    )
    K = _kernels.K_table(4, kF=KF, epsilon_bg=EPS_B, kappa=KAPPA)
    gam = _kernels.gamma_linear(K, m_star=M_STAR, T=T0, E_F=E_F)
    r2 = fs.ee_scattering.L_coeff[3, 0, 0].item() / gam[2].item()
    r4 = fs.ee_scattering.L_coeff[7, 0, 0].item() / gam[4].item()
    assert abs(r2 - 1.299) < 0.03  # reference evaluator: 1.2985
    assert abs(r4 - 1.156) < 0.03  # reference evaluator: 1.155
    assert fs.ee_scattering.L_coeff[1, 0, 0] == 0.0  # momentum: exact null projection
    assert fs.ee_scattering.L_coeff[0, 0, 0] == 0.0  # number
    # small genuine odd-m relaxation, positive:
    assert 0.0 <= fs.ee_scattering.L_coeff[5, 0, 0] < 0.3 * fs.ee_scattering.L_coeff[3, 0, 0]


def test_exact_rates_radial_tower():
    """Nr > 1: conservation nulls, PSD, and the hydrodynamic hierarchy."""
    fs = make_fs(
        M_theta=2, Nr=3,
        ee=dict(epsilon_bg=EPS_B, nonlinear=False,
                n_xi=24, n_phi=512, n_xi_proj=12),
    )
    L = fs.ee_scattering.L_coeff.to(torch.float64).cpu()
    # symmetry + PSD:
    for c in range(L.shape[0]):
        assert torch.allclose(L[c], L[c].T, atol=1e-18)
        assert torch.linalg.eigvalsh(L[c]).min() > -1e-18
    # exact nulls in the code's discrete radial measure:
    Ttm = fs.radial.T_to_modes.to(torch.float64).cpu()
    ones_c = Ttm @ torch.ones(3, dtype=torch.float64)
    x_c = Ttm @ fs.radial.xi.to(torch.float64).cpu()
    t_ratio = T0 / E_F
    k_c = Ttm @ torch.sqrt(1 + t_ratio * fs.radial.xi.to(torch.float64).cpu())
    scale = L[3].diag().max()
    assert (L[0] @ ones_c).abs().max() < 1e-12 * scale  # number
    assert (L[0] @ x_c).abs().max() < 1e-12 * scale  # energy
    assert (L[1] @ k_c).abs().max() < 1e-12 * scale  # momentum
    # hierarchy: energy modes fastest, shear slowest nonzero:
    e_energy = torch.linalg.eigvalsh(L[0])[-1]
    e_heat = sorted(torch.linalg.eigvalsh(L[1]).tolist())[1]
    e_shear = sorted(torch.linalg.eigvalsh(L[3]).tolist())[0]
    assert e_energy > e_heat > e_shear > 0


def test_material_integration():
    """FermiSurface.rho_dot with ee: shapes, decay, density conservation."""
    fs = make_fs(
        M_theta=6, tau_p=np.inf,
        ee=dict(epsilon_bg=EPS_B, nonlinear=True,
                n_xi=16, n_phi=256, n_xi_proj=8),
    )
    Nk = fs.angular.N_theta
    torch.manual_seed(0)
    rho = 1e-4 * torch.randn(5, 7, Nk, dtype=fs.v.dtype, device=rc.device)
    rho_dot = fs.rho_dot(rho, 0.0, 0)
    assert rho_dot.shape == rho.shape
    # density (m=0) exactly conserved at every spatial point (the residual is
    # only the machine-precision roundoff of the nodal<->modal transforms; the
    # m=0 modal rate is identically zero):
    n_dot = rho_dot.mean(dim=-1)
    assert n_dot.abs().max() < 1e-14 * rho_dot.abs().max()
    # total free-energy-like norm decays in the linear (small-amplitude)
    # regime, where the PSD linear operator dominates over the cubic (the
    # exact cubic/quadratic are higher order in the deformation amplitude and
    # not individually sign-definite -- they redistribute, not dissipate, the
    # quadratic norm):
    rho_small = 1e-8 * torch.randn(5, 7, Nk, dtype=fs.v.dtype, device=rc.device)
    a_s = fs.to_modes(rho_small)
    a_dot_s = fs.to_modes(fs.rho_dot(rho_small, 0.0, 0))
    assert (a_s * a_dot_s).sum() < 0
    # tau_ee conflict is rejected:
    with pytest.raises(Exception):
        make_fs(tau_ee=1.0, ee=dict(epsilon_bg=EPS_B))


def test_L_blocks_pointwise_vs_reference():
    """L_blocks (init path) against the independent reference evaluator."""
    x_chk = torch.tensor([-2.0, 1.0], dtype=torch.float64)
    common = dict(kF=KF, m_star=M_STAR, T=T0, epsilon_bg=EPS_B, kappa=KAPPA)
    R = _kernels.L_blocks(
        x_nodes=x_chk, psi_coeff=torch.ones(1, 1, dtype=torch.float64),
        m_list=[2], n_xi=16, xi_cut=9.0, n_phi=256, **common,
    )
    w_occ = lambda x: 0.25 / torch.cosh(x / 2) ** 2
    df = lambda x, phi: w_occ(x) * torch.cos(2 * phi) / T0
    fdot = _kernels.exact_collision_reference(
        df, x_chk, torch.zeros(2, dtype=torch.float64), linearize=True,
        n_xi=16, xi_cut=9.0, n_phi=256, chunk=2, **common,
    )
    phidot = fdot * 4 * T0 * torch.cosh(x_chk / 2) ** 2
    assert torch.allclose(R[0, :, 0], -phidot, rtol=1e-10)


def test_unreduced_vs_reduced_reference():
    """Independent, reduction-FREE validation of the kinematic reduction.

    ``exact_collision_reference`` and the production vertices share ONE analytic
    reduction of the collision integral: the momentum delta is resolved in the
    azimuths, giving the Jacobian ``1/(k2 k4 |sin(phi4-phi2)|)``, two ``phi2``
    roots and a van-Hove caustic.  Two codes that both use it cannot catch a
    conceptual error *in* it.  ``unreduced_collision_reference`` shares none of
    it -- it keeps the energy delta as a narrow Gaussian and integrates
    ``(x2, phi2, x3, phi3)`` directly, with no kinematic Jacobian and no
    root-finding (so no caustic).  Agreement therefore tests the reduction
    itself; a wrong Jacobian, root multiplicity or phase-space prefactor would
    disagree at O(1), not at the few-% quadrature level.

    The evaluator is deterministic float64 quadrature (no RNG), so the residuals
    are reproducible across machines.  This uses a fast CI config; the method
    converges to ``exact_collision_reference`` as ``sigma -> 0`` with
    ``n_xi2 ~ 1/sigma`` (demonstrated offline for these GaAs params: ``sigma =
    0.3`` well resolved -> ``< 1 %``; Richardson ``sigma -> 0`` -> ``< 0.4 %``,
    for both the linear and the full nonlinear bracket)."""
    common = dict(kF=KF, m_star=M_STAR, T=T0, epsilon_bg=EPS_B, kappa=KAPPA)
    w_occ = lambda x: 0.25 / torch.cosh(x / 2) ** 2
    x1 = torch.zeros(1, dtype=torch.float64)
    phi1 = torch.tensor([0.3], dtype=torch.float64)  # generic (off-axis) point

    # (1) Linearized operator on a shear (cos 2phi) Fermi-surface mode:
    df = lambda x, phi: w_occ(x) * torch.cos(2 * phi) / T0
    red = _kernels.exact_collision_reference(
        df, x1, phi1, linearize=True, n_xi=32, xi_cut=9.0, n_phi=2048,
        chunk=1, **common)[0].item()
    unr = _kernels.unreduced_collision_reference(
        df, x1, phi1, linearize=True, sigma=0.4, n_phi=128, n_xi3=24,
        xi_cut=9.0, **common)[0].item()
    rel_lin = abs(unr - red) / abs(red)
    assert rel_lin < 0.015, (
        f"linear reduction mismatch: reduced={red:.4e} unreduced={unr:.4e}"
        f" (rel={rel_lin:.2e})")

    # (2) FULL nonlinear bracket at a physical O(1) deformation (Phi = cos 2phi
    # so delta_f = w_occ cos 2phi, |delta_f| <= 0.25, occupations stay in [0,1]).
    # First confirm the nonlinear content is substantial (so this genuinely
    # exercises the cubic/quadratic reduction, not just the linear part), then
    # check the reduction-free value reproduces the reduced one:
    dfn = lambda x, phi: w_occ(x) * torch.cos(2 * phi)
    redL = _kernels.exact_collision_reference(
        dfn, x1, phi1, linearize=True, n_xi=32, xi_cut=9.0, n_phi=2048,
        chunk=1, **common)[0].item()
    redF = _kernels.exact_collision_reference(
        dfn, x1, phi1, linearize=False, n_xi=32, xi_cut=9.0, n_phi=2048,
        chunk=1, **common)[0].item()
    assert abs(redF - redL) / abs(redL) > 0.10, (
        "nonlinear content too small to discriminate the reduction")
    unrF = _kernels.unreduced_collision_reference(
        dfn, x1, phi1, linearize=False, sigma=0.4, n_phi=128, n_xi3=24,
        xi_cut=9.0, **common)[0].item()
    rel_nl = abs(unrF - redF) / abs(redF)
    assert rel_nl < 0.08, (
        f"nonlinear reduction mismatch: reduced={redF:.4e} unreduced={unrF:.4e}"
        f" (rel={rel_nl:.2e})")


def test_a_dot_regression_baseline():
    """The optimized (convolution-form) a_dot reproduces the pre-optimization
    operator bit-for-bit (<= 1e-11).  Baseline saved by the regression harness
    at _scratch_ee/baseline_adot.npz (keys '{M}_{Nr}_a', '{M}_{Nr}_out' built
    with rates='exact', nonlinear=True, n_xi=16, n_phi=256, n_xi_proj=8);
    skipped gracefully if the file is absent."""
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    # repo root is .../qimpy-collisions; the baseline lives under _scratch_ee
    # (test is at src/qimpy/transport/material/fermi_surface/scattering -> 6 up):
    root = os.path.abspath(
        os.path.join(here, "..", "..", "..", "..", "..", "..")
    )
    npz = os.path.join(root, "_scratch_ee", "baseline_adot.npz")
    if not os.path.exists(npz):
        pytest.skip(f"regression baseline absent ({npz})")
    data = np.load(npz)
    for (M, Nr) in ((4, 2), (6, 2)):
        key_a, key_o = f"{M}_{Nr}_a", f"{M}_{Nr}_out"
        if key_a not in data:
            continue
        a = torch.as_tensor(data[key_a])
        out_ref = torch.as_tensor(data[key_o])
        fs = make_fs(
            M_theta=M, Nr=Nr,
            ee=dict(epsilon_bg=EPS_B, nonlinear=True,
                    n_xi=16, n_phi=256, n_xi_proj=8),
        )
        a_dev = a.to(dtype=fs.v.dtype, device=rc.device)
        out = fs.ee_scattering.a_dot(a_dev).to(dtype=out_ref.dtype, device="cpu")
        rel = (out - out_ref).abs().max().item() / out_ref.abs().max().item()
        assert rel <= 1e-11, f"a_dot regression (M={M}, Nr={Nr}): rel={rel:.3e}"


# ---------------------------------------------------------------------------
# Matrix-free (kinematic-generator) nonlinear operator: storage-optimal,
# L-independent representation that must be numerically identical to the dense
# vertex path to quadrature precision.
# ---------------------------------------------------------------------------
def _structured_field(M, Nr, seed=7, scale=1e-2):
    """A random energy-structured (all radial modes) modal field
    ``(n_batch, Nr*dim)`` for the matrix-free vs dense comparisons."""
    dim = 2 * M + 1
    torch.manual_seed(seed)
    return scale * torch.randn(4, Nr * dim, dtype=torch.float64, device=rc.device)


def _surface_field(M, Nr, scale=1e-2):
    """A pure on-Fermi-surface (l=0) modal field with a few harmonics."""
    dim = 2 * M + 1
    a = torch.zeros(4, Nr, dim, dtype=torch.float64, device=rc.device)
    torch.manual_seed(11)
    a[:, 0, :] = scale * torch.randn(4, dim, dtype=torch.float64, device=rc.device)
    return a.reshape(4, Nr * dim)


def test_matrix_free_vs_kernel_reference():
    """The matrix-free a_dot reproduces the dense-vertex kinematics to roundoff.
    The dense cubic/quadratic vertices (``_kernels.cubic_vertex`` /
    ``quadratic_vertex``) are validated against the brute-force reference
    pointwise by ``test_cubic_vertex_*_vs_reference`` /
    ``test_quadratic_vertex_vs_reference``.  Here we build the dense nonlinear
    a_dot inline from those same kernels (with the operator's own radial Galerkin
    + null projection) and check the matrix-free generator path matches it
    term-by-term, transferring the reference guarantee to the production operator
    without a second backend.  Matched quadrature -> agreement to float64
    roundoff (far below the 1e-9 floor)."""
    common = dict(kF=KF, m_star=M_STAR, T=T0, epsilon_bg=EPS_B, kappa=KAPPA)

    def dense_nonlinear(ee, a4):
        """Dense nonlinear a_dot from the validated vertex kernels + the
        operator's Galerkin/null projection (the old dense apply, inline)."""
        fs = ee.fermi_surface
        M = fs.M_theta
        psi_coeff, x_fine, P, Ginv, _ = ee._radial_galerkin(T0)
        psi_coeff, x_fine = psi_coeff.cpu(), x_fine.cpu()
        P, Ginv = P.cpu(), Ginv.cpu()
        kin = dict(M=M, well_width=ee.well_width, n_xi=ee.n_xi,
                   xi_cut=ee.xi_cut, n_phi=ee.n_phi, **common)
        Vc = _kernels.cubic_vertex(x_nodes=x_fine, psi_coeff=psi_coeff, **kin)
        Vq = _kernels.quadratic_vertex(x_nodes=x_fine, psi_coeff=psi_coeff, **kin)
        GP = Ginv @ P
        V_cubic = torch.einsum("lf,fcxaybzd->lcxaybzd", GP, Vc)
        V_quad = torch.einsum("lf,fcxayb->lcxayb", GP, Vq)
        Proj = ee._radial_null_projectors(T0).cpu()
        V_cubic = torch.einsum("cLl,lcxaybzd->Lcxaybzd", Proj, V_cubic)
        V_quad = torch.einsum("cLl,lcxayb->Lcxayb", Proj, V_quad)
        a4 = a4.cpu()
        cub = torch.einsum(
            "lcxaybzd,...xa,...yb,...zd->...lc", V_cubic, a4, a4, a4)
        qd = torch.einsum("lcxayb,...xa,...yb->...lc", V_quad, a4, a4)
        # kernels output the "-Phi_dot" convention; production folds +conv so
        # nl = +Phi_dot_NL -- negate here to mirror it:
        return -(cub + qd).reshape(*a4.shape[:-2], fs.Nr * fs.angular.dim)

    for (M, Nr) in ((3, 1), (3, 2), (4, 2)):
        dim = 2 * M + 1
        fs = make_fs(M_theta=M, Nr=Nr, ee=dict(  # force the matrix-free path
            epsilon_bg=EPS_B, nonlinear=True, n_xi=12, n_phi=128, n_xi_proj=8,
            backend="matrix_free"))
        ee = fs.ee_scattering
        torch.manual_seed(7)
        a = 1e-2 * torch.randn(4, Nr * dim, dtype=fs.v.dtype, device=rc.device)
        a4 = a.reshape(4, Nr, dim)
        nl_mf = ee.a_dot(a) + torch.einsum(
            "cij,...jc->...ic", ee.L_coeff, a4).reshape(4, Nr * dim)
        nl_ref = dense_nonlinear(ee, a4)
        rel = (nl_mf.cpu() - nl_ref).abs().max().item() / nl_ref.abs().max().item()
        assert rel < 1e-9, f"matrix-free vs kernel (M={M}, Nr={Nr}): rel={rel:.2e}"


def test_dense_backend_vs_matrix_free():
    """The dense (precontracted-vertex) and matrix-free backends are the same
    operator: a_dot agrees to roundoff.  Also checks that 'auto' picks dense when
    the vertex fits the storage cap and matrix_free when it would not."""
    common = dict(epsilon_bg=EPS_B, nonlinear=True, n_xi=12, n_phi=128,
                  n_xi_proj=8)
    for (M, Nr) in ((3, 1), (4, 2)):
        dim = 2 * M + 1
        fd = make_fs(M_theta=M, Nr=Nr, ee=dict(backend="dense", **common))
        fm = make_fs(M_theta=M, Nr=Nr, ee=dict(backend="matrix_free", **common))
        assert fd.ee_scattering.backend == "dense"
        assert fm.ee_scattering.backend == "matrix_free"
        torch.manual_seed(7)
        a = 1e-2 * torch.randn(4, Nr * dim, dtype=fd.v.dtype, device=rc.device)
        od, om = fd.ee_scattering.a_dot(a), fm.ee_scattering.a_dot(a)
        rel = (od - om).abs().max().item() / om.abs().max().item()
        assert rel < 1e-9, f"dense vs matrix_free (M={M}, Nr={Nr}): rel={rel:.2e}"
    # 'auto' picks dense when the compact kernel fits, matrix_free when not
    # (the selection-compact kernel is Nr^4 dim^3, so the crossover is at large
    # Nr / large M):
    auto_small = make_fs(M_theta=4, Nr=1, ee=dict(backend="auto", **common))
    auto_big = make_fs(M_theta=32, Nr=8, ee=dict(backend="auto", **common))
    assert auto_small.ee_scattering.backend == "dense"
    assert auto_big.ee_scattering.backend == "matrix_free"


def test_matrix_free_conservation():
    """Criterion 3: the matrix-free nonlinear output annihilates number,
    momentum and energy nulls (post the SAME null projection as the dense path)
    at Nr=1 and Nr>=2."""
    for Nr in (1, 3):
        M = 2
        fs = make_fs(
            M_theta=M, Nr=Nr,
            ee=dict(epsilon_bg=EPS_B,
                    nonlinear=True, backend="matrix_free",
                    n_xi=16, n_phi=256, n_xi_proj=8),
        )
        dim = fs.angular.dim
        Ttm = fs.radial.T_to_modes.to(torch.float64).cpu()
        ones_c = Ttm @ torch.ones(Nr, dtype=torch.float64)
        t_ratio = T0 / E_F
        k_c = Ttm @ torch.sqrt(
            1.0 + t_ratio * fs.radial.xi.to(torch.float64).cpu()
        )
        torch.manual_seed(3)
        a = 1e-2 * torch.randn(4, Nr * dim, dtype=fs.v.dtype, device=rc.device)
        a_lin = -torch.einsum(
            "cij,...jc->...ic", fs.ee_scattering.L_coeff, a.reshape(4, Nr, dim),
        ).reshape(4, Nr * dim)
        nl = (fs.ee_scattering.a_dot(a) - a_lin).reshape(4, Nr, dim)
        scale = nl.abs().max().item()
        num = torch.einsum("bl,l->b", nl[..., 0], ones_c.to(nl.device, nl.dtype))
        assert num.abs().max().item() < 1e-10 * scale, "number"
        if Nr > 1:
            x_c = Ttm @ fs.radial.xi.to(torch.float64).cpu()
            ene = torch.einsum("bl,l->b", nl[..., 0], x_c.to(nl.device, nl.dtype))
            assert ene.abs().max().item() < 1e-10 * scale, "energy"
        mc = k_c.to(nl.device, nl.dtype)
        momx = torch.einsum("bl,l->b", nl[..., 1], mc)
        momy = torch.einsum("bl,l->b", nl[..., 2], mc)
        assert momx.abs().max().item() < 1e-10 * scale, "momentum x"
        assert momy.abs().max().item() < 1e-10 * scale, "momentum y"


def test_matrix_free_storage_flat_in_Nr():
    """The matrix-free generator footprint is independent of the radial depth
    L = Nr (it stores only the field-independent kinematic quadrature, plus the
    tiny psi table), whereas the dense vertex grows steeply with Nr.  Check that
    doubling Nr leaves the generator's dominant arrays unchanged in size."""
    sizes = {}
    for Nr in (2, 4):
        fs = make_fs(
            M_theta=4, Nr=Nr,
            ee=dict(epsilon_bg=EPS_B, nonlinear=True, backend="matrix_free",
                    n_xi=8, n_phi=64, n_xi_proj=6),
        )
        ee = fs.ee_scattering
        gen = sum(
            getattr(ee, f"_mf_{k}").numel()
            for k in ("x4", "dphi2", "dphi4", "Wk")
        )
        psi = ee._mf_psi_coeff.numel()
        sizes[Nr] = (gen, psi)
    # generator quadrature arrays identical across Nr; psi table tiny (~Nr^2):
    assert sizes[2][0] == sizes[4][0], (
        f"generator size changed with Nr: {sizes}"
    )
    assert sizes[4][1] <= 64, f"psi table not tiny: {sizes[4][1]}"


def test_matrix_free_rho_dot_integration():
    """The matrix-free operator drives FermiSurface.rho_dot end-to-end: density
    conserved exactly per spatial point, and the small-amplitude free-energy
    norm decays (the PSD linear operator dominates)."""
    fs = make_fs(
        M_theta=6, tau_p=np.inf,
        ee=dict(epsilon_bg=EPS_B, nonlinear=True, backend="matrix_free",
                n_xi=16, n_phi=256, n_xi_proj=8),
    )
    Nk = fs.angular.N_theta
    torch.manual_seed(0)
    rho = 1e-4 * torch.randn(5, 7, Nk, dtype=fs.v.dtype, device=rc.device)
    rho_dot = fs.rho_dot(rho, 0.0, 0)
    assert rho_dot.shape == rho.shape
    n_dot = rho_dot.mean(dim=-1)
    assert n_dot.abs().max() < 1e-14 * rho_dot.abs().max()
    rho_small = 1e-8 * torch.randn(5, 7, Nk, dtype=fs.v.dtype, device=rc.device)
    a_s = fs.to_modes(rho_small)
    a_dot_s = fs.to_modes(fs.rho_dot(rho_small, 0.0, 0))
    assert (a_s * a_dot_s).sum() < 0


def test_a_dot_nonlinear_signed():
    """END-TO-END SIGNED: the assembled a_dot's cubic channel vs the governing
    equation, sign and magnitude.  The 2026-07 audit found a sign flip at the
    assembly seam that every layer-wise test missed (kernels validated in their
    own -Phi_dot convention, backends against each other, invariants sign-blind);
    this is the only test that crosses that seam.  Pointwise-in-x comparisons are
    NOT valid here: the cubic's cos(2 phi) overlap changes sign across the shell,
    so the reference must be Galerkin-projected exactly like the operator."""
    torch.set_default_dtype(torch.float64)
    fs = make_fs(M_theta=4, Nr=1, ee=dict(epsilon_bg=EPS_B, nonlinear=True,
                                          check_convergence=False))
    ee = fs.ee_scattering
    dim = fs.angular.dim
    amp = 2.0 * T0
    a = torch.zeros(dim, dtype=torch.float64, device=rc.device)
    a[3] = amp                                    # m=2 cosine channel
    a4 = a.reshape(1, dim)
    lin = (-torch.einsum("cij,jc->ic", ee.L_coeff, a4)).reshape(-1)
    cub = 0.5 * (ee.a_dot(a) - ee.a_dot(-a)) - lin
    # scaling sanity: the odd channel is pure cubic (series terminates)
    cub_h = 0.5 * (ee.a_dot(0.5 * a) - ee.a_dot(-0.5 * a)) \
        - (-torch.einsum("cij,jc->ic", ee.L_coeff, 0.5 * a4)).reshape(-1)
    assert abs(float(cub[3] / cub_h[3]) - 8.0) < 1e-6
    # governing-equation reference: eps^3 Richardson, Galerkin-projected
    # (w_eq_RB * conv = 1 -> unit x-weight; Ginv = T0/Gband at Nr=1).
    # Whole reference on CPU inside a device context: the suite's global
    # default-device wrapper otherwise breaks the reference's numpy interop.
    with torch.device("cpu"):
        xg = torch.tensor(np.linspace(-8.0, 8.0, 25), dtype=torch.float64)
        phi1 = torch.tensor(np.linspace(0, 2 * np.pi, 16, endpoint=False),
                            dtype=torch.float64)

        def df(x, phi):
            return (0.25 / torch.cosh(x / 2) ** 2) * (amp * torch.cos(2 * phi)) / T0

        common2 = dict(kF=KF, m_star=M_STAR, T=T0, epsilon_bg=EPS_B, kappa=KAPPA)
        refkw = dict(n_xi=20, xi_cut=10.0, n_phi=128, chunk=2, **common2)
        X, P = torch.meshgrid(xg, phi1, indexing="ij")
        Xf, Pf = X.reshape(-1), P.reshape(-1)
        fd = {s: _kernels.exact_collision_reference(
                (lambda s_: (lambda x, phi: df(x, phi) * s_))(s),
                Xf, Pf, linearize=False, **refkw).reshape(len(xg), len(phi1))
              for s in (1.0, 2.0, -1.0, -2.0)}
        C3 = (0.5 * (fd[2.0] - fd[-2.0]) - (fd[1.0] - fd[-1.0])) / 6.0
        ov = 2.0 * (C3 * torch.cos(2 * phi1)[None, :]).mean(1)
        dx = float(xg[1] - xg[0])
        Gband = float((0.25 / torch.cosh(xg / 2) ** 2).sum() * dx)
        cub2_ref = float(ov.sum() * dx / Gband * T0)
    ratio = float(cub[3].cpu()) / cub2_ref
    assert ratio > 0, f"nonlinear sign FLIPPED vs governing equation: {ratio:.3f}"
    assert 0.5 < ratio < 1.6, f"cubic magnitude off vs reference: {ratio:.3f}"


def test_cartesian_local_te_rates():
    """A heated cell (recovered T_e = 2T) relaxes its ripple (T_e/T)^2 = 4x
    faster with local_te_rates=True than False (identical state; the ratio
    isolates the per-cell rescale exactly).  Lives here for the module's
    no-default-device fixture (builds an EEScattering)."""
    from qimpy.transport.material import FermiSurface
    torch.set_default_dtype(torch.float64)
    T = 0.02
    ee = dict(epsilon_bg=12.9, nonlinear=False, on_shell=False,
              check_convergence=False, n_xi=8, n_phi=64, n_xi_proj=6)
    pg = ProcessGrid(rc.comm, "rk", (-1, 1))

    def mk(local):
        return FermiSurface(
            kF=1.0, vF=1.5, M_theta=6, Nr=1, T=T, xi_max=6.0,
            cartesian=dict(dk=T / (3.0 * 1.5), local_te_rates=local),
            ee_scattering=dict(ee), process_grid=pg)

    outs = {}
    for tag, local in (("on", True), ("off", False)):
        fs = mk(local)
        rep = fs.representation
        eps = rep.eps_k
        f0 = rep._f0_lab
        f_hot = torch.special.expit(-(eps - fs.mu) / (2 * T))
        th = torch.atan2(rep.k[:, 1], rep.k[:, 0])
        ripple = 1e-6 * f_hot * (1 - f_hot) / (2 * T) * torch.cos(2 * th)
        rho = (f_hot - f0 + ripple)[None]
        outs[tag] = fs.rho_dot(rho, 0.0, 0)
    ratio = float(outs["on"].abs().max() / outs["off"].abs().max())
    assert 3.5 < ratio < 4.5, f"(T_e/T)^2 rescale broken: ratio={ratio:.2f}"


def test_local_te_ensemble_exact():
    """EXACT local-T_e (the factorization C^(d) = T_e^2 T^(1-d) M_d(t_e)): the
    ensemble-evaluated linear rate at te = 2T matches an operator FRESHLY BUILT
    at 2T (for d=1 the rate is convention-free, directly comparable at Nr=1),
    reproduces the material operator at te = T, and beats the scalar
    (T_e/T)^2 fallback, which misses the O(t) shape drift of M_1."""
    from qimpy.transport.material import FermiSurface
    torch.set_default_dtype(torch.float64)
    pg = ProcessGrid(rc.comm, "rk", (-1, 1))
    T = 0.02
    ee_base = dict(epsilon_bg=12.9, nonlinear=False, on_shell=False,
                   check_convergence=False, n_xi=16, n_phi=128, n_xi_proj=8)

    def mk(Tm, local=None):
        eed = dict(ee_base)
        if local:
            eed["local_te"] = local
        return FermiSurface(kF=1.0, vF=1.5, M_theta=4, Nr=1, T=Tm, xi_max=6.0,
                            ee_scattering=eed, process_grid=pg)

    fs = mk(T, local=dict(n_nodes=6, te_fac_min=0.5, te_fac_max=3.0))
    fs2 = mk(2 * T)
    dim = fs.angular.dim
    a = torch.zeros(1, dim, dtype=torch.float64, device=rc.device)
    a[0, 3] = 1.0                                       # m=2 channel
    te = torch.full((1,), 2 * T, dtype=torch.float64, device=rc.device)
    gam_ens = -float(fs._modal_collision(a, te=te)[0, 3])
    gam_fresh = float(fs2.ee_scattering.L_coeff[3, 0, 0])
    rel = abs(gam_ens - gam_fresh) / gam_fresh
    assert rel < 5e-3, f"ensemble(2T) vs fresh build: rel={rel:.1e}"
    # the scalar fallback misses the shape drift -- it must be WORSE:
    gam_scalar = 4.0 * float(fs.ee_scattering.L_coeff[3, 0, 0])
    assert abs(gam_scalar - gam_fresh) / gam_fresh > rel
    # and at te = T the ensemble reproduces the material operator:
    teT = torch.full((1,), T, dtype=torch.float64, device=rc.device)
    gam_T = -float(fs._modal_collision(a, te=teT)[0, 3])
    gam_0 = float(fs.ee_scattering.L_coeff[3, 0, 0])
    assert abs(gam_T - gam_0) / gam_0 < 5e-3


def test_spin_degeneracy_scales_the_rate() -> None:
    """g_s enters the golden rule as an overall factor on the collision
    integral, so every rate must be exactly linear in it -- and the SCREENING
    constant must use the same degeneracy (kappa = g_s m*/eps_bg), since kappa
    is 2 pi e^2 nu_2D/eps_bg with the spin-degenerate 2D DOS.  Before this was
    a parameter the two were inconsistent: kappa was built with g_s = 2 while
    the collision prefactor carried g_s = 1."""
    ee1 = dict(epsilon_bg=EPS_B, kappa=KAPPA, nonlinear=False, g_s=1.0)
    ee2 = dict(epsilon_bg=EPS_B, kappa=KAPPA, nonlinear=False, g_s=2.0)
    L1 = make_fs(M_theta=2, Nr=1, ee=ee1).ee_scattering.L_coeff[3, 0, 0].item()
    L2 = make_fs(M_theta=2, Nr=1, ee=ee2).ee_scattering.L_coeff[3, 0, 0].item()
    assert abs(L2 / L1 - 2.0) < 1e-12, f"rate not linear in g_s: {L2/L1}"
    # default is the unpolarized 2DEG, and kappa follows the same degeneracy
    fs = make_fs(M_theta=2, Nr=1, ee=dict(epsilon_bg=EPS_B, nonlinear=False))
    assert fs.ee_scattering.g_s == 2.0
    assert abs(fs.ee_scattering.kappa - 2 * M_STAR / EPS_B) < 1e-15


def test_L1_finite_difference_vs_reference() -> None:
    """The LINEARIZED bracket (eq 6) against a finite difference of the
    UNEXPANDED B - F.

    Why this test exists.  `L_blocks`, `exact_collision_reference(
    linearize=True)` and `unreduced_collision_reference` all implement the SAME
    hand-derived formula  W (Phi3 + Phi4 - Phi1 - Phi2),  W = f1 f2 (1-f3)(1-f4).
    It is common-mode across every evaluator in the repository, so
    `test_L_blocks_pointwise_vs_reference` -- which compares two of them --
    cannot see an error in the linearization itself.  Yet L1 IS the production
    linear operator, hence tau_ee, l_ee and every linear transport coefficient.

    Here the reference is run with linearize=False (raw B - F) and the linear
    term is extracted by the amplitude stencil, L1 = (8 o1 - o2)/6, which is
    EXACT in exact arithmetic because B - F terminates at cubic order.  A wrong
    hand derivation shows up immediately; only float cancellation limits it.
    """
    # AMPLITUDE MATTERS.  The stencil is exact in exact arithmetic, but it
    # extracts L1 by cancelling the C3 contributions, and C3/L1 ~ A^2.  The
    # natural field w_occ(x)/T0 is delta-f ~ 1.9e4 -- absurdly nonlinear -- and
    # at that amplitude the cancellation alone gives 1.2e-3.  Measured scan
    # (rel vs amplitude A): 1e0 -> 1.17e-3, 1e-1 -> 4.0e-7, 1e-2 -> 2.2e-9,
    # 1e-3 -> 1.5e-12, 1e-4 -> 3.0e-15, 1e-5 -> 3.5e-15; flat in n_phi.  Falls
    # as A^2 and floors at round-off => cancellation, NOT a derivation error.
    AMP = 1e-4
    n_xi, xi_cut, n_phi = 16, 9.0, 256
    x_chk = torch.tensor([-2.0, 1.0], dtype=torch.float64)
    common = dict(kF=KF, m_star=M_STAR, T=T0, epsilon_bg=EPS_B, kappa=KAPPA)
    R = _kernels.L_blocks(
        x_nodes=x_chk, psi_coeff=torch.ones(1, 1, dtype=torch.float64),
        m_list=[2], n_xi=n_xi, xi_cut=xi_cut, n_phi=n_phi, **common,
    )
    w_occ = lambda x: 0.25 / torch.cosh(x / 2) ** 2
    df = lambda x, phi: AMP * w_occ(x) * torch.cos(2 * phi) / T0
    L1 = _reference_eps_terms(
        df, x_chk, torch.zeros(2, dtype=torch.float64), n_phi, order=1,
        n_xi=n_xi, xi_cut=xi_cut,
    )
    phidot = L1 * 4 * T0 * torch.cosh(x_chk / 2) ** 2 / AMP
    rel = float((R[0, :, 0] + phidot).abs().max()
                / phidot.abs().max().clamp(min=1e-300))
    assert rel < 1e-12, f"hand-derived L1 disagrees with FD of raw B-F: {rel:.2e}"

    # the stencil must also reproduce the OTHER orders on the same data, i.e.
    # L1 + Q2 + C3 == the full bracket at unit amplitude (no missing piece)
    Q2 = _reference_eps_terms(df, x_chk, torch.zeros(2, dtype=torch.float64),
                              n_phi, order=2, n_xi=n_xi, xi_cut=xi_cut)
    C3 = _reference_eps_terms(df, x_chk, torch.zeros(2, dtype=torch.float64),
                              n_phi, order=3, n_xi=n_xi, xi_cut=xi_cut)
    full = _kernels.exact_collision_reference(
        df, x_chk, torch.zeros(2, dtype=torch.float64), linearize=False,
        n_xi=n_xi, xi_cut=xi_cut, n_phi=n_phi, chunk=2, **common,
    )
    closes = float((full - (L1 + Q2 + C3)).abs().max()
                   / full.abs().max().clamp(min=1e-300))
    assert closes < 1e-10, f"L1+Q2+C3 != B-F: {closes:.2e}"
