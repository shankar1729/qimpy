"""Generalized harness (v2): ARBITRARY multi-mode inputs, full cos+sin projection,
modal -> field reconstruction, and conserved moments.

v1 (`q_common.py`) is frozen: single cosine input mode, cosine-only output
harmonics.  That was enough to settle the Q(1,4) question but it leaves three
things structurally untestable:

  * the nonlinear vertices were only ever evaluated as Q(a, a) and C(a, a, a)
    with ONE mode, so the unordered-pair / unordered-triple packing in
    `_build_dense_vertices` ran only in its DIAGONAL branch (multiplicity 1).
    The multiplicity-2 pairs and the multiplicity-3 and -6 triples -- exactly
    where a permutation bug would live -- were never exercised;
  * sin harmonics were never fed in, so the real<->complex fold and the
    Hermitian completion `Fhat[..., :M] = Fhalf[..., 1:].flip(-1).conj()` ran
    only on inputs whose imaginary part vanishes identically;
  * only modal COEFFICIENTS were compared, never the reconstructed field, so
    there is no statement that the truncated operator CONVERGES to eq (1).

Angular basis convention (matches FermiSurface):  e_0 = 1,
e_{2m-1} = cos(m phi), e_{2m} = sin(m phi);  dim = 2 M + 1.
Modal field:  Phi(xi, phi) = sum_{l,c} a_{l,c} psi_l(xi) e_c(phi),
              delta_f = w_eq Phi,  w_eq = 0.25 sech^2(xi/2) / T.
"""
import numpy as np
import torch

M_STAR, EPS_B, KF, T0 = 0.067, 12.9, 7.5e-3, 1.33e-5
E_F = 0.5 * KF**2 / M_STAR
KAPPA = 2 * M_STAR / EPS_B


def w_occ(x):
    return 0.25 / torch.cosh(x / 2) ** 2


def psi_eval(x, psi_coeff):
    """psi_l(x) in the model's OWN hybrid feature basis {1, x, v^2, v, ...}."""
    Nr = psi_coeff.shape[1]
    v = torch.tanh(0.5 * x)
    cols = [torch.ones_like(x), x]
    pe, po = 2, 1
    while len(cols) < Nr:
        cols.append(v ** pe); pe += 2
        if len(cols) < Nr:
            cols.append(v ** po); po += 2
    return torch.stack(cols[:Nr], dim=-1) @ psi_coeff


def ang_index(m, kind="cos"):
    if m == 0:
        return 0
    return 2 * m - 1 if kind == "cos" else 2 * m


def ang_eval(m, kind, phi):
    if m == 0:
        return torch.ones_like(phi)
    return torch.cos(m * phi) if kind == "cos" else torch.sin(m * phi)


# The input is specified in the RADIAL FEATURE basis {1, xi, v^2, v, ...},
# v = tanh(xi/2) -- NOT in the psi basis.  This matters for the truncation
# ladder: psi_l comes out of `_radial_galerkin` and therefore DEPENDS ON Nr, so
# a fixed modal vector is a DIFFERENT physical field at each Nr and the ladder
# would be comparing apples to oranges.  A feature-space input with p < 2 is
# exactly representable at every Nr >= 2, so the input field is identical across
# the whole ladder and what the ladder measures is purely the OUTPUT truncation.
FEATS = {
    0: lambda x: torch.ones_like(x),
    1: lambda x: x,
    2: lambda x: torch.tanh(x / 2) ** 2,
    3: lambda x: torch.tanh(x / 2),
}


def field_from_features(feats, T=T0):
    """feats = [(p, m, kind, amp), ...] -> delta_f(x, phi) in OCCUPATION units.
    Basis-independent: this is the object BOTH reference evaluators see.

    T MUST be the run temperature.  w_eq = w_occ / T, so hard-coding the module
    constant here silently rescales the field whenever T != T0 -- the reference
    then sees a different physical Phi from the one production's coefficients
    encode, and the comparison is invalid in a way no resolution knob can fix."""
    def df(x, p_):
        out = torch.zeros_like(x)
        for pi, m, kind, amp in feats:
            out = out + FEATS[pi](x) * (amp * ang_eval(m, kind, p_))
        return w_occ(x) * out / T         # w_eq = w_occ / T
    return df


def modes_from_features(feats, psi_coeff, dim, device=None):
    """Same field, expressed EXACTLY in the operator's own (Nr, dim) basis.
    psi_l(x) = sum_p psi_coeff[p, l] phi_p(x)  =>  a = psi_coeff^{-1} c."""
    Nr = psi_coeff.shape[1]
    inv = torch.linalg.inv(psi_coeff)                     # (l, p)
    a = torch.zeros(Nr * dim, dtype=torch.float64)
    for pi, m, kind, amp in feats:
        if pi >= Nr:
            raise ValueError(f"feature {pi} not representable at Nr = {Nr}")
        for l in range(Nr):
            a[l * dim + ang_index(m, kind)] += float(inv[l, pi]) * amp
    return a.to(device) if device is not None else a


def scale_features(feats, peak_df, xi_cut=9.0, T=T0):
    """Rescale so peak |delta_f| = peak_df on a dense probe (Pauli-safe <~ 0.4)."""
    xg = torch.linspace(-xi_cut, xi_cut, 241, dtype=torch.float64)
    ph = torch.linspace(0.0, 2 * np.pi, 97, dtype=torch.float64)[:-1]
    X, P = torch.meshgrid(xg, ph, indexing="ij")
    cur = float(field_from_features(feats, T)(X, P).abs().max())
    s = peak_df / cur
    return [(p, m, k, a * s) for p, m, k, a in feats]


# ---------------------------------------------------------------- projection
def grids(nx, xi_grid, n_az):
    xg = torch.tensor(np.linspace(-xi_grid, xi_grid, nx), dtype=torch.float64)
    ph = torch.tensor(np.linspace(0.0, 2 * np.pi, n_az, endpoint=False),
                      dtype=torch.float64)
    X, P = torch.meshgrid(xg, ph, indexing="ij")
    return xg, ph, X.reshape(-1), P.reshape(-1)


def orders_from_stencil(fd):
    """B - F terminates at cubic and its order-0 part vanishes on shell:
        o_s = [F(s) - F(-s)]/2 = s L + s^3 C
        L = (8 o1 - o2)/6,  Q = [F(1) + F(-1)]/2,  C = (o2 - 2 o1)/6"""
    o1 = 0.5 * (fd[1.0] - fd[-1.0])
    o2 = 0.5 * (fd[2.0] - fd[-2.0])
    return {"L": (8.0 * o1 - o2) / 6.0,
            "Q": 0.5 * (fd[1.0] + fd[-1.0]),
            "C": (o2 - 2.0 * o1) / 6.0}


def project(arr, xg, ph, psi_coeff, M, T=T0):
    """fdot field (nx, n_az) -> modal (Nr, dim) in the operator's own convention.

    EXACT in the angle (uniform DFT, both parities) provided n_az > 2 M; the
    radial part is a trapezoid Galerkin against psi_n under the w_eq measure --
    spectrally accurate for this exponentially-decaying analytic integrand.
    """
    nx, n_az = len(xg), len(ph)
    dim = 2 * M + 1
    psi_g = psi_eval(xg, psi_coeff)
    dx = float(xg[1] - xg[0])
    W = w_occ(xg)
    Ginv = torch.linalg.inv(torch.einsum("xn,xm,x->nm", psi_g, psi_g, W) * dx)
    cols = [torch.ones_like(ph) / n_az]
    for m in range(1, M + 1):
        cols += [2.0 / n_az * torch.cos(m * ph), 2.0 / n_az * torch.sin(m * ph)]
    Wm = torch.stack(cols, dim=1)                       # (n_az, dim)
    Am = arr.reshape(nx, n_az) @ Wm
    Phi = Am * (4 * T * torch.cosh(xg / 2) ** 2)[:, None]    # fdot -> Phi_dot
    return torch.einsum("nm,xm,x,xk->nk", Ginv, psi_g, W * dx, Phi)


def reconstruct(a, xg, ph, psi_coeff, M, T=T0):
    """modal (Nr, dim) Phi_dot coefficients -> fdot field (nx, n_az).

    The inverse of `project` in the continuum; comparing the RECONSTRUCTED field
    to the reference field is the only comparison that is meaningful across
    different truncations (M, Nr), because modal coefficients live in different
    spaces."""
    psi_g = psi_eval(xg, psi_coeff)                      # (nx, Nr)
    cols = [torch.ones_like(ph)]
    for m in range(1, M + 1):
        cols += [torch.cos(m * ph), torch.sin(m * ph)]
    E = torch.stack(cols, dim=1)                         # (n_az, dim)
    Phi = torch.einsum("xl,lc,pc->xp", psi_g, a, E)
    return Phi * (w_occ(xg) / T)[:, None]                # Phi_dot -> fdot


# --------------------------------------------------------------- invariants
def moments(arr, xg, ph, t=None):
    """Conserved moments of an fdot field, in the measure d^2k = m* T dx dphi.

    Returns (N, Px, Py, E) up to a common positive constant -- all four must
    vanish for eq (1) REGARDLESS of any projector, which is what makes this
    non-tautological: the existing conservation tests contract against a
    line-for-line copy of the projector's own null covectors.
    """
    t = T0 / E_F if t is None else t
    nx, n_az = len(xg), len(ph)
    a = arr.reshape(nx, n_az)
    dx = float(xg[1] - xg[0])
    dphi = 2 * np.pi / n_az
    k = torch.sqrt(torch.clamp(1.0 + t * xg, min=0.0))    # |k| / kF
    w = dx * dphi
    N = (a.sum(1) * w).sum()
    Px = ((a * torch.cos(ph)[None, :]).sum(1) * k * w).sum()
    Py = ((a * torch.sin(ph)[None, :]).sum(1) * k * w).sum()
    E = ((a.sum(1)) * xg * w).sum()
    scale = (a.abs().sum() * w)
    return (torch.stack([N, Px, Py, E]),
            torch.stack([N, Px, Py, E]) / scale.clamp_min(1e-300))
