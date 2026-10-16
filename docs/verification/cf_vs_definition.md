# C[f] verified against eq (1) — full benchmark record

**2026-08-03/04. Branch `dg-reconstruct`, commit `4453efe2`. GaAs 2DEG of the notes:
kF = 7.5e-3, m\* = 0.067, ε_bg = 12.9, T = 1.33e-5, κ = 2m\*/ε_bg, g_s = 2 (atomic units).
E_F = 4.198e-4, t = T/E_F = 0.031683.**

**Verdict: the production e-e collision operator reproduces the Boltzmann definition.
The previously reported "20 % error in the quadratic vertex" was an error in the
reduction-free *reference*, and the σ-ladder used to certify it was structurally
incapable of detecting it. That claim is retracted.**

---

## 0. Provenance and what is where

| item | location |
|---|---|
| code changes | committed as `4453efe2` on `dg-reconstruct`, **on the shelved instance `qimpy-gpu`** (`~/qimpy`) |
| patched sources (copies) | `patched_source/{_kernels.py, _ee.py, test_ee.py}` in this folder |
| benchmark harness | `probes/` in this folder |
| raw stdout logs | **not retained** — the instance was already shutting down when I tried to pull them. Every number below is transcribed from the run output; each is reproducible with the commands given. |
| superseded harnesses | `../lqc_verify.py`, `../q_sigma_ladder.py` — these produced the **retracted** 20 % result. Kept for the record; do not reuse without reading §2. |

All instances shelved.

---

## 1. The setup common to every number

One input field, one projection, used identically by every evaluator:

    Phi_in = psi_{n_in}(xi) cos(m_in phi),   n_in = 0, m_in = 2
    delta_f = w_eq Phi,   w_eq = 0.25 sech^2(xi/2T) / T
    amplitude fixed so peak |delta_f| = 0.30   =>  a = 4.36546545543638e-3
    M_theta = 6, Nr = 3  (13 angular x 3 radial modes)

Orders are separated by the **signed amplitude stencil**, exact because `B − F` is a
polynomial in the deviations that terminates at cubic and whose order-0 part vanishes
on shell:

    o_s = [F(s) − F(−s)]/2 = s L + s^3 C
    L = (8 o1 − o2)/6      Q = [F(1) + F(−1)]/2      C = (o2 − 2 o1)/6

Angular extraction is an **exact DFT**: a single cos(2φ) input generates a closed,
finite harmonic set — L at m = 2, Q at m ∈ {0,4}, C at m ∈ {2,6} — so a uniform
azimuth grid resolves every coefficient with no quadrature error. Radial extraction is
a trapezoid Galerkin against the model's own ψ_n under the w_eq measure.

**The selection rules come out exactly** in the reduction-free reference: every entry
outside the sets above is ~1e-23 (numerical zero). That alone validates the amplitude
stencil and the harmonic bookkeeping.

### The three evaluators

| name | shares with production |
|---|---|
| **production** | everything — the assembled operator the solver integrates |
| `exact_collision_reference` | the kinematic **reduction** only (same azimuth roots, same `_beta_grid`). No vertex algebra, no modal assembly, no radial Galerkin build, independent quadrature. |
| `unreduced_collision_reference` | **nothing**. Momentum delta resolved trivially (k4 = k1+k2−k3), energy delta as a nascent Gaussian of width σ, direct 4-D (x2,φ2,x3,φ3) quadrature. No Jacobian, no roots. |

---

## 2. Root cause of the retracted 20 %

### 2.1 Why the σ-ladder lied

The σ-ladder was run at **fixed nodes-per-σ** — n_xi2 = 240, 320, 480, 640 at
σ = 0.40, 0.30, 0.20, 0.15, i.e. all four at exactly 0.188 σ spacing. That is the
*correct* way to keep the Gaussian resolved as σ shrinks, and it is precisely what
makes every error on the **other** axes σ-independent by construction. The ratio was
flat because it was stably wrong.

> **Flatness in the one knob you varied is not convergence. Ask which axis your ladder
> is blind to.**

### 2.2 The mechanism

Having no Jacobian *relocates* the caustic; it does not remove it. After the fine x2
rule integrates the Gaussian across the shell, what survives is the reciprocal slope

    1/|∂e/∂x2| = 1 / |1 − k4 cos(φ4 − φ2) / k2|

which degenerates on the collinear locus **k2 ∥ k4** — the same van-Hove configuration
the reduced form carries as `1/|sin(φ4 − φ2)|`. It is not a true singularity (the
Gaussian stops localizing once |∂e/∂x2| < σ/X, X = the x2 range) but it leaves a peak
in the **azimuths** of height ~X/σ and width ~√(2σ/X) ≈ 0.18 rad, on a grid that is
**uniform** and shared between φ2 and φ3, at only n_phi = 112 → Δφ = 0.056 rad →
**3.3 nodes across the peak**.

The peak's *area* goes as √σ · (1/√σ), i.e. it is nearly σ-independent. Hence the flat
signature.

### 2.3 Measured — the reference's own n_phi ladder

Q(n=1, m=4), σ = 0.30, xi_cut = 9, n_xi2 = 320, n_xi3 = 20, NX = 13, N_AZ = 5:

| n_phi | Q(0,4) | **Q(1,4)** | Q(2,4) | nodes across peak |
|---|---|---|---|---|
| 112 | 8.199492e-11 | **5.091069e-10** | −3.686981e-11 | 3.3 |
| 224 | 1.186870e-10 | **5.970999e-10** | −1.547056e-11 | 6.5 |
| 448 | 1.193894e-10 | **6.104017e-10** | −1.167335e-11 | 13.0 |
| 896 | 1.189528e-10 | **6.108617e-10** | −1.167796e-11 | 26 |

**18 % low at n_phi = 112, converged by ~448.** The same n_phi = 112 reproduces the
*linear* rate to ~1 %. Q is ~10× more sensitive because its coefficients are first
differences of f0 between legs, which weight the collinear region far more than L1's
W or the f0-free C3.

Other reference axes at n_phi = 448, σ = 0.30 (all Q(1,4)):

| axis | value | shift |
|---|---|---|
| baseline (n_xi3 = 20, xi_cut = 9) | 6.104017e-10 | — |
| n_xi3 = 40 | 6.10489e-10 | **+0.014 %** |
| xi_cut = 12 (n_xi2 = 427) | 6.13365e-10 | +0.49 % |

σ at the converged n_phi = 896, xi_cut = 9:

| σ | n_xi2 | Q(1,4) |
|---|---|---|
| 0.30 | 320 | 6.108617e-10 |
| 0.15 | 640 | 6.124090e-10 |
| Richardson σ→0 | | **6.12925e-10** (drift 0.25 % — now genuinely asymptotic) |

---

## 3. The decisive pointwise measurement: the reduction is correct to 0.10 %

`probes/q_reduction_probe.py`. One point (x1 = 0, φ1 = 0.3), linearized bracket on a
cos(2φ) shear mode, **no projection at all**. Reduced value (n_xi = 48, n_phi = 2048):
**−5.852961e-03**.

| σ | n_phi = 128 (old fixed) | rel | n_phi = 0 (auto) | rel |
|---|---|---|---|---|
| 0.40 | −5.823798e-03 | 0.498 % | −6.005012e-03 | 2.598 % |
| 0.30 | −5.727348e-03 | 2.146 % | −5.942262e-03 | 1.526 % |
| 0.20 | −5.694218e-03 | 2.712 % | −5.895332e-03 | 0.724 % |
| 0.15 | −5.726365e-03 | 2.163 % | −5.879822e-03 | 0.459 % |
| **Richardson (0.4, 0.2)** | −5.651025e-03 | **3.450 %** | −5.858772e-03 | **0.099 %** |
| **Richardson (0.3, 0.15)** | −5.726037e-03 | 2.169 % | −5.859009e-03 | **0.103 %** |

At the coarse n_phi the σ trend is **not even monotone** and Richardson **extrapolates
the contaminant**, landing worse (3.45 %) than the raw σ = 0.4 point (0.50 %) — and
0.50 % is what the old test asserted against, i.e. it was passing by cancellation.
Resolved, the trend is clean O(σ²) and the two independent Richardson pairs agree with
each other to **0.004 %**.

⇒ The azimuth roots, the `1/(k2 k4 |sin(φ4−φ2)|)` Jacobian, the van-Hove excision and
the phase-space prefactor are all correct to 0.1 %.

---

## 4. Three-way comparison, all 13 channels

Matched xi_cut = 9, NX = 13, identical projection. Production = dense backend,
n_xi = 16, n_phi = 254, n_xi_proj = 96 (the auto values), cubic enabled.
Reduced = `exact_collision_reference` n_xi = 96, n_phi = 1024.
Reduction-free = `unreduced_collision_reference` n_phi = 448, σ = 0.30, N_AZ = 14.

| ch | production | reduced | reduction-free | prod−reduced, % of order peak | prod−red.free, % of peak |
|---|---|---|---|---|---|
| L(0,2) | −2.16770e-09 | −2.17639e-09 | −2.20000e-09 | 0.40 | 1.48 |
| L(1,2) | −1.90885e-10 | −1.84521e-10 | −1.82188e-10 | 0.29 | 0.40 |
| L(2,2) | −6.47600e-10 | −6.90649e-10 | −6.64578e-10 | 1.98 | 0.78 |
| Q(2,0) | −6.94542e-11 | −7.07239e-11 | −5.80149e-11 | 0.21 | 1.87 |
| Q(0,4) | +1.11236e-10 | +1.15149e-10 | +1.18961e-10 | 0.64 | 1.26 |
| **Q(1,4)** | **+6.18440e-10** | **+6.11514e-10** | **+6.09931e-10** | **1.13** | **1.39** |
| Q(2,4) | −2.99427e-12 | −2.59528e-12 | −1.17332e-11 | 0.07 | 1.43 |
| C(0,2) | −1.51594e-10 | −1.67858e-10 | −1.65936e-10 | **2.48** | **2.19** |
| C(1,2) | +5.21595e-11 | +4.94184e-11 | +4.58347e-11 | 0.42 | 0.96 |
| C(2,2) | −6.56755e-10 | −6.56236e-10 | −6.22680e-10 | 0.08 | 5.19 |
| C(0,6) | +4.45336e-11 | +4.17989e-11 | +4.29039e-11 | 0.42 | 0.25 |
| C(1,6) | +3.70249e-11 | +3.64917e-11 | +3.68375e-11 | 0.08 | 0.03 |
| C(2,6) | −6.21616e-11 | −6.34544e-11 | −5.00707e-11 | 0.20 | 1.84 |

Order peaks used: |L| = 2.176e-9, |Q| = 6.115e-10, |C| = 6.562e-10.
**Max residual: 2.5 % of peak vs the reduced brute force, 5.2 % vs the definition.**
(The 5.19 % on C(2,2) is a *reference-vs-reference* difference — the reduction-free
value there is still moving with n_phi; 224→448 shifted it 1.4 %.)

The two production backends are **bit-identical**: dense and matrix-free both give
Q(1,4) = +6.180242e-10 (at xi_cut = 10) — two independent implementations of the
vertex apply agreeing to the last digit.

---

## 5. Convergence ladders

### 5.1 Production (dense, cubic zeroed for speed; the even stencil removes it exactly)
xi_cut = 10 unless noted. Verified: zeroing the cubic leaves Q bit-identical.

| n_xi | n_phi | xi_cut | n_xi_proj | Q(0,4) | Q(1,4) | Q(2,4) |
|---|---|---|---|---|---|---|
| 16 | 254 | 10 | 96 | 1.133095e-10 | 6.180242e-10 | −5.276799e-12 |
| 32 | 254 | 10 | 96 | 1.141027e-10 | 6.214916e-10 | −4.597100e-12 |
| 48 | 254 | 10 | 96 | 1.115792e-10 | 6.229655e-10 | +1.203004e-13 |
| 16 | 508 | 10 | 96 | 1.147377e-10 | 6.155543e-10 | −5.153303e-12 |
| 16 | 1016 | 10 | 96 | 1.217853e-10 | 6.203896e-10 | −4.787068e-12 |
| 16 | 254 | 14 | 96 | 1.247618e-10 | 6.008902e-10 | −1.143988e-11 |
| 16 | 254 | 10 | 160 | 9.714171e-11 | 6.164087e-10 | +1.529384e-11 |
| 16 | 254 | 9 | 96 | 1.112362e-10 | 6.184401e-10 | −2.994267e-12 |
| 16 | 254 | 12 | 96 | 1.152991e-10 | 6.167839e-10 | +1.990115e-12 |

**Spread ±1.8 % about ~6.16e-10 — converged.**

### 5.2 Production with the cubic enabled, xi_cut = 9

| n_xi | n_phi | n_xi_proj | build | Q(1,4) | L(0,2) | **C(0,2)** |
|---|---|---|---|---|---|---|
| 16 | 254 | 96 | 10.1 min | 6.184401e-10 | −2.167698e-09 | **−1.515943e-10** |
| 16 | 508 | 96 | 16.3 min | 6.133114e-10 | −2.168077e-09 | **−1.568334e-10** |
| 16 | 1016 | 96 | 33.0 min | 6.159699e-10 | −2.167745e-09 | **−1.619591e-10** |
| 24 | 254 | 96 | 21.6 min | 6.236478e-10 | −2.169997e-09 | **−1.630917e-10** |
| 16 | 254 | 160 | 16.2 min | 6.174408e-10 | −2.168969e-09 | **−1.589399e-10** |
| **24** | **508** | **160** | 63.6 min | 6.167973e-10 | −2.168345e-09 | **−1.573483e-10** |

Note the last row: refining all three axes together does **not** give the sum of the
individual shifts — they partly **cancel**.

### 5.3 `exact_collision_reference`, xi_cut = 9, NX = 13

| n_xi | n_phi | L(0,2) | Q(1,4) | C(0,2) | C(2,2) |
|---|---|---|---|---|---|
| 24 | 1024 | −2.17472e-09 | +6.06811e-10 | −1.65863e-10 | −6.51297e-10 |
| 48 | 1024 | −2.17642e-09 | +6.12192e-10 | −1.67404e-10 | −6.49034e-10 |
| 96 | 1024 | −2.17639e-09 | +6.11514e-10 | −1.67858e-10 | −6.56236e-10 |
| 48 | 2048 | −2.17623e-09 | +6.10474e-10 | −1.62889e-10 | −6.52906e-10 |
| 48 | 1024 (NX = 25) | −2.17296e-09 | +6.13292e-10 | −1.66624e-10 | −6.73739e-10 |

Leading channels converged to ≤1 %. The NX row measures the **harness** (the 13-point
trapezoid Galerkin), not either evaluator: it moves L(2,2) by 3.2 % and C(2,2) by
3.8 %, which is the floor of any projected comparison at this grid.

---

## 6. Test suite (after the fixes) — 115 passed, 1 skipped

| test | measured |
|---|---|
| `test_unreduced_vs_reduced_reference` (linear) | **0.065 %** |
| `test_unreduced_vs_reduced_reference` (full nonlinear bracket) | **1.406 %** |
| `test_full_Cf_vs_unreduced_definition` m = 2 | **0.201 %** (was 0.291 %) |
| `test_full_Cf_vs_unreduced_definition` m = 3 | **0.942 %** (was 1.112 %) |
| `test_full_Cf_QUADRATIC_vs_unreduced_definition` *(new)* | **2.72 %**, σ-drift 0.1 % |
| `test_full_Cf_NONLINEAR_vs_unreduced_definition` m = 6 | **0.92 %** (was 4.54 %), drift 0.9 % |
| `test_full_Cf_NONLINEAR_vs_unreduced_definition` m = 2 | 7.06 % (see §7), drift 0.2 % |
| `test_unreduced_reference_angular_convergence` *(new)* | 1.03 % on 1.5× n_phi |

The single skip is `test_a_dot_regression_baseline` — `baseline_adot.npz` is still
absent, so that test has never actually run. Pre-existing; still open.

---

## 7. The one channel that does not close

**Cubic (n = 0, m = 2).** Production −1.573e-10 (best config: n_xi = 24, n_phi = 508,
n_xi_proj = 160) against −1.666e-10 reduction-free and −1.679e-10 reduced. As a ratio
on that channel it is ~6 %; as a fraction of the cubic order's peak |C(2,2)| = 6.56e-10
— on which the two evaluators agree to **0.08 %** — it is **1.6 %**.

Neither side is converged on it: production moves it 3.5–7.6 % across n_phi / n_xi /
n_xi_proj and those shifts partly cancel; the reference moves it 1–2 % across its own
axes; the projection harness alone has a ~4 % floor on n = 2 channels at NX = 13.

Deliberately left visible in the test tolerance rather than hidden. It is the largest
open residual in the C[f] chain.

---

## 8. What was changed in the code (commit `4453efe2`)

1. **`unreduced_collision_reference`**: `n_phi = 0` auto rule calibrated to ~14 nodes
   across the collinear peak, `2·ceil(π / (0.07 √(2σ/X)))` ≈ 430 at σ = 0.3 (vs the
   112–160 used before); prominent warning documenting the mechanism, the measured
   ladder, and that a σ-ladder cannot see it.
2. **`_check_quadrature_convergence`** now refines n_phi for the **quadratic and cubic
   vertices**, not just the linear blocks — priced at one output node (~2 % of the
   build). Linear-only was a real hole: at the auto n_phi = 254 the linear rates are
   converged to 0.4 % while the cubic (0,2) still moves 3.5 % on n_phi → 508.
3. **New `test_full_Cf_QUADRATIC_vs_unreduced_definition`** — the quadratic vertex had
   **no definitional test at all**. Requires Nr ≥ 2 and reads (n = 1, m = 4): ψ₁ ∝ ξ is
   the non-cancelling projection of a particle-hole-odd Q2, while ψ₀ sees only the
   near-cancellation (which is why Q(0,4) looked like a σ-artifact).
4. **New `test_unreduced_reference_angular_convergence`** pinning the auto rule.
5. `test_unreduced_vs_reduced_reference` now Richardson-extrapolates in σ at the auto
   n_phi; tolerance 1.5 % → 0.5 %.
6. All reference call-sites moved to `rc.device`. **The evaluator is pure tensor ops
   and is ~100× faster on the GPU** — the n_phi = 112 point that cost a multi-hour
   26-shard CPU campaign takes 0.7 min and reproduces it to all 16 digits.
7. **`g_s` forwarding fixed in three places** where it was silently dropped: the
   matrix-free `kin` dict in `_ee.py`, `cubic_vertex`, `quadratic_vertex`. Inert at the
   default g_s = 2, but this operator has been bitten by exactly this omission before.

### Verified but *not* changed
The vertex algebra was re-derived from scratch (by hand and by an adversarial agent
audit) and is exact in **both** implementations — the six Q2 pair coefficients
(F3+F4−1, F2−F4, F2−F3, F1−F4, F1−F3, −(F1+F2−1)) and C3 = d1d2(d3+d4) − d3d4(d1+d2).
The mode-space assembly (unordered-pair enumeration, multiplicity, symmetrization,
Hermitian completion) is exact, and for the disputed channel the contributing pair is
diagonal, so no multiplicity bug could have produced the discrepancy.

One real structural observation that turned out **not** to be the cause: `_beta_grid`
clusters at β = π, but the caustic actually sits at β = π ± 2√(t(x3−x2)), since
cos_arg(β = π) = −k3/k2 **exactly**. Refuted as the explanation because it predicts
error ∝ n_phi^(−1/2) and production is flat from n_phi 254 → 1016. Worth revisiting if
the cubic (0,2) residual is ever chased further.

---

## 9. Reproducing

On a GPU box with qimpy at `4453efe2`, from `probes/`:

```bash
export CUDA_VISIBLE_DEVICES=0

# production side (writes /tmp/qprod_<tag>.npz, needed by the others)
QTAG=ref QB=dense QZC=1               python3 q_prod.py   # 36 s, cubic zeroed
QTAG=pfull QZC=0 QXIC=9               python3 q_prod.py   # 10 min, with cubic
QTAG=p_nxi32 QNXI=32                  python3 q_prod.py   # ladder any axis

# reduced brute force (independent vertex algebra) -- ~0.5 min
QTAG=e48 QNXI=48 QNPHI=1024 QXIC=9 QNX=13 QXG=9 python3 q_exact.py

# reduction-free definition -- 0.7 min at n_phi=112, 20 min at 896
QTAG=np448 QNPHI=448 QSIG=0.30 QNXI2=320 QNXI3=20        python3 q_unred.py
QFULL=1 QNAZ=14 QNX=13 QTAG=full448 QNPHI=448            python3 q_unred.py

# the decisive pointwise cross-ladder -- ~2 min
python3 q_reduction_probe.py
```

`q_common.py` holds the shared field, amplitude convention, stencil and projection —
edit it and every evaluator moves together, which is the point.
