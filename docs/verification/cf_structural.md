# C[f] structural verification — beyond one input, one truncation, one temperature

**2026-08-04/05. Commits `33251c81` (tests) and `be4bd5b6` (Galerkin domain fix) on
top of `4453efe2`, branch `clean-transport`. Same GaAs 2DEG as `RESULTS.md`.
Both instances shelved.**

`RESULTS.md` established that production reproduces eq (1) for **one input mode
(n=0, m=2), at one truncation (M=6, N_r=3), at one temperature**. This round closes
those three axes.

**Verdict: production reproduces eq (1) on every axis tested. One real production
defect was found and fixed along the way — the radial Galerkin domain was frozen at
|ξ| ≤ 8 regardless of N_r (§6b) — and everything else resolved to the verification
harness or to the truncation.** Two new permanent tests guard the results.

⚠ §2 below contains a claim I later RETRACTED; see §6b.

---

## 1. Nonlinear packing — the audit's explicit blind spot

The dense path assembles Q and C from **unordered** leg tuples with explicit
multiplicities (1, 2 for pairs; 1, 3, 6 for triples). The matrix-free path evaluates
Q2 and C3 **pointwise on the quadrature grid and never packs anything**. With a
*single-mode* input the only contributing tuple is the diagonal one (multiplicity 1),
so every off-diagonal branch is dead code — which is why the earlier audit could say
"no multiplicity bug can produce the 20%".

Input: three modes, mixed radial features, **including a sin harmonic** (so the
real↔complex fold and the Hermitian completion see a non-vanishing imaginary part).

| order | max\|dense\| | dense vs matrix-free |
|---|---|---|
| L | 3.605119e-09 | **0.000e+00** |
| Q | 8.771880e-10 | **1.5e-14** |
| C | 2.034183e-10 | **1.3e-14** |
| full | 3.323176e-09 | **4.2e-15** |

⇒ the multiplicity and permutation bookkeeping is verified against an implementation
that has none. Guarded by `test_nonlinear_packing_dense_vs_matrix_free`.

---

## 2. Truncation convergence — the actual "converging to eq (1)" statement

Three numbers, and the distinction is the point:

* **basis** — ‖rec(proj(ref)) − ref‖/‖ref‖: the error you would get if the operator
  were *exact* and you Galerkin-projected the true answer the same way. Production
  appears nowhere in it.
* **prod** — ‖rec(a_prod) − ref‖/‖ref‖: the total error.
* **coef** — ‖a_prod − proj(ref)‖/‖proj(ref)‖ over m ≥ 2: the operator's own error
  inside the span. (m = 0, 1 carry the conserved directions, which production
  projects out and the reference does not — comparing them would need the projector
  on both sides, i.e. the tautology this whole exercise exists to avoid.)

Multi-mode input, reference = reduction-sharing brute force:

| M | N_r | basis | prod | coef |
|---|---|---|---|---|
| 4 | 2 | 0.850 | 0.824 | 0.031 |
| 6 | 2 | 0.843 | 0.817 | 0.031 |
| 6 | 3 | 0.705 | 0.672 | 0.032 |
| 6 | 4 | 0.399 | 0.318 | 0.094 |
| **8** | **4** | **0.399** | **0.318** | **0.094** |
| 6 | 6 | 0.237 | 0.199 | 0.176 |

* **prod < basis at every rung** — production's reconstructed field is closer to the
  definition than the Galerkin projection's own error. The operator contributes no
  more error than the projection does.
* **M is irrelevant past 4**: (6,4) and (8,4) agree to four digits.
* **N_r drives everything**: 0.82 → 0.67 → 0.32 → 0.20.
* **At the documented N_r = 3, ~70% of C[f]'s output field lies outside the retained
  span.** (The L² field norm weights all of ξ equally, whereas transport observables
  weight the Fermi surface — so this is *not* a 70% error in the shear rate. It is a
  statement about basis completeness.)

### ⚠ RETRACTED: "the operator's in-span error grows with N_r"

`coef` = 0.031, 0.031, 0.032, **0.094**, 0.094, **0.176**, and I originally read this
as a crossover past which N_r costs more than it buys. **That reading was wrong** —
see §6b. The FIELD error falls monotonically throughout; the growth was (i) a real
but fixable production defect in the Galerkin domain and (ii) non-orthogonality of
the odd radial modes, which makes the coefficient split — not the field —
ill-determined. The numbers in this table are pre-fix.

---

## 3. Conservation, non-tautologically

Moments {1, **k**, ε} of the reference **field**, normalized by the field's own L1
norm — no null covectors anywhere.

Reduction-sharing reference, multi-mode input:

| order | N | Px | Py | E |
|---|---|---|---|---|
| L | +1.7e-14 | −5.7e-06 | −7.5e-15 | **+1.9e-14** |
| Q | −9.4e-04 | −1.1e-03 | −6.5e-04 | −8.9e-03 |
| C | +2.1e-03 | −1.7e-04 | +1.5e-03 | +3.6e-03 |

For single-mode inputs the linear and cubic orders conserve all four to
**1e-14 … 1e-16**. Corroborating this, the **raw** m=1 linear block — straight out of
the quadrature, before symmetrization, null projection or PSD clamp — has smallest
eigenvalue **1.6e-24**.

**The reduction-free evaluator shows a much larger energy residual in Q (1.5e-1 vs
8.9e-3).** That is the nascent-delta artifact, not a defect: smearing δ(Δε) with a
Gaussian breaks exact energy conservation, and Q is the energy-odd order. Confirmed
by the reduced evaluator, which imposes energy conservation exactly via root-finding,
conserving to 1e-14.

Incidentally: the raw L blocks come out **symmetric to 3e-18 and already PSD** at
every m — so the PSD clamp is inert, and symmetry/positivity are properties of the
exact evaluation rather than of the imposition.

---

## 4. Input-mode sweep

Six single-mode inputs through one operator build (a_dot does not depend on the
input, so the sweep costs one build plus six applies). Reference = reduction-sharing
brute force.

| input | basis | prod | coef | L | Q | C |
|---|---|---|---|---|---|---|
| (p=0, m=0) μ-shift | 0.807 | 0.791 | 0.034 | **null** | 0.413 | 0.032 |
| (p=0, m=1) | 0.632 | 0.623 | 0.025 | 0.034 | 0.011 | 0.032 |
| (p=0, m=3) odd | 0.579 | 0.554 | 0.069 | 0.164 | 0.008 | 0.032 |
| (p=1, m=2) ∝ξ | 0.744 | 0.740 | 0.024 | 0.029 | 0.077 | 1.398 |
| (p=2, m=2) | 0.474 | 0.444 | 0.031 | 0.019 | 0.021 | 0.249 |
| (p=0, m=2) sin | 0.344 | 0.324 | 0.031 | 0.028 | 0.012 | 0.038 |

* **`prod < basis` for every input.**
* **The density mode is annihilated exactly**: production gives identically 0 against
  a reference L peak of **1.14e-23** (numerical zero — the input is a collision
  invariant at linear order).
* **`L_coeff` vs the amplitude stencil agrees to ≤ 4.6e-16 on all six** — nothing
  quadratic or cubic leaks into what the solver integrates as linear.

### The two apparent outliers, both resolved as method

**C = 1.398 for the ∝ξ input.** Not production quadrature (n_phi 254→1016: 1.398 →
1.375), not reference quadrature (n_xi 48→96, n_phi 1024→2048: → 1.402), and the
reduction-free reference agrees with the reduced one (1.87 vs 1.40 — they disagree
with *each other*, the signature of an unstable comparison). The `basis` column
explains it: **81% of that cubic field is outside the N_r=3 span**, so the projected
coefficients are a small unstable residue. Laddering N_r collapses it:

| N_r | 3 | 4 | 6 |
|---|---|---|---|
| C coef | 1.402 | **0.468** | **0.389** |
| field error | 0.740 | 0.289 | 0.211 |

**L = 0.164 for the odd m=3 input.** The n=2 coefficient reads −8.89e-11 vs −1.73e-10
(19% of peak) and does not improve with N_r. But the m=3 block's smallest eigenvalue
is **4.06e-8 against a largest of 2.29e-6 — 57× smaller**: n=2 is a nearly-null
direction, where coefficient-level comparison is ill-conditioned by construction. In
the *field* metric production is below the projection's own error (0.397 vs 0.437) for
this input as for every other.

> **Method lesson: compare fields, not coefficients, whenever the truncation or a
> near-null direction is in play.**

---

## 5. Temperature

⚠ **The first T-sweep was wrong, and the bug was mine.** `qv2_common` hard-coded the
module constant `T0` in the w_eq conversion, so at T ≠ 1.33e-5 the reference saw a
field built with one temperature while production interpreted its coefficients with
another. Symptom: the middle T clean (Q 0.029) and *both* edges catastrophic (Q 1.004
at T/2, 0.492 at 2T), immune to every resolution knob on both sides — which is what
finally identified it as a harness inconsistency rather than physics.

With the temperature threaded correctly, over a factor of 4 in T:

| T | t = T/E_F | prod n_phi | L | Q | C | prod field | basis field |
|---|---|---|---|---|---|---|---|
| 0.665e-5 | 0.0158 | 506 | 0.031 | 0.027 | 0.040 | 0.689 | 0.720 |
| 1.33e-5 | 0.0317 | 254 | 0.030 | 0.029 | 0.056 | 0.671 | 0.707 |
| 2.66e-5 | 0.0634 | 128 | 0.030 | 0.039 | 0.105 | 0.663 | 0.698 |

The mild growth of C with T is the auto rule: `n_phi = max(6M+2, 8/t)` **decreases**
with t (506 → 254 → 128), starving the cubic at the hot end. The nonlinear leg added
to `_check_quadrature_convergence` in `4453efe2` warns about exactly this.

---

## 6. Detailed balance and the H-theorem

Exact properties of eq (1), independent of every quadrature choice.

| | value | |
|---|---|---|
| max\|C[f_le]\| / max\|C[f_neq]\| | **1.74e-14** | drifted **and heated** Fermi-Dirac annihilated |
| Ṡ[f_le] | +5.39e-21 | ≈ 0 at equilibrium |
| Ṡ[f_neq] | **+1.46e-05 > 0** | H-theorem satisfied |

This is a **nonlinear** null state — the linearized operator annihilates only the four
collision invariants — so it can only be tested against `a_dot`, never `L_coeff`.
Guarded by `test_detailed_balance_and_H_theorem`.

Production's residual vs truncation (θ = T/T_e = 1, pure drift + μ shift):

| N_r | 2 | 3 | 4 | 6 |
|---|---|---|---|---|
| residual | 2.71e-2 | 1.53e-2 | **9.89e-3** | 2.44e-2 |
| basis repr. error of f_le | 6.34e-2 | 3.99e-2 | 5.76e-3 | 7.73e-4 |

The residual tracks the representation error down to N_r = 4, then **rises** at N_r = 6
while the representation error keeps falling to 7.7e-4 — the operator's own error has
become the floor. Independent corroboration of §2's growing `coef`.

(With heating, θ = 0.769, the residual is noisier: Φ_le = δf/w_eq grows like
e^{(1−θ)ξ}, so the basis represents it poorly — 8–29% — and that, not the operator,
sets the residual.)

---

## 6b. Both residuals chased to ground (commit `be4bd5b6`)

### ⚠ RETRACTION of §2's "crossover"

§2 recorded that "the operator's in-span error grows with N_r, so there is a
crossover past which N_r costs more than it buys". **That is wrong.** The FIELD
error falls monotonically (0.824, 0.672, 0.318, 0.199 at N_r = 2, 3, 4, 6). Only
coefficient-space metrics grew, for two separate reasons below. The corroboration
I cited — §6's null residual rising at N_r=6 — is itself a coefficient max-norm and
is subject to the same effect.

**Q and C never degraded with N_r at all**: 0.023–0.036 and 0.040–0.070 across
N_r = 3–6, before and after. The nonlinear vertices were never implicated. Only L.

### A real production defect: the radial Galerkin domain

`_radial_galerkin` used `x_span = max(8, |ξ_c|max + 2)` — **8 for every N_r** at the
default xi_max = 6. The overlap integrand is w_eq ψ_l Φ̇ ~ |ξ|e^{−|ξ|} (Φ̇ is O(1) at
large |ξ|, ψ₁ ~ ξ *grows*, tanh powers saturate only near |ξ| ~ 6), so truncating at
X leaves ~(1+X)e^{−X} = 3e-3 at X = 8 — and the higher modes carry more of it.
Measured against a reference projected over a converged domain (|x| ≤ 14, verified
by laddering 8/9/12/14):

| N_r | 3 | 4 | 6 |
|---|---|---|---|
| L coef, x_span = 8 | 0.0538 | 0.1307 | 0.2186 |
| L coef, **x_span = 16** | **0.0162** | **0.0637** | **0.1756** |

3.3× at N_r=3, 2.1× at N_r=4. Worst single row was n=1 — ψ₁ is the ξ mode, exactly
the one whose overlap integrand grows — improving 5.3× (2.00e-10 → 3.75e-11).
**117 passed, 1 skipped**: every existing tolerance holds.

### Excluded, each by measurement

| candidate | result |
|---|---|
| Gram conditioning | cond(G) = 1.06, 1.06, 1.31, 1.71, 2.29 at N_r = 2…8 — far too small |
| operator energy quadrature | n_xi 16 → 64 at N_r=6: L spectra unchanged in the 3rd digit |
| Galerkin fine rule | n_xi_proj 96 → 160: 0.0637 → 0.0599 |
| PSD clamp | raw and stored spectra identical; raw blocks already symmetric to 3e-18 and PSD |

### The residual: the odd radial modes are non-orthogonal

Features {1, ξ, v², **v**, v⁴, **v³**}, v = tanh(ξ/2), and v ≈ ξ/2 near the surface —
so ψ₁, ψ₃, ψ₅ are near-parallel there and separated only by their tails. Measured
overlaps ⟨ψ₁,ψ₃⟩ = −0.140, ⟨ψ₃,ψ₅⟩ = +0.319, ⟨ψ₁,ψ₅⟩ = −0.167, against ≤ 0.033 for
every even pair. Per-row coefficient errors **cancel 1.46× (N_r=4) and 2.13× (N_r=6)**
when reconstructed as a field: part of what the coefficient metric charges as error
is redistribution among near-parallel modes.

**Practical consequence:** raise N_r freely — field accuracy improves monotonically —
but do not interpret individual high-N_r radial coefficients, especially odd ones,
in isolation.

### The cubic (n=0, m=2) is closed

With the reference's projection domain matched to production's, the whole cubic
order for that input is **flat in N_r: coef = 0.0127, 0.0131, 0.0137** at N_r = 3, 4, 6,
and the disputed channel's 1.76% of the cubic peak is ordinary within it. It was
never an outlier — it looked like one because I compared a single coefficient
across a domain mismatch.

---

## 7. What remains open
1. **|M_q|², the well-width form factor, and the golden-rule prefactor convention**
   remain common-mode to both evaluators. No internal test reaches them; this needs
   an external anchor (a published 2D Fermi-liquid quasiparticle rate at comparable
   r_s, or a measured ℓ_ee).
2. **The run-time modifiers** — the residual closure and the (T_e/T)² rescale — have
   still never been compared against the definition.
3. `baseline_adot.npz` is still absent, so that regression test has never run.

---

## 8. Files

* `probes/qv2_common.py` — generalized harness: arbitrary multi-mode inputs specified
  in the radial **feature** basis (so a fixed physical field survives an N_r ladder —
  ψ_l depends on N_r, a fixed modal vector does not describe a fixed field),
  full cos+sin projection, field reconstruction, conserved moments.
* `probes/qv2_prod.py` — production side; `QFEATSETS` runs many inputs off one build.
* `probes/qv2_ref.py` — either reference evaluator on the same field.
* `probes/qv2_equil.py` — detailed balance + H-theorem.
* `probes/qv2_analyze.py` — channel table, basis/prod/coef decomposition, moments.
* `patched_source/test_ee.py` — includes the two new tests.

Reproduce (GPU, from `probes/`):

```bash
export CUDA_VISIBLE_DEVICES=0
QTAG=multi QEVAL=unred QNX=21 QNAZ=26 QNPHI=0 python3 qv2_ref.py    # ~75 min
QTAG=t_M6N3 QM=6 QNR=3 python3 qv2_prod.py                          # ~9 min
python3 qv2_analyze.py /tmp/v2ref_multi.npz /tmp/v2prod_t_M6N3_0.npz
QTAG=eq QM=6 QNR=3 python3 qv2_equil.py                             # ~13 min
```
