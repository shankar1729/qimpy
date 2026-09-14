# C[f] verification records

Two narrative records of how the e-e collision operator was checked against
equation (1). They are kept because each one ends in a **retraction**, and a
retraction that is not written down gets re-derived by the next person.

* `cf_vs_definition.md` — production reproduces the Boltzmann definition. The
  reported "20% error in the quadratic vertex" was an error in the
  reduction-free *reference*, not in production, and the σ-ladder used to
  certify it could not have detected it.
* `cf_structural.md` — the same check across input mode, truncation and
  temperature. Found one real production defect: the radial Galerkin domain was
  frozen at |ξ| ≤ 8 regardless of `Nr`. Its own §2 is marked retracted in §6b.

These are records, not a runnable harness: both refer to `patched_source/`
copies and to two superseded probe scripts that are not in this tree. The
probes that produced the numbers were deleted — they carried no assertions and
printed ladders for a human to read. What survives as executable is in
`src/qimpy/transport/material/fermi_surface/scattering/test_ee.py`, which is
run by `make test-validate`.

Not part of the Sphinx build.

## Status of the open items

`cf_structural.md` §7 lists three open items. Records are not rewritten, so
their current status is here instead:

1. **Absolute scale** (|M_q|², the well-width form factor, the golden-rule
   prefactor) — still open, and no internal test can close it: both evaluators
   share it. `test_form_factor` pins the shape and
   `test_spin_degeneracy_scales_the_rate` pins the g_s consistency, but the
   overall multiplier needs an external anchor — a published 2D Fermi-liquid
   quasiparticle rate at comparable r_s, or a measured ℓ_ee.
2. **Run-time modifiers** — closed. `test_residual_closure_rate_bounds_the_band_it_replaces`
   and `test_te_rescale_fallback_is_a_leading_form_not_an_identity`.
   The first found that the closure's "cannot over-damp" guarantee is not
   exact: γ_m is monotone over the even harmonics for the closed form, at
   Nr = 1, and for the fastest radial channel, but **not** for the slowest
   channel — which is the one the closure reads. The inversion is ≤0.3%, sits
   in the m ≈ 6–8 plateau, and survives quadrature refinement.
3. **`baseline_adot`** — closed. It had never run: the path resolved outside
   the repo and the test skipped, reading green. The baseline is now committed
   beside the test as JSON (`*.npz` is gitignored by the mesh purge) and the
   test asserts its presence instead of skipping. It is a forward-looking
   freeze, not the original pre-optimization comparison, which is gone.
