Electron-electron collisions (2D Fermi liquid)
==============================================

This page is a self-contained derivation and implementation guide for the
microscopic electron--electron (e-e) collision operator of an isotropic 2D
Fermi liquid, as implemented in
``qimpy.transport.material.fermi_surface.scattering``
(``_kernels.py`` for the kinematics/vertices, ``_ee.py`` for the
operator and its two apply backends).  It explains *every* mathematical step,
*every* implementation option, and the reason for each design choice, so the
code and the math can be reviewed together without external notes.

Units are Hartree atomic units (:math:`\hbar = k_B = 1`); temperature
:math:`T` is therefore an energy.


.. contents:: Contents
    :local:
    :depth: 2


1. Physical model
-----------------

Two electrons :math:`1,2` scatter to :math:`3,4` on a single isotropic band
:math:`\varepsilon(\mathbf{k})` with Fermi wavevector :math:`k_F`, Fermi
velocity :math:`v_F` and effective mass :math:`m^\ast = k_F/v_F` (so
:math:`E_F = k_F^2 / 2 m^\ast`).  The interaction is the statically screened
2D Coulomb potential

.. math::

   M_q = \frac{2\pi\, F(q)}{\epsilon_b \,\bigl(q + \kappa\, F(q)\bigr)},
   \qquad
   |M_q|^2 \ \text{enters the rate},

with background dielectric constant :math:`\epsilon_b`, 2D Thomas--Fermi
screening wavevector :math:`\kappa` (default :math:`2 m^\ast/\epsilon_b`, the
degeneracy-2 2DEG value), and an optional quantum-well form factor
:math:`F(q)` (:math:`F=1` for an ideal zero-thickness sheet; the finite-width
infinite-square-well :math:`F(qW)` is ``well_form_factor``).  This is
``matrix_element_sq``.

The occupation is split into an equilibrium part and a deviation,

.. math::

   f_i = f^0_i + \delta f_i, \qquad
   f^0(\xi) = \frac{1}{e^{\xi/T}+1}, \qquad \xi = \varepsilon - \mu,

and we use the dimensionless energy :math:`x = \xi/T`.  The solver works with a
rescaled field :math:`\Phi` defined by

.. math::

   \delta f = w_{\rm eq}\,\Phi, \qquad
   w_{\rm eq}(x) = \frac{1}{4T}\,\operatorname{sech}^2\!\frac{x}{2}
                 = \frac{f^0(1-f^0)}{T}.

:math:`w_{\rm eq}` is the (positive) equilibrium response weight; writing the
unknown as :math:`\Phi` makes the linearized operator symmetric and keeps the
modal basis well conditioned across the thermal shell.


2. The collision operator and its exact expansion
-------------------------------------------------

The collision integral for leg 1 is the gain minus loss bracket

.. math::

   \dot f_1 = \frac{(m^\ast)^3}{(2\pi)^3}\!\int\! d\xi_2\,d\xi_3\,d\phi_3
   \sum_{\rm roots}
   \frac{|M_q|^2}{k_2 k_4\,|\sin(\phi_4-\phi_2)|}\,(B-F),

.. math::

   B = f_3 f_4 (1-f_1)(1-f_2), \qquad
   F = f_1 f_2 (1-f_3)(1-f_4),

(the prefactor, Jacobian, roots and phase space are derived in
:ref:`ee-kinematics`).  :math:`B` is the scattering-in (gain) term, :math:`F`
the scattering-out (loss) term.

**Exact termination at cubic order.**  Substituting
:math:`f_i = f^0_i + d_i` (with :math:`d_i \equiv \delta f_i`) and expanding,

.. math::

   B - F = L_1[\delta f] + Q_2[\delta f] + C_3[\delta f].

There is no constant term (equilibrium detailed balance,
:math:`f^0_1 f^0_2 (1-f^0_3)(1-f^0_4) = f^0_3 f^0_4 (1-f^0_1)(1-f^0_2)` on the
energy shell, makes :math:`B-F=0` at :math:`\delta f=0`), and the quartic term
cancels identically: the :math:`d_1 d_2 d_3 d_4` coefficient is :math:`+1` from
:math:`B` (via :math:`(1-f_1)(1-f_2)\to d_1 d_2`) and :math:`+1` from
:math:`F`, so :math:`B-F` contributes :math:`1-1=0`.  **The expansion is
therefore exact and finite** -- three terms capture the full nonlinear
operator for arbitrary :math:`\delta f`.

**Linear term** (the standard linearized collision operator):

.. math::

   L_1 = W\left(\frac{d_3}{f^0_3(1-f^0_3)} + \frac{d_4}{f^0_4(1-f^0_4)}
               - \frac{d_1}{f^0_1(1-f^0_1)} - \frac{d_2}{f^0_2(1-f^0_2)}\right),
   \quad W = f^0_1 f^0_2 (1-f^0_3)(1-f^0_4),

which in the :math:`\Phi` variable is :math:`L_1 = (W/T)(\Phi_3+\Phi_4
-\Phi_1-\Phi_2)` since :math:`d_i/[f^0_i(1-f^0_i)] = \Phi_i/T`.

**Cubic term** (collecting the degree-3 terms of :math:`B-F`; every
:math:`f^0` cancels):

.. math::

   C_3 = d_1 d_2 (d_3 + d_4) - d_3 d_4 (d_1 + d_2).

:math:`C_3` is **independent of** :math:`f^0`.  Intuitively, each leg that is
*left out* of a triple enters once through the gain (carrying :math:`f^0`) and
once through the loss (carrying :math:`1-f^0`); the two sum to 1.  Because it
is :math:`f^0`-independent, :math:`C_3` is manifestly temperature-finite and
carries the leading drive nonlinearity.

**Quadratic term** (degree-2; coefficients are particle-hole odd):

.. math::

   Q_2 = {}& d_1 d_2 (f^0_3+f^0_4-1) + d_1 d_3 (f^0_2-f^0_4)
           + d_1 d_4 (f^0_2-f^0_3) \\
         &+ d_2 d_3 (f^0_1-f^0_4) + d_2 d_4 (f^0_1-f^0_3)
           - d_3 d_4 (f^0_1+f^0_2-1).

Every coefficient is a combination of :math:`f^0-\tfrac12` factors, so
:math:`Q_2` **vanishes on the Fermi surface** (where all :math:`f^0=\tfrac12`)
and is :math:`O(T/E_F)`.  It is the particle-hole-odd (thermoelectric) vertex
and genuinely couples opposite energy parities.

The threshold for nonlinearity is :math:`\Phi \sim O(1)`, i.e.
:math:`\delta f \sim w_{\rm eq}`; in transport terms a drift
:math:`v_d^\ast = T/(m^\ast v_F)` or bias :math:`V^\ast \sim T` (reported at
construction).


.. _ee-kinematics:

3. Kinematic reduction
----------------------

The collision integral over :math:`\mathbf{k}_2,\mathbf{k}_3,\mathbf{k}_4`
carries momentum and energy delta functions.  Momentum conservation removes
:math:`\mathbf{k}_4 = \mathbf{k}_1+\mathbf{k}_2-\mathbf{k}_3`.  Working on the
energy shell, write each leg on the Fermi circle of its energy,

.. math::

   k_i = k_F\sqrt{1 + t\,x_i}, \qquad t = T/E_F,

(``k_fermi``; :math:`x_4 = x_1 + x_2 - x_3` from energy conservation).
Choosing :math:`\phi_1=0` by isotropy and parameterizing leg 3 by its azimuth
:math:`\beta = \phi_3-\phi_1`, the momentum transfer is

.. math::

   q^2 = k_1^2 + k_3^2 - 2 k_1 k_3 \cos\beta,
   \qquad \mathbf{P} = \mathbf{k}_1 - \mathbf{k}_3,

and the remaining energy delta fixes :math:`\phi_2` from
:math:`|\mathbf{P}+\mathbf{k}_2| = k_4`:

.. math::

   \cos(\phi_2-\phi_P) = \frac{k_4^2 - q^2 - k_2^2}{2 q k_2}
   \equiv \cos\,\delta,
   \qquad \phi_P = \operatorname{atan2}(P_y,P_x).

This has **two roots** :math:`\phi_2 = \phi_P \pm \delta` (the
:math:`\pm` is the ``sgn`` loop in the code), each contributing a Jacobian

.. math::

   \frac{1}{k_2 k_4 |\sin(\phi_4-\phi_2)|}

from resolving the energy delta in the azimuth.  Putting it together, the
reduced rate is the expression in Section 2, an integral over
:math:`(\xi_2, \xi_3, \beta)` with the two-root sum, the prefactor
:math:`(m^\ast)^3/(2\pi)^3`, and the thermal phase space :math:`T^2` (from
:math:`d\xi_2\,d\xi_3 = T^2\,dx_2\,dx_3`).  All of this is assembled once per
:math:`(x_1, x_3, \text{root})` by ``_shell_geometry``, which returns the
full kinematic weight

.. math::

   W_k = \frac{|M_q|^2}{k_2 k_4 |\sin(\phi_4-\phi_2)|}\;
         w_2\, w_\beta\, w_3\, T^2\, \frac{(m^\ast)^3}{(2\pi)^3}

(zero off the kinematically allowed shell) together with the leg angles
:math:`\phi_2, \phi_4`.

**Independent check of the reduction.**  Two evaluators that *share* this
reduction cannot catch an error in it.  ``unreduced_collision_reference``
therefore avoids it entirely: it keeps the energy delta as a narrow Gaussian
of width :math:`\sigma` and integrates :math:`(x_2,\phi_2,x_3,\phi_3)`
directly, with **no** Jacobian and **no** root finding (hence no caustic).  It
agrees with the reduced ``exact_collision_reference`` to :math:`<1.5\%`
(linear) and :math:`<8\%` (full nonlinear) at the test resolution, and to
:math:`<0.4\%` under :math:`\sigma\to 0` Richardson extrapolation -- validating
the reduction itself (Jacobian, root multiplicity, phase-space prefactor).


.. _ee-vanhove:

4. The van-Hove edge and the grading trick
------------------------------------------

At backscattering, :math:`\beta = \pi`, the two :math:`\phi_2` roots coalesce
(:math:`\delta\to 0`), the factor :math:`|\sin(\phi_4-\phi_2)|\to 0`, and the
Jacobian diverges.  This is an integrable inverse-square-root **van-Hove edge**:
near :math:`\beta=\pi` the integrand behaves as
:math:`1/\sqrt{\pi-\beta}`.  Plain Gauss--Legendre converges only
algebraically and non-monotonically there.

The fix (``_beta_grid``) is a quadratic **node grading** that maps a
Gauss--Legendre node :math:`s\in(0,2\pi)` through

.. math::

   \beta = \pi\bigl(1 + u\,|u|\bigr), \qquad u = \frac{s-\pi}{\pi},
   \qquad \frac{d\beta}{ds} = 2|u| \xrightarrow[\beta\to\pi]{} 0.

The Jacobian of the variable change, :math:`|d\beta/ds| = 2|u| \sim
\sqrt{|\beta-\pi|}`, **exactly cancels** the :math:`1/\sqrt{|\beta-\pi|}`
edge: the integrand times :math:`|d\beta/ds|` is regular at
:math:`\beta=\pi`, restoring smooth, monotone (effectively spectral)
convergence.  This is a fixed numerical scheme -- like the choice of
Gauss--Legendre itself -- with no physical free parameter.  The grid is
symmetric about :math:`\pi`, which is also used by the reflection analysis in
Section 8.3.  The same grid is shared by the linear blocks, the nonlinear
vertices and the brute-force reference, so all comparisons stay exact at
:math:`\phi_1=0`.

A second, milder edge is the root-coalescence point :math:`|\cos\delta|\to 1`,
where the arccos derivative diverges.  A thin sliver
:math:`|\cos\delta| > 1-\texttt{EDGE\_EPS}` (with
:math:`\texttt{EDGE\_EPS}=10^{-6}`) is excluded by the root mask; being
:math:`1/\sqrt{}`-integrable it biases the result by only
:math:`O(\sqrt{\texttt{EDGE\_EPS}})` while preventing
:math:`O(1/\sqrt{\texttt{EDGE\_EPS}})` node spikes.

Odd angular harmonics: :math:`K_m = \int_0^{2\pi} |M_q|^2 (1-\cos m\alpha)/
|\sin\alpha|\,d\alpha` (``K_table``, :math:`q = 2k_F\sin(\alpha/2)`) is
finite only for even :math:`m`; for odd :math:`m` the integrand is
non-integrable at :math:`\alpha=\pi` and the harmonic does not relax at
leading order, so odd-:math:`m` entries are gated to zero.


5. Modal discretization
-----------------------

The field is expanded in angular harmonics and a radial (energy) Galerkin
basis,

.. math::

   \delta f(x,\phi) = w_{\rm eq}(x)\,
   \sum_{l<N_r}\psi_l(x)\sum_{|m|\le M} a_{l,m}\, e_m(\phi),

with real cos/sin angular functions :math:`e_0=1`, :math:`e_{2m-1}=\cos
m\phi`, :math:`e_{2m}=\sin m\phi` (dimension :math:`\dim = 2M+1`) and complex
harmonics :math:`\hat a[m]` related by the unitary maps
``_real_to_complex`` / ``_complex_to_real``.  The radial basis
:math:`\psi_l(x) = \sum_p c_{p,l}\,x^p` is the code's energy-mode basis;
``EEScattering._radial_galerkin`` builds the fine Galerkin nodes
:math:`x_{\rm fine}`, the projection covector :math:`P[l,f] = \langle\psi_l|
\cdot\rangle_w` and the inverse Gram :math:`G^{-1}`.

Because the legs carry the *physical* weight :math:`w_{\rm eq}\psi_l`, the raw
integral returns the occupation rate :math:`\dot f`; converting to the solver
field uses :math:`\dot\Phi = \dot f/w_{\rm eq} = 4T\cosh^2(x_1/2)\,\dot f`.
This factor, with the decay sign, is folded into the node-to-mode projector
:math:`\texttt{GPc} = (G^{-1}P)\cdot(-\texttt{conv})`,
:math:`\texttt{conv}=4T\cosh^2(x_1/2)`.

**Automatic quadrature.**  From the truncation :math:`(M,N_r)`, the degeneracy
:math:`t=T/E_F` and a target tolerance ``tol``, the resolution is set
automatically: :math:`n_\xi \sim \max(16, 4N_r)` (radial), :math:`n_{\xi,\rm
proj}\ge N_r` (Galerkin projection), and the binding angular axis
:math:`n_\phi = \max(6M{+}2,\ 8/t)` -- it must dealias the cubic's
:math:`3M` harmonics *and* resolve the collinear edge of width :math:`\sim t`.
A construction-time check recomputes the representative rates at
:math:`1.5\,n_\phi` and warns if they move by more than ``tol``.


6. Linear operator
------------------

The linear block :math:`L^{(m)}` (radial matrix per harmonic :math:`m`) is the
Galerkin projection of :math:`L_1` over the thermal shell
(``L_blocks``, then :math:`L = G^{-1}P\,R`).  Two options:

* ``on_shell=False`` (default): the **exact** finite-:math:`T` shell
  quadrature, including the full radial tower and the collinear-log
  enhancement of the energy/heat modes (e.g. :math:`\gamma_2` is :math:`\sim
  1.3\times` the surface value at :math:`t\approx 0.03`).
* ``on_shell=True``: the leading-order closed form
  :math:`\gamma_m = (m^\ast)^2 T^2 K_m/(16\pi E_F)` (:math:`N_r=1` only).

The discrete operator is then made **exactly conserving and dissipative**:

* number/energy (harmonic :math:`m=0`) and momentum (:math:`m=1`) are exact
  collision invariants; they are projected out by
  :math:`L \to \Pi L \Pi` with :math:`\Pi = I - QQ^\top`, :math:`Q` an
  orthonormal basis of the null covectors (``_null_covectors``);
* :math:`L` is symmetrized (detailed balance) and its eigenvalues are clipped
  to :math:`\ge 0` (H-theorem / positive semidefiniteness).

These steps change :math:`L` from the raw bracket by :math:`O(\text{quadrature
error})`; the underlying linear kinematics match the brute-force
``exact_collision_reference`` (linearized) to roundoff at matched
quadrature.


7. Nonlinear operator: common structure
---------------------------------------

The nonlinear part (:math:`Q_2+C_3`) is delivered by two interchangeable
backends that compute the **same operator** to roundoff and differ only in the
speed/memory trade-off.  Both share:

* complex angular harmonics on the legs; the output harmonic is fixed by the
  **additive selection rule** :math:`m_o = m_a+m_b+m_c` (cubic) or
  :math:`m_a+m_b` (quadratic) -- a consequence of azimuthal isotropy;
* the output-energy leg 1 sits at :math:`\phi_1=0`, so its harmonic phase is
  constant and its harmonic index *becomes* the output harmonic (a
  convolution; the "leg-1 elision" in the dense path, the "leg-1 hoist" in the
  matrix-free path);
* the same finishing step ``_finalize_nonlinear``: the per-output-harmonic
  null projection (:math:`m_o\in\{0,\pm1\}`) followed by the complex
  :math:`\to` real fold, so the nonlinear terms conserve exactly like the
  linear block.


8. Nonlinear operator: backends
-------------------------------

8.1 Dense (sparse-symmetric packed kernel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The dense backend precontracts the vertices over the quadrature into a modal
kernel applied per cell as a gather + scatter.  Naively the cubic kernel is a
rank-3 tensor over the flat leg index :math:`p = l\cdot \dim + (m{+}M)`
(:math:`P = N_r\dim` values per leg), i.e. :math:`O(P^4)` to store and apply.
**Four exact symmetries** collapse this:

#. **Additive selection + truncation.**  Only output harmonics
   :math:`|m_o|\le M` are kept, and :math:`m_o=\sum m`, so only input triples
   with :math:`\sum m \in[-M,M]` survive.
#. **Output reality.**  :math:`\dot f` is real, so :math:`\hat F[-m_o] =
   \hat F[m_o]^\ast`; only the :math:`m_o\ge0` half is stored.  Combined with
   (1) this keeps triples with :math:`0\le\sum m\le M` -- a fraction
   :math:`\tfrac16` of harmonic space (Irwin--Hall :math:`n=3`).
#. **Input-leg permutation.**  The three input legs carry the same field, so
   only one entry per *unordered* triple :math:`(p_a\le p_b\le p_c)` is
   stored, with the leg-ordering multiplicity (6/3/1 for all-distinct/
   two-equal/all-equal) folded into the value.
#. **Reflection** :math:`\phi\to-\phi` makes the *kernel real*.  The discrete
   kinematic measure is reflection-invariant -- the :math:`\beta` grid is
   symmetric about :math:`\pi` and the reflection swaps the two
   :math:`\phi_2` roots, so contributions pair exactly.  Writing the kernel as
   :math:`\text{Tc}[m_a,m_b,m_c]=\int d\mu\,W\,
   e^{i(m_a\phi_2+m_b\phi_3+m_c\phi_4)}` with real radial factors, the
   substitution :math:`\phi\to-\phi` (invariant :math:`d\mu\,W`) returns the
   complex conjugate of the *same* integral, so :math:`\text{Tc}` is **real**
   (verified to :math:`\sim10^{-14}`; the code stores ``.real``).  This halves
   storage and gives a real-kernel :math:`\times` complex-field apply.  This is
   a *field-independent* property of the precontracted kernel and is exact for
   any input; it is **not** the same as halving the matrix-free quadrature loop,
   which is *not* exploitable (see the matrix-free note,
   :ref:`reflection halving <ee-reflection-mf>`).

The result is the packed kernel :math:`\texttt{\_sp\_S}[l_o, t]` over kept
unordered triples :math:`t`, with storage and per-cell apply both

.. math::

   \sim \frac{N_r\,(N_r\dim)^3}{16}
   \;=\; \frac{(N_r\dim)^4}{16\,\dim}

(measured compression :math:`\approx 16\dim`, :math:`\to 18\dim`
asymptotically).  The quadratic kernel is analogous over unordered pairs and
is :math:`O((N_r\dim)^2)` (negligible).  The energy-parity ("even-:math:`\sigma`")
radial selection is **not** used: it holds only to leading order in
:math:`T/E_F` (broken by band curvature), so it would be an approximation, not
an exact symmetry.

**Apply** (``_apply_dense``): transform the field to complex harmonics,
gather the triple/pair products, multiply by the packed kernel, scatter into
the :math:`m_o\ge0` half by the additive rule, mirror :math:`m_o<0` by
conjugation, then ``_finalize_nonlinear``.

**Direct slab-packed build** (``cubic_packed_node``).  Building the packed
kernel by forming the full :math:`(N_r\dim)^3` per-node complex vertex and
gathering from it needs :math:`O((N_r\dim)^3)` memory plus a
:math:`g\cdot(N_r\dim)^2` GEMM intermediate (:math:`g = n_\xi n_\phi`) -- both
explode at large :math:`M`.  Instead the build writes straight into packed
form:

* the :math:`(2,3,4)` triple :math:`\texttt{Tc\_full}[a,b,c] = \sum_g (-W_k)
  p_2[a]p_3[b]p_4[c]` is accumulated **slab by slab** over the first leg index
  :math:`a` (one BLAS matmul per slab); for each slab the symmetric value is
  gathered for the three permutation groups whose first index lies in the slab.
  Gather is linear, so accumulating per :math:`(x_3,\text{root})` and per slab
  reproduces the full symmetric quadrature sum;
* the leg-1 part (leg 1 radial-only, harmonic elided) is the small
  :math:`(N_r, (N_r\dim)^2)` tensor, formed in full and gathered;
* the quadratic kernel is :math:`(N_r\dim)^2` and is formed in full.

Peak build memory is then :math:`\sim` the packed kernel plus a slab
transient that the slab size bounds (verified: identical result for
:math:`\texttt{slab}=` full/4/1, peak RSS scaling with the slab).  The result
is bit-for-bit (to roundoff, :math:`\sim10^{-15}`) the full-tensor gather.
Build cost is :math:`\sim n_{\xi,\rm proj}\,(N_r\dim)^3\,n_q` FLOPs, where
:math:`n_q = 2 n_\xi^2 n_\phi`.


8.2 Matrix-free (kinematic generator)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The matrix-free backend stores only the field-independent kinematic generator
-- for every output node and quadrature point :math:`(x_2,x_3,\beta,
\text{root})` it keeps the leg energies, the leg relative angles
(:math:`\Delta\phi_1=0`, :math:`\Delta\phi_3=\beta`, and
:math:`\Delta\phi_2,\Delta\phi_4` from the two roots) and the kinematic weight
:math:`W_k`.  **None of this depends on** :math:`N_r` (only the tiny radial
table :math:`\psi` does), so storage is flat in :math:`N_r` -- the only
feasible option at large :math:`(M,N_r)`.

**Apply** (``_apply_matrix_free_chunk``): for each leg reconstruct
:math:`\delta f` at an output-angle grid of :math:`N_{\rm out}=6M+2` points
(enough to resolve harmonics up to :math:`3M` without aliasing), evaluate
:math:`C_3` and :math:`Q_2`, integrate over the quadrature with weights
:math:`W_k`, project the output-energy axis to radial modes with
:math:`\texttt{GPc}`, DFT in :math:`\phi_1` to harmonics, and
``_finalize_nonlinear``.  Three exact optimizations:

* **Real (Hermitian) reconstruction.**  Since the field is real,
  :math:`\hat a[-m]=\hat a[m]^\ast`, and the leg phase
  :math:`e^{im\Delta\phi}` preserves this, so the leg harmonics
  :math:`B_m` are Hermitian and the reconstructed
  :math:`\Phi(\phi_1)=\sum_m B_m e^{im\phi_1}` is **real**:

  .. math::

     \Phi = \operatorname{Re}B_0
          + 2\!\sum_{m>0}\!\bigl[\operatorname{Re}B_m\cos m\phi_1
                              - \operatorname{Im}B_m\sin m\phi_1\bigr].

  Synthesizing from the :math:`m\ge0` half with real
  :math:`(M{+}1, N_{\rm out})` cos/sin matrices (or a half-spectrum
  ``irfft``) halves the leg contraction and replaces a complex
  :math:`(2M{+}1, N_{\rm out})` GEMM with two real
  :math:`(M{+}1, N_{\rm out})` ones (:math:`\sim 5\times` fewer FLOPs than
  naive).
* **Leg-1 hoist.**  Leg 1 has :math:`\Delta\phi_1=0` and a
  quadrature-independent radial factor, so :math:`d_1` is reconstructed once
  and broadcast instead of per quadrature point.
* **Reconstruction backend** ``recon`` = ``gemm``/``fft``/``auto``: the real
  cos/sin matmuls vs the half-spectrum ``irfft``; ``auto`` micro-benchmarks
  both at construction and keeps the faster (adapts to :math:`M`, dtype,
  CPU/GPU).

.. _ee-reflection-mf:

**Not used: reflection halving.**  Under :math:`\phi\to-\phi` a quadrature
point :math:`(\beta,\text{root})` maps to :math:`(2\pi-\beta,\text{other
root})`.  The *same* reflection invariance that makes the dense kernel real
holds here too -- so the *full* matrix-free quadrature sum is real (the code
takes ``.real``).  What is **not** exploitable is using it to evaluate only
*half* the :math:`\beta` grid and reconstruct the other half: that would relate
the operator on :math:`\delta f` to the operator on the *reflected* field
:math:`\delta f(-\phi)`, and for a general (non-reflection-symmetric) input the
two halves give genuinely different contributions (the leg harmonics transform
as :math:`C_m e^{-im\Delta\phi}`, not the conjugate/reverse of the originals).
The crucial difference from the dense path is that there the symmetry acts on
the *precomputed, field-independent* kernel (making it real once, for all
inputs), whereas here it would have to relate per-quadrature-point
contributions of the *specific* field -- which it does not.  So, like the
energy-parity selection, reflection halving of the quadrature is left out.

Per-cell apply cost is :math:`\sim n_{\xi,\rm proj}\, n_q\, M\, N_{\rm out}`
mul-adds; storage is :math:`\sim n_{\xi,\rm proj}\, n_q` (flat in :math:`N_r`).


8.3 Automatic backend selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``backend='auto'`` chooses **dense** when its packed kernel fits a fixed cap
(:math:`\sim0.5` GiB) *and* the one-time build (:math:`\sim n_{\xi,\rm
proj}(N_r\dim)^3 n_q` FLOPs) is quick; otherwise **matrix-free**.  Both give
the same operator, so this only trades speed vs memory.


9. Precision and device
-----------------------

The transport-level ``precision`` knob (``float64`` default / ``float32``)
sets the working dtype.  The kinematics are always evaluated in
:math:`\texttt{float64}`; ``work_dtype``/``work_device`` then set the
precision/device of the build GEMMs and accumulators and of the stored kernel
or generator.  ``float32`` (complex64) roughly halves memory and doubles apply
throughput; the quadrature-sum error
(:math:`\sim\sqrt{n_q}\,\epsilon_{32}\sim10^{-4}`) stays well under the
:math:`\sim10^{-3}` quadrature tolerance, and conservation remains exact (the
null projection is applied regardless of dtype).  For large-:math:`M`
matrix-free runs, ``float32`` on a GPU is recommended.


10. Complexity and cost
-----------------------

With :math:`P=N_r\dim`, :math:`n_q = 2 n_\xi^2 n_\phi`,
:math:`N_{\rm tri}\approx P^3/16`:

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - quantity
     - dense
     - matrix-free
   * - storage
     - :math:`\sim N_r N_{\rm tri}\sim P^4/(16\,\dim)`
     - :math:`\sim n_{\xi,\rm proj}\, n_q` (flat in :math:`N_r`)
   * - build FLOPs
     - :math:`\sim n_{\xi,\rm proj}\, P^3 n_q`
     - trivial (quadrature tables)
   * - build memory
     - :math:`\sim` packed kernel (bounded)
     - :math:`\sim n_{\xi,\rm proj}\, n_q`
   * - apply / cell
     - :math:`\sim N_r N_{\rm tri}`
     - :math:`\sim n_{\xi,\rm proj}\, n_q M N_{\rm out}`

The per-cell **apply** is far cheaper for dense (it precomputes the
:math:`\sim10^5`--:math:`10^6`-point quadrature): the matrix-free/dense apply
ratio is :math:`\sim n_{\xi,\rm proj} n_\xi^2/N_r^4` (hundreds to thousands;
:math:`\sim3000\times` at :math:`M=128,N_r=8`, dropping fast with
:math:`N_r`).  But dense storage and build grow as :math:`N_r^4\dim^3`, so
dense is viable only at small/moderate :math:`(M,N_r)` -- exactly where
``auto`` selects it.  At large :math:`(M,N_r)` matrix-free is the only feasible
backend (flat storage, trivial build), at a higher but tractable per-cell cost
(``float32`` + GPU recommended).

As a concrete extreme (:math:`M=128,N_r=8`): dense would need
:math:`\sim52` GB of kernel storage and a :math:`\sim2\times10^{17}`-FLOP build
(GPU-hours), so it is impractical; matrix-free stores :math:`\sim0.4` GB
(float32), builds in seconds, and applies at :math:`\sim0.2`--:math:`0.7`
s/cell on a GPU in float32.


11. Validation
--------------

The operator is validated independently at several levels (see
``test_ee.py``):

* the angular kernels :math:`K_m` and the closed-form rate against the
  derivation targets;
* the cubic and quadratic vertices pointwise against the brute-force
  ``exact_collision_reference`` (including an energy-structured field that
  fails an :math:`l=0`-only restriction, proving the radial legs are exact);
* the reduction itself against the assumption-free
  ``unreduced_collision_reference`` (no shared Jacobian/roots);
* dense :math:`\equiv` matrix-free for arbitrary random fields (to
  :math:`\sim10^{-14}`), and both :math:`\equiv` the brute-force nonlinear
  bracket (to :math:`\sim10^{-13}` at matched quadrature);
* exact conservation (number/energy/momentum) of the nonlinear output;
* the direct slab-packed build :math:`\equiv` the full-tensor gather (to
  :math:`\sim10^{-15}`), and a bit-for-bit (:math:`\le10^{-11}`) regression
  baseline of the full ``a_dot``.
