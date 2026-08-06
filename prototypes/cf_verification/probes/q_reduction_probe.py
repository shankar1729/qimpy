"""The sigma x n_phi CROSS-ladder -- the single most decisive measurement.

Everything else in this campaign compares PROJECTED modal coefficients, which
drags in the radial Galerkin harness.  This one is pointwise: one evaluation
point (x1 = 0, phi1 = 0.3), the linearized bracket on a cos(2 phi) shear mode,
reduced vs reduction-free, no projection at all.

It ladders BOTH sigma and n_phi and shows why laddering sigma alone certifies
nothing:

    n_phi = 128 (the old fixed value)      n_phi = 0 (calibrated auto)
      sigma=0.40   0.498 %                   sigma=0.40   2.598 %
      sigma=0.30   2.146 %                   sigma=0.30   1.526 %
      sigma=0.20   2.712 %                   sigma=0.20   0.724 %
      sigma=0.15   2.163 %                   sigma=0.15   0.459 %
      Richardson(0.4,0.2)  3.450 %           Richardson(0.4,0.2)  0.099 %
      Richardson(0.3,0.15) 2.169 %           Richardson(0.3,0.15) 0.103 %

At the coarse n_phi the sigma trend is NOT MONOTONE and Richardson EXTRAPOLATES
THE CONTAMINANT, landing worse (3.45 %) than the raw sigma = 0.4 point (0.50 %,
which was pure cancellation and was what the old test asserted against).
Resolved, the trend is clean O(sigma^2) and the two independent Richardson pairs
agree with each other to 0.004 %.

=> the kinematic reduction (azimuth roots, the 1/(k2 k4 |sin(phi4-phi2)|)
Jacobian, the van-Hove excision, the phase-space prefactor) is correct to 0.1 %.

Run on the GPU: this evaluator is pure tensor ops and is ~100x faster there.
    CUDA_VISIBLE_DEVICES=0 python3 q_reduction_probe.py
"""
import torch
import numpy as np

torch.set_default_dtype(torch.float64)
from qimpy.transport.material.fermi_surface.scattering import _kernels

M_STAR, EPS_B, KF, T0 = 0.067, 12.9, 7.5e-3, 1.33e-5
common = dict(kF=KF, m_star=M_STAR, T=T0, epsilon_bg=EPS_B,
              kappa=2 * M_STAR / EPS_B)
w = lambda x: 0.25 / torch.cosh(x / 2) ** 2
df = lambda x, phi: w(x) * torch.cos(2 * phi) / T0

torch.set_default_device("cuda")
x1 = torch.zeros(1)
phi1 = torch.tensor([0.3])          # generic, off-axis

red = _kernels.exact_collision_reference(
    df, x1, phi1, linearize=True, n_xi=48, xi_cut=9.0, n_phi=2048,
    chunk=1, **common)[0].item()
print(f"reduced (n_xi=48, n_phi=2048) = {red:.6e}")

res = {}
for sg in (0.4, 0.3, 0.2, 0.15):
    for npx in (128, 0):            # 0 = the calibrated auto rule
        u = _kernels.unreduced_collision_reference(
            df, x1, phi1, linearize=True, sigma=sg, n_phi=npx, n_xi3=24,
            xi_cut=9.0, x2chunk=2, **common)[0].item()
        res[(sg, npx)] = u
        print(f"  sigma={sg}  n_phi={'auto' if npx == 0 else npx:>5}  {u:.6e}"
              f"   rel {abs(u / red - 1):.3%}", flush=True)

for npx in (128, 0):
    tag = "auto" if npx == 0 else npx
    r = (4 * res[(0.2, npx)] - res[(0.4, npx)]) / 3
    print(f"Richardson(0.4,0.2)  n_phi={tag}: {r:.6e}  rel {abs(r/red-1):.3%}")
    r2 = (4 * res[(0.15, npx)] - res[(0.3, npx)]) / 3
    print(f"Richardson(0.3,0.15) n_phi={tag}: {r2:.6e}  rel {abs(r2/red-1):.3%}")
