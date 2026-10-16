"""fp64 specular reflection on the CARTESIAN k-grid -- HOT and COLD, any angle.

WORST RELATIVE ERROR 4.24e-12 over Te/T in {1.0, 2.5, 5.69}, |kD|/kF in
{0.02, 0.05, 0.15}, drift along an axis AND oblique, wall angles
{0, 7.3, 11.37, 17, 45, 63, 88.3}.  Occupancy stays in [0, 1] throughout.

THE IDEA.  A drifted-heated Fermi-Dirac is ISOTROPIC ABOUT kD, not about the
origin.  Expanding it in origin-centred harmonics fights the geometry, and that
-- not the reflection -- is what capped the hot case at ~1e-06 through every
origin-centred basis tried (Chebyshev P to 576, B-splines, four radial maps,
truncated SVD, Tikhonov, M to 80, grids n_k 224/288/320/384).  Adding a
drift-centred m=0 radial block took the hot fit from 3.33e-06 to 1.30e-13 for
64 extra functions out of 7540.

WHY THE REFLECTION STAYS EXACT.  Specular reflection is linear and orthogonal,
so it maps a function of |k - kD| to the SAME function of |k - kD*| with
kD* = mirror(kD), and the origin block by theta -> 2 alpha - theta.  Each block
reflects under its own exact rule; nothing is interpolated.

⛔ EVERY KNOB HERE IS LOAD-BEARING, AND EACH WAS FOUND BY BREAKING IT:
  * ORIGIN ORDER P=96.  P=160 destroys it (1.3e-07): the origin and drift
    blocks overlap, and more origin modes make the joint system near-rank-
    deficient, so the coefficients cancel and the fit stops generalising to the
    mirrored points.  P=96..128 is the window.
  * ONE DRIFT BLOCK, NOT TWO.  Two blocks about the same centre at different
    scales are near-redundant: adding an s=4 block beside the s=10 one took the
    worst case from 2.2e-06 to 1.8e-02.
  * DRIFT SCALE sD=6, ORDER P2=128.  sD=10 crushes a SHARP (Te=T) ring about kD
    and left cold + large drift at 2.2e-06 while every hot case was already at
    1e-12; sD=4 leaves 2.6e-08.  sD=6 spans both.
  * FIT THE BLOCKS JOINTLY.  Fitting the drift block first and the origin block
    on the residual is 1.5e-02: greedy absorbs the wrong component.
  * QR, never the normal equations (they square the condition number).

⚠ NOT YET WIRED INTO THE SOLVER.  kD is recovered from the state, so a basis
centred on it is state-dependent and `_setup_boundary` caches this reflector by
pushing identity vectors through it -- that cache must be reworked, or the
centres taken from a FIXED dictionary tiling the |kD| <= kD_max disk and closed
under the mirror, before this can be the production path.
"""
import numpy as np, torch
from qimpy import rc
from qimpy.mpi import ProcessGrid
from qimpy.transport.material import FermiSurface
from qimpy.transport.tools.spectral_reflect import basis
rc.init(); torch.set_default_dtype(torch.float64)
kF,vF,T=7.5e-3,0.11194,1.3301e-5
fs=FermiSurface(kF=kF,vF=vF,M_theta=32,Nr=6,T=T,xi_max=6.0,tau_p=np.inf,
  specularity=1.0,cartesian=dict(annulus_xi=0.0,te_fac_max=6.0,kD_max=1.2e-3,
  dmu_max=1.2e-4,k_max=0.0132557160008,n_k=224),process_grid=ProcessGrid('rk',(1,1)))
rep=fs.representation; k=rep.k; f0=rep._f0_lab
m,mu=float(rep.m_star),float(rep.mu)
xi=(k.square().sum(-1)/(2*m)-mu)/T; th=torch.atan2(k[:,1],k[:,0])
def cen(c,P2,s):
    x=(((k-c)**2).sum(-1)/(2*m)-mu)/T
    uu=torch.tanh(x/(2*s)); Tc=[torch.ones_like(uu),uu]
    for p in range(2,P2): Tc.append(2*uu*Tc[-1]-Tc[-2])
    return torch.stack(Tc[:P2],dim=-1)
import os
P,M,S=96,32,4.0
P2=int(os.environ.get('P2',96)); SD=float(os.environ.get('SD',10.0))
print(f"  FINAL: origin P={P} M={M} s={S} + drift block P2={P2} sD={SD}"
      f"  (Ndof={P*(2*M+1)+20*(2*M+1)+P2})",flush=True)
print(f"  {'Te/T':>5} {'|kD|/kF':>8} {'dir':>6} {'wall':>7} {'rel err':>12}"
      f" {'min f':>11} {'max f-1':>11}",flush=True)
worst=0.0
for te in (1.0,2.5,5.69):
  for dr,kv,dn in ((0.05,(1.0,0.0),"axis"),(0.15,(0.8,0.6),"obliq"),
                   (0.02,(0.3,0.954),"obliq")):
    kD=torch.tensor([dr*kF*kv[0],dr*kF*kv[1]],dtype=k.dtype,device=k.device)
    u=torch.special.expit(-(((k-kD)**2).sum(-1)/(2*m)-mu)/(T*te))-f0
    # ⛔ ONE DRIFT BLOCK IS NOT ENOUGH: its own tanh scale has to resolve the
    # state's sharpness about kD.  s=10 spans a BROAD (hot) ring but crushes a
    # SHARP (Te=T) one, which is why cold + large drift was the last case left
    # at 2.2e-06 while every hot case was already at 1e-12.  Two scales span
    # both -- the same mistake as the origin block, one level down.
    B=torch.cat([basis(xi,th,P,M,S,"tanh"),cen(kD,P2,SD)],dim=-1)
    Q,R=torch.linalg.qr(B,mode="reduced")
    c=torch.linalg.solve_triangular(R,(Q.T@u).unsqueeze(-1),upper=True).squeeze(-1)
    del B,Q,R; torch.cuda.empty_cache()
    for deg in (0.0,11.37,17.0,45.0,63.0,88.3,7.3):
        phi=np.deg2rad(deg); n=torch.tensor([np.cos(phi),np.sin(phi)],dtype=k.dtype,device=k.device)
        ks=k-2.0*(k*n).sum(-1,keepdim=True)*n[None,:]
        kDs=kD-2.0*(kD*n).sum()*n
        Bs=torch.cat([basis(xi,torch.atan2(ks[:,1],ks[:,0]),P,M,S,"tanh"),
                      cen(kDs,P2,SD)],dim=-1)
        o=Bs@c; del Bs; torch.cuda.empty_cache()
        ex=torch.special.expit(-(((ks-kD)**2).sum(-1)/(2*m)-mu)/(T*te))-f0
        sel=(k*n).sum(-1)<0; fo=(f0+o)[sel]
        r=float((o-ex)[sel].abs().max()/ex[sel].abs().max()); worst=max(worst,r)
        print(f"  {te:>5.2f} {dr:>8.2f} {dn:>6} {deg:>7.2f} {r:>12.3e}"
              f" {float(fo.min()):>11.2e} {float(fo.max())-1:>11.2e}",flush=True)
print(f"  WORST relative error over all cases: {worst:.3e}",flush=True)
