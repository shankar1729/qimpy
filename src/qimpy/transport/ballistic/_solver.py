"""Exact steady ballistic transport by the method of characteristics.

Steady ballistic Boltzmann is ``v.grad f = 0``: f is constant along straight
characteristics that reflect specularly off walls.  So the steady state needs
no time integration at all -- trace each ray BACKWARD until it reaches a
contact and read off that reservoir's Fermi-Dirac.  The transport operator is
linear and the only nonlinearity lives in the boundary data, so sampling the
exact FD (not its linearisation) gives the exact NONLINEAR ballistic answer at
any bias.

This is an INDEPENDENT check on the finite-volume solver: no mesh cells, no
k-grid, no time step, no limiter, no wall closure -- none of the machinery
whose correctness it is used to test.

⛔ TRAPPED TRAJECTORIES ARE PHYSICS, NOT A BUG.  In a specular polygon a finite
measure of rays never reaches a contact; for those f is set by history, not by
the boundary data.  They are counted and reported (``unresolved``) rather than
quietly filled in, which is what numerical diffusion does inside an FV solver.
The unresolved fraction decays only as ~N^(-1/2) in bounce count, so it is the
dominant uncertainty of this method and must be reported with every number.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ._polygon import AU, EPS, Polygon

KB = 3.166811563e-6  #: Hartree per Kelvin


class Ballistic:
    """Exact ballistic solver on a qimpy mesh.

    Parameters mirror :class:`~qimpy.transport.material.FermiSurface`: a
    parabolic band with Fermi wavevector ``kF``, Fermi velocity ``vF`` and
    temperature ``T`` (Hartree), driven by per-contact ``dmu``.
    """

    def __init__(self, mesh_file: str, *, kF: float, vF: float, T: float,
                 contacts: dict[str, float], spin: float = 2.0,
                 device: Optional[torch.device] = None) -> None:
        dev = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.poly = Polygon(mesh_file, dev)
        self.kF, self.vF, self.T, self.spin = kF, vF, T, spin
        self.m_star = kF / vF
        self.E_F = 0.5 * kF * vF
        unknown = set(contacts) - set(self.poly.code)
        if unknown:
            raise KeyError(f"contacts {sorted(unknown)} are not markers in the "
                           f"mesh (has {sorted(self.poly.code)})")
        # every contact present in the mesh but undriven sits at equilibrium
        self.dmu = {nm: float(contacts.get(nm, 0.0))
                    for nm in self.poly.contact_names}
        self.device = dev

    # -- energy moments ----------------------------------------------------
    def _moments(self, dmu: float, n_xi: int = 96, xi_max: float = 12.0):
        """``(Wn, Wj)``: density and current energy weights for a shift dmu.

        ``Wn = int deps [FD(eps; mu+dmu) - FD(eps; mu)] * m*`` and
        ``Wj = int deps [..] * k(eps)``.  Both are relative to equilibrium, so
        the inert sea below the window cancels and only the shell contributes.
        """
        xg, xw = np.polynomial.legendre.leggauss(n_xi)
        xi = torch.as_tensor(xi_max * xg, dtype=torch.float64, device=self.device)
        w = torch.as_tensor(xi_max * xw, dtype=torch.float64, device=self.device)
        df = torch.special.expit(-(xi - dmu / self.T)) - torch.special.expit(-xi)
        eps = self.E_F + self.T * xi
        k = torch.sqrt((2 * self.m_star * eps).clamp(min=0.0))
        return (float((w * df).sum() * self.T * self.m_star),
                float((w * df * k).sum() * self.T))

    # -- characteristics ---------------------------------------------------
    @torch.no_grad()
    def trace_back(self, p0: torch.Tensor, v0: torch.Tensor,
                   max_bounce: int = 12800):
        """Backward-trace rays; returns ``(contact_id, alive)``.

        ``contact_id == 0`` means the ray never reached a contact within
        ``max_bounce`` reflections -- trapped, and therefore unresolved.
        """
        poly = self.poly
        p, v = p0.clone(), v0.clone()
        cid = torch.zeros(p.shape[0], dtype=torch.int64, device=p.device)
        alive = torch.ones(p.shape[0], dtype=torch.bool, device=p.device)
        for _ in range(max_bounce):
            if not alive.any():
                break
            ia = torch.where(alive)[0]
            s, idx = poly.hit(p[ia], v[ia])
            good = torch.isfinite(s)
            alive[ia[~good]] = False          # numerically escaped => trapped
            ia, s, idx = ia[good], s[good], idx[good]
            if ia.numel() == 0:
                continue
            p[ia] = p[ia] + s[:, None] * v[ia]
            kind = poly.kind[idx]
            at_contact = kind > 0
            cid[ia[at_contact]] = kind[at_contact]
            alive[ia[at_contact]] = False
            wl = ~at_contact
            if wl.any():
                iw = ia[wl]
                nh = poly.n_hat[idx[wl]]
                vv = v[iw]
                v[iw] = vv - 2.0 * (vv * nh).sum(1, keepdim=True) * nh
                p[iw] = p[iw] + EPS * v[iw]   # nudge off the wall
        return cid, alive

    # -- observables -------------------------------------------------------
    @torch.no_grad()
    def contact_current(self, name: str, *, n_ang: int = 1024,
                        n_edge: int = 48, max_bounce: int = 12800,
                        chunk_rays: int = 2_000_000) -> tuple[float, float]:
        """Net current through contact ``name``, in atomic units.

        Returns ``(I, unresolved_fraction)``.  Outward directions carry f
        traced back into the device; inward directions carry this contact's own
        reservoir.  The equilibrium part integrates to zero by angular
        symmetry, so deviations suffice.
        """
        poly = self.poly
        dev = self.device
        sel = poly.contact_segments(name)
        if sel.numel() == 0:
            return 0.0, 0.0
        a, b, nh = poly.a[sel], poly.b[sel], poly.n_hat[sel]
        seg_len = (b - a).norm(dim=1)
        u = (torch.arange(n_edge, device=dev, dtype=torch.float64) + 0.5) / n_edge
        P = (a[:, None, :] + u[None, :, None] * (b - a)[:, None, :]).reshape(-1, 2)
        NH = nh[:, None, :].expand(-1, n_edge, -1).reshape(-1, 2)
        dl = (seg_len[:, None] / n_edge).expand(-1, n_edge).reshape(-1)
        th = (torch.arange(n_ang, device=dev, dtype=torch.float64) + 0.5) \
            * (2 * np.pi / n_ang)
        vdir = torch.stack([torch.cos(th), torch.sin(th)], dim=1)

        Wj = {0: 0.0}
        for nm, dmu in self.dmu.items():
            Wj[poly.code[nm]] = self._moments(dmu)[1]
        self_w = Wj[poly.code[name]]

        tot = 0.0
        n_trap = 0
        n_out = 0
        per = max(1, chunk_rays // max(n_ang, 1))
        for lo in range(0, P.shape[0], per):
            hi = min(lo + per, P.shape[0])
            pp, nn, ll = P[lo:hi], NH[lo:hi], dl[lo:hi]
            cosang = (vdir[None, :, :] * nn[:, None, :]).sum(-1)
            out = cosang > 0                                   # leaving device
            p0 = (pp[:, None, :] - 1e-7 * nn[:, None, :]).expand(-1, n_ang, 2)
            cid, _ = self.trace_back(
                p0.reshape(-1, 2).contiguous(),
                (-vdir)[None].expand(hi - lo, n_ang, 2).reshape(-1, 2).contiguous(),
                max_bounce)
            cid = cid.view(hi - lo, n_ang)
            w = torch.zeros_like(cosang)
            for c, val in Wj.items():
                w = torch.where(cid == c, torch.full_like(w, val), w)
            w = torch.where(out, w, torch.full_like(w, self_w))
            tot += float((ll[:, None] * cosang * w).sum())
            n_trap += int(((cid == 0) & out).sum())
            n_out += int(out.sum())
        I = self.spin / (2 * np.pi) ** 2 * (2 * np.pi / n_ang) * tot * AU
        return I, (n_trap / max(n_out, 1))

    def contact_currents(self, **kw) -> dict[str, float]:
        """Current through every contact in the mesh, in atomic units."""
        return {nm: self.contact_current(nm, **kw)[0]
                for nm in self.poly.contact_names}
