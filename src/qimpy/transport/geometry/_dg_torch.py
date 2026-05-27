"""Torch mirror of the DG2D advection RHS (the form qimpy needs: torch, GPU-ready).

Conservative weak (divergence) form with the per-element curved mass matrix.
The state evolved by the time-stepper is the mass-weighted variable w = M_k u
(see `rhs_w`): dw/dt = vol - surf carries no mass-inverse in the conserved
direction. Density u = M_k^{-1} w is recovered for fluxes, collisions, render, IO.

Boundary conditions on physical (non-periodic) faces are applied by overriding the
exterior trace uP at boundary nodes (the upwind flux then uses uP on inflow
channels, uM on outflow channels):
  - reflective walls : uP = reflector(uM)   (specular/diffuse, from material)
  - contacts         : uP = contactor(t)    (dmu-shifted distribution, from material)
  - default (none)   : uP = uM              (outflow)
"""
import numpy as np
import torch

class AdvectTorch:
    def __init__(self, dg, device="cpu", dtype=torch.float64):
        t = lambda a: torch.as_tensor(np.asarray(a), device=device, dtype=dtype)
        ti = lambda a: torch.as_tensor(np.asarray(a), device=device, dtype=torch.long)
        self.Np, self.K, self.Nfp, self.Nfaces = dg.Np, dg.K, dg.Nfp, dg.Nfaces
        self.Dr, self.Ds = t(dg.Dr), t(dg.Ds)
        self.Mref, self.Emat = t(dg.Mref), t(dg.Emat)
        self.Mass = t(dg.Mass)
        self.Minv_mass = t(dg.Minv_mass)
        self.xs, self.ys = t(dg.xs), t(dg.ys)
        self.xr, self.yr = t(dg.xr), t(dg.yr)
        self.sJ = t(dg.sJ)
        self.nxf = t(dg.nx.flatten(order='F')); self.nyf = t(dg.ny.flatten(order='F'))
        self.vmapM, self.vmapP = ti(dg.vmapM), ti(dg.vmapP)
        self.mapB = ti(dg.mapB)
        self.device, self.dtype = device, dtype
        # boundary operators (set via set_boundary); default = pure outflow
        self._reflect_sel = None      # LongTensor: indices into mapB that reflect
        self._reflector = None        # callable(rho[1,Nsel,C]) -> [1,Nsel,C]
        self._specularity = 1.0       # 1 = specular; <1 adds flux-consistent diffuse
        self._contacts = []           # list of (sel LongTensor into mapB, contactor(t))
        # surface quadrature weight per boundary face-node (for boundary integrals
        # such as contact currents): column-sums of the lift operator times sJ.
        ecs = self.Emat.sum(0)                       # (Nfaces*Nfp,)
        self.face_quad_w = (ecs[None, :] * self.sJ.T).reshape(-1)  # (K*Nfaces*Nfp,)

    def set_boundary(self, reflect_sel=None, reflector=None, contacts=(),
                     specularity=1.0):
        """Register physical-boundary operators.
        reflect_sel : indices into self.mapB whose nodes reflect (rest = outflow)
        reflector   : PURE-SPECULAR material reflector on those nodes' normals
        contacts    : iterable of (sel_into_mapB, contactor(t)) Dirichlet patches
        specularity : 1 = specular; <1 mixes in flux-consistent diffuse reflection
        """
        if reflect_sel is not None:
            reflect_sel = torch.as_tensor(reflect_sel, device=self.device,
                                          dtype=torch.long)
        self._reflect_sel = reflect_sel
        self._reflector = reflector
        self._specularity = float(specularity)
        self._contacts = [(torch.as_tensor(s, device=self.device, dtype=torch.long), c)
                          for s, c in contacts]

    def boundary_normals(self, sel=None):
        """Outward unit normals at boundary nodes (all mapB, or a sub-selection)."""
        pos = self.mapB if sel is None else self.mapB[
            torch.as_tensor(sel, device=self.device, dtype=torch.long)]
        return torch.stack([self.nxf[pos], self.nyf[pos]], dim=-1)

    def apply_mass(self, u):
        return torch.einsum('kij,jkc->ikc', self.Mass, u)

    def apply_mass_inv(self, w):
        return torch.einsum('kij,jkc->ikc', self.Minv_mass, w)

    def _residual(self, u, ax, ay, t=0.0, coupling=None, return_fstar=False):
        Np, K, Nfp, Nfaces = self.Np, self.K, self.Nfp, self.Nfaces
        C = u.shape[-1]
        uf = u.permute(1, 0, 2).reshape(Np * K, C)
        scalar = coupling is None
        adn = (self.nxf[:, None] * ax[None, :] + self.nyf[:, None] * ay[None, :]
               ) if scalar else None
        uM = uf[self.vmapM]; uP = uf[self.vmapP].clone()
        if self.mapB.numel():
            uP[self.mapB] = uM[self.mapB]                       # default: outflow
            if self._reflector is not None and self._reflect_sel.numel():
                pos = self.mapB[self._reflect_sel]
                uM_b = uM[pos]                                    # (Nsel, C)
                spec_uP = self._reflector(uM_b[None])[0]          # pure specular
                s = self._specularity
                if scalar and s < 1.0:
                    # flux-consistent diffuse: incoming trace = outgoing flux spread
                    # over incoming directions by |v.n|, so net wall flux stays 0.
                    adn_b = adn[pos]                              # (Nsel, C)
                    phi_out = (adn_b.clamp(min=0.0) * uM_b).sum(1, keepdim=True)
                    w_in = (-adn_b.clamp(max=0.0)).sum(1, keepdim=True).clamp(min=1e-300)
                    D = phi_out / w_in                           # (Nsel, 1)
                    uP[pos] = s * spec_uP + (1.0 - s) * D        # outflow uP ignored
                else:
                    uP[pos] = spec_uP                            # specular
            for sel, contactor in self._contacts:
                if sel.numel():
                    pos = self.mapB[sel]
                    cval = contactor(t)
                    uP[pos] = torch.as_tensor(cval, device=uP.device, dtype=uP.dtype) \
                        * torch.ones_like(uP[pos])               # broadcast (Nsel,C)
        if scalar:
            # discrete-ordinate / scalar channels: A_n is diagonal (= v.n per channel)
            fstar_flat = adn * 0.5 * (uM + uP) + 0.5 * adn.abs() * (uM - uP)
            Fx = ax * u; Fy = ay * u
        else:
            # coupled linear system: A_n = nx Ax + ny Ay, characteristic-upwind flux
            #   f* = 1/2 A_n (uM+uP) + 1/2 |A_n| (uM-uP),  |A_n| via face rotation.
            phi = torch.atan2(self.nyf, self.nxf)
            An_sum = (self.nxf[:, None] * ((uM + uP) @ coupling.Ax.T)
                      + self.nyf[:, None] * ((uM + uP) @ coupling.Ay.T))
            fstar_flat = 0.5 * An_sum + 0.5 * coupling.abs_flux(uM - uP, phi)
            Fx = u @ coupling.Ax.T; Fy = u @ coupling.Ay.T
        fstar = fstar_flat.reshape(K, Nfaces * Nfp, C).permute(1, 0, 2)
        a = self.ys[..., None] * Fx - self.xs[..., None] * Fy
        b = -self.yr[..., None] * Fx + self.xr[..., None] * Fy
        Ma = torch.einsum('ij,jkc->ikc', self.Mref, a)
        Mb = torch.einsum('ij,jkc->ikc', self.Mref, b)
        vol = (torch.einsum('ji,jkc->ikc', self.Dr, Ma)
               + torch.einsum('ji,jkc->ikc', self.Ds, Mb))
        surf = torch.einsum('if,fkc->ikc', self.Emat, self.sJ[..., None] * fstar)
        res = vol - surf
        if return_fstar:
            return res, fstar_flat[self.mapB]            # boundary numerical flux
        return res

    def rhs(self, u, ax, ay, t=0.0, coupling=None):
        """d u/dt in the nodal density u (= M_k^{-1}(vol - surf))."""
        return self.apply_mass_inv(self._residual(u, ax, ay, t, coupling))

    def rhs_w(self, w, ax, ay, t=0.0, coupling=None):
        """d w/dt for the conservative variable w = M_k u (= vol - surf)."""
        return self._residual(self.apply_mass_inv(w), ax, ay, t, coupling)
