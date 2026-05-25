"""
Nodal Discontinuous Galerkin on triangles, 2D — reference engine.

Faithful NumPy port of the operators in Hesthaven & Warburton,
"Nodal Discontinuous Galerkin Methods" (Springer, 2008), specialized to
the operator qimpy.transport needs on the spatial domain:

    d rho_k / dt  +  v_k . grad rho_k  =  0           (per k-channel)

with v_k a *constant* 2-vector (qimpy's material.transport_velocity row).
Collision is added pointwise elsewhere, so the spatial solver is a bank of
constant-coefficient linear scalar advections, vectorized over channels.

Every operation here is a dense matmul or an index-gather, so the port to
torch (qimpy's array library, CPU/GPU) is mechanical.
"""

from __future__ import annotations
import numpy as np

# --------------------------------------------------------------------------- #
#  1D building blocks: Jacobi polynomials, Gauss / Gauss-Lobatto nodes        #
# --------------------------------------------------------------------------- #

def jacobi_p(x, alpha, beta, N):
    """Orthonormal Jacobi polynomial P_N^{(alpha,beta)} evaluated at x (vector)."""
    x = np.asarray(x, dtype=float).ravel()
    PL = np.zeros((N + 1, x.size))
    from math import gamma
    g0 = (2.0 ** (alpha + beta + 1) / (alpha + beta + 1)
          * gamma(alpha + 1) * gamma(beta + 1) / gamma(alpha + beta + 1))
    PL[0] = 1.0 / np.sqrt(g0)
    if N == 0:
        return PL[0]
    g1 = (alpha + 1) * (beta + 1) / (alpha + beta + 3) * g0
    PL[1] = ((alpha + beta + 2) * x / 2 + (alpha - beta) / 2) / np.sqrt(g1)
    if N == 1:
        return PL[N]
    aold = 2.0 / (2 + alpha + beta) * np.sqrt(
        (alpha + 1) * (beta + 1) / (alpha + beta + 3))
    for i in range(1, N):
        h1 = 2 * i + alpha + beta
        anew = 2.0 / (h1 + 2) * np.sqrt(
            (i + 1) * (i + 1 + alpha + beta) * (i + 1 + alpha) * (i + 1 + beta)
            / (h1 + 1) / (h1 + 3))
        bnew = -(alpha ** 2 - beta ** 2) / h1 / (h1 + 2)
        PL[i + 1] = (1.0 / anew) * (-aold * PL[i - 1] + (x - bnew) * PL[i])
        aold = anew
    return PL[N]


def grad_jacobi_p(r, alpha, beta, N):
    """Derivative of orthonormal Jacobi polynomial."""
    r = np.asarray(r, dtype=float).ravel()
    if N == 0:
        return np.zeros_like(r)
    return np.sqrt(N * (N + alpha + beta + 1)) * jacobi_p(r, alpha + 1, beta + 1, N - 1)


def jacobi_gq(alpha, beta, N):
    """N+1 Gauss-quadrature nodes/weights for weight (1-x)^a (1+x)^b (Golub-Welsch)."""
    if N == 0:
        return np.array([(alpha - beta) / (alpha + beta + 2)]), np.array([2.0])
    h1 = 2 * np.arange(N + 1) + alpha + beta
    d = -(alpha ** 2 - beta ** 2) / (h1 * (h1 + 2) + (h1 == 0))
    j = np.diag(d)
    i = np.arange(1, N + 1)
    h1i = h1[:N]
    e = (2.0 / (h1i + 2) * np.sqrt(
        i * (i + alpha + beta) * (i + alpha) * (i + beta)
        / (h1i + 1) / (h1i + 3)))
    j = j + np.diag(e, 1) + np.diag(e, -1)
    if alpha + beta < 10 * np.finfo(float).eps:
        j[0, 0] = 0.0
    x, V = np.linalg.eigh(j)
    return x, (V[0] ** 2)


def jacobi_gl(alpha, beta, N):
    """N+1 Gauss-Lobatto nodes for weight (1-x)^a (1+x)^b."""
    if N == 1:
        return np.array([-1.0, 1.0])
    xint, _ = jacobi_gq(alpha + 1, beta + 1, N - 2)
    return np.concatenate(([-1.0], np.sort(xint), [1.0]))


def vandermonde_1d(N, r):
    r = np.asarray(r, dtype=float).ravel()
    V = np.zeros((r.size, N + 1))
    for j in range(N + 1):
        V[:, j] = jacobi_p(r, 0, 0, j)
    return V


# --------------------------------------------------------------------------- #
#  Reference-triangle nodal set (warp & blend) and orthonormal basis          #
# --------------------------------------------------------------------------- #

_ALPOPT = np.array([0.0, 0.0, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999, 1.2832,
                    1.3648, 1.4773, 1.4959, 1.5743, 1.5770, 1.6223, 1.6258])


def warp_factor(N, rout):
    rout = np.asarray(rout, dtype=float).ravel()
    LGLr = jacobi_gl(0, 0, N)
    req = np.linspace(-1, 1, N + 1)
    Veq = vandermonde_1d(N, req)
    Pmat = np.zeros((N + 1, rout.size))
    for i in range(N + 1):
        Pmat[i] = jacobi_p(rout, 0, 0, i)
    Lmat = np.linalg.solve(Veq.T, Pmat)
    warp = Lmat.T @ (LGLr - req)
    zerof = (np.abs(rout) < 1.0 - 1.0e-10).astype(float)
    sf = 1.0 - (zerof * rout) ** 2
    return warp / sf + warp * (zerof - 1.0)


def nodes2d(N):
    """Warp-and-blend nodes on the equilateral triangle, returned as (r,s)."""
    alpha = _ALPOPT[N - 1] if N < 16 else 5.0 / 3.0
    Np = (N + 1) * (N + 2) // 2
    L1 = np.zeros(Np); L2 = np.zeros(Np); L3 = np.zeros(Np)
    sk = 0
    for n in range(N + 1):
        for m in range(N + 1 - n):
            L1[sk] = n / N
            L3[sk] = m / N
            sk += 1
    L2 = 1.0 - L1 - L3
    x = -L2 + L3
    y = (-L2 - L3 + 2 * L1) / np.sqrt(3.0)
    blend1 = 4 * L2 * L3
    blend2 = 4 * L1 * L3
    blend3 = 4 * L1 * L2
    warpf1 = warp_factor(N, L3 - L2)
    warpf2 = warp_factor(N, L1 - L3)
    warpf3 = warp_factor(N, L2 - L1)
    warp1 = blend1 * warpf1 * (1 + (alpha * L1) ** 2)
    warp2 = blend2 * warpf2 * (1 + (alpha * L2) ** 2)
    warp3 = blend3 * warpf3 * (1 + (alpha * L3) ** 2)
    x = x + warp1 + np.cos(2 * np.pi / 3) * warp2 + np.cos(4 * np.pi / 3) * warp3
    y = y + 0 * warp1 + np.sin(2 * np.pi / 3) * warp2 + np.sin(4 * np.pi / 3) * warp3
    return xy_to_rs(x, y)


def xy_to_rs(x, y):
    L1 = (np.sqrt(3.0) * y + 1.0) / 3.0
    L2 = (-3.0 * x - np.sqrt(3.0) * y + 2.0) / 6.0
    L3 = (3.0 * x - np.sqrt(3.0) * y + 2.0) / 6.0
    r = -L2 + L3 - L1
    s = -L2 - L3 + L1
    return r, s


def rs_to_ab(r, s):
    """Collapsed coordinates for the orthonormal basis (singular at top vertex)."""
    denom = np.where(np.abs(s - 1.0) > 1e-10, 1.0 - s, 1.0)
    a = np.where(np.abs(s - 1.0) > 1e-10, 2.0 * (1.0 + r) / denom - 1.0, -1.0)
    b = s
    return a, b


def simplex2d_p(a, b, i, j):
    h1 = jacobi_p(a, 0, 0, i)
    h2 = jacobi_p(b, 2 * i + 1, 0, j)
    return np.sqrt(2.0) * h1 * h2 * (1.0 - b) ** i


def grad_simplex2d_p(a, b, idx, jdx):
    fa = jacobi_p(a, 0, 0, idx); dfa = grad_jacobi_p(a, 0, 0, idx)
    gb = jacobi_p(b, 2 * idx + 1, 0, jdx); dgb = grad_jacobi_p(b, 2 * idx + 1, 0, jdx)
    # r-derivative
    dmodedr = dfa * gb
    if idx > 0:
        dmodedr = dmodedr * ((0.5 * (1 - b)) ** (idx - 1))
    # s-derivative
    dmodeds = dfa * (gb * (0.5 * (1 + a)))
    if idx > 0:
        dmodeds = dmodeds * ((0.5 * (1 - b)) ** (idx - 1))
    tmp = dgb * ((0.5 * (1 - b)) ** idx)
    if idx > 0:
        tmp = tmp - 0.5 * idx * gb * ((0.5 * (1 - b)) ** (idx - 1))
    dmodeds = dmodeds + fa * tmp
    dmodedr = 2.0 ** (idx + 0.5) * dmodedr
    dmodeds = 2.0 ** (idx + 0.5) * dmodeds
    return dmodedr, dmodeds


def vandermonde2d(N, r, s):
    a, b = rs_to_ab(r, s)
    Np = (N + 1) * (N + 2) // 2
    V = np.zeros((r.size, Np))
    sk = 0
    for i in range(N + 1):
        for j in range(N + 1 - i):
            V[:, sk] = simplex2d_p(a, b, i, j)
            sk += 1
    return V


def grad_vandermonde2d(N, r, s):
    a, b = rs_to_ab(r, s)
    Np = (N + 1) * (N + 2) // 2
    Vr = np.zeros((r.size, Np)); Vs = np.zeros((r.size, Np))
    sk = 0
    for i in range(N + 1):
        for j in range(N + 1 - i):
            Vr[:, sk], Vs[:, sk] = grad_simplex2d_p(a, b, i, j)
            sk += 1
    return Vr, Vs


def dmatrices2d(N, r, s, V):
    Vr, Vs = grad_vandermonde2d(N, r, s)
    Vinv = np.linalg.inv(V)
    return Vr @ Vinv, Vs @ Vinv  # Dr, Ds


# --------------------------------------------------------------------------- #
#  Element geometry, face nodes, connectivity, LIFT                           #
# --------------------------------------------------------------------------- #

class DG2D:
    """Operator bundle + mesh maps for order-N nodal DG on a triangle mesh."""

    def __init__(self, N, VX, VY, EToV):
        self.N = N
        self.Np = (N + 1) * (N + 2) // 2
        self.Nfp = N + 1
        self.Nfaces = 3
        self.VX = np.asarray(VX, float); self.VY = np.asarray(VY, float)
        self.EToV = np.asarray(EToV, int)
        self.K = self.EToV.shape[0]

        self.r, self.s = nodes2d(N)
        self.V = vandermonde2d(N, self.r, self.s)
        self.invV = np.linalg.inv(self.V)
        self.Dr, self.Ds = dmatrices2d(N, self.r, self.s, self.V)
        self._face_nodes()
        self.LIFT = self._lift()
        self._coordinates()
        self._geometric_factors()
        self._normals()
        self._connect()
        self._build_maps()
        self._build_cubature()
        self._build_mass()

    def _build_cubature(self):
        """Collapsed-coordinate tensor Gauss cubature on the reference triangle,
        exact to high order so curved (varying-J) integrals are computed exactly."""
        nq = 2 * self.N + 1
        a, wa = np.polynomial.legendre.leggauss(nq)   # nodes/weights on [-1,1], sum(w)=2
        b, wb = np.polynomial.legendre.leggauss(nq)
        A, B = np.meshgrid(a, b, indexing="ij")
        WA, WB = np.meshgrid(wa, wb, indexing="ij")
        rc = (0.5 * (1 + A) * (1 - B) - 1).flatten()
        sc = B.flatten()
        Wc = (WA * WB * (1 - B) / 2).flatten()       # (r,s)<-(a,b) collapse Jacobian
        self._Wc = Wc
        self._Icub = vandermonde2d(self.N, rc, sc) @ self.invV  # (Ncub, Np)

    def _build_mass(self):
        """Per-element curved mass matrix M_k = I_c^T diag(W_c J_k) I_c and inverse,
        plus its row sums (the exact integration weights 1^T M_k)."""
        Icub = self._Icub
        self.Jcub = Icub @ self.J                    # (Ncub, K)
        WIcub = self._Wc[:, None] * Icub             # (Ncub, Np)
        M = np.einsum("qi,qk,qj->kij", WIcub, self.Jcub, Icub)  # (K, Np, Np)
        self.Mass = M
        self.Minv_mass = np.linalg.inv(M)
        self.mass_rows = M.sum(axis=2).T             # (Np, K) = 1^T M_k

    # node indices on the three edges of the reference triangle
    def _face_nodes(self):
        N, r, s = self.N, self.r, self.s
        tol = 1e-10
        f1 = np.where(np.abs(s + 1) < tol)[0]
        f2 = np.where(np.abs(r + s) < tol)[0]
        f3 = np.where(np.abs(r + 1) < tol)[0]
        self.Fmask = np.vstack([f1, f2, f3])  # 3 x Nfp
        self.fmask_flat = self.Fmask.flatten()

    def _lift(self):
        N, Np, Nfp, Nfaces = self.N, self.Np, self.Nfp, self.Nfaces
        Emat = np.zeros((Np, Nfaces * Nfp))
        for f, edge in enumerate(self.Fmask):
            if f == 0:
                faceR = self.r[edge]
            elif f == 1:
                faceR = self.r[edge]
            else:
                faceR = self.s[edge]
            V1d = vandermonde_1d(N, faceR)
            massEdge = np.linalg.inv(V1d @ V1d.T)
            Emat[edge, f * Nfp:(f + 1) * Nfp] = massEdge
        self.Emat = Emat                       # raw face-mass extraction (weak form)
        self.Mref = np.linalg.inv(self.V @ self.V.T)  # reference mass matrix
        return self.V @ (self.V.T @ Emat)

    def _coordinates(self):
        va, vb, vc = self.EToV[:, 0], self.EToV[:, 1], self.EToV[:, 2]
        r, s = self.r[:, None], self.s[:, None]
        self.x = 0.5 * (-(r + s) * self.VX[va] + (1 + r) * self.VX[vb] + (1 + s) * self.VX[vc])
        self.y = 0.5 * (-(r + s) * self.VY[va] + (1 + r) * self.VY[vb] + (1 + s) * self.VY[vc])

    def _geometric_factors(self):
        xr = self.Dr @ self.x; xs = self.Ds @ self.x
        yr = self.Dr @ self.y; ys = self.Ds @ self.y
        J = -xs * yr + xr * ys
        self.J = J
        self.xr = xr; self.xs = xs; self.yr = yr; self.ys = ys  # cofactors (weak form)
        self.rx = ys / J; self.sx = -yr / J
        self.ry = -xs / J; self.sy = xr / J

    def _normals(self):
        N, Nfp, K = self.N, self.Nfp, self.K
        fm = self.fmask_flat
        xr = self.Dr @ self.x; xs = self.Ds @ self.x
        yr = self.Dr @ self.y; ys = self.Ds @ self.y
        fxr = xr[fm]; fxs = xs[fm]; fyr = yr[fm]; fys = ys[fm]
        nx = np.zeros((3 * Nfp, K)); ny = np.zeros((3 * Nfp, K))
        f1 = slice(0, Nfp); f2 = slice(Nfp, 2 * Nfp); f3 = slice(2 * Nfp, 3 * Nfp)
        nx[f1] = fyr[f1]; ny[f1] = -fxr[f1]
        nx[f2] = fys[f2] - fyr[f2]; ny[f2] = -fxs[f2] + fxr[f2]
        nx[f3] = -fys[f3]; ny[f3] = fxs[f3]
        sJ = np.sqrt(nx ** 2 + ny ** 2)
        self.nx = nx / sJ; self.ny = ny / sJ; self.sJ = sJ
        self.Fscale = sJ / self.J[fm]

    def _connect(self):
        K, Nfaces = self.K, self.Nfaces
        TotalFaces = Nfaces * K
        vn = np.array([[0, 1], [1, 2], [2, 0]])
        # sparse face-to-vertex via hashing of sorted vertex pairs
        face_id = {}
        EToE = np.tile(np.arange(K)[:, None], (1, Nfaces))
        EToF = np.tile(np.arange(Nfaces)[None, :], (K, 1))
        for k in range(K):
            for f in range(Nfaces):
                key = tuple(sorted((self.EToV[k, vn[f, 0]], self.EToV[k, vn[f, 1]])))
                if key in face_id:
                    k2, f2 = face_id[key]
                    EToE[k, f] = k2; EToF[k, f] = f2
                    EToE[k2, f2] = k; EToF[k2, f2] = f
                else:
                    face_id[key] = (k, f)
        self.EToE, self.EToF = EToE, EToF

    def _build_maps(self):
        K, Nfp, Nfaces, Np = self.K, self.Nfp, self.Nfaces, self.Np
        nodeids = np.arange(K * Np).reshape(K, Np)  # global node id, element-major
        vmapM = np.zeros((Nfp, Nfaces, K), dtype=int)
        vmapP = np.zeros((Nfp, Nfaces, K), dtype=int)
        for k in range(K):
            for f in range(Nfaces):
                vmapM[:, f, k] = nodeids[k, self.Fmask[f]]
        xf = self.x.flatten(order='F'); yf = self.y.flatten(order='F')
        for k in range(K):
            for f in range(Nfaces):
                k2 = self.EToE[k, f]; f2 = self.EToF[k, f]
                vidM = vmapM[:, f, k]; vidP = vmapM[:, f2, k2]
                xM, yM = xf[vidM], yf[vidM]
                xP, yP = xf[vidP], yf[vidP]
                # match by geometric distance (handles opposite ordering)
                D = (xM[:, None] - xP[None, :]) ** 2 + (yM[:, None] - yP[None, :]) ** 2
                j = np.argmin(D, axis=1)
                vmapP[:, f, k] = vidP[j]
        self.vmapM = vmapM.flatten(order='F')
        self.vmapP = vmapP.flatten(order='F')
        self.mapB = np.where(self.vmapM == self.vmapP)[0]
        self.vmapB = self.vmapM[self.mapB]
        # boundary face-node Cartesian coords (for inflow data)
        self.xb = xf[self.vmapB]; self.yb = yf[self.vmapB]

    # --------------------------------------------------------------------- #
    #  Advection RHS:  d u/dt = -(ax u_x + ay u_y) + surface upwind          #
    #  u has shape (Np, K[, C]); bc(xb, yb) returns exterior trace at        #
    #  inflow boundary nodes (used only where a.n < 0).                      #
    # --------------------------------------------------------------------- #
    def _weak_residual(self, u, ax, ay, bc=None):
        """Un-inverted weak RHS residual  (vol - surf)  of shape (Np,K,C).
        du/dt = M_k^{-1}(vol - surf). The conserved projection 1^T(vol - surf)
        telescopes to the boundary flux using ONLY fixed reference operators
        (no per-element mass inverse). u: (Np,K,C); ax,ay: (C,)."""
        Np, K, Nfp, Nfaces = self.Np, self.K, self.Nfp, self.Nfaces
        C = u.shape[-1]
        uf = u.reshape(Np * K, C, order='F')
        nxf = self.nx.flatten(order='F'); nyf = self.ny.flatten(order='F')
        adn = nxf[:, None] * ax[None, :] + nyf[:, None] * ay[None, :]   # (Nf, C)
        uM = uf[self.vmapM]
        uP = uf[self.vmapP].copy()
        if self.mapB.size:
            if bc is not None:
                ext = np.asarray(bc(self.xb, self.yb, uM[self.mapB]), float)
                if ext.ndim == 1:
                    ext = np.broadcast_to(ext[:, None], (self.mapB.size, C))
                inflow = adn[self.mapB] < 0.0
                uP[self.mapB] = np.where(inflow, ext, uM[self.mapB])
            else:
                uP[self.mapB] = uM[self.mapB]
        du = uM - uP
        fstar = adn * 0.5 * (uM + uP) + 0.5 * np.abs(adn) * du
        fstar = fstar.reshape(Nfaces * Nfp, K, C, order='F')
        Fx = ax * u; Fy = ay * u
        a = self.ys[..., None] * Fx - self.xs[..., None] * Fy
        b = -self.yr[..., None] * Fx + self.xr[..., None] * Fy
        Ma = np.einsum('ij,jkc->ikc', self.Mref, a)
        Mb = np.einsum('ij,jkc->ikc', self.Mref, b)
        vol = (np.einsum('ji,jkc->ikc', self.Dr, Ma)
               + np.einsum('ji,jkc->ikc', self.Ds, Mb))
        surf = np.einsum('if,fkc->ikc', self.Emat, self.sJ[..., None] * fstar)
        return vol - surf

    def apply_mass(self, u):
        """Mass-weighted variable w = M_k u (per element, per channel)."""
        sq = (u.ndim == 2); uu = u[..., None] if sq else u
        w = np.einsum('kij,jkc->ikc', self.Mass, uu)
        return w[..., 0] if sq else w

    def apply_mass_inv(self, w):
        """Recover density u = M_k^{-1} w (per element, per channel)."""
        sq = (w.ndim == 2); ww = w[..., None] if sq else w
        u = np.einsum('kij,jkc->ikc', self.Minv_mass, ww)
        return u[..., 0] if sq else u

    def advec_rhs(self, u, ax, ay, bc=None):
        """RHS of d u/dt + a.grad u = 0 (upwind), in the nodal density u.

        u   : (Np, K) or (Np, K, C) nodal field(s)
        ax,ay : scalar, or (C,) per-channel constant velocity components
        bc  : optional callable(xb, yb, uM) -> exterior trace at boundary nodes.
        """
        squeeze = (u.ndim == 2)
        if squeeze:
            u = u[..., None]
        C = u.shape[-1]
        ax = np.broadcast_to(np.atleast_1d(np.asarray(ax, float)), (C,))
        ay = np.broadcast_to(np.atleast_1d(np.asarray(ay, float)), (C,))
        res = self._weak_residual(u, ax, ay, bc)
        rhs = np.einsum('kij,jkc->ikc', self.Minv_mass, res)
        return rhs[..., 0] if squeeze else rhs

    def advec_rhs_w(self, w, ax, ay, bc=None):
        """RHS for the conservative variable w = M_k u:  d w/dt = vol - surf.

        Recovers u = M_k^{-1} w internally for the fluxes, but the returned
        d w/dt is NOT multiplied by any inverse, so 1^T(dw/dt) telescopes to the
        boundary flux with no mass inverse in the conserved direction. Evolving w
        is the natural conservative-variable form for the advection conservation
        law: 1^T w = sum_k 1^T M_k u_k is then advanced only by the (fixed-operator)
        face-flux balance.
        """
        squeeze = (w.ndim == 2)
        if squeeze:
            w = w[..., None]
        C = w.shape[-1]
        ax = np.broadcast_to(np.atleast_1d(np.asarray(ax, float)), (C,))
        ay = np.broadcast_to(np.atleast_1d(np.asarray(ay, float)), (C,))
        res = self._weak_residual(self.apply_mass_inv(w), ax, ay, bc)
        return res[..., 0] if squeeze else res

    def curve_boundary(self, curved_faces, project):
        """Make boundary elements isoparametric while preserving conformity.

        curved_faces : iterable of (element_k, local_face_f) on the true boundary
        project(x, y) -> (x, y) : snap points onto the exact boundary curve.

        The boundary face is bulged onto the curve and the displacement is blended
        into the element interior by a transfinite (Gordon-Hall) factor that
        vanishes on the other two edges and at all three vertices. The two edge
        endpoints (mesh vertices) are held fixed, so the other two edges -- and
        hence every shared interior face and neighbour -- stay exactly conforming.
        """
        r, s = self.r, self.s
        lam = [-(r + s) / 2.0, (1 + r) / 2.0, (1 + s) / 2.0]  # vertex barycentrics
        fv = {0: (0, 1, 2), 1: (1, 2, 0), 2: (2, 0, 1)}       # (ia, ib, opp) per face
        eps = 1e-12
        for k, f in curved_faces:
            ia, ib, ic = fv[f]
            fn = self.Fmask[f]
            xs0 = self.x[fn, k].copy(); ys0 = self.y[fn, k].copy()
            xc, yc = project(xs0, ys0)
            dxf = np.asarray(xc, float) - xs0; dyf = np.asarray(yc, float) - ys0
            dxf[0] = dxf[-1] = 0.0; dyf[0] = dyf[-1] = 0.0   # hold endpoints (vertices) fixed
            # 1D interpolation along the face (edge parameter t in [0,1] -> xi in [-1,1])
            la_f = lam[ia][fn]; lb_f = lam[ib][fn]
            xi_node = 2.0 * (lb_f / (la_f + lb_f)) - 1.0
            invV1d = np.linalg.inv(vandermonde_1d(self.N, xi_node))
            # blend displacement to every volume node
            La = lam[ia]; Lb = lam[ib]; denom = La + Lb
            t = np.where(denom > eps, Lb / np.where(denom > eps, denom, 1.0), 0.0)
            interp = vandermonde_1d(self.N, 2.0 * t - 1.0) @ invV1d   # (Np, Nfp)
            self.x[:, k] += (interp @ dxf) * denom    # denom = La+Lb = 1 - L_opp
            self.y[:, k] += (interp @ dyf) * denom
        self._geometric_factors()
        self._normals()
        self._build_mass()   # J changed: recompute curved mass matrices

    def integrate(self, f):
        """Exact domain integral of nodal field f (Np, K[, ...]) using curved
        cubature: integral = sum_q W_q J_q (I_c f)_q. Equals sum_k 1^T M_k f_k,
        the quantity the weak-form scheme conserves to machine precision."""
        fc = np.tensordot(self._Icub, f, axes=([1], [0]))    # (Ncub, K, ...)
        return np.einsum("q,qk,qk...->...", self._Wc, self.Jcub, fc)

    def make_periodic(self, lattice, tol=1e-6):
        """Link periodic boundary faces by pairing whole faces under the lattice
        shift, then matching their nodes reciprocally. Operating per face (not per
        node) guarantees every node of a periodic face is linked, with no orphans
        at shared seam vertices."""
        Nfp, Nfaces, K = self.Nfp, self.Nfaces, self.K
        self._lattice = np.atleast_2d(np.asarray(lattice, float))  # for periodicity queries
        nxf = self.nx.reshape(Nfp, Nfaces, K, order='F')
        nyf = self.ny.reshape(Nfp, Nfaces, K, order='F')
        vmapM3 = self.vmapM.reshape(Nfp, Nfaces, K, order='F')
        xf = self.x.flatten(order='F'); yf = self.y.flatten(order='F')
        bfaces = [(k, f) for k in range(K) for f in range(Nfaces)
                  if self.EToE[k, f] == k]

        def flatpos(f, k):
            return np.ravel_multi_index(
                (np.arange(Nfp), np.full(Nfp, f), np.full(Nfp, k)),
                (Nfp, Nfaces, K), order='F')

        linked_pos = []
        done: set = set()
        for L in np.atleast_2d(np.asarray(lattice, float)):
            Lh = L / np.linalg.norm(L)
            fn = {(k, f): np.array([nxf[0, f, k], nyf[0, f, k]]) for (k, f) in bfaces}
            plus = [(k, f) for (k, f) in bfaces
                    if (k, f) not in done and fn[(k, f)] @ Lh > 0.5]
            minus = [(k, f) for (k, f) in bfaces
                     if (k, f) not in done and fn[(k, f)] @ Lh < -0.5]
            mcent = {(k, f): np.array([xf[vmapM3[:, f, k]].mean(),
                                       yf[vmapM3[:, f, k]].mean()]) for (k, f) in minus}
            for (k, f) in plus:
                if (k, f) in done:
                    continue
                gp = vmapM3[:, f, k]; xp = xf[gp]; yp = yf[gp]
                target = np.array([xp.mean(), yp.mean()]) - L
                best, bd = None, 1e18
                for kf2 in minus:
                    if kf2 in done:
                        continue
                    dd = ((mcent[kf2] - target) ** 2).sum()
                    if dd < bd:
                        bd, best = dd, kf2
                if best is None or bd > tol * tol:
                    continue
                k2, f2 = best; gm = vmapM3[:, f2, k2]
                pos_p = flatpos(f, k); pos_m = flatpos(f2, k2)
                for a in range(Nfp):
                    b = int(np.argmin((xf[gm] - (xp[a] - L[0])) ** 2
                                      + (yf[gm] - (yp[a] - L[1])) ** 2))
                    self.vmapP[pos_p[a]] = gm[b]
                    self.vmapP[pos_m[b]] = gp[a]
                    linked_pos += [pos_p[a], pos_m[b]]
                done.add((k, f)); done.add((k2, f2))
        linked_pos = np.array(linked_pos, int)
        mask = ~np.isin(self.mapB, linked_pos)
        self.mapB = self.mapB[mask]
        self.vmapB = self.vmapM[self.mapB]
        self.xb = self.xb[mask]; self.yb = self.yb[mask]
        return int(np.unique(linked_pos).size)

    @property
    def dt_scale(self):
        """h_min/(N+1)^2 style CFL scale; multiply by 1/|v| for dt_max."""
        # inradius estimate from Fscale (= sJ/J): dt ~ min over nodes of 1/Fscale
        return float(1.0 / np.max(self.Fscale)) / (self.N + 1) ** 2


# --------------------------------------------------------------------------- #
#  Low-storage 5-stage 4th-order Runge-Kutta (Carpenter-Kennedy)              #
# --------------------------------------------------------------------------- #
_RK4A = np.array([0.0,
                  -567301805773.0 / 1357537059087.0,
                  -2404267990393.0 / 2016746695238.0,
                  -3550918686646.0 / 2091501179385.0,
                  -1275806237668.0 / 842570457699.0])
_RK4B = np.array([1432997174477.0 / 9575080441755.0,
                  5161836677717.0 / 13612068292357.0,
                  1720146321549.0 / 2090206949498.0,
                  3134564353537.0 / 4481467310338.0,
                  2277821191437.0 / 14882151754819.0])
_RK4C = np.array([0.0,
                  1432997174477.0 / 9575080441755.0,
                  2526269341429.0 / 6820363962896.0,
                  2006345519317.0 / 3224310063776.0,
                  2802321613138.0 / 2924317926251.0])


def lserk4_step(dg, u, ax, ay, dt, t, resu, bc=None):
    for i in range(5):
        rhs = dg.advec_rhs(u, ax, ay, bc=bc)
        resu = _RK4A[i] * resu + dt * rhs
        u = u + _RK4B[i] * resu
    return u, resu
