"""Boundary polygon for the exact ballistic solver, from a qimpy mesh file."""
from __future__ import annotations

import numpy as np
import torch

AU = 18897.0  #: bohr per micron -- the same scale factor the FV meshes use
EPS = 1e-9  #: ray-start exclusion, in microns


class Polygon:
    """Boundary segments of a qimpy mesh, with a marker id per segment.

    Reads exactly the same ``.npz`` the finite-volume geometry reads
    (``vertices``, ``boundary_edges``, ``boundary_markers``), so any mesh that
    can be run through :class:`~qimpy.transport.Transport` can be run through
    the analytic ballistic solver without conversion.

    Marker ids are assigned from the mesh's own marker strings: ``wall`` is
    always 0, and every other distinct marker gets 1, 2, ... in sorted order.

    ⛔ Marker names are NOT hardcoded to source/drain.  The scratch version of
    this solver had ``code = {"wall": 0, "source": 1, "drain": 2}`` and raised
    KeyError on any other name, so it could not run the seven- and
    nine-terminal devices at all -- exactly the geometries where an independent
    check is most valuable.
    """

    def __init__(self, mesh_file: str, device: torch.device) -> None:
        d = np.load(mesh_file, allow_pickle=True)
        V = d["vertices"] / AU  # microns
        BE = np.asarray(d["boundary_edges"])
        BM = np.array([str(s) for s in d["boundary_markers"]])
        # ⛔ str(), not the raw numpy scalars: np.load gives np.str_, which
        # prints as "np.str_('drain')" and compares unequal to a plain
        # "drain" in a dict key round-trip.
        names = sorted(str(x) for x in set(BM) - {"wall"})
        self.contact_names: list[str] = names
        self.code: dict[str, int] = {"wall": 0}
        self.code.update({nm: i + 1 for i, nm in enumerate(names)})

        self.a = torch.as_tensor(V[BE[:, 0]], dtype=torch.float64, device=device)
        self.b = torch.as_tensor(V[BE[:, 1]], dtype=torch.float64, device=device)
        self.kind = torch.as_tensor([self.code[m] for m in BM],
                                    dtype=torch.int64, device=device)
        e = self.b - self.a
        L = e.norm(dim=1, keepdim=True)
        assert float(L.min()) > 0.0, "mesh has a zero-length boundary edge"
        self.t_hat = e / L
        n = torch.stack([e[:, 1], -e[:, 0]], dim=1) / L

        # ⛔ Orient outward by an actual point-in-polygon test, NOT by
        # (centre - centroid).n > 0.  That heuristic is nearly tangent on a
        # re-entrant cross: for the junction geometry the dot product falls to
        # ~1% of |centre| at small edge length and flips sign, silently
        # inverting the specular reflection on those segments.
        cen = 0.5 * (self.a + self.b)
        step = 1e-4 * float(L.min())
        probe = (cen + step * n).cpu().numpy()
        flip = torch.as_tensor(self._inside(V, BE, probe), device=device)[:, None]
        self.n_hat = torch.where(flip, -n, n)
        self.vertices = V
        self.device = device

    @staticmethod
    def _inside(V: np.ndarray, BE: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Even-odd crossing test on raw arrays (runs before self is built)."""
        a, b = V[BE[:, 0]], V[BE[:, 1]]
        cnt = np.zeros(len(q), dtype=int)
        for i in range(len(a)):
            y0, y1 = a[i, 1], b[i, 1]
            cond = (y0 > q[:, 1]) != (y1 > q[:, 1])
            with np.errstate(divide="ignore", invalid="ignore"):
                xi = a[i, 0] + (q[:, 1] - y0) * (b[i, 0] - a[i, 0]) / (y1 - y0)
            cnt += (cond & (q[:, 0] < xi)).astype(int)
        return cnt % 2 == 1

    def hit(self, p: torch.Tensor, v: torch.Tensor):
        """First boundary hit of rays ``(p, v)``; returns ``(s, segment)``.

        Closed-form ray/segment intersection, vectorised over all rays x all
        segments.  Rays leave from a boundary point, so ``s < EPS`` (the
        segment they start on) is excluded.
        """
        a, b = self.a, self.b
        e = b - a
        det = v[:, None, 0] * (-e[None, :, 1]) - v[:, None, 1] * (-e[None, :, 0])
        rhs = a[None, :, :] - p[:, None, :]
        s = (rhs[..., 0] * (-e[None, :, 1])
             - rhs[..., 1] * (-e[None, :, 0])) / det
        u = (v[:, None, 0] * rhs[..., 1] - v[:, None, 1] * rhs[..., 0]) / det
        ok = (s > EPS) & (u >= -1e-12) & (u <= 1 + 1e-12) & det.abs().gt(0)
        s = torch.where(ok, s, torch.full_like(s, float("inf")))
        return s.min(dim=1)

    def contact_segments(self, name: str) -> torch.Tensor:
        """Indices of the boundary segments carrying contact ``name``."""
        if name not in self.code:
            raise KeyError(f"{name!r} not in mesh markers {sorted(self.code)}")
        return torch.where(self.kind == self.code[name])[0]
