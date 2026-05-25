"""DG triangle-mesh container and external-mesh I/O for TriSet.

qimpy does NOT generate meshes. The triangle mesh is produced by external
tooling (e.g. Shewchuk's `triangle`, gmsh, or a hand-written generator) and
supplied to ``TriSet`` as a file. This module defines the in-memory container
that ``DG2D`` consumes (``MeshResult``) and the loader/saver for the external
mesh format.

External mesh format (NumPy ``.npz``)
-------------------------------------
    vertices          (Nv, 2) float   node coordinates
    triangles         (K, 3)  int     triangle connectivity (CCW)
    boundary_edges    (Nb, 2) int     vertex-index pairs on the physical boundary
    boundary_markers  (Nb,)   str     marker name per boundary edge; a name that
                                       matches a key in the ``contacts`` dict is a
                                       contact, anything else (e.g. 'wall') reflects
    lattice           (nL, 2) float   OPTIONAL periodic displacement vectors

Only ``vertices`` and ``triangles`` are strictly required; without boundary
markers every physical face defaults to a reflective wall.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class MeshResult:
    """The mesh as DG2D and TriSet consume it (output of :func:`load_mesh`)."""
    VX: np.ndarray
    VY: np.ndarray
    EToV: np.ndarray
    edge_marker: dict              # sorted (vi, vj) -> marker id (>0)
    marker_names: list             # id -> name (id 0 reserved/unused)
    projectors: dict               # id -> curve-projection fn, or None (straight)
    _lattice: Optional[list] = None

    def face_markers(self, dg):
        """For each boundary (element, local-face), return (name, marker_id)."""
        vn = np.array([[0, 1], [1, 2], [2, 0]])
        out = {}
        for k in range(dg.K):
            for f in range(dg.Nfaces):
                if dg.EToE[k, f] != k:
                    continue                       # interior face
                a = dg.EToV[k, vn[f, 0]]; b = dg.EToV[k, vn[f, 1]]
                m = self.edge_marker.get(tuple(sorted((int(a), int(b)))), 0)
                name = self.marker_names[m] if m < len(self.marker_names) else "wall"
                out[(k, f)] = (name, m)
        return out


def load_mesh(path: str) -> MeshResult:
    """Read an external triangle mesh (see module docstring for the format)."""
    d = np.load(path, allow_pickle=True)
    V = np.asarray(d["vertices"], float)
    EToV = np.asarray(d["triangles"], int)
    VX = V[:, 0].copy(); VY = V[:, 1].copy()

    edge_marker: dict = {}
    marker_names = ["_"]                            # id 0 reserved
    if "boundary_edges" in d and "boundary_markers" in d:
        be = np.asarray(d["boundary_edges"], int)
        bn = [str(x) for x in np.asarray(d["boundary_markers"]).ravel()]
        name_id: dict = {}
        for (a, b), name in zip(be, bn):
            if name not in name_id:
                name_id[name] = len(marker_names)
                marker_names.append(name)
            edge_marker[tuple(sorted((int(a), int(b))))] = name_id[name]
    projectors = {i: None for i in range(len(marker_names))}

    mesh = MeshResult(VX, VY, EToV, edge_marker, marker_names, projectors)
    if "lattice" in d:
        lat = np.asarray(d["lattice"], float)
        if lat.size:
            mesh._lattice = [row.copy() for row in lat]
    return mesh


def save_mesh(path: str, vertices, triangles, boundary_edges=None,
              boundary_markers=None, lattice=None) -> None:
    """Write an external triangle mesh in the format :func:`load_mesh` reads.

    Convenience for external mesh generators; qimpy itself never calls this
    during a run. ``boundary_edges``/``boundary_markers`` tag physical faces
    (a name matching a contact key becomes that contact; others reflect).
    """
    out = dict(vertices=np.asarray(vertices, float),
               triangles=np.asarray(triangles, int))
    if boundary_edges is not None:
        out["boundary_edges"] = np.asarray(boundary_edges, int)
        out["boundary_markers"] = np.asarray(boundary_markers, dtype=object)
    if lattice is not None:
        out["lattice"] = np.asarray(lattice, float)
    np.savez(path, **out)
