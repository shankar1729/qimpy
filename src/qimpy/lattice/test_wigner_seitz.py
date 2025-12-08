from __future__ import annotations
from typing import Sequence
from tempfile import mkstemp
import subprocess
import os

import pytest
import torch

from qimpy import rc, log
from qimpy.io import log_config
from . import WignerSeitz


def get_test_lattices() -> Sequence[tuple[torch.Tensor, int, int, int]]:
    """Generate test lattices with known Wigner-Seitz face, edge and vertex counts"""
    return [
        (5.9 * torch.eye(3), 6, 12, 8),
        (3.8 * torch.tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]]), 12, 24, 14),
        (4.7 * torch.tensor([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]), 14, 36, 24),
        (3.0 * torch.tensor([[1, -0, 1], [1, -10, 0], [0, 1, 1]]), 8, 18, 12),
    ]


@pytest.mark.mpi_skip
@pytest.mark.parametrize("v_basis, n_faces, n_edges, n_vertices", get_test_lattices())
def test_wigner_seitz(
    v_basis: torch.Tensor, n_faces: int, n_edges: int, n_vertices: int
) -> None:
    ws = WignerSeitz(v_basis)
    assert len(ws.faces) == n_faces
    assert len(ws.edges) == n_edges
    assert len(ws.vertices) == n_vertices


def main():
    log_config()
    rc.init()
    assert rc.n_procs == 1  # No MPI needed/supported for these tests
    x3d_viewer = os.environ.get("X3D_VIEWER", "view3dscene")
    log.info(f"Using '{x3d_viewer}' to view x3d files; export X3D_VIEWER to override.")
    for v_basis, n_faces, n_edges, n_vertices in get_test_lattices():
        ws = WignerSeitz(v_basis)
        log.info(
            f"Created Wigner-Seitz cell with {len(ws.faces)} faces, {len(ws.edges)}"
            f" edges and {len(ws.vertices)} vertices. (Expected {n_faces} faces,"
            f" {n_edges} edges and {n_vertices} vertices.)"
        )
        file_handle, x3d_filename = mkstemp(suffix=".x3d")
        os.close(file_handle)  # WignerSeitz.write_x3d will reopen file
        ws.write_x3d(x3d_filename)
        subprocess.run([x3d_viewer, x3d_filename], capture_output=True)
        os.remove(x3d_filename)


if __name__ == "__main__":
    main()
