from __future__ import annotations

import torch
import numpy as np
from lxml import etree


def get_combinations(N: int, p: int) -> torch.Tensor:
    """Return unique p-tuple indices up to N (N_C_p x p tensor)."""
    i_single = torch.arange(N)
    i_tuples = torch.stack(
        torch.meshgrid(*([i_single] * p), indexing="ij")
    ).reshape(p, -1)
    unique = torch.all(i_tuples[:-1] < i_tuples[1:], dim=0)
    return i_tuples.T[unique]


def weld_points(points: torch.Tensor, tol: float) -> torch.Tensor:
    """Return unique points based on distance < tol from N x d tensor."""
    distances = torch.linalg.norm(points[:, None] - points[None, :], dim=-1)
    lowest_equivalent_index = torch.argmax(torch.where(distances < tol, 1, 0), dim=0)
    _, unique_indices = torch.unique(lowest_equivalent_index, return_inverse=True)
    return points[unique_indices]


def get_grid(basis: torch.Tensor) -> torch.Tensor:
    """Return grid containing linear combinations of basis vectors sufficient to cover Wigner-Seitz cell"""
    a, b, c = basis[:, 0], basis[:, 1], basis[:, 2]
    # Find maximum radius of sphere within parallelopiped from basis lattice vectors
    Rmax = np.max(
        [torch.linalg.norm(_vec) for _vec in [a, b, c, a + b, b + c, a + c, a + b + c]])
    # This sphere should be within shape_min in each direction
    # Therefore shape_min >= Rmax * (R^-1 or G^-1)
    shape_min = (Rmax * (torch.linalg.inv(basis).norm(dim=0)))
    shape_min = torch.ceil(shape_min)
    iGridX = np.arange(-shape_min[0], shape_min[0] + 1)
    iGridY = np.arange(-shape_min[1], shape_min[1] + 1)
    iGridZ = np.arange(-shape_min[2], shape_min[2] + 1)
    # Reduce grid for any orthogonal basis vector
    if np.dot(a, b) == 0 == np.dot(a, c):
        iGridX = np.arange(-1, 2)
    if np.dot(b, a) == 0 == np.dot(b, c):
        iGridY = np.arange(-1, 2)
    if np.dot(c, a) == 0 == np.dot(c, b):
        iGridZ = np.arange(-1, 2)
    iGrid = np.array([x.flatten() for x in np.meshgrid(iGridX, iGridY, iGridZ)]).T
    iGrid = iGrid[np.where((iGrid ** 2).sum(axis=1) > 0)[0]]  # eliminate origin
    return torch.tensor(iGrid) @ basis.T

def getWignerSeitz(basis: torch.Tensor, tol: float = 1e-6):
    basisGrid = get_grid(basis)
    print(f"Initial lattice vectors considered = {len(basisGrid)}")

    def get_plane_distance(r):
        """Signed distance of r from each perpendicular bisector plane from 0->basisGrid."""
        bgrid_mag = torch.linalg.norm(basisGrid, dim=-1)
        bgrid_hat = basisGrid / bgrid_mag[:,
                                None]  # outward normal of the perpendicular bisector planes
        return r @ bgrid_hat.T - 0.5 * bgrid_mag

    def ws_boundary_distance(r):
        """Distance to WS boundary (Cartesian coords in axis=-1): <0 == inside, >0 == outside."""
        return torch.max(get_plane_distance(r), dim=-1)[0]

    def ws_plane_count(r):
        """Number of planes each point within r is on."""
        return torch.count_nonzero(torch.abs(get_plane_distance(r)) < tol, dim=-1)

    # Down-select to planes that lie on WS boundary:
    basisGrid = basisGrid[torch.where(ws_boundary_distance(0.5 * basisGrid) < tol)[0]]
    # Down-select to WS faces with finite area:
    basisGrid = basisGrid[
        ws_plane_count(0.5 * basisGrid) == 1]  # must be only face with 0 distance
    n_faces = len(basisGrid)
    print(f"Number of WS faces = {n_faces}")
    assert 6 <= n_faces <= 14
    assert n_faces % 2 == 0

    # Compute vertices by 3-way intersection of planes
    i_face_triplets = get_combinations(n_faces, 3)
    Lhs = 0.5 * basisGrid[i_face_triplets]  # n_triplets x 3 x 3
    intersect = (torch.abs(torch.linalg.det(Lhs)) > tol)
    # i_face_triplets = i_face_triplets[intersect]
    Lhs = Lhs[intersect]
    rhs = (Lhs ** 2).sum(dim=-1)
    vertices = torch.linalg.solve(Lhs, rhs)

    # Down-select vertices to WS boundary:
    on_boundary = (torch.abs(ws_boundary_distance(vertices)) < tol)
    # i_face_triplets = i_face_triplets[on_boundary]
    vertices = vertices[on_boundary]
    vertices = weld_points(vertices, tol)
    n_vertices = len(vertices)
    print(f"Number of WS vertices = {n_vertices}")

    # Find edges of WS boundary:
    i_vertex_pairs = get_combinations(n_vertices, 2)
    edge_midpoints = vertices[i_vertex_pairs].mean(axis=1)
    on_boundary = (ws_plane_count(edge_midpoints) == 2)
    edges = i_vertex_pairs[on_boundary]
    n_edges = len(edges)
    print(f"Number of WS edges = {n_edges}")

    write_x3d("wigner_seitz.x3d", np.array(vertices), np.array(edges))
    print("(HINT: run \"view3dscene wigner_seitz.x3d\" to visualize the outputted Wigner-Seitz cell)")


def write_x3d(filename: str, vertices: np.ndarray, edges: np.ndarray) -> None:
    NSMAP = {'xsd': 'http://www.w3.org/2001/XMLSchema-instance'}
    qname = etree.QName('http://www.w3.org/2001/XMLSchema-instance',
                        'noNamespaceSchemaLocation')
    data = etree.Element('X3D', nsmap=NSMAP)
    data.set(qname, 'http://www.web3d.org/specifications/x3d-3.2.xsd')
    data.set('profile', 'Interchange')
    data.set('version', '3.2')
    Scene = etree.SubElement(data, 'Scene')
    etree.SubElement(Scene, 'Background', {"skyColor": "1 1 1"})
    Shape = etree.SubElement(Scene, 'Shape', {"DEF": "BZ"})
    Appearance = etree.SubElement(Shape, 'Appearance')
    etree.SubElement(Appearance, 'Material',
                     {'diffuseColor': '0 0 0', 'specularColor': '0 0 0'})
    coordIndex = np.array2string(
        np.hstack((edges, np.full((edges.shape[0], 1), -1))).flatten(),
        max_line_width=np.inf)[1:-1]
    Lineset = etree.SubElement(Shape, 'IndexedLineSet', {'coordIndex': coordIndex})
    points = np.array2string(vertices.flatten(), max_line_width=np.inf)[1:-1]
    etree.SubElement(Lineset, 'Coordinate', {'point': points})
    docinfo = '<!DOCTYPE X3D PUBLIC "ISO//Web3D//DTD X3D 3.2//EN"\n"http://www.web3d.org/specifications/x3d-3.2.dtd">'
    with open(filename, "wb") as f:
        f.write(etree.tostring(data, xml_declaration=True, encoding="UTF-8",
                               pretty_print=True, doctype=docinfo))




a_cubic = 10.0
#getWignerSeitz(a_cubic * torch.eye(3))
#getWignerSeitz(a_cubic*0.5*torch.tensor([[1,-0,0],[0,10,0],[0,0,1]]))
#getWignerSeitz(a_cubic*0.5*torch.tensor([[0,1,1],[1,0,1],[1,1,0]]))
#getWignerSeitz(a_cubic*0.5*torch.tensor([[1,1,-1],[1,-1,1],[-1,1,1]]))
getWignerSeitz(a_cubic*0.5*torch.tensor([[1,-0,1],[1,-10,0],[0,1,1]]))
