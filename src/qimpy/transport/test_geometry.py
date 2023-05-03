from . import Geometry
import matplotlib.pyplot as plt
import torch


def test_geometry():
    vertices = [
        [0.0, 0.0],
        [3.0, 0.0],
        [11.0, 0.0],
        [17.0, 20.0],
        [10.0, 18.0],
        [3.0, 16.0],
        [0.0, 15.0],
        [5.0, 12.0],
        [3.0, 5.0],
        [9.0, 8.0],
        [3.0, 9.0],
        [7.0, 4.0],
        [8.0, 11.0],
    ]
    edges = [
        [0, 1, -1, 15],
        [1, 2, -1, 20],
        [2, 3, -1, 15],
        [3, 4, -1, 15],
        [4, 5, -1, 15],
        [5, 6, -1, 15],
        [6, 0, -1, 15],
        [7, 8, 10, 10],
        [8, 9, 11, 10],
        [9, 7, 12, 10],
        [7, 5, -1, 10],
        [8, 6, -1, 10],
        [8, 1, -1, 10],
        [9, 2, -1, 10],
        [9, 4, -1, 10],
    ]
    quads = [
        [0, 12, 11, 6],
        [1, 13, 8, 12],
        [2, 3, 14, 13],
        [9, 14, 4, 10],
        [7, 10, 5, 11],
    ]
    geometry = Geometry(vertices=vertices, edges=edges, quads=quads)

    plt.figure()
    for i_edge, spline in enumerate(geometry.edge_splines):
        x, y = spline.points.T
        plt.plot(x, y, marker="+")
        n_label = spline.n_points // 3  # avoid collision with vertex at midpoint
        plt.text(x[n_label], y[n_label], f"E{i_edge}")

    for i_vertex, (x, y) in enumerate(geometry.vertices):
        plt.text(x, y, f"V{i_vertex}")

    for i_quad, quad in enumerate(geometry.quads):
        splines = [geometry.edge_splines[i_edge] for i_edge in quad]
        midpoints = [spline.points[spline.n_points // 2] for spline in splines]
        centroid = torch.stack(midpoints).mean(dim=0)
        plt.text(centroid[0], centroid[1], f"Q{i_quad}")
        for midpoint in midpoints:
            x, y = torch.stack((centroid, midpoint)).T
            plt.plot(x, y, color="k", lw=1, ls="dotted")

    plt.gca().axis("equal")
    plt.show()


if __name__ == "__main__":
    test_geometry()
