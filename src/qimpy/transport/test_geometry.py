from . import Geometry
import matplotlib.pyplot as plt


def test_geometry():
    vertices = [[0.0, 0.0], [10.0, 0.0], [15.0, 20.0], [0.0, 15.0], [2.0, 7.0]]
    edges = [
        [0, 1, -1, 15],
        [1, 2, -1, 20],
        [2, 3, -1, 15],
        [3, 0, 4, 20],
    ]
    geometry = Geometry(vertices=vertices, edges=edges)

    plt.figure()
    for spline in geometry.edge_splines:
        x, y = spline.points.T
        plt.plot(x, y, marker="+")
    plt.show()


if __name__ == "__main__":
    test_geometry()
