import qimpy as qp
from _advect import Advect, to_numpy


@qp.utils.stopwatch(name="plot_streamlines")
def plot_streamlines(advect, plt, contour_kwargs, stream_kwargs):
    contour_kwargs.setdefault("levels", 100)
    contour_kwargs.setdefault("cmap", "bwr")
    stream_kwargs.setdefault("density", 2.0)
    stream_kwargs.setdefault("linewidth", 1.0)
    stream_kwargs.setdefault("color", "k")
    stream_kwargs.setdefault("arrowsize", 1.0)
    x = to_numpy(advect.x[advect.non_ghost])
    y = to_numpy(advect.y[advect.non_ghost])
    v = to_numpy(advect.velocity)
    rho = to_numpy(advect.density)
    plt.contourf(x, y, rho.T, **contour_kwargs)
    plt.streamplot(x, y, v[..., 0].T, v[..., 1].T, **stream_kwargs)


def main():
    import matplotlib.pyplot as plt

    qp.utils.log_config()
    qp.rc.init()
    assert qp.rc.n_procs == 1  # MPI not yet supported

    x_y_bottom_left = [0, 0]
    x_y_bottom_right = [1.0, 0.0]
    x_y_top_right = [1.5, 1.0]
    x_y_top_left = [0.5, 1.0]

    x_y_corners = [x_y_bottom_left, x_y_bottom_right, x_y_top_right, x_y_top_left]
    sim = Advect(x_y_corners)
    for time_step in range(256):
        qp.log.info(f"{time_step = }")
        sim.time_step()

    # Plot only at end (for easier performance benchmarking of time steps):
    qp.log.info("Plotting density and streamlines")
    plot_streamlines(sim, plt, dict(levels=100), dict(linewidth=1.0))
    plt.gca().set_aspect("equal")
    plt.savefig("contour_final.png", bbox_inches="tight", dpi=200)

    qp.utils.StopWatch.print_stats()


if __name__ == "__main__":
    main()
