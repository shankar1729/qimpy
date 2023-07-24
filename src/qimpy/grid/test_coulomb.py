import matplotlib.pyplot as plt
import numpy as np

from qimpy import rc
from qimpy.io import log_config


def plot_nyquist():
    """Plot dependence of Nyquist frequency with broadening
    to inform heuristic for best ion width selection"""
    dx = 0.2  # Typical grid spacing at 100 Eh plane-wave spacing
    rTest = 3.0  # Worst-case ion-fluid spacing (H in H3O+, NonlinearPCM)
    N = 128
    x = dx * np.arange(N)
    L = dx * N
    G = (2 * np.pi / L) * np.concatenate((np.arange(N // 2), np.arange(N // 2 - N, 0)))
    sigma_by_dx = np.arange(1.0, 4.0, 0.1)
    sigma = dx * sigma_by_dx
    fTilde = np.exp(-0.5 * (sigma[:, None] * G[None, :]) ** 2) * (1.0 / L)
    f = np.fft.fft(fTilde, axis=-1).real

    # Visualize f near test radius:
    plt.figure()
    for i, f_i in enumerate(f):
        plt.plot(x, np.abs(f_i), label=r"$\sigma$/dx=" + f"{sigma_by_dx[i]:.1f}")
    plt.axvline(rTest, color="k", ls="dotted", lw=1)
    plt.xlim(0, 2 * rTest)
    plt.ylim(1e-14, 1.0)
    plt.yscale("log")
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")

    # MAE between rTest and 2*rTest vs sigma:
    sel = np.where(np.logical_and(x >= rTest, x <= 2 * rTest))
    mae = np.abs(f[:, sel]).mean(axis=-1)
    plt.figure()
    plt.plot(sigma_by_dx, mae)
    plt.yscale("log")
    plt.ylabel("MAE")
    plt.xlabel(r"$\sigma$/dx")
    plt.show()


def main():
    log_config()
    rc.init()
    if rc.is_head:
        plot_nyquist()


if __name__ == "__main__":
    main()
