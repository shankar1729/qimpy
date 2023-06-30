import qimpy as qp
import numpy as np
import torch
import pytest
from scipy.special import spherical_jn


@pytest.mark.mpi_skip
def test_jl(plt=None):
    x = np.logspace(-3, 3, 6000)
    l_max = 6
    jl_ref = [spherical_jn(l, x) / (x**l) for l in range(l_max + 1)]
    jl_test = qp.ions.spherical_bessel.jl_by_xl(
        l_max, torch.tensor(x, device=qp.rc.device)
    ).to(qp.rc.cpu)
    err_mean_all = 0.0
    err_max_all = 0.0
    for l in range(l_max + 1):
        err_scale = np.maximum(x ** (l + 1), 1)  # to match forms near 0 and infty
        err = np.abs(jl_test[l] - jl_ref[l]) * err_scale
        err_mean = err.mean()
        err_max = err.max()
        if plt is not None:
            plt.plot(x, err, label=f"l = {l}")
            qp.log.info(f"l: {l}  ErrMean: {err_mean:.3e}  ErrMax: {err_max:.3e}")
        err_mean_all += err_mean / (l_max + 1)
        err_max_all = max(err_max_all, err_max)
    qp.log.info(f"l<={l_max}  ErrMean: {err_mean_all:.3e}  ErrMax: {err_max_all:.3e}")
    assert err_mean_all <= 1e-16
    assert err_max_all <= 2e-15


def main():
    """Invoke test_jl with plots and per-l errors."""
    import matplotlib.pyplot as plt

    qp.utils.log_config()
    qp.rc.init()
    test_jl(plt)
    plt.xscale("log")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
