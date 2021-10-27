"""Calculate spherical Bessel functions."""
# List exported symbols for doc generation
__all__ = ["jl_by_xl"]

import torch


def jl_by_xl(l_max: int, x: torch.Tensor) -> torch.Tensor:
    """Compute spherical bessel functions j_l(x)/x^l for each l <= l_max.
    This is optimized to calculate j_l up to l = 6 efficiently combining
    recursion relations and Taylor expansions to achieve both absolute
    and relative errors < 1e-15 for all x. (The errors will grow beyond
    l = 6 due to instability of the efficient recursion relation chosen,
    so do not use this routine for higher l without testing.)
    """
    result = torch.empty((l_max + 1,) + x.shape, dtype=x.dtype, device=x.device)
    taylor_prefac = 1.0
    for l in range(l_max + 1):
        result_l = result[l]
        x_cut = 1.0 + 0.7 * l  # cutoff for Taylor expansion

        # Taylor series for small x:
        sel = torch.where(x <= x_cut)
        x_sel = x[sel]
        taylor_prefac /= 2 * l + 1
        term = torch.full_like(x_sel, taylor_prefac)  # first non-zero term
        series = term.clone().detach()
        x_sel_sq = x_sel * x_sel
        for i in range(1, 9 + 2 * l):
            term *= x_sel_sq * (-0.25 / (i * (i + l + 0.5)))
            series += term
        result_l[sel] = series

        # Trigonometric formula for larger x:
        sel = torch.where(x > x_cut)
        x_sel = x[sel]
        if l == 0:
            result_l[sel] = torch.sin(x_sel) / x_sel  # j_0(x)
        elif l == 1:
            result_l[sel] = (
                result[0][sel] - torch.cos(x_sel)  # j_0(x) = sin(x)/x from before
            ) / (
                x_sel * x_sel
            )  # to j_1/x
        else:
            result_l[sel] = ((2 * l - 1) * result[l - 1][sel] - result[l - 2][sel]) / (
                x_sel * x_sel
            )  # j_l/x^l by recursion for l>1
    return result


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.special import spherical_jn
    import qimpy as qp
    import numpy as np

    qp.utils.log_config()
    rc = qp.utils.RunConfig()
    x = np.logspace(-3, 3, 6000)
    l_max = 6
    jl_ref = [spherical_jn(l, x) / (x ** l) for l in range(l_max + 1)]
    jl_test = jl_by_xl(l_max, torch.tensor(x, device=rc.device)).to(rc.cpu)
    for l in range(l_max + 1):
        err_scale = np.maximum(x ** (l + 1), 1)  # to match forms near 0 and infty
        err = np.abs(jl_test[l] - jl_ref[l]) * err_scale
        plt.plot(x, err, label=f"l = {l}")
        print(f"l: {l}  ErrMean: {err.mean():.3e}  ErrMax: {err.max():.3e}")
    plt.xscale("log")
    plt.legend()
    plt.show()
