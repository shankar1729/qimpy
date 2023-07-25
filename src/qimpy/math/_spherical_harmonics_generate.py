from typing import Sequence, Any

import numpy as np
from scipy.special import sph_harm

from qimpy import log, rc
from qimpy.io import log_config


def get_harmonics_ref(l_max: int, r: np.ndarray) -> np.ndarray:
    """Reference real solid harmonics based on SciPy spherical harmonics."""
    rMag = np.linalg.norm(r, axis=-1)
    theta = np.arccos(r[..., 2] / rMag)
    phi = np.arctan2(r[..., 1], r[..., 0])
    phi += np.where(phi < 0.0, 2 * np.pi, 0)
    results = []
    for l in range(l_max + 1):
        result = np.zeros((2 * l + 1,) + r.shape[:-1])
        for m in range(0, l + 1):
            ylm = ((-1) ** m) * (rMag**l) * sph_harm(m, l, phi, theta)
            if m == 0:
                result[l] = ylm.real
            else:
                result[l + m] = np.sqrt(2) * ylm.real
                result[l - m] = np.sqrt(2) * ylm.imag
        results.append(result)
    return np.concatenate(results, axis=0)


def get_lm(l_max: int) -> list[tuple[int, int]]:
    """Get list of all (l,m) in order up to (and including) l_max"""
    return [(l, m) for l in range(l_max + 1) for m in range(-l, l + 1)]


def format_array(array: Sequence[Any], fmt: str) -> str:
    """Convert `array` to string with format `fmt` for each entry."""
    return "[" + ", ".join(fmt.format(elem) for elem in array) + "]"


def generate_harmonic_coefficients(l_max_hlf: int) -> None:
    """Generate tables of recursion coefficients for computing real
    solid harmonics up to l_max = 2 * l_max_hlf, as well as tables of
    product coefficients (Clebsch-Gordon coefficients) for real solid
    harmonics up to order l_max_hlf. Print results formatted as Python
    code that can be pasted into _spherical_harmonics_data.py."""
    l_max = 2 * l_max_hlf
    log.info(
        f"L_MAX: int = {l_max}  # Maximum l for harmonics\n"
        f"L_MAX_HLF: int = {l_max_hlf}  # Maximum l for products"
    )
    # Calculate solid harmonics on a mesh covering unit cube:
    grids1d = 3 * (np.linspace(-1.0, 1.0, 2 * l_max),)  # avoids zero
    r = np.array(np.meshgrid(*grids1d)).reshape(3, -1).T
    r_sq = (r**2).sum(axis=-1)
    ylm = get_harmonics_ref(l_max, r)
    # Calculate recursion coefficients:
    ERR_TOL = 1e-14
    COEFF_TOL = 1e-8
    log.info(
        "CooIndices = tuple[list[int], list[int], list[float]]\n\n"
        "# Recursion coefficients for computing real harmonics at l>1\n"
        "# from products of those at l = 1 and l-1. The integers index\n"
        "# a sparse matrix with (2l+1) rows and 3*(2l-1) columns.\n"
        "YLM_RECUR: list[CooIndices] = ["
    )
    Y_00 = np.sqrt(0.25 / np.pi)
    Y_1m_prefac = np.sqrt(0.75 / np.pi)
    log.info(f"    ([], [], [{Y_00:.16f}]), ([], [], [{Y_1m_prefac:.16f}]),")
    for l in range(2, l_max + 1):
        l_minus_1_slice = slice((l - 1) ** 2, l**2)
        y_product = ylm[l_minus_1_slice, None, :] * ylm[None, 1:4, :]
        y_product = y_product.reshape((2 * l - 1) * 3, -1)
        index_row = []
        index_col = []
        values = []
        for m in range(-l, l + 1):
            # List pairs of m at l = 1 and l-1 that can add up to m:
            m_pairs_all = set(
                [
                    (sign * m + dsign * dm, dm)
                    for sign in (-1, 1)
                    for dsign in (-1, 1)
                    for dm in (-1, 0, 1)
                ]
            )
            m_pairs = [m_pair for m_pair in m_pairs_all if abs(m_pair[0]) < l]
            m_pair_indices = [3 * (l - 1 + m) + (1 + dm) for m, dm in m_pairs]
            # Solve for coefficients of the linear combination:
            for n_sel in range(1, len(m_pair_indices) + 1):
                # Try increasing numbers till we get one:
                y_product_allowed = y_product[m_pair_indices[:n_sel]]
                y_target = ylm[l * (l + 1) + m]
                coeff = np.linalg.lstsq(y_product_allowed.T, y_target, rcond=None)[0]
                residual = np.dot(coeff, y_product_allowed) - y_target
                err = np.linalg.norm(residual) / np.linalg.norm(y_target)
                if err < ERR_TOL:
                    break
            assert err < ERR_TOL
            # Select non-zero coefficients to form product expansion:
            sel = np.where(np.abs(coeff) > COEFF_TOL * np.linalg.norm(coeff))[0]
            indices = np.array(m_pair_indices)[sel]
            coeff = coeff[sel]
            # Sort by index and add to lists for current l:
            sort_index = indices.argsort()
            index_row += [l + m] * len(sort_index)
            index_col += list(indices[sort_index])
            values += list(coeff[sort_index])
        # Format as python code:
        log.info(
            f"    ("
            f"{format_array(index_row, '{:d}')}, "
            f"{format_array(index_col, '{:d}')}, "
            f"{format_array(values, '{:.16f}')}),"
        )
    log.info("]\n")
    # Calculate Clebsch-Gordon coefficients:
    lm_hlf = get_lm(l_max_hlf)
    log.info(
        "# Clebsch-Gordon coefficients for products of real harmonics.\n"
        "# The integer indices correspond to l*(l+1)+m for each (l,m).\n"
        "YLM_PROD: dict[tuple[int, int],"
        " tuple[list[int], list[float]]] = {"
    )
    for ilm1, (l1, m1) in enumerate(lm_hlf):
        for ilm2, (l2, m2) in enumerate(lm_hlf[: ilm1 + 1]):
            # List (l,m) pairs allowed by angular momentum addition rules:
            m_allowed = {m1 + m2, m1 - m2, m2 - m1, -(m1 + m2)}
            l_allowed = range(l1 - l2, l1 + l2 + 1, 2)
            lm_all = np.array(
                [(l, m) for l in l_allowed for m in m_allowed if (abs(m) <= l)]
            )
            l_all = lm_all[:, 0]
            m_all = lm_all[:, 1]
            ilm = l_all * (l_all + 1) + m_all  # flattened index
            # Solve for coefficients of the linear combination:
            y_product = ylm[ilm1] * ylm[ilm2]
            y_terms = ylm[ilm] * (r_sq[None, :] ** ((l1 + l2 - l_all) // 2)[:, None])
            results = np.linalg.lstsq(y_terms.T, y_product, rcond=None)
            coeff = results[0]
            err = np.sqrt(results[1][0]) / np.linalg.norm(y_product)
            assert err < ERR_TOL
            # Select non-zero coefficients to form product expansion:
            sel = np.where(np.abs(coeff) > COEFF_TOL * np.linalg.norm(coeff))[0]
            ilm = ilm[sel]
            coeff = coeff[sel]
            # Sort by (l,m):
            sort_index = ilm.argsort()
            ilm = ilm[sort_index]
            coeff = coeff[sort_index]
            # Format as python code:
            log.info(
                f"    ({ilm1}, {ilm2}): ("
                f"{format_array(ilm, '{:d}')}, "
                f"{format_array(coeff, '{:.16f}')}),"
            )
    log.info("}")


def main():
    rc.init()
    assert rc.n_procs == 1  # no MPI
    log_config()  # after rc to suppress header messages
    generate_harmonic_coefficients(l_max_hlf=3)


if __name__ == "__main__":
    main()
