"""Calculate spherical harmonics and their product expansions."""
from __future__ import annotations
import qimpy.ions._spherical_harmonics_data as shdata
import torch
from typing import List, Tuple, Dict

# List exported symbols for doc generation
__all__ = ["L_MAX", "L_MAX_HLF", "get_harmonics"]


# Versions of shdata converted to torch.Tensors on appropriate device
_YLM_RECUR: List[torch.Tensor] = []
_YLM_PROD: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
L_MAX: int = shdata.L_MAX  #: Maximum l for calculation of harmonics
L_MAX_HLF: int = shdata.L_MAX_HLF  #: Maximum l of harmonics in products


def _initialize_device(device: torch.device) -> None:
    """Initialize spherical harmonic data as torch tensors on device"""
    global _YLM_RECUR, _YLM_PROD
    # Recurrence coefficients:
    _YLM_RECUR.clear()
    for l, (i1, i2, coeff) in enumerate(shdata.YLM_RECUR):
        if l < 2:
            _YLM_RECUR.append(torch.tensor(coeff, device=device))
        else:
            indices = torch.tensor((i1, i2), device=device)
            _YLM_RECUR.append(
                torch.sparse_coo_tensor(
                    indices, coeff, size=(2 * l + 1, 3 * (2 * l - 1)), device=device
                )
            )
    # Product coefficients:
    _YLM_PROD.clear()
    for ilm_pair, (ilm, coeff) in shdata.YLM_PROD.items():
        _YLM_PROD[ilm_pair] = (
            torch.tensor(ilm, device=device),
            torch.tensor(coeff, device=device),
        )


def get_harmonics(l_max: int, r: torch.Tensor) -> torch.Tensor:
    """Compute real solid harmonics :math:`r^l Y_{lm}(r)` for each l <= l_max.
    Contains l=0, followed by all m for l=1, and so on till l_max."""
    if not _YLM_PROD:
        _initialize_device(r.device)
    assert l_max <= shdata.L_MAX
    result = torch.empty(
        ((l_max + 1) ** 2,) + r.shape[:-1], dtype=r.dtype, device=r.device
    )
    if l_max >= 0:
        # l = 0: constant
        result[0] = _YLM_RECUR[0]
    if l_max >= 1:
        # l = 1: proportional to (y, z, x) for m = (-1, 0, +1):
        Y1 = (_YLM_RECUR[1] * r.flatten(0, -2).T[(1, 2, 0), :]).view(
            (3,) + r.shape[:-1]
        )
        result[1:4] = Y1
        Yprev = Y1
    for l in range(2, l_max + 1):
        # l > 1: compute from product of harmonics at l = 1 and l - 1:
        Yl = (
            _YLM_RECUR[l]
            @ (Yprev[:, None, :] * Y1[None, :, :]).view(3 * (2 * l - 1), -1)
        ).view((2 * l + 1,) + r.shape[:-1])
        result[l ** 2 : (l + 1) ** 2] = Yl
        Yprev = Yl
    return result


def get_harmonics_t(l_max: int, G: torch.Tensor) -> torch.Tensor:
    """Same as :func:`get_harmonics`, but in reciprocal space. The result
    is a complex tensor containing :math:`(iG)^l Y_{lm}(G)`, where the extra
    phase factor is from the Fourier transform of spherical harmonics. This is
    required for the corresponding real-space version to be real."""
    # Prepare phase factors:
    phase = []
    phase_cur = 1.0 + 0.0j
    for l in range(l_max + 1):
        phase.extend([phase_cur] * (2 * l + 1))  # repeated for m
        phase_cur *= 1.0j
    # Multiply real harmonics:
    return get_harmonics(l_max, G) * torch.tensor(phase, device=G.device).view(
        (len(phase),) + (1,) * (len(G.shape) - 1)
    )


if __name__ == "__main__":
    import argparse
    import qimpy as qp
    import numpy as np
    from scipy.special import sph_harm
    from typing import Sequence, Any

    def get_harmonics_ref(l_max: int, r: np.ndarray) -> np.ndarray:
        """Reference real solid harmonics based on SciPy spherical harmonics"""
        rMag = np.linalg.norm(r, axis=-1)
        theta = np.arccos(r[..., 2] / rMag)
        phi = np.arctan2(r[..., 1], r[..., 0])
        phi += np.where(phi < 0.0, 2 * np.pi, 0)
        results = []
        for l in range(l_max + 1):
            result = np.zeros((2 * l + 1,) + r.shape[:-1])
            for m in range(0, l + 1):
                ylm = ((-1) ** m) * (rMag ** l) * sph_harm(m, l, phi, theta)
                if m == 0:
                    result[l] = ylm.real
                else:
                    result[l + m] = np.sqrt(2) * ylm.real
                    result[l - m] = np.sqrt(2) * ylm.imag
            results.append(result)
        return np.concatenate(results, axis=0)

    def get_lm(l_max: int) -> List[Tuple[int, int]]:
        """Get list of all (l,m) in order up to (and including) l_max"""
        return [(l, m) for l in range(l_max + 1) for m in range(-l, l + 1)]

    def print_array(
        array: Sequence[Any], line: str, padding: int, fmt: str, width: int = 79
    ) -> str:
        """PEP8-compatible printing of array, where line is pending text yet to
        be printed, and padding controls where array starts if wrapped to next
        lien based on width. Each entry will be formatted according to fmt."""
        # Convert all entries to strings and compute total length:
        fmt += ", "
        strings = [fmt.format(elem) for elem in array]
        total_len = sum([len(string) for string in strings]) + 3
        if len(line) + total_len < width:
            # Will fit on same line:
            return line + "[" + "".join(strings).rstrip(", ") + "]"
        else:
            # Need multiple lines:
            qp.log.info(line.rstrip())
            line = (" " * padding) + "["
            for string in strings:
                if len(line) + len(string) >= width:  # wrap
                    qp.log.info(line.rstrip())
                    line = " " * (padding + 1)
                line += string
            return line.rstrip(", ") + "]"

    def generate_harmonic_coefficients(l_max_hlf: int) -> None:
        """Generate tables of recursion coefficients for computing real
        solid harmonics up to l_max = 2 * l_max_hlf, as well as tables of
        product coefficients (Clebsch-Gordon coefficients) for real solid
        harmonics up to order l_max_hlf. Print results formatted as Python
        code that can be pasted into _spherical_harmonics_data.py."""
        l_max = 2 * l_max_hlf
        qp.log.info(
            "from typing import List, Tuple, Dict\n\n"
            f"L_MAX: int = {l_max}  # Maximum l for harmonics\n"
            f"L_MAX_HLF: int = {l_max_hlf}  # Maximum l for products"
        )
        # Calculate solid harmonics on a mesh covering unit cube:
        grids1d = 3 * (np.linspace(-1.0, 1.0, 2 * l_max),)  # avoids zero
        r = np.array(np.meshgrid(*grids1d)).reshape(3, -1).T
        r_sq = (r ** 2).sum(axis=-1)
        ylm = get_harmonics_ref(l_max, r)
        # Calculate recursion coefficients:
        ERR_TOL = 1e-14
        COEFF_TOL = 1e-8
        qp.log.info(
            "CooIndices = Tuple[List[int], List[int], List[float]]\n\n"
            "# Recursion coefficients for computing real harmonics at l>1\n"
            "# from products of those at l = 1 and l-1. The integers index\n"
            "# a sparse matrix with (2l+1) rows and 3*(2l-1) columns.\n"
            "YLM_RECUR: List[CooIndices] = ["
        )
        Y_00 = np.sqrt(0.25 / np.pi)
        Y_1m_prefac = np.sqrt(0.75 / np.pi)
        line = f"    ([], [], [{Y_00:.16f}]), ([], [], [{Y_1m_prefac:.16f}]),"
        for l in range(2, l_max + 1):
            l_minus_1_slice = slice((l - 1) ** 2, l ** 2)
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
                    coeff = np.linalg.lstsq(y_product_allowed.T, y_target, rcond=None)[
                        0
                    ]
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
            qp.log.info(line)  # pending data from previous entry
            line = "    ("
            padding = len(line)
            line = print_array(index_row, line, padding, "{:d}") + ", "
            line = print_array(index_col, line, padding, "{:d}") + ", "
            line = print_array(values, line, padding, "{:.16f}") + "),"
        qp.log.info(line.rstrip(", ") + "]\n")
        # Calculate Clebsch-Gordon coefficients:
        lm_hlf = get_lm(l_max_hlf)
        qp.log.info(
            "# Clebsch-Gordon coefficients for products of real harmonics.\n"
            "# The integer indices correspond to l*(l+1)+m for each (l,m)."
        )
        line = "YLM_PROD: Dict[Tuple[int, int]," " Tuple[List[int], List[float]]] = {"
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
                y_terms = ylm[ilm] * (
                    r_sq[None, :] ** ((l1 + l2 - l_all) // 2)[:, None]
                )
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
                qp.log.info(line)  # pending data from previous entry
                line = f"    ({ilm1}, {ilm2}): ("
                padding = len(line)
                line = print_array(ilm, line, padding, "{:d}") + ", "
                line = print_array(coeff, line, padding, "{:.16f}") + "),"
        qp.log.info(line.rstrip(", ") + "}")

    def main():
        # Parse command line:
        parser = argparse.ArgumentParser(
            description="Generate / test real spherical harmonic coefficients"
        )
        # --- mutually-exclusive group of generate or test
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            "-g", "--generate", action="store_true", help="generate coefficients"
        )
        group.add_argument(
            "-t", "--test", action="store_true", help="test coefficients"
        )
        # ---
        args = parser.parse_args()

        if args.generate:
            rc = qp.utils.RunConfig()
            assert rc.n_procs == 1
            qp.utils.log_config()  # after rc to suppress header messages
            l_max_hlf = 3
            generate_harmonic_coefficients(l_max_hlf)

        if args.test:
            qp.utils.log_config()
            rc = qp.utils.RunConfig()
            assert rc.n_procs == 1
            r = torch.randn(10, 10000, 3, device=rc.device)
            r_sq = (r ** 2).sum(dim=-1)

            # Test spherical harmonics:
            qp.log.info("Testing spherical harmonics:")
            rel_err_all = []
            watch = qp.utils.StopWatch("get_harmonics", rc)
            ylm = get_harmonics(shdata.L_MAX, r)
            watch.stop()
            watch_ref = qp.utils.StopWatch("get_harmonics_ref", rc)
            ylm_ref = torch.from_numpy(
                get_harmonics_ref(shdata.L_MAX, r.to(rc.cpu).numpy())
            ).to(rc.device)
            watch_ref.stop()
            for l in range(shdata.L_MAX + 1):
                l_slice = slice(l ** 2, (l + 1) ** 2)
                err = (ylm[l_slice] - ylm_ref[l_slice]).norm().item()
                rel_err = err / (ylm_ref[l_slice]).norm().item()
                rel_err_all.append(rel_err)
                qp.log.info(f"  l: {l} Err: {rel_err:9.3e}")
            rel_err_overall = np.sqrt((np.array(rel_err_all) ** 2).mean())
            qp.log.info(f"  Overall Err: {rel_err_overall:9.3e}\n")

            # Test product coefficients:
            qp.log.info("Testing product coefficients:")
            if not _YLM_PROD:
                _initialize_device(rc.device)
            rel_err_all = []
            dl_shape = (-1,) + (1,) * len(r_sq.shape)  # bcast with r_sq[None]
            for l1 in range(shdata.L_MAX_HLF + 1):
                l1_slice = slice(l1 ** 2, (l1 + 1) ** 2)
                for l2 in range(l1 + 1):
                    l2_slice = slice(l2 ** 2, (l2 + 1) ** 2)
                    product_ref = ylm[l1_slice, None, :] * ylm[None, l2_slice, :]
                    product = torch.zeros_like(product_ref)
                    for m1 in range(-l1, l1 + 1):
                        ilm1 = l1 * (l1 + 1) + m1
                        for m2 in range(-l2, l2 + 1):
                            ilm2 = l2 * (l2 + 1) + m2
                            index, coeffs = _YLM_PROD[
                                (max(ilm1, ilm2), min(ilm1, ilm2))
                            ]
                            l_net = index.sqrt().floor().to(torch.int)
                            dl_by_2 = (l1 + l2 - l_net).div(2, rounding_mode="floor")
                            prod_terms = ylm[index] * (
                                r_sq[None, ...] ** dl_by_2.view(dl_shape)
                            )
                            product[l1 + m1, l2 + m2] = torch.tensordot(
                                coeffs, prod_terms, dims=1
                            )
                    err = (product - product_ref).norm().item()
                    rel_err = err / product_ref.norm().item()
                    rel_err_all.append(rel_err)
                    qp.log.info(f"  l: {l1} {l2} Err: {rel_err:9.3e}")
            rel_err_overall = np.sqrt((np.array(rel_err_all) ** 2).mean())
            qp.log.info(f"  Overall Err: {rel_err_overall:9.3e}")
            qp.utils.StopWatch.print_stats()

    main()
