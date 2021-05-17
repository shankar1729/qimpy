import qimpy.ions._spherical_harmonics_data as shdata
from scipy.sparse import coo_matrix  # TODO: replace with torch COO tensors


def get_harmonics(l_max, r):
    assert(l_max <= shdata.L_MAX)
    results = []
    if l_max >= 0:
        results.append(shdata.YLM_RECUR[0] * np.ones((1,) + r.shape[:-1]))
    if l_max >= 1:
        results.append(shdata.YLM_RECUR[1] * r[..., (1, 2, 0)].T)
    for l in range(2, l_max+1):
        # Compute from product of harmonics at l = 1 and l-1:
        if isinstance(shdata.YLM_RECUR[l], tuple):
            # Convert to sparse matrix on first use:
            shdata.YLM_RECUR[l] = coo_matrix(
                (shdata.YLM_RECUR[l][2],
                    (shdata.YLM_RECUR[l][0], shdata.YLM_RECUR[l][1])),
                shape=(2*l + 1, 3*(2*l - 1)))
        product = results[l-1][:, None, :] * results[1][None, :, :]
        product = product.reshape(3*(2*l - 1), -1)
        results.append(shdata.YLM_RECUR[l].dot(product))
    return results


if __name__ == "__main__":
    import argparse
    import numpy as np
    from scipy.special import sph_harm

    def get_harmonics_ref(l_max, r):
        'Reference real solid harmonics based on SciPy spherical harmonics'
        rMag = np.linalg.norm(r, axis=-1)
        theta = np.arccos(r[..., 2]/rMag)
        phi = np.arctan2(r[..., 1], r[..., 0])
        phi += np.where(phi < 0., 2*np.pi, 0)
        results = []
        for l in range(l_max+1):
            result = np.zeros((2*l + 1,) + r.shape[:-1])
            for m in range(0, l + 1):
                ylm = ((-1)**m) * (rMag**l) * sph_harm(m, l, phi, theta)
                if m == 0:
                    result[l] = ylm.real
                else:
                    result[l+m] = np.sqrt(2) * ylm.real
                    result[l-m] = np.sqrt(2) * ylm.imag
            results.append(result)
        return results

    def get_lm(l_max):
        'Get list of all (l,m) in order up to (and including) l_max'
        return [(l, m)
                for l in range(l_max + 1)
                for m in range(-l, l+1)]

    def print_array(array, line, padding, fmt, width=79):
        '''PEP8-compatible printing of array, where line is pending text yet to
        be printed, and padding controls where array starts if wrapped to next
        lien based on width. Each entry will be formatted according to fmt.'''
        # Convert all entries to strings and compute total length:
        fmt += ', '
        strings = [fmt.format(elem) for elem in array]
        total_len = sum([len(string) for string in strings]) + 3
        if len(line) + total_len < width:
            # Will fit on same line:
            return line + '[' + ''.join(strings).rstrip(', ') + ']'
        else:
            # Need multiple lines:
            print(line.rstrip())
            line = (' '*padding) + '['
            for string in strings:
                if len(line) + len(string) >= width:  # wrap
                    print(line.rstrip())
                    line = ' '*(padding+1)
                line += string
            return line.rstrip(', ') + ']'

    def generate_harmonic_coefficients(l_max_hlf):
        '''Generate tables of recursion coefficients for computing real
        solid harmonics up to l_max = 2 * l_max_hlf, as well as tables of
        product coefficients (Clebsch-Gordon coefficients) for real solid
        harmonics up to order l_max_hlf. Print results formatted as Python
        code that can be pasted into _spherical_harmonics_data.py.'''
        l_max = 2 * l_max_hlf
        print('L_MAX =', l_max, '     # Maximum l with harmonics supported')
        print('L_MAX_HLF =', l_max_hlf, ' # Maximum l with products supported')
        print()
        # Calculate solid harmonics on a mesh covering unit cube:
        grids1d = 3 * (np.linspace(-1., 1., 2*l_max), )  # avoids zero
        r = np.array(np.meshgrid(*grids1d)).reshape(3, -1).T
        r_sq = (r**2).sum(axis=-1)
        ylm = get_harmonics_ref(l_max, r)
        # Calculate recursion coefficients:
        ERR_TOL = 1e-14
        COEFF_TOL = 1e-8
        print('# Recursion coefficients for computing real harmonics at l>1')
        print('# from products of those at l = 1 and l-1. The integers index')
        print('# a sparse matrix with (2l+1) rows and 3*(2l-1) columns.')
        Y_00 = np.sqrt(0.25/np.pi)
        Y_1m_prefac = np.sqrt(0.75/np.pi)
        line = f'YLM_RECUR = [\n    {Y_00:.16f}, {Y_1m_prefac:.16f},'
        for l in range(2, l_max + 1):
            y_product = ylm[l-1][:, None, :] * ylm[1][None, :, :]
            y_product = y_product.reshape((2*l - 1) * 3, -1)
            index_row = []
            index_col = []
            values = []
            for m in range(-l, l + 1):
                # List pairs of m at l = 1 and l-1 that can add up to m:
                m_pairs = set([(sign*m + dsign*dm, dm)
                               for sign in (-1, 1)
                               for dsign in (-1, 1)
                               for dm in (-1, 0, 1)])
                m_pairs = [m_pair for m_pair in m_pairs if abs(m_pair[0]) < l]
                m_pair_indices = [3*(l - 1 + m) + (1 + dm)
                                  for m, dm in m_pairs]
                # Solve for coefficients of the linear combination:
                for n_sel in range(1, len(m_pair_indices)+1):
                    # Try increasing numbers till we get one:
                    y_product_allowed = y_product[m_pair_indices[:n_sel]]
                    y_target = ylm[l][l + m]
                    coeff = np.linalg.lstsq(y_product_allowed.T, y_target,
                                            rcond=None)[0]
                    residual = np.dot(coeff, y_product_allowed) - y_target
                    err = np.linalg.norm(residual) / np.linalg.norm(y_target)
                    if err < ERR_TOL:
                        break
                assert(err < ERR_TOL)
                # Select non-zero coefficients to form product expansion:
                sel = np.where(np.abs(coeff)
                               > COEFF_TOL * np.linalg.norm(coeff))[0]
                indices = np.array(m_pair_indices)[sel]
                coeff = coeff[sel]
                # Sort by index and add to lists for current l:
                sort_index = indices.argsort()
                index_row += [l + m] * len(sort_index)
                index_col += list(indices[sort_index])
                values += list(coeff[sort_index])
            # Format as python code:
            print(line)  # pending data from previous entry
            line = '    ('
            padding = len(line)
            line = print_array(index_row, line, padding, '{:d}') + ', '
            line = print_array(index_col, line, padding, '{:d}') + ', '
            line = print_array(values, line, padding, '{:.16f}') + '),'
        print(line.rstrip(', ') + ']')
        print()
        # Calculate Clebsch-Gordon coefficients:
        lm_hlf = get_lm(l_max_hlf)
        ylm = np.vstack(ylm)  # flatten into a single array with all (l,m)
        print('# Clebsch-Gordon coefficients for products of real harmonics.')
        print('# The integer indices correspond to l*(l+1)+m for each (l,m).')
        line = 'YLM_PROD = {'
        for ilm1, (l1, m1) in enumerate(lm_hlf):
            for ilm2, (l2, m2) in enumerate(lm_hlf[:ilm1+1]):
                # List (l,m) pairs allowed by angular momentum addition rules:
                m_allowed = {m1 + m2, m1 - m2, m2 - m1, -(m1 + m2)}
                l_allowed = range(l1 - l2, l1 + l2 + 1, 2)
                lm_all = np.array([(l, m)
                                   for l in l_allowed
                                   for m in m_allowed
                                   if (abs(m) <= l)])
                l_all = lm_all[:, 0]
                m_all = lm_all[:, 1]
                ilm = l_all*(l_all + 1) + m_all  # flattened index
                # Solve for coefficients of the linear combination:
                y_product = ylm[ilm1] * ylm[ilm2]
                y_terms = ylm[ilm] * (
                    r_sq[None, :] ** ((l1 + l2 - l_all)//2)[:, None])
                results = np.linalg.lstsq(y_terms.T, y_product, rcond=None)
                coeff = results[0]
                err = np.sqrt(results[1][0]) / np.linalg.norm(y_product)
                assert(err < ERR_TOL)
                # Select non-zero coefficients to form product expansion:
                sel = np.where(np.abs(coeff)
                               > COEFF_TOL * np.linalg.norm(coeff))[0]
                ilm = ilm[sel]
                coeff = coeff[sel]
                # Sort by (l,m):
                sort_index = ilm.argsort()
                ilm = ilm[sort_index]
                coeff = coeff[sort_index]
                # Format as python code:
                print(line)  # pending data from previous entry
                line = f'    ({ilm1}, {ilm2}): ('
                padding = len(line)
                line = print_array(ilm, line, padding, '{:d}') + ', '
                line = print_array(coeff, line, padding, '{:.16f}') + '),'
        print(line.rstrip(', ') + '}')

    # Parse command line:
    parser = argparse.ArgumentParser(
        description='Generate / test real spherical harmonic coefficients')
    # --- mutually-exclusive group of generate or test
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-g', '--generate', action='store_true',
                       help='generate coefficients')
    group.add_argument('-t', '--test', action='store_true',
                       help='test coefficients')
    # ---
    args = parser.parse_args()

    if args.generate:
        l_max_hlf = 3
        generate_harmonic_coefficients(l_max_hlf)

    if args.test:
        r = np.random.randn(1000, 3)
        r_sq = (r**2).sum(axis=-1)

        # Test spherical harmonics:
        print('Testing spherical harmonics:')
        rel_err_all = []
        ylm = get_harmonics(shdata.L_MAX, r)
        ylm_ref = get_harmonics_ref(shdata.L_MAX, r)
        for l in range(shdata.L_MAX+1):
            err = np.linalg.norm(ylm[l] - ylm_ref[l])
            rel_err = err / np.linalg.norm(ylm_ref[l])
            rel_err_all.append(rel_err)
            print(f"  l: {l} Err: {rel_err:9.3e}")
        rel_err_overall = np.sqrt((np.array(rel_err_all)**2).mean())
        print(f'  Overall Err: {rel_err_overall:9.3e}\n')

        # Test product coefficients:
        print('Testing product coefficients:')
        ylm_all = np.vstack(ylm)  # flatten into a single array with all (l,m)
        rel_err_all = []
        for l1 in range(shdata.L_MAX_HLF+1):
            for l2 in range(l1+1):
                product_ref = ylm[l1][:, None, :] * ylm[l2][None, :, :]
                product = np.zeros_like(product_ref)
                for m1 in range(-l1, l1+1):
                    ilm1 = l1*(l1 + 1) + m1
                    for m2 in range(-l2, l2+1):
                        ilm2 = l2*(l2 + 1) + m2
                        index, coeffs = shdata.YLM_PROD[(max(ilm1, ilm2),
                                                         min(ilm1, ilm2))]
                        dl = l1 + l2 - np.floor(np.sqrt(index)).astype(int)
                        prod_terms = ylm_all[index] * (
                            r_sq[None, :] ** (dl[:, None]//2))
                        product[l1+m1, l2+m2] = np.dot(coeffs, prod_terms)
                err = np.linalg.norm(product - product_ref)
                rel_err = err / np.linalg.norm(product_ref)
                rel_err_all.append(rel_err)
                print(f"  l: {l1} {l2} Err: {rel_err:9.3e}")
        rel_err_overall = np.sqrt((np.array(rel_err_all)**2).mean())
        print(f'  Overall Err: {rel_err_overall:9.3e}')
