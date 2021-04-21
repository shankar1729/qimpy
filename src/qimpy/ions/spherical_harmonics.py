from qimpy.ions._spherical_harmonics_data import YLM_PROD


def Y4(x, y, z):
    return 1.0925484305920792*x*y


def Y5(x, y, z):
    return 1.0925484305920792*y*z


def Y6(x, y, z):
    return -0.31539156525252005*(x*x + y*y - 2.*z*z)


def Y7(x, y, z):
    return 1.0925484305920792*x*z


def Y8(x, y, z):
    return 0.5462742152960396*(x-y)*(x+y)


def Y9(x, y, z):
    return -0.5900435899266435*y*(y*y-3.*x*x)


def Y10(x, y, z):
    return 2.890611442640554*x*y*z


def Y11(x, y, z):
    return -0.4570457994644658*y*(x*x + y*y - 4.*z*z)


def Y12(x, y, z):
    return 0.3731763325901154*z*(2.*z*z - 3.*(x*x + y*y))


def Y13(x, y, z):
    return -0.4570457994644658*x*(x*x + y*y - 4.*z*z)


def Y14(x, y, z):
    return 1.445305721320277*(x-y)*(x+y)*z


def Y15(x, y, z):
    return 0.5900435899266435*x*(x*x - 3.*y*y)


def Y16(x, y, z):
    return 2.5033429417967046*x*y*(x-y)*(x+y)


def Y17(x, y, z):
    return -1.7701307697799304*y*z*(y*y - 3.*x*x)


def Y18(x, y, z):
    return -0.9461746957575601*x*y*(x*x + y*y - 6.*z*z)


def Y19(x, y, z):
    return -0.6690465435572892*y*z*(3.*(x*x + y*y) - 4.*z*z)


def Y20(x, y, z):
    x2y2 = x*x + y*y
    z2 = z*z
    return 0.03526184897173477*(9.*x2y2*(x2y2 - 8*z2) + 24.*z2*z2)


def Y21(x, y, z):
    return -0.6690465435572892*x*z*(3.*(x*x + y*y) - 4.*z*z)


def Y22(x, y, z):
    x2 = x*x
    y2 = y*y
    return -0.47308734787878004*(x2 - y2)*(x2 + y2 - 6.*z*z)


def Y23(x, y, z):
    return 1.7701307697799304*x*z*(x*x - 3.*y*y)


def Y24(x, y, z):
    x2 = x*x
    y2 = y*y
    return 0.6258357354491761*(x2*(x2 - 6.*y2) + y2*y2)


def Y25(x, y, z):
    x2 = x*x
    y2 = y*y
    return 0.6563820568401701*y*(5.*x2*(x2 - 2.*y2) + y2*y2)


def Y26(x, y, z):
    return 8.302649259524166*x*y*z*(x-y)*(x+y)


def Y27(x, y, z):
    x2 = x*x
    y2 = y*y
    return 0.4892382994352504*y*(y2 - 3.*x2)*(x2 + y2 - 8.*z*z)


def Y28(x, y, z):
    return -4.793536784973324*x*y*z*(x*x + y*y - 2.*z*z)


def Y29(x, y, z):
    x2y2 = x*x + y*y
    z2 = z*z
    return 0.45294665119569694*y*(x2y2*(x2y2 - 12.*z2) + 8.*z2*z2)


def Y30(x, y, z):
    x2y2 = x*x + y*y
    z2 = z*z
    return 0.1169503224534236*z*(15.*x2y2*x2y2 - 8.*z2*(5.*x2y2 - z2))


def Y31(x, y, z):
    x2y2 = x*x + y*y
    z2 = z*z
    return 0.45294665119569694*x*(x2y2*(x2y2 - 12.*z2) + 8.*z2*z2)


def Y32(x, y, z):
    x2 = x*x
    y2 = y*y
    return -2.396768392486662*(x2-y2)*z*(x2 + y2 - 2.*z*z)


def Y33(x, y, z):
    x2 = x*x
    y2 = y*y
    return -0.4892382994352504*x*(x2 - 3.*y2)*(x2 + y2 - 8.*z*z)


def Y34(x, y, z):
    x2 = x*x
    y2 = y*y
    return 2.0756623148810416*z*(x2*(x2 - 6.*y2) + y2*y2)


def Y35(x, y, z):
    x2 = x*x
    y2 = y*y
    return 0.6563820568401701*x*(x2*(x2 - 10.*y2) + 5.*y2*y2)


def Y36(x, y, z):
    x2 = x*x
    y2 = y*y
    return 1.3663682103838286*x*y*(x2*(3.*x2 - 10.*y2) + 3.*y2*y2)


def Y37(x, y, z):
    x2 = x*x
    y2 = y*y
    return 2.366619162231752*y*z*(5.*x2*(x2 - 2.*y2) + y2*y2)


def Y38(x, y, z):
    x2 = x*x
    y2 = y*y
    return -2.0182596029148967*x*y*(x2 - y2)*(x2 + y2 - 10.*z*z)


def Y39(x, y, z):
    x2 = x*x
    y2 = y*y
    return 0.9212052595149236*y*z*(y2 - 3.*x2)*(3.*(x2 + y2) - 8.*z*z)


def Y40(x, y, z):
    x2y2 = x*x + y*y
    z2 = z*z
    return 0.9212052595149236*x*y*(x2y2*(x2y2 - 16.*z2) + 16.*z2*z2)


def Y41(x, y, z):
    x2y2 = x*x + y*y
    z2 = z*z
    return 0.5826213625187314*y*z*(5.*x2y2*(x2y2 - 4.*z2) + 8.*z2*z2)


def Y42(x, y, z):
    x2y2 = x*x + y*y
    z2 = z*z
    return 0.06356920226762842*(5.*x2y2*x2y2*(18.*z2 - x2y2)
                                + 8.*z2*z2*(2.*z2 - 15.*x2y2))


def Y43(x, y, z):
    x2y2 = x*x + y*y
    z2 = z*z
    return 0.5826213625187314*x*z*(5.*x2y2*(x2y2 - 4.*z2) + 8.*z2*z2)


def Y44(x, y, z):
    x2 = x*x
    y2 = y*y
    x2y2 = x2 + y2
    z2 = z*z
    return 0.4606026297574618*(x2-y2)*(x2y2*(x2y2 - 16.*z2) + 16.*z2*z2)


def Y45(x, y, z):
    x2 = x*x
    y2 = y*y
    return -0.9212052595149236*x*z*(x2 - 3.*y2)*(3.*(x2 + y2) - 8.*z*z)


def Y46(x, y, z):
    x2 = x*x
    y2 = y*y
    return -0.5045649007287242*(x2*(x2 - 6.*y2) + y2*y2)*(x2 + y2 - 10.*z*z)


def Y47(x, y, z):
    x2 = x*x
    y2 = y*y
    return 2.366619162231752*x*z*(x2*(x2 - 10.*y2) + 5.*y2*y2)


def Y48(x, y, z):
    x2 = x*x
    y2 = y*y
    return 0.6831841051919143*(x2*x2*(x2 - 15.*y2) + y2*y2*(15.*x2 - y2))


Yarr = [
    [],
    [],
    [Y4, Y5, Y6, Y7, Y8],
    [Y9, Y10, Y11, Y12, Y13, Y14, Y15],
    [Y16, Y17, Y18, Y19, Y20, Y21, Y22, Y23, Y24],
    [Y25, Y26, Y27, Y28, Y29, Y30, Y31, Y32, Y33, Y34, Y35],
    [Y36, Y37, Y38, Y39, Y40, Y41, Y42, Y43, Y44, Y45, Y46, Y47, Y48]]


def get_harmonics(l_max, r):
    assert(l_max <= 6)
    results = []
    if l_max >= 0:
        results.append(0.28209479177387814 * np.ones((1,) + r.shape[:-1]))
    if l_max >= 1:
        results.append(0.4886025119029199 * r[..., (1, 2, 0)].T)
    for l in range(2, l_max+1):
        result = np.zeros((2*l + 1,) + r.shape[:-1])
        for lm in range(2*l + 1):
            result[lm] = Yarr[l][lm](r[..., 0], r[..., 1], r[..., 2])
        results.append(result)
    return results


if __name__ == "__main__":
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

    def generate_harmonic_coefficients(l_max_hlf):
        '''Generate tables of recurrence coefficients for computing
        real solid harmonics up to l_max = 2 * l_max_hlf, as well as
        tables of product coefficients (Clebsch-Gordon coefficients)
        for real solid harmonics up to order l_max_hlf.
        for real solid harmonics till l_max used above.
        Print results formatted as the Python code above.'''
        l_max = 2 * l_max_hlf
        lm_all = [(l, m)
                  for l in range(l_max + 1)
                  for m in range(-l, l+1)]
        lm_hlf = [(l, m)
                  for l in range(l_max_hlf + 1)
                  for m in range(-l, l+1)]
        # Calculate solid harmonics on a mesh covering unit cube:
        grids1d = 3 * (np.linspace(-1., 1., 2*l_max), )  # avoids zero
        r = np.array(np.meshgrid(*grids1d)).reshape(3, -1).T
        r_sq = (r**2).sum(axis=-1)
        ylm = np.vstack(get_harmonics_ref(l_max, r))
        # Calculate Clebsch-Gordon coefficients:
        print('# Clebsch-Gordon coefficients for products of real harmonics.')
        print('# The integer indices correspond to l*(l+1)+m for each (l,m).')
        print('YLM_PROD = {')
        for ilm1, (l1, m1) in enumerate(lm_hlf):
            for ilm2, (l2, m2) in enumerate(lm_hlf[:ilm1+1]):
                # List (l,m) pairs allowed by angular momentum addition rules:
                m_all = {m1 + m2, m1 - m2, m2 - m1, -(m1 + m2)}
                l_all = range(l1 - l2, l1 + l2 + 1, 2)
                lm_all = np.array([(l, m)
                                   for l in l_all
                                   for m in m_all
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
                assert(err < 1e-14)
                # Select non-zero coefficients to form product expansion:
                sel = np.where(np.abs(coeff) > 1e-8 * np.linalg.norm(coeff))[0]
                ilm = ilm[sel]
                coeff = coeff[sel]
                # Sort by (l,m):
                sort_index = ilm.argsort()
                ilm = ilm[sort_index]
                coeff = coeff[sort_index]
                # Format as python code:
                prefix = '    ({:d}, {:d}): ('.format(ilm1, ilm2)
                indices = '[' + ', '.join([str(i) for i in ilm]) + '],'
                len_available = 79 - len(prefix) - len(indices)
                single_line = (len_available > 21*len(coeff) + 5)
                if single_line:
                    indices += ' ['
                    indices += ', '.join(['{:.16f}'.format(c) for c in coeff])
                    indices += ']),'
                else:
                    padding = len(prefix)
                    line = (' '*padding) + '['
                    for c in coeff:
                        if len(line) + 22 >= 79:  # wrap
                            indices += '\n' + line
                            line = ' '*padding
                        line += '' if (line[-1] == '[') else ' '
                        line += '{:.16f},'.format(c)
                    indices += '\n' + line[:-1] + ']),'
                print(prefix + indices)
        print('}')
    generate_harmonic_coefficients(3)
    exit()

    r = np.random.randn(1000, 3)
    rel_err_all = []
    l_max = 6
    ylm = get_harmonics(l_max, r)
    ylm_ref = get_harmonics_ref(l_max, r)
    for l in range(l_max+1):
        err = np.linalg.norm(ylm[l] - ylm_ref[l])
        rel_err = err / np.linalg.norm(ylm_ref[l])
        rel_err_all.append(rel_err)
        print("l: {:d} Err: {:9.3e}".format(l, rel_err))
    print('Overall Err:', np.sqrt((np.array(rel_err_all)**2).mean()))
