import xml.etree.ElementTree as ET
import qimpy as qp
import numpy as np
import torch
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._pseudopotential import Pseudopotential
    from ..utils import RunConfig


def _read_upf(self: 'Pseudopotential', filename: str, rc: 'RunConfig'):
    """Read a UPF pseudopotential.
    Note that only norm-conserving UPF files are currently supported.

    Parameters
    ----------
    filename : str
        Full path to the UPF file to read.
    rc : qimpy.utils.RunConfig
        Current run configuration.
    """
    watch = qp.utils.StopWatch('read_upf', rc)
    qp.log.info(f"\nReading '{filename}':")
    upf = ET.fromstring(open(filename, 'r').read().replace('&', '&amp;'))
    assert(upf.tag == 'UPF')

    # Read header first:
    section = upf.find('PP_HEADER')
    assert section is not None

    # --- Get element:
    try:
        self.element = section.attrib["element"].strip()
        self.atomic_number = \
            qp.ions.symbols.ATOMIC_NUMBERS[self.element]
    except KeyError:
        qp.log.error(
            "  Could not determine atomic number for element '"
            + self.element + "'.\n  Please edit pseudopotential to"
            "use the standard chemical symbol.")
        raise ValueError('Invalid chemical symbol in '+filename)
    qp.log.info(f"  '{self.element}' pseudopotential,"
                f" '{section.attrib['functional']}' functional")

    # --- Non essential info:
    def optional_attrib(name, prefix='  ', suffix='\n'):
        attrib = section.attrib.get(name, None)
        return ((prefix+attrib+suffix) if attrib else '')
    optionals = (optional_attrib('generated')
                 + optional_attrib('comment')
                 + optional_attrib('author', '  Author: ', '')
                 + optional_attrib('date', '  Date: ', ''))
    if optionals:
        qp.log.info(optionals.rstrip('\n'))

    # --- Check for unsupported types:
    self.is_paw = (
        str(section.attrib.get("is_paw")).lower() in ['t', 'true'])
    if self.is_paw:
        qp.log.error("  PAW datasets are not yet supported.")
        raise ValueError('PAW dataset in '+filename+' unsupported')

    # --- Valence properties:
    self.Z = float(section.attrib["z_valence"])
    self.l_max = int(section.attrib["l_max"])
    n_grid = int(section.attrib["mesh_size"])
    n_beta = int(section.attrib["number_of_proj"])
    n_psi = int(section.attrib["number_of_wfc"])
    qp.log.info(f"  {self.Z:g} valence electrons, {n_psi}"
                f" orbitals, {n_beta} projectors, {n_grid}"
                f" radial grid points, with l_max = {self.l_max}")

    # Get radial grid and integration weight before any radial functions:
    section = upf.find('PP_MESH')
    assert section is not None
    r = None
    for entry in section:
        if entry.tag == 'PP_R':
            r = np.fromstring(entry.text, sep=' ')
            if not r[0]:  # avoid divide by 0 below
                r[0] = 1e-3 * r[1]
            self.r = torch.tensor(r, device=rc.device)
        elif entry.tag == 'PP_RAB':
            self.dr = torch.tensor(np.fromstring(entry.text, sep=' '),
                                   device=rc.device)
        else:
            qp.log.info(f"  NOTE: ignored section '{entry.tag}'")
    assert r is not None

    # Read all remaining sections (order not relevant):
    for section in upf:

        if section.tag in ('PP_INFO', 'PP_HEADER', 'PP_MESH'):
            pass  # not needed / already parsed above

        elif section.tag == 'PP_NLCC':
            # Nonlinear / partial core correction (optional):
            self.n_core = qp.ions.RadialFunction(
                self.r, self.dr, np.fromstring(section.text, sep=' '))

        elif section.tag == 'PP_LOCAL':
            # Local potential:
            self.ion_width = np.inf  # No range separation yet
            self.Vloc = qp.ions.RadialFunction(
                self.r, self.dr, 0.5 * np.fromstring(section.text, sep=' '))
            # Note: 0.5 above converts from Ry to Eh

        elif section.tag == 'PP_NONLOCAL':
            beta = np.zeros((n_beta, len(r)))  # projectors
            l_beta = np.zeros(n_beta, dtype=int)  # angular momenta
            for entry in section:
                if entry.tag.startswith('PP_BETA.'):
                    # Check projector number:
                    i_beta = int(entry.tag[8:]) - 1
                    assert((i_beta >= 0) and (i_beta < n_beta))
                    # Get projector angular momentum:
                    l_beta[i_beta] = entry.attrib['angular_momentum']
                    assert(l_beta[i_beta] <= self.l_max)
                    # Read projector (contains factor of r removed below):
                    beta[i_beta] = np.fromstring(entry.text, sep=' ')
                elif entry.tag == 'PP_DIJ':
                    # Get descreened 'D' matrix of pseudopotential:
                    if n_beta:
                        self.D = torch.tensor(
                            np.fromstring(entry.text, sep=' ') * 0.5,
                            device=rc.device).reshape(n_beta, n_beta)
                        # Note: 0.5 above converts from Ry to Eh
                    else:
                        # np.fromstring misbehaves for an empty string
                        self.D = torch.zeros((0, 0), device=rc.device)
                else:
                    qp.log.info(f"  NOTE: ignored section '{entry.tag}'")
            # Create projector radial function:
            self.beta = qp.ions.RadialFunction(
                self.r, self.dr,
                beta * (r[None, :] ** -(l_beta + 1)[:, None]),
                l_beta)

        elif section.tag == 'PP_PSWFC':
            psi = np.zeros((n_psi, len(r)))  # orbitals
            l_psi = np.zeros(n_psi, dtype=int)  # angular momenta
            self.eig_psi = np.zeros(n_psi)  # eigenvalue by orbital
            for entry in section:
                if entry.tag.startswith('PP_CHI.'):
                    # Check orbital number:
                    i_psi = int(entry.tag[7:]) - 1
                    assert((i_psi >= 0) and (i_psi < n_psi))
                    # Get orbital angular momentum:
                    l_psi[i_psi] = entry.attrib['l']
                    assert(l_psi[i_psi] <= self.l_max)
                    # Report orbital:
                    occ = float(entry.attrib["occupation"])
                    label = entry.attrib["label"]
                    self.eig_psi[i_psi] = float(entry.attrib.get(
                        "pseudo_energy", "NaN")) * 0.5  # convert from Ry to Eh
                    qp.log.info(f"    {label}   l: {l_psi[i_psi]}'"
                                f"   occupation: {occ:4.1f}"
                                f"   eigenvalue: {self.eig_psi[i_psi]}")
                    # Read orbital (contains factor of r removed below):
                    psi[i_psi] = np.fromstring(entry.text, sep=' ')
                else:
                    qp.log.info(f"  NOTE: ignored section '{entry.tag}'")
            # Create orbitals radial function:
            self.psi = qp.ions.RadialFunction(
                self.r, self.dr,
                psi * (r[None, :] ** -(l_psi + 1)[:, None]),
                l_psi)

        elif section.tag == 'PP_RHOATOM':
            # Read atom electron density (removing 4 pi r^2 factor in PS file):
            self.rho_atom = qp.ions.RadialFunction(
                self.r, self.dr,
                np.fromstring(section.text, sep=' ') / (4*np.pi*(r**2)))

        elif section.tag == 'PP_SPIN_ORB':
            self.j_beta = np.zeros(n_beta)   # j for each projector
            self.j_psi = np.zeros(n_psi)     # j for each orbital
            for entry in section:
                if entry.tag.startswith('PP_RELBETA.'):
                    # Check projector number:
                    i_beta = int(entry.tag[11:]) - 1
                    assert((i_beta >= 0) and (i_beta < n_beta))
                    # Get projector's total angular momentum:
                    self.j_beta[i_beta] = entry.attrib['jjj']
                elif entry.tag.startswith('PP_RELWFC.'):
                    # Check orbital number:
                    i_psi = int(entry.tag[10:]) - 1
                    assert((i_psi >= 0) and (i_psi < n_psi))
                    # Get orbital's total angular momentum:
                    self.j_psi[i_psi] = entry.attrib['jchi']
                else:
                    qp.log.info(f"  NOTE: ignored section '{entry.tag}'")

        else:
            qp.log.info(f"  NOTE: ignored section '{section.tag}'")

    # Make sure some common entries are set:
    assert hasattr(self, 'Vloc')
    if not hasattr(self, 'rho_atom'):
        self.rho_atom = qp.ions.RadialFunction(self.r, self.dr)
    if not hasattr(self, 'n_core'):
        self.n_core = qp.ions.RadialFunction(self.r, self.dr)
    watch.stop()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    qp.utils.log_config()
    rc = qp.utils.RunConfig()

    psPath = '/home/shankar/DFT/Pseudopotentials/PSlib'
    psNames = [
        'H.pbe-n-nc.UPF',  # NC example
        'H.pbe-rrkjus_psl.0.1.UPF',  # US example
        'Pt.pbe-n-rrkjus_psl.0.1.UPF',  # US example with nlcc
    ]

    for psName in psNames:

        # Read pseudopotential:
        ps = qp.ions.Pseudopotential(psPath+'/'+psName, rc)

        # Plot local potential and densities:
        r = ps.r.to(rc.cpu)
        plt.figure()
        plt.title(psName + ' density/potential')
        plt.plot(r, ps.rho_atom.f.to(rc.cpu)[0],
                 label=r'$\rho_{\mathrm{atom}}(r)$')
        if hasattr(ps, 'nCore'):
            plt.plot(r, ps.n_core.f.to(rc.cpu)[0],
                     label=r'$n_{\mathrm{core}}(r)$')
        plt.plot(r, r * ps.Vloc.f.to(rc.cpu)[0],
                 label=r'$r V_{\mathrm{loc}}(r)$')
        plt.xlabel(r'$r$')
        plt.xlim(0, 10.)
        plt.legend()

        # Plot projectors:
        plt.figure()
        plt.title(psName + ' projectors')
        for i, beta_i in enumerate(ps.beta.f.to(rc.cpu)):
            l_i = int(ps.beta.l[i].item())
            plt.plot(r, beta_i, label=r'$\beta_'+'spdf'[l_i]+f'(r)/r^{l_i}$')
        plt.xlabel(r'$r$')
        plt.xlim(0, 10.)
        plt.legend()

        # Plot projectors:
        plt.figure()
        plt.title(psName + ' orbitals')
        for i, psi_i in enumerate(ps.psi.f.to(rc.cpu)):
            l_i = int(ps.psi.l[i].item())
            plt.plot(r, psi_i, label=r'$\psi_'+'spdf'[l_i]+f'(r)/r^{l_i}$')
        plt.xlabel(r'$r$')
        plt.xlim(0, 10.)
        plt.legend()
    plt.show()
