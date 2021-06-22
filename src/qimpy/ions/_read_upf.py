import xml.etree.ElementTree as ET
import qimpy as qp
import numpy as np
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

    # Loop over sections in UPF file:
    for section in upf:

        if section.tag == 'PP_INFO':
            pass

        elif section.tag == 'PP_HEADER':
            # Get element:
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

            # Non essential info:
            def optional_attrib(name, prefix='  ', suffix='\n'):
                attrib = section.attrib.get(name, None)
                return ((prefix+attrib+suffix) if attrib else '')
            optionals = (optional_attrib('generated')
                         + optional_attrib('comment')
                         + optional_attrib('author', '  Author: ', '')
                         + optional_attrib('date', '  Date: ', ''))
            if optionals:
                qp.log.info(optionals.rstrip('\n'))

            # Check for unsupported types:
            self.is_paw = (
                str(section.attrib.get("is_paw")).lower() in ['t', 'true'])
            if self.is_paw:
                qp.log.error("  PAW datasets are not yet supported.")
                raise ValueError('PAW dataset in '+filename+' unsupported')

            # Valence properties:
            self.Z = float(section.attrib["z_valence"])
            self.l_max = int(section.attrib["l_max"])
            n_grid = int(section.attrib["mesh_size"])
            n_beta = int(section.attrib["number_of_proj"])
            n_psi = int(section.attrib["number_of_wfc"])
            qp.log.info(f"  {self.Z:g} valence electrons, {n_psi}"
                        f" orbitals, {n_beta} projectors, {n_grid}"
                        f" radial grid points, with l_max = {self.l_max}")

        elif section.tag == 'PP_MESH':
            # Radial grid and integration weight:
            for entry in section:
                if entry.tag == 'PP_R':
                    self.r_grid = np.fromstring(entry.text, sep=' ')
                    if not self.r_grid[0]:  # avoid divide by 0 below
                        self.r_grid[0] = 1e-3 * self.r_grid[1]
                elif entry.tag == 'PP_RAB':
                    self.dr_grid = np.fromstring(entry.text, sep=' ')
                else:
                    qp.log.info(f"  NOTE: ignored section '{entry.tag}'")

        elif section.tag == 'PP_NLCC':
            # Nonlinear / partial core correction (optional):
            self.n_core = np.fromstring(section.text, sep=' ')

        elif section.tag == 'PP_LOCAL':
            # Local potential:
            self.Vloc = np.fromstring(section.text, sep=' ')
            self.Vloc *= 0.5  # Convert from Ry to Eh
            self.Vloc += np.where(
                self.r_grid >= 0, self.Z/self.r_grid, 0.)  # remove Z/r part

        elif section.tag == 'PP_NONLOCAL':
            self.beta = np.zeros((n_beta, len(self.r_grid)))  # projectors
            self.l_beta = np.zeros(n_beta, dtype=int)  # angular momenta

            for entry in section:

                if entry.tag.startswith('PP_BETA.'):
                    # Check projector number:
                    i_beta = int(entry.tag[8:]) - 1
                    assert((i_beta >= 0) and (i_beta < n_beta))
                    # Get projector angular momentum:
                    self.l_beta[i_beta] = entry.attrib['angular_momentum']
                    assert(self.l_beta[i_beta] <= self.l_max)
                    # Read projector (and remove 1/r factor stored in PS):
                    self.beta[i_beta] = np.fromstring(entry.text, sep=' ')
                    self.beta[i_beta] *= np.where(
                        self.r_grid >= 0, 1./self.r_grid, 0.)

                elif entry.tag == 'PP_DIJ':
                    # Get descreened 'D' matrix of pseudopotential:
                    if n_beta:
                        self.D = np.fromstring(entry.text, sep=' ')
                        self.D = self.D.reshape(n_beta, n_beta) * 0.5
                        # Note: 0.5 converts from Ry to Eh
                    else:
                        # np.fromstring misbehaves for an empty string
                        self.D = np.zeros((0, 0))

                else:
                    qp.log.info(f"  NOTE: ignored section '{entry.tag}'")

        elif section.tag == 'PP_PSWFC':
            self.psi = np.zeros((n_psi, len(self.r_grid)))  # orbitals
            self.l_psi = np.zeros(n_psi, dtype=int)  # angular momenta
            self.eig_psi = np.zeros(n_psi)  # eigenvalue by orbital
            for entry in section:
                if entry.tag.startswith('PP_CHI.'):
                    # Check orbital number:
                    i_psi = int(entry.tag[7:]) - 1
                    assert((i_psi >= 0) and (i_psi < n_psi))
                    # Get orbital angular momentum:
                    self.l_psi[i_psi] = entry.attrib['l']
                    assert(self.l_psi[i_psi] <= self.l_max)
                    # Report orbital:
                    occ = float(entry.attrib["occupation"])
                    label = entry.attrib["label"]
                    self.eig_psi[i_psi] = float(entry.attrib.get(
                        "pseudo_energy", "NaN")) * 0.5  # convert from Ry to Eh
                    qp.log.info(f"    {label}   l: {self.l_psi[i_psi]}'"
                                f"   occupation: {occ:4.1f}"
                                f"   eigenvalue: {self.eig_psi[i_psi]}")
                    # Read orbital (and remove 1/r factor stored in PS):
                    self.psi[i_psi] = np.fromstring(entry.text, sep=' ')
                    self.psi[i_psi] *= np.where(
                        self.r_grid >= 0, 1./self.r_grid, 0.)
                else:
                    qp.log.info(f"  NOTE: ignored section '{entry.tag}'")

        elif section.tag == 'PP_RHOATOM':
            # Read atom electron density (removing 4 pi r^2 factor in PS file):
            self.rho_atom = np.fromstring(section.text, sep=' ')
            self.rho_atom *= np.where(
                self.r_grid >= 0, 1./(4*np.pi*self.r_grid**2), 0.)

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
            qp.log.info(f"  NOTE: ignored section '{entry.tag}'")
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
        plt.figure()
        plt.title(psName + ' density/potential')
        plt.plot(ps.r_grid, ps.rho_atom, label=r'$\rho_{\mathrm{atom}}(r)$')
        if hasattr(ps, 'nCore'):
            plt.plot(ps.r_grid, ps.n_core, label=r'$n_{\mathrm{core}}(r)$')
        plt.plot(ps.r_grid, ps.r_grid * ps.Vloc,
                 label=r'$r V_{\mathrm{loc}}(r)$')
        plt.xlabel(r'$r$')
        plt.xlim(0, 10.)
        plt.legend()

        # Plot projectors:
        plt.figure()
        plt.title(psName + ' projectors')
        for i_beta, beta in enumerate(ps.beta):
            plt.plot(ps.r_grid, beta,
                     label=r'$\beta_'+'spdf'[ps.l_beta[i_beta]]+'(r)$')
        plt.xlabel(r'$r$')
        plt.xlim(0, 10.)
        plt.legend()

        # Plot projectors:
        plt.figure()
        plt.title(psName + ' orbitals')
        for i_psi, psi in enumerate(ps.psi):
            plt.plot(ps.r_grid, psi,
                     label=r'$\psi_'+'spdf'[ps.l_psi[i_psi]]+'(r)$')
        plt.xlabel(r'$r$')
        plt.xlim(0, 10.)
        plt.legend()
    plt.show()
