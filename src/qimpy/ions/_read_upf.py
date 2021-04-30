import xml.etree.ElementTree as ET
import qimpy as qp
import numpy as np
import re


def _read_upf(self, filename, rc):
    '''Read a UPF pseudopotential.
    Note that only norm-conserving UPF files are currently supported.

    Parameters
    ----------
    filename : str
        Full path to the UPF file to read.
    rc : qimpy.utils.RunConfig
        Current run configuration.
    '''
    watch = qp.utils.StopWatch('read_upf', rc)
    qp.log.info("\nReading '{:s}':".format(filename))
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
                self.atomicNumber = \
                    qp.ions.symbols.ATOMIC_NUMBERS[self.element]
            except KeyError:
                qp.log.error(
                    "  Could not determine atomic number for element '"
                    + self.element + "'.\n  Please edit pseudopotential to"
                    "use the standard chemical symbol.")
                raise ValueError('Invalid chemical symbol in '+filename)
            qp.log.info("  '{:s}' pseudopotential, '{:s}' functional".format(
                self.element, section.attrib["functional"]))

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
            self.isPaw = (
                section.attrib.get("is_paw").lower() in ['t', 'true'])
            if self.isPaw:
                qp.log.error("  PAW datasets are not yet supported.")
                raise ValueError('PAW dataset in '+filename+' unsupported')

            # Valence properties:
            self.Z = float(section.attrib["z_valence"])
            self.lMax = int(section.attrib["l_max"])
            self.nGrid = int(section.attrib["mesh_size"])
            self.nBeta = int(section.attrib["number_of_proj"])
            self.nPsi = int(section.attrib["number_of_wfc"])
            qp.log.info((
                "  {:g} valence electrons, {:d} orbitals, {:d} projectors, "
                + "{:d} radial grid points, with lMax = {:d}").format(
                self.Z, self.nPsi, self.nBeta, self.nGrid, self.lMax))

        elif section.tag == 'PP_MESH':
            # Radial grid and integration weight:
            for entry in section:
                if entry.tag == 'PP_R':
                    self.rGrid = np.fromstring(entry.text, sep=' ')
                    if not self.rGrid[0]:
                        self.rGrid[0] = 1e-3 * self.rGrid[1]  # avoid 1/0 below
                elif entry.tag == 'PP_RAB':
                    self.drGrid = np.fromstring(entry.text, sep=' ')
                else:
                    qp.log.info("  NOTE: ignored section '{:s}'".format(
                        entry.tag))

        elif section.tag == 'PP_NLCC':
            # Nonlinear / partial core correction (optional):
            self.nCore = np.fromstring(section.text, sep=' ')

        elif section.tag == 'PP_LOCAL':
            # Local potential:
            self.Vloc = np.fromstring(section.text, sep=' ')
            self.Vloc *= 0.5  # Convert from Ry to Eh
            self.Vloc += np.where(
                self.rGrid >= 0, self.Z/self.rGrid, 0.)  # remove Z/r part

        elif section.tag == 'PP_NONLOCAL':
            self.beta = np.zeros((self.nBeta, len(self.rGrid)))  # projectors
            self.lBeta = np.zeros(self.nBeta, dtype=int)  # angular momenta

            for entry in section:

                if entry.tag.startswith('PP_BETA.'):
                    # Check projector number:
                    iBeta = int(entry.tag[8:]) - 1
                    assert((iBeta >= 0) and (iBeta < self.nBeta))
                    # Get projector angular momentum:
                    self.lBeta[iBeta] = entry.attrib['angular_momentum']
                    assert(self.lBeta[iBeta] <= self.lMax)
                    # Read projector (and remove 1/r factor stored in PS):
                    self.beta[iBeta] = np.fromstring(entry.text, sep=' ')
                    self.beta[iBeta] *= np.where(
                        self.rGrid >= 0, 1./self.rGrid, 0.)

                elif entry.tag == 'PP_DIJ':
                    # Get descreened 'D' matrix of pseudopotential:
                    if self.nBeta:
                        self.D = np.fromstring(entry.text, sep=' ')
                        self.D = self.D.reshape(self.nBeta, self.nBeta) * 0.5
                        # Note: 0.5 converts from Ry to Eh
                    else:
                        # np.fromstring misbehaves for an empty string
                        self.D = np.zeros((0, 0))

                else:
                    qp.log.info("  NOTE: ignored section '{:s}'".format(
                        entry.tag))

        elif section.tag == 'PP_PSWFC':
            self.psi = np.zeros((self.nPsi, len(self.rGrid)))  # orbitals
            self.lPsi = np.zeros(self.nPsi, dtype=int)  # angular momenta
            self.eigPsi = np.zeros(self.nPsi)  # eigenvalue by orbital
            for entry in section:
                if entry.tag.startswith('PP_CHI.'):
                    # Check orbital number:
                    iPsi = int(entry.tag[7:]) - 1
                    assert((iPsi >= 0) and (iPsi < self.nPsi))
                    # Get orbital angular momentum:
                    self.lPsi[iPsi] = entry.attrib['l']
                    assert(self.lPsi[iPsi] <= self.lMax)
                    # Report orbital:
                    occ = float(entry.attrib["occupation"])
                    label = entry.attrib["label"]
                    self.eigPsi[iPsi] = float(entry.attrib.get(
                        "pseudo_energy", "NaN")) * 0.5  # convert from Ry to Eh
                    qp.log.info((
                        "    {:3s}   l: {:d}   occupation: {:4.1f}   "
                        + "eigenvalue: {:f}").format(
                        label, self.lPsi[iPsi], occ, self.eigPsi[iPsi]))
                    # Read orbital (and remove 1/r factor stored in PS):
                    self.psi[iPsi] = np.fromstring(entry.text, sep=' ')
                    self.psi[iPsi] *= np.where(
                        self.rGrid >= 0, 1./self.rGrid, 0.)
                else:
                    qp.log.info(
                        "  NOTE: ignored section '{:s}'".format(entry.tag))

        elif section.tag == 'PP_RHOATOM':
            # Read atom electron density (removing 4 pi r^2 factor in PS file):
            self.rhoAtom = np.fromstring(section.text, sep=' ')
            self.rhoAtom *= np.where(
                self.rGrid >= 0, 1./(4*np.pi*self.rGrid**2), 0.)

        elif section.tag == 'PP_SPIN_ORB':
            self.jBeta = np.zeros(self.nBeta)   # j for each projector
            self.jPsi = np.zeros(self.nPsi)     # j for each orbital
            for entry in section:
                if entry.tag.startswith('PP_RELBETA.'):
                    # Check projector number:
                    iBeta = int(entry.tag[11:]) - 1
                    assert((iBeta >= 0) and (iBeta < self.nBeta))
                    # Get projector's total angular momentum:
                    self.jBeta[iBeta] = entry.attrib['jjj']
                elif entry.tag.startswith('PP_RELWFC.'):
                    # Check orbital number:
                    iPsi = int(entry.tag[10:]) - 1
                    assert((iPsi >= 0) and (iPsi < self.nPsi))
                    # Get orbital's total angular momentum:
                    self.jPsi[iPsi] = entry.attrib['jchi']
                else:
                    qp.log.info(
                        "  NOTE: ignored section '{:s}'".format(entry.tag))

        else:
            qp.log.info("  NOTE: ignored section '{:s}'".format(section.tag))
    watch.stop()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    qp.log_config()

    psPath = '/home/shankar/DFT/Pseudopotentials/PSlib'
    psNames = [
        'H.pbe-n-nc.UPF',  # NC example
        'H.pbe-rrkjus_psl.0.1.UPF',  # US example
        'Pt.pbe-n-rrkjus_psl.0.1.UPF',  # US example with nlcc
    ]

    for psName in psNames:

        # Read pseudopotential:
        ps = qp.ions.Pseudopotential(psPath+'/'+psName)

        # Plot local potential and densities:
        plt.figure()
        plt.title(psName + ' density/potential')
        plt.plot(ps.rGrid, ps.rhoAtom, label=r'$\rho_{\mathrm{atom}}(r)$')
        if hasattr(ps, 'nCore'):
            plt.plot(ps.rGrid, ps.nCore, label=r'$n_{\mathrm{core}}(r)$')
        plt.plot(ps.rGrid, ps.rGrid * ps.Vloc,
                 label=r'$r V_{\mathrm{loc}}(r)$')
        plt.xlabel(r'$r$')
        plt.xlim(0, 10.)
        plt.legend()

        # Plot projectors:
        plt.figure()
        plt.title(psName + ' projectors')
        for iBeta, beta in enumerate(ps.beta):
            plt.plot(ps.rGrid, beta,
                     label=r'$\beta_'+'spdf'[ps.lBeta[iBeta]]+'(r)$')
        plt.xlabel(r'$r$')
        plt.xlim(0, 10.)
        plt.legend()

        # Plot projectors:
        plt.figure()
        plt.title(psName + ' orbitals')
        for iPsi, psi in enumerate(ps.psi):
            plt.plot(ps.rGrid, psi,
                     label=r'$\psi_'+'spdf'[ps.lPsi[iPsi]]+'(r)$')
        plt.xlabel(r'$r$')
        plt.xlim(0, 10.)
        plt.legend()
    plt.show()
