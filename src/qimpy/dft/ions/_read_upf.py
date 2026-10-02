from __future__ import annotations
import xml.etree.ElementTree as ET

import numpy as np
import torch

from qimpy import log, rc
from qimpy.math import RadialFunction
from qimpy.dft import ions
from . import symbols


def read_upf(ps: ions.Pseudopotential, filename: str) -> None:
    """Read a UPF pseudopotential.
    Note that only norm-conserving and ultrasoft UPF files are supported (not PAW).

    Parameters
    ----------
    ps
        Pseudopotential object to read into.
    filename
        Full path to the UPF file to read.
    """
    log.info(f"\nReading '{filename}':")
    upf = ET.fromstring(open(filename, "r").read().replace("&", "&amp;"))
    assert upf.tag == "UPF"

    # Read header first:
    section = upf.find("PP_HEADER")
    assert section is not None

    # --- Get element:
    try:
        ps.element = section.attrib["element"].strip()
        ps.atomic_number = symbols.ATOMIC_NUMBERS[ps.element]
    except KeyError:
        log.error(
            "  Could not determine atomic number for element '"
            + ps.element
            + "'.\n  Please edit pseudopotential to"
            "use the standard chemical symbol."
        )
        raise ValueError("Invalid chemical symbol in " + filename)
    log.info(
        f"  '{ps.element}' pseudopotential,"
        f" '{section.attrib['functional']}' functional"
    )

    # --- Non essential info:
    def optional_attrib(name, prefix="  ", suffix="\n"):
        attrib = section.attrib.get(name, None)
        return (prefix + attrib + suffix) if attrib else ""

    optionals = (
        optional_attrib("generated")
        + optional_attrib("comment")
        + optional_attrib("author", "  Author: ", "")
        + optional_attrib("date", "  Date: ", "")
    )
    if optionals:
        log.info(optionals.rstrip("\n"))

    # --- Check for unsupported types:
    ps.is_paw = str(section.attrib.get("is_paw")).lower() in ["t", "true"]
    if ps.is_paw:
        log.error("  PAW datasets are not yet supported.")
        raise ValueError("PAW dataset in " + filename + " unsupported")

    ps.is_ultrasoft = str(section.attrib.get("is_ultrasoft")).lower() in ["t", "true"]
    augmentation = None  # read from PP_NONLOCAL/PP_AUGMENTATION if ultrasoft

    # --- Valence properties:
    ps.Z = float(section.attrib["z_valence"])
    ps.l_max = int(section.attrib["l_max"])
    n_grid = int(section.attrib["mesh_size"])
    n_beta = int(section.attrib["number_of_proj"])
    n_psi = int(section.attrib["number_of_wfc"])
    log.info(
        f"  {ps.Z:g} valence electrons, {n_psi}"
        f" orbitals, {n_beta} projectors, {n_grid}"
        f" radial grid points, with l_max = {ps.l_max}"
    )

    # --- relativity:
    ps.is_relativistic = section.attrib["relativistic"] == "full"
    ps.j_psi = None
    ps.j_beta = None

    # Get radial grid and integration weight before any radial functions:
    section = upf.find("PP_MESH")
    assert section is not None
    r = None
    for entry in section:
        if entry.tag == "PP_R":
            assert entry.text is not None
            r = np.fromstring(entry.text, sep=" ")
            if not r[0]:  # avoid divide by 0 below
                r[0] = 1e-3 * r[1]
            ps.r = torch.tensor(r, device=rc.device)
        elif entry.tag == "PP_RAB":
            assert entry.text is not None
            ps.dr = torch.tensor(np.fromstring(entry.text, sep=" "), device=rc.device)
        else:
            log.info(f"  NOTE: ignored section '{entry.tag}'")
    assert r is not None

    # Read all remaining sections (order not relevant):
    for section in upf:

        if section.tag in ("PP_INFO", "PP_HEADER", "PP_MESH"):
            pass  # not needed / already parsed above

        elif section.tag == "PP_NLCC":
            # Nonlinear / partial core correction (optional):
            assert section.text is not None
            ps.n_core = RadialFunction(
                ps.r, ps.dr, np.fromstring(section.text, sep=" ")
            )

        elif section.tag == "PP_LOCAL":
            # Local potential:
            assert section.text is not None
            ps.ion_width = np.inf  # No range separation yet
            ps.Vloc = RadialFunction(
                ps.r, ps.dr, 0.5 * np.fromstring(section.text, sep=" ")
            )
            # Note: 0.5 above converts from Ry to Eh

        elif section.tag == "PP_NONLOCAL":
            beta = np.zeros((n_beta, len(r)))  # projectors
            l_beta = np.zeros(n_beta, dtype=int)  # angular momenta
            for entry in section:
                if entry.tag.startswith("PP_BETA."):
                    assert entry.text is not None
                    # Check projector number:
                    i_beta = int(entry.tag[8:]) - 1
                    assert (i_beta >= 0) and (i_beta < n_beta)
                    # Get projector angular momentum:
                    l_beta[i_beta] = entry.attrib["angular_momentum"]
                    assert l_beta[i_beta] <= ps.l_max
                    # Read projector (contains factor of r removed below):
                    beta[i_beta] = np.fromstring(entry.text, sep=" ")
                elif entry.tag == "PP_DIJ":
                    # Get descreened 'D' matrix of pseudopotential:
                    if n_beta:
                        assert entry.text is not None
                        ps.D = torch.tensor(
                            np.fromstring(entry.text, sep=" ") * 0.5,
                            device=rc.device,
                        ).reshape(n_beta, n_beta)
                        # Note: 0.5 above converts from Ry to Eh
                    else:
                        # np.fromstring misbehaves for an empty string
                        ps.D = torch.zeros((0, 0), device=rc.device)
                elif entry.tag == "PP_AUGMENTATION":
                    augmentation = _read_augmentation(entry, n_beta, len(r), ps.l_max)
                else:
                    log.info(f"  NOTE: ignored section '{entry.tag}'")
            # Create projector radial function:
            ps.beta = RadialFunction(
                ps.r, ps.dr, beta * (r[None, :] ** -(l_beta + 1)[:, None]), l_beta
            )

        elif section.tag == "PP_PSWFC":
            assert section.text is not None
            psi = np.zeros((n_psi, len(r)))  # orbitals
            l_psi = np.zeros(n_psi, dtype=int)  # angular momenta
            ps.eig_psi = np.zeros(n_psi)  # eigenvalue by orbital
            for entry in section:
                if entry.tag.startswith("PP_CHI."):
                    assert entry.text is not None
                    # Check orbital number:
                    i_psi = int(entry.tag[7:]) - 1
                    assert (i_psi >= 0) and (i_psi < n_psi)
                    # Get orbital angular momentum:
                    l_psi[i_psi] = entry.attrib["l"]
                    assert l_psi[i_psi] <= ps.l_max
                    # Report orbital:
                    occ = float(entry.attrib["occupation"])
                    label = entry.attrib["label"]
                    ps.eig_psi[i_psi] = (
                        float(entry.attrib.get("pseudo_energy", "NaN")) * 0.5
                    )  # convert from Ry to Eh
                    log.info(
                        f"    {label}   l: {l_psi[i_psi]}"
                        f"  occupation: {occ:4.1f}"
                        f"  eigenvalue: {ps.eig_psi[i_psi]}"
                    )
                    # Read orbital (contains factor of r removed below):
                    psi[i_psi] = np.fromstring(entry.text, sep=" ")
                else:
                    log.info(f"  NOTE: ignored section '{entry.tag}'")
            # Create orbitals radial function:
            ps.psi = RadialFunction(
                ps.r, ps.dr, psi * (r[None, :] ** -(l_psi + 1)[:, None]), l_psi
            )

        elif section.tag == "PP_RHOATOM":
            # Read atom electron density (removing 4 pi r^2 factor in PS file):
            assert section.text is not None
            ps.rho_atom = RadialFunction(
                ps.r,
                ps.dr,
                np.fromstring(section.text, sep=" ") / (4 * np.pi * (r**2)),
            )

        elif section.tag == "PP_SPIN_ORB":
            assert ps.is_relativistic
            j_beta = np.zeros(n_beta)  # j for each projector
            j_psi = np.zeros(n_psi)  # j for each orbital
            for entry in section:
                if entry.tag.startswith("PP_RELBETA."):
                    # Check projector number:
                    i_beta = int(entry.tag[11:]) - 1
                    assert (i_beta >= 0) and (i_beta < n_beta)
                    # Get projector's total angular momentum:
                    j_beta[i_beta] = entry.attrib["jjj"]
                elif entry.tag.startswith("PP_RELWFC."):
                    # Check orbital number:
                    i_psi = int(entry.tag[10:]) - 1
                    assert (i_psi >= 0) and (i_psi < n_psi)
                    # Get orbital's total angular momentum:
                    j_psi[i_psi] = entry.attrib["jchi"]
                else:
                    log.info(f"  NOTE: ignored section '{entry.tag}'")
            ps.j_beta = torch.tensor(j_beta, device=rc.device)
            ps.j_psi = torch.tensor(j_psi, device=rc.device)
        else:
            log.info(f"  NOTE: ignored section '{section.tag}'")

    # Make sure some common entries are set:
    assert hasattr(ps, "Vloc")
    if not hasattr(ps, "rho_atom"):
        ps.rho_atom = RadialFunction(ps.r, ps.dr)
    if not hasattr(ps, "n_core"):
        ps.n_core = RadialFunction(ps.r, ps.dr)
    if ps.is_ultrasoft:
        if augmentation is None:
            raise ValueError("Ultrasoft pseudopotential without PP_AUGMENTATION")
        ps.set_augmentation(*augmentation)


def _read_augmentation(
    section: ET.Element, n_beta: int, n_grid: int, l_max: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Parse PP_AUGMENTATION of an ultrasoft UPF v2 file.
    Returns the arguments of :meth:`Pseudopotential.set_augmentation`."""
    # Logicals appear as T/F, true/false or .true./.false. depending on QE version:
    q_with_l = section.attrib.get("q_with_l", "F").strip().lower() in (
        "t",
        "true",
        ".true.",
    )
    n_qf = int(section.attrib.get("nqf", 0))  # Taylor coefficients (if pseudized)
    n_L = int(section.attrib.get("nqlc", 0)) or (2 * l_max + 1)
    shape_Q = (n_beta, n_beta) + ((n_L,) if q_with_l else ()) + (n_grid,)
    r2_Q = np.zeros(shape_Q)  # r^2 Q_ij(r), per L only if q_with_l
    q_file = np.zeros((n_beta, n_beta))
    r_inner = Q_taylor = None
    for entry in section:
        data = np.fromstring(entry.text or "", sep=" ")
        if entry.tag == "PP_Q":
            q_file = data.reshape(n_beta, n_beta)
        elif entry.tag == "PP_QFCOEF":
            Q_taylor = data.reshape(n_beta, n_beta, n_L, n_qf)
        elif entry.tag == "PP_RINNER":
            r_inner = data
        elif entry.tag.startswith("PP_QIJ"):
            # PP_QIJ.i.j (q_with_l false) or PP_QIJL.i.j.L (true), 1-based i, j.
            # The tag and the attribute must agree: otherwise functions of
            # different L would be silently mixed up, so refuse instead.
            indices = [int(index) for index in entry.tag.split(".")[1:]]
            if (len(indices) == 3) != q_with_l:
                raise ValueError(
                    f"{entry.tag} inconsistent with q_with_l = {q_with_l} in"
                    " PP_AUGMENTATION"
                )
            i, j = indices[0] - 1, indices[1] - 1
            L_index = tuple(indices[2:])  # empty unless q_with_l
            r2_Q[(i, j) + L_index][: len(data)] = data
            r2_Q[(j, i) + L_index][: len(data)] = data
        else:
            log.info(f"  NOTE: ignored section '{entry.tag}'")
    if q_with_l or (not n_qf):
        r_inner = Q_taylor = None  # inner pseudization only for q_with_l false
    return r2_Q, q_file, r_inner, Q_taylor
