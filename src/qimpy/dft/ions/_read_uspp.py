from __future__ import annotations

import numpy as np
import torch

from qimpy import log, rc
from qimpy.math import RadialFunction
from qimpy.dft import ions
from . import symbols


def read_uspp(ps: ions.Pseudopotential, filename: str) -> None:
    """Read an ultrasoft pseudopotential in Vanderbilt's '.uspp' format.
    This is the Fortran sequential-binary output of the USPP generator (version
    7.3 or newer), as used by the GBRV library. The record layout follows the
    USPP writer (and the JDFTx reader, SpeciesInfo_readUspp.cpp).

    Parameters
    ----------
    ps
        Pseudopotential object to read into.
    filename
        Full path to the '.uspp' file to read.
    """
    log.info(f"\nReading '{filename}':")
    records = _FortranRecords(filename)

    # Header (unused entries: XC code, reference energy):
    version, date = records.read(("<i4", 3), ("<i4", 3))
    title, Z_all_electron, ps.Z, _, n_valence, n_grid, _ = records.read(
        "S20", "<f8", "<f8", "<f8", "<i4", "<i4", "<f8"
    )
    if tuple(version[:2]) < (7, 3):
        raise ValueError(f"USPP format {version} older than 7.3 in {filename}")
    ps.atomic_number = int(round(Z_all_electron))
    ps.element = _ELEMENTS[ps.atomic_number]
    log.info(
        f"  '{ps.element}' pseudopotential ({title.decode().strip()}),"
        f" USPP {'.'.join(map(str, version))} from {'-'.join(map(str, date))}"
    )

    # Valence orbitals (code nlm, i.e. l is the second digit), core and l_max:
    orbital_dtype = np.dtype([("code", "<i4"), ("occupation", "<f8"), ("eig", "<f8")])
    (orbitals,) = records.read((orbital_dtype, n_valence))
    _, has_core, _ = records.read("<i4", "<i4", "<f8")
    l_max_plus_1, _, _, _, n_qf, _ = records.read(
        "<i4", "<i4", "<f8", "<i4", "<i4", "<f8"
    )
    n_L = 2 * l_max_plus_1 - 1  # L = 0 ... 2 l_max in products of projector Ylm's
    (r_inner,) = records.read(("<f8", n_L))  # radius of inner pseudization of Q^L
    records.read("<i4")  # relativity of generation (at most scalar-relativistic)
    records.read(("<f8", l_max_plus_1))  # projector cutoff radii (unused)

    # Projectors and their pair data (upper triangle i <= j, one record per pair).
    # (Unused: projector reference energies, screened D of the reference atom.)
    n_beta, n_grid_beta = records.read("<i4", "<i4")
    l_beta = np.zeros(n_beta, dtype=int)
    r_beta = np.zeros((n_beta, n_grid))  # r beta(r), zero beyond n_grid_beta
    D0 = np.zeros((n_beta, n_beta))  # descreened D matrix in Ry
    q_file = np.zeros((n_beta, n_beta))  # overlap coefficients q_ij
    r2_Q = np.zeros((n_beta, n_beta, n_grid))  # r^2 Q_ij(r), same for all L
    Q_taylor = np.zeros((n_beta, n_beta, n_L, n_qf))  # inner pseudization per L
    for i in range(n_beta):
        l_beta[i], _, r_beta[i, :n_grid_beta] = records.read(
            "<i4", "<f8", ("<f8", n_grid_beta)
        )
        for j in range(i, n_beta):
            D0_ij, _, q_ij, r2_Q_ij, taylor_ij = records.read(
                "<f8", "<f8", "<f8", ("<f8", n_grid_beta), ("<f8", n_L * n_qf)
            )
            D0[i, j] = D0[j, i] = D0_ij
            q_file[i, j] = q_file[j, i] = q_ij
            r2_Q[i, j, :n_grid_beta] = r2_Q[j, i, :n_grid_beta] = r2_Q_ij
            Q_taylor[i, j] = Q_taylor[j, i] = taylor_ij.reshape(n_L, n_qf)
    records.read(("<i4", n_beta), "<i4", "<f8")  # generation details (unused)

    # Radial functions on the full log grid:
    _, r_Vloc = records.read("<f8", ("<f8", n_grid))  # descreened r V_loc in Ry
    r2_n_core = np.zeros(n_grid)  # 4 pi r^2 n_core(r)
    if has_core:
        records.read("<f8")  # core pseudization radius
        (r2_n_core,) = records.read(("<f8", n_grid))
    records.read(("<f8", n_grid))  # screened r V_loc of reference atom (unused)
    (r2_rho_atom,) = records.read(("<f8", n_grid))  # 4 pi r^2 rho_atom(r)
    (r,) = records.read(("<f8", n_grid))
    (dr,) = records.read(("<f8", n_grid))
    (n_psi,) = records.read("<i4")
    (r_psi,) = records.read(("<f8", n_psi * n_grid))  # r psi(r) of each orbital
    records.check_end()

    # Store in the conventions of read_upf (Hartree units, r^l removed):
    l_psi = (orbitals["code"] // 10) % 10
    ps.l_max = int(max(l_beta.max(initial=0), l_psi.max(initial=0)))
    log.info(
        f"  {ps.Z:g} valence electrons, {n_psi} orbitals, {n_beta} projectors,"
        f" {n_grid} radial grid points, with l_max = {ps.l_max}"
    )
    ps.is_paw = False
    ps.is_relativistic = False  # generated at most scalar-relativistically
    ps.j_beta = ps.j_psi = None
    if not r[0]:
        r[0] = 1e-3 * r[1]  # avoid divide by 0 below (as in read_upf)
    ps.r = torch.tensor(r, device=rc.device)
    ps.dr = torch.tensor(dr, device=rc.device)
    ps.ion_width = np.inf  # No range separation yet
    ps.Vloc = RadialFunction(ps.r, ps.dr, 0.5 * r_Vloc / r)  # 0.5 converts Ry to Eh
    ps.n_core = RadialFunction(ps.r, ps.dr, r2_n_core / (4 * np.pi * r**2))
    ps.rho_atom = RadialFunction(ps.r, ps.dr, r2_rho_atom / (4 * np.pi * r**2))
    ps.beta = RadialFunction(
        ps.r, ps.dr, r_beta * (r[None, :] ** -(l_beta + 1)[:, None]), l_beta
    )
    ps.D = torch.tensor(0.5 * D0, device=rc.device)  # 0.5 converts Ry to Eh
    ps.eig_psi = 0.5 * orbitals["eig"]  # 0.5 converts Ry to Eh
    for code, occupation, eig in zip(
        orbitals["code"], orbitals["occupation"], ps.eig_psi
    ):
        log.info(
            f"    |{code}>  l: {(code // 10) % 10}  occupation: {occupation:4.1f}"
            f"  eigenvalue: {eig}"
        )
    ps.psi = RadialFunction(
        ps.r,
        ps.dr,
        r_psi.reshape(n_psi, n_grid) * (r[None, :] ** -(l_psi + 1)[:, None]),
        l_psi,
    )
    ps.set_augmentation(r2_Q, q_file, r_inner=r_inner, Q_taylor=Q_taylor)


class _FortranRecords:
    """Consecutive records of a Fortran sequential binary file. Each record is
    stored as [int32 n_bytes][payload][int32 n_bytes] (4-byte markers, as written
    by both gfortran and ifort, which covers the USPP generator builds in use)."""

    def __init__(self, filename: str) -> None:
        with open(filename, "rb") as file:
            self.buffer = file.read()
        self.offset = 0

    def read(self, *fields: str | tuple[str | np.dtype, int]) -> list:
        """Read the next record as consecutive `fields`. Each field is either a
        dtype string, for a scalar returned as a python number (or bytes), or a
        (dtype, count) pair for an array. Raise an error unless the fields fill
        the record exactly, so that a layout mismatch is caught where it occurs."""
        n_bytes = self._marker(self.offset)
        start = self.offset + 4
        if self._marker(start + n_bytes) != n_bytes:
            raise ValueError("Corrupt Fortran record markers in '.uspp' file")
        results = []
        offset = start
        for field in fields:
            if isinstance(field, str):
                dtype, count = np.dtype(field), 1
            else:
                dtype, count = np.dtype(field[0]), field[1]
            values = np.frombuffer(self.buffer, dtype, count, offset)
            offset += dtype.itemsize * count
            results.append(
                values[0].item() if isinstance(field, str) else values.copy()
            )
        if offset != start + n_bytes:
            raise ValueError(
                f"Unexpected '.uspp' record size {n_bytes} (expected {offset - start})"
            )
        self.offset = offset + 4
        return results

    def check_end(self) -> None:
        """Ensure that the entire file has been read."""
        if self.offset != len(self.buffer):
            raise ValueError(f"{len(self.buffer) - self.offset} unread bytes in file")

    def _marker(self, offset: int) -> int:
        return int(np.frombuffer(self.buffer, "<i4", 1, offset)[0])


#: Chemical symbol of each atomic number (inverse of `symbols.ATOMIC_NUMBERS`)
_ELEMENTS: dict[int, str] = {Z: symbol for symbol, Z in symbols.ATOMIC_NUMBERS.items()}
