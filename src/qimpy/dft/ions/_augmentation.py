from __future__ import annotations
from functools import cache

import numpy as np
import torch
import torch.distributed as dist

from qimpy import rc, dft
from qimpy.profiler import stopwatch
from qimpy.grid import Grid, FieldH
from qimpy.math import RadialFunction, spherical_harmonics as sh
from qimpy.math.quintic_spline import Interpolator
from . import Pseudopotential

ATOM_CHUNK_BYTES: int = 300_000_000  #: Memory limit on per-atom-lm intermediates


class Augmentation:
    """Density augmentation from ultrasoft pseudopotentials on charge density grid."""

    ions: dft.ions.Ions  #: Ions whose augmentation charges these are
    grid: Grid  #: Density grid (local half-space portion used)
    q_all: torch.Tensor  #: Overlap augmentation
    n_spins: int  #: Number of spin channels
    n_spinor: int  #: Number of spinor components
    D_aug: torch.Tensor  #: Augmentation of `Ions.D_all`
    indexH: torch.Tensor  #: Non-zero indices into FieldH data for ke-cutoff sphere
    iG: torch.Tensor  #: Local reciprocal lattice vectors (integer coordinates)
    G: torch.Tensor  #: Local reciprocal lattice vectors (Cartesian)
    Gmag: torch.Tensor  #: Local magnitudes of `G`.
    Ginterp: Interpolator  #: Interpolates radial functions at \|G\|
    in_sphere: torch.Tensor  #: Local G within the density cutoff sphere (augmented)
    L_max: int  #: Largest L of all augmentation functions
    Y: tuple[torch.Tensor, torch.Tensor]  #: Even and odd signed harmonics
    species: list[Species]  #: Data of each ultrasoft species

    def __init__(self, system: dft.System, q_all: torch.Tensor) -> None:
        """Prepare density augmentation for specified dft `system`."""
        self.ions = system.ions
        self.grid = system.grid
        self.q_all = q_all
        self.n_spins = system.electrons.n_spins
        self.n_spinor = system.electrons.n_spinor
        D_shape = (self.n_spins, 1) + q_all.shape
        self.D_aug = torch.zeros(D_shape, dtype=q_all.dtype, device=rc.device)

        # Augment only within the density cutoff sphere, |G|^2 <= 2 grid.ke_cutoff
        R_lengths = system.lattice.Rbasis.norm(dim=0)
        G_cut = np.pi * min(n / float(a) for n, a in zip(self.grid.shape, R_lengths))
        if ke_cutoff := self.grid.ke_cutoff:
            G_cut = min(G_cut, np.sqrt(2.0 * ke_cutoff))

        # Construct grid index and interpolation:
        iG = self.grid.get_mesh("H").to(torch.double).flatten(0, 2)
        G = iG @ self.grid.lattice.Gbasis.T
        Gmag = G.norm(dim=-1)
        self.indexH = torch.where(Gmag <= G_cut)[0]  # index into flattened FieldH
        self.iG = iG[self.indexH]
        self.G = G[self.indexH]
        self.Gmag = Gmag[self.indexH]
        self.Ginterp = Interpolator(self.Gmag, RadialFunction.DG, need_grad=True)

        # Initialize harmonics for max product L needed:
        self.L_max = max(
            int(ps.Q.l.max()) for ps in self.ions.pseudopotentials if (ps.Q is not None)
        )
        self.Y = self._get_harmonics()

        # Collect data of ultrasoft species, locating their atoms and projectors:
        self.species = []
        i_proj_start = 0
        for ps, atom_slice in zip(self.ions.pseudopotentials, self.ions.slices):
            n_atoms = atom_slice.stop - atom_slice.start
            n_proj_s = ps.n_projectors * self.n_spinor
            proj_slice = slice(i_proj_start, i_proj_start + n_atoms * n_proj_s)
            i_proj_start = proj_slice.stop
            if ps.Q is not None:
                self.species.append(Species(atom_slice, proj_slice, ps, self))

    def _get_harmonics(self, prime: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Get harmonics sorted by even and odd, and with i^(L - L%2) sign."""
        L = torch.arange(self.L_max + 1, device=rc.device)
        L = L.repeat_interleave(2 * L + 1)  # repeat for M
        is_even = L % 2 == 0
        if prime:
            Y_prime = sh.get_harmonics_and_prime(self.L_max, self.G)[1]
            Y_prime[:, (L // 2) % 2 == 1] *= -1.0  # i^L relative to L = 0, 1
            return Y_prime[:, is_even], Y_prime[:, ~is_even]
        else:
            Y = sh.get_harmonics(self.L_max, self.G)
            Y[(L // 2) % 2 == 1] *= -1.0  # i^L relative to L = 0, 1 for even, odd L
            return Y[is_even], Y[~is_even]

    @stopwatch
    def density(
        self, C: dft.electrons.Wavefunction, f: torch.Tensor, need_Mvec: bool
    ) -> FieldH:
        """Compute density augmentation from wavefunction projections `beta_C`
        and weighted occupations `wf`, with k-point reduction over `group_k`.
        Return 4 components including magnetization if `need_Mvec` (spinor case)."""
        beta_C = C.proj
        basis = C.basis
        wf = f * basis.w_sk
        group_k = basis.kpoints.group
        nG = len(self.G)
        n_densities = 4 if need_Mvec else self.n_spins
        n_aug = torch.zeros((n_densities, nG), device=rc.device, dtype=torch.cdouble)
        for species in self.species:
            species.update_density_matrix(beta_C, wf, group_k)
            coeff = species.radial_coefficients(n_densities)  # coeff, density, atom, lm
            for atoms in species.atom_chunks(self.n_spins):
                radial = self.Ginterp(coeff[:, :, atoms])  # density, atom, LM, grid
                n_aug_atoms = species.combine(radial, self.Y)  # density, atom, grid
                phase = species.phase(atoms)  # atom, grid
                n_aug += torch.einsum("daG, aG -> dG", n_aug_atoms, phase)
        inv_volume = 1.0 / self.grid.lattice.volume
        n_aug_tilde = FieldH(self.grid, shape_batch=(n_densities,))
        n_aug_tilde.data.flatten(-3, -1)[:, self.indexH] = n_aug * inv_volume
        return n_aug_tilde

    @stopwatch
    def density_grad(self, V: FieldH) -> None:
        """Gradient propagation corresponding to `density`, leading to matrix `D_aug`
        on all projectors (n_spins x 1 x n_proj x n_proj) that adds to `Ions.D_all`."""
        V_w = (V.data.conj() * self.grid.weight2H).flatten(-3, -1)[:, None, self.indexH]
        for species in self.species:
            # Propagate V from grid to atomic density matrix:
            coeff_grad = []
            for atoms in species.atom_chunks(self.n_spins):
                V_atoms = V_w * species.phase(atoms)  # d, atom, grid
                radial_grad = species.combine_grad(V_atoms, self.Y)  # d, atom, LM, grid
                coeff_grad.append(self.Ginterp.grad(radial_grad))
            D = species.radial_coefficients_grad(torch.cat(coeff_grad, dim=2))
            if self.grid.is_split:
                D = D.contiguous()
                dist.all_reduce(D, group=self.grid.group)  # sum over local G
            # Set to diagonal blocks in D_aug:
            n_atoms, n_proj = species.n_atoms, species.n_proj_s
            shape = (self.n_spins, n_atoms, n_proj, n_atoms, n_proj)
            block = self.D_aug[:, 0, species.proj_slice, species.proj_slice].view(shape)
            block.diagonal(dim1=1, dim2=3).copy_(D.permute(0, 2, 3, 1))

    @stopwatch
    def geometry_grad(self, V: FieldH) -> None:
        """Accumulate force and stress contributions due to density augmentation."""
        ions = self.ions
        lattice = self.grid.lattice
        need_forces = ions.positions.requires_grad
        need_stress = lattice.requires_grad
        if not (need_forces or need_stress):
            return
        V_w = (V.data.conj() * self.grid.weight2H).flatten(-3, -1)[:, None, self.indexH]
        pos_grad = torch.zeros_like(ions.positions)
        lattice_grad = torch.zeros((3, 3), device=rc.device)
        volume_term = torch.zeros((), device=rc.device)  # volume term in stress
        if need_stress:
            # Derivatives of radial parts and harmonics with respect to G:
            Gmag = self.Gmag
            Ginterp_prime = Interpolator(Gmag, RadialFunction.DG, deriv=1)
            Ghat = self.G * torch.where(Gmag == 0.0, 0.0, 1.0 / Gmag)[..., None]
            Y_prime = self._get_harmonics(prime=True)
            Y_prime = [(Y_prime[0][i_dir], Y_prime[1][i_dir]) for i_dir in range(3)]

        for species in self.species:
            coeff = species.radial_coefficients(len(V_w))  # at current density matrix
            for atoms in species.atom_chunks(self.n_spins * (4 if need_stress else 1)):
                radial = self.Ginterp(coeff[:, :, atoms])  # d, atom, LM, grid
                V_atoms = V_w * species.phase(atoms)  # d, atom, grid
                V_n_atoms = V_atoms * species.combine(radial, self.Y)
                if need_forces:
                    start = species.atom_slice.start
                    pos_grad[start + atoms.start : start + atoms.stop] += torch.einsum(
                        "daG, Gi -> ai", V_n_atoms.imag, self.iG
                    )
                if need_stress:
                    volume_term += V_n_atoms.real.sum()
                    radial_prime = Ginterp_prime(coeff[:, :, atoms])
                    n_aug_atoms_prime = species.combine(radial_prime, self.Y)
                    for i_dir in range(3):  # one row at a time to reduce memory
                        dV_n_atoms_dG = n_aug_atoms_prime * Ghat[..., i_dir]
                        dV_n_atoms_dG += species.combine(radial, Y_prime[i_dir])
                        lattice_grad[i_dir] -= torch.einsum(
                            "daG, Gj -> j", (V_atoms * dV_n_atoms_dG).real, self.G
                        )
        lattice_grad -= volume_term * torch.eye(3, device=rc.device)
        if self.grid.is_split:
            dist.all_reduce(pos_grad, group=self.grid.group)
            dist.all_reduce(lattice_grad, group=self.grid.group)
        if need_forces:
            assert ions.positions.grad is not None
            ions.positions.grad += pos_grad * (2 * np.pi)  # 2 pi from dphase/dpositions
        if need_stress:
            lattice.grad += lattice_grad


class Species:
    """Augmentation data of one ultrasoft species."""

    atom_slice: slice  #: Atoms of this species within `Ions`
    proj_slice: slice  #: Projectors of these atoms within `Ions.beta` and `D_all`
    ps: Pseudopotential  #: Corresponding pseudopotential
    augmentation: Augmentation  #: Parent augemntation object for shared quantities
    n_proj: int  #: Number of projectors per atom (including m, but not spinors)
    n_proj_s: int  #: Number of projectors per atom (including m and spinors)
    n_even_LM: int  #: Number of (L, M) at even L
    n_odd_LM: int  #: Number of (L, M) at odd L
    gaunt: torch.Tensor  #: Gaunt map to product functions: (n_Q, n_LM, n_proj, n_proj)
    rho: torch.Tensor  #: Atomic density matrix in projector basis

    def __init__(
        self,
        atom_slice: slice,
        proj_slice: slice,
        ps: Pseudopotential,
        augmentation: Augmentation,
    ) -> None:
        self.atom_slice = atom_slice
        self.proj_slice = proj_slice
        self.n_proj = ps.n_projectors
        self.n_proj_s = self.n_proj * augmentation.n_spinor
        self.ps = ps
        self.augmentation = augmentation

        # Split gaunt coefficients into even and odd pieces:
        gaunt = ps.gaunt
        self.n_even_LM = gaunt[0].shape[1]
        self.n_odd_LM = gaunt[1].shape[1]
        self.gaunt = torch.cat(gaunt, dim=1)

        # Extra transformations for relativistic case:
        if augmentation.n_spinor == 2:
            self.PAULI = ps.pqn_beta.PAULI()  # 4 x 2 x 2
            self.overlaps_lj = ps.pqn_beta.overlaps_lj(diagonal_only=True)

    @property
    def n_atoms(self) -> int:
        return self.atom_slice.stop - self.atom_slice.start

    def update_density_matrix(
        self, beta_C: torch.Tensor, wf: torch.Tensor, group_k: dist.ProcessGroup
    ) -> None:
        """Set atomic density matrix from wavefunction projections `beta_C`
        and weighted occupations `wf`."""
        proj = beta_C[:, :, self.proj_slice].unflatten(2, (self.n_atoms, self.n_proj_s))
        self.rho = torch.einsum("skain, skn, skajn -> saij", proj, wf, proj.conj())
        dist.all_reduce(self.rho, group=group_k)

    def radial_coefficients(self, n_densities: int) -> torch.Tensor:
        """Compute radial spline coefficients (density component, atom, lm)
        for density augmentation from atomic density matrix `rho` (spin, atom, i, j)."""
        rho = self.rho
        if self.augmentation.n_spinor == 2:
            if self.ps.is_relativistic:
                rho = self.overlaps_lj @ rho @ self.overlaps_lj
            rho = rho.reshape(self.n_atoms, self.n_proj, 2, self.n_proj, 2)
            rho = torch.einsum("dsS, aisjS -> daij", self.PAULI[:n_densities], rho)
        elif self.augmentation.n_spins == 2:
            rho = torch.einsum("ds, saij -> daij", spin_map(), rho)
        Q_coeff = self.ps.Q.f_tilde_coeff
        return torch.einsum("cq, qLij, daij -> cdaL", Q_coeff, self.gaunt, rho.real)

    def radial_coefficients_grad(self, coeff_grad: torch.Tensor) -> torch.Tensor:
        """Gradient propagation from spline coefficients to atomic density matrix,
        corresponding to `radial_coefficients`."""
        Q_coeff = self.ps.Q.f_tilde_coeff[: coeff_grad.shape[0]]  # rows used by Ginterp
        V_rho = torch.einsum("qLij, cq, cdaL -> daij", self.gaunt, Q_coeff, coeff_grad)
        if self.augmentation.n_spinor == 2:
            V_rho = torch.einsum("dsS, daij -> aisjS", self.PAULI[: len(V_rho)], V_rho)
            V_rho = V_rho.reshape(1, self.n_atoms, self.n_proj_s, self.n_proj_s)
            if self.ps.is_relativistic:
                return self.overlaps_lj @ V_rho @ self.overlaps_lj
        elif self.augmentation.n_spins == 2:
            V_rho = torch.einsum("ds, daij -> saij", spin_map(), V_rho)
        return V_rho

    def combine(
        self, radial: torch.Tensor, Y: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Combine radial and angular parts, leveraging purely real and imaginary
        contributions from even and odd L respectively in the angular part."""
        n_even = self.n_even_LM
        n_odd = self.n_odd_LM
        return torch.complex(
            torch.einsum("daLG, LG -> daG", radial[:, :, :n_even], Y[0][:n_even]),
            torch.einsum("daLG, LG -> daG", radial[:, :, n_even:], Y[1][:n_odd]),
        )

    def combine_grad(
        self, U: torch.Tensor, Y: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Gradient propagation corresponding to `combine`."""
        n_even = self.n_even_LM
        n_odd = self.n_odd_LM
        U = U[:, :, None]  # add broadcasting dimensions for below
        return torch.cat((U.real * Y[0][:n_even], -U.imag * Y[1][:n_odd]), dim=2)

    def phase(self, atoms: slice) -> torch.Tensor:
        """Translation phases (atom, grid) of `atoms` (slice within `atom_slice`)."""
        iG = self.augmentation.iG
        ions = self.augmentation.ions
        start = self.atom_slice.start
        atom_slice_sub = slice(start + atoms.start, start + atoms.stop)
        return ions.translation_phase(iG, atom_slice_sub).T

    def atom_chunks(self, n_batch: int) -> list[slice]:
        """Split the atoms of `species` so that per-chunk grid intermediates of
        n_batch x LM real values per atom and grid point fit `ATOM_CHUNK_BYTES`."""
        nG = np.prod(self.augmentation.grid.shapeH_mine)
        bytes_per_atom = 8 * n_batch * max(1, nG * self.gaunt.shape[1])
        n_chunk = max(1, ATOM_CHUNK_BYTES // bytes_per_atom)
        n_atoms = self.n_atoms
        return [slice(i, min(i + n_chunk, n_atoms)) for i in range(0, n_atoms, n_chunk)]


@cache
def spin_map() -> torch.Tensor:
    """Map from spin channels (up, down) to the density components (n, M_z)."""
    return torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=rc.device)
