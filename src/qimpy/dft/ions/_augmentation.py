from __future__ import annotations

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
    species: list[Species]  #: Data of each ultrasoft species
    iG: torch.Tensor  #: Local reciprocal lattice vectors (integer coordinates)
    G: torch.Tensor  #: Local reciprocal lattice vectors (Cartesian)
    Ginterp: Interpolator  #: Interpolates radial functions at \|G\|
    in_sphere: torch.Tensor  #: Local G within the density cutoff sphere (augmented)
    L_max: int  #: Largest L of all augmentation functions
    Y: tuple[torch.Tensor, torch.Tensor]  #: Even and odd signed harmonics
    q_all: torch.Tensor  #: Overlap augmentation
    D_aug: torch.Tensor  #: Augmentation of `Ions.D_all`

    def __init__(self, ions: dft.ions.Ions, grid: Grid, q_all: torch.Tensor) -> None:
        """Prepare density augmentation on `grid` for atom positions in `ions`."""
        self.ions = ions
        self.grid = grid
        self.iG = grid.get_mesh("H").to(torch.double)
        self.G = self.iG @ grid.lattice.Gbasis.T
        Gmag = self.G.norm(dim=-1)
        self.Ginterp = Interpolator(Gmag, RadialFunction.DG, need_grad=True)

        # Augment only within the density cutoff sphere, |G|^2 <= 2 grid.ke_cutoff
        R_lengths = grid.lattice.Rbasis.norm(dim=0)
        G_inscribed = np.pi * min(n / float(a) for n, a in zip(grid.shape, R_lengths))
        self.in_sphere = Gmag < G_inscribed
        if grid.ke_cutoff:
            self.in_sphere &= Gmag <= np.sqrt(2.0 * grid.ke_cutoff)

        # Initialize harmonics for max product L needed:
        self.L_max = max(
            int(ps.Q.l.max()) for ps in ions.pseudopotentials if (ps.Q is not None)
        )
        self.Y = self._get_harmonics()
        self.q_all = q_all
        self.D_aug = torch.zeros_like(q_all)

        # Collect data of ultrasoft species, locating their atoms and projectors:
        self.species = []
        i_proj_start = 0
        for ps, atom_slice in zip(ions.pseudopotentials, ions.slices):
            n_atoms = atom_slice.stop - atom_slice.start
            proj_slice = slice(i_proj_start, i_proj_start + n_atoms * ps.n_projectors)
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
            Y_prime *= self.in_sphere  # restrict to ke-cutoff sphere
            Y_prime[:, (L // 2) % 2 == 1] *= -1.0  # i^L relative to L = 0, 1
            return Y_prime[:, is_even], Y_prime[:, ~is_even]
        else:
            Y = sh.get_harmonics(self.L_max, self.G)
            Y *= self.in_sphere  # restrict to ke-cutoff sphere
            Y[(L // 2) % 2 == 1] *= -1.0  # i^L relative to L = 0, 1 for even, odd L
            return Y[is_even], Y[~is_even]

    @stopwatch
    def density(self, C: dft.electrons.Wavefunction, f: torch.Tensor) -> FieldH:
        """Compute density augmentation from wavefunction projections `beta_C`
        and weighted occupations `wf`, with k-point reduction over `group_k`."""
        beta_C = C.proj
        basis = C.basis
        wf = f * basis.w_sk
        group_k = basis.kpoints.group
        n_spins = beta_C.shape[0]
        n_aug = FieldH(self.grid, shape_batch=(n_spins,))
        inv_volume = 1.0 / self.grid.lattice.volume
        for species in self.species:
            species.update_density_matrix(beta_C, wf, group_k)
            coeff = species.radial_coefficients()  # spline, density, atom, lm
            for atoms in species.atom_chunks(n_spins):
                radial = self.Ginterp(coeff[:, :, atoms])  # density, atom, LM, grid
                T = species.combine(radial, self.Y)  # density, atom, grid
                phase = species.phase(atoms)  # atom, grid
                n_aug.data += inv_volume * torch.einsum("da..., a... -> d...", T, phase)
        return n_aug

    @stopwatch
    def density_grad(self, V: FieldH) -> None:
        """Gradient propagation corresponding to `density`, leading to matrix `D_aug`
        on all projectors (n_spins x 1 x n_proj x n_proj) that adds to `Ions.D_all`."""
        n_spins = V.data.shape[0]
        n_proj_tot = self.ions.n_projectors
        D_aug = torch.zeros(
            (n_spins, 1, n_proj_tot, n_proj_tot), dtype=torch.cdouble, device=rc.device
        )
        V_weighted = V.data.conj() * self.grid.weight2H  # Hermitian-symmetry weights
        for species in self.species:
            coeff_grad = []
            for atoms in species.atom_chunks(n_spins):
                U = V_weighted[:, None] * species.phase(atoms)  # d, atom, grid
                radial_grad = species.combine_grad(U, self.Y)  # d, atom, LM, grid
                coeff_grad.append(self.Ginterp.grad(radial_grad))
            D = species.radial_coefficients_grad(torch.cat(coeff_grad, dim=2))
            if self.grid.is_split:
                D = D.contiguous()
                dist.all_reduce(D, group=self.grid.group)  # sum over local G
            # Write each atom's block on the diagonal (view the species block as
            # spin, atom, i, atom', j and set its atom = atom' diagonal):
            n_atoms, n_proj = species.n_atoms, species.n_proj
            block = D_aug[:, 0, species.proj_slice, species.proj_slice]
            block = block.unflatten(1, (n_atoms, n_proj)).unflatten(
                3, (n_atoms, n_proj)
            )
            block.diagonal(dim1=1, dim2=3).copy_(D.permute(0, 2, 3, 1))
        self.D_aug = D_aug

    @stopwatch(name="Augmentation.geometry_grad")
    def accumulate_geometry_grad(self, V: FieldH) -> None:
        """Accumulate force and stress contributions due to density augmentation."""
        ions = self.ions
        lattice = self.grid.lattice
        need_forces = ions.positions.requires_grad
        need_stress = lattice.requires_grad
        if not (need_forces or need_stress):
            return
        n_spins = V.data.shape[0]
        V_weighted = V.data.conj() * self.grid.weight2H
        pos_grad = torch.zeros_like(ions.positions)
        lattice_grad = torch.zeros((3, 3), device=rc.device)
        if need_stress:
            # Derivatives of radial parts and harmonics with respect to G:
            Gmag = self.G.norm(dim=-1)
            Ginterp_prime = Interpolator(Gmag, RadialFunction.DG, deriv=1)
            Ghat = self.G * torch.where(Gmag == 0.0, 0.0, 1.0 / Gmag)[..., None]
            Y_prime = self._get_harmonics(prime=True)

        for species in self.species:
            coeff = species.radial_coefficients()  # at current density matrix
            for atoms in species.atom_chunks(n_spins * (4 if need_stress else 1)):
                radial = self.Ginterp(coeff[:, :, atoms])  # d, atom, LM, grid
                U = V_weighted[:, None] * species.phase(atoms)  # d, atom, grid
                # E = sum Re(U T), with d(phase)/d(position) = -2 pi i iG phase:
                UT = U * species.combine(radial, self.Y)
                if need_forces:
                    atom_start = species.atom_slice.start
                    pos_grad[atom_start + atoms.start : atom_start + atoms.stop] += (
                        2 * np.pi
                    ) * torch.einsum("da..., ...k -> ak", UT.imag, self.iG)
                if need_stress:
                    # Volume factor 1/Omega of the density at fixed rho_beta
                    # (unlike the wavefunction density in QimPy's convention):
                    lattice_grad -= torch.eye(3, device=rc.device) * UT.real.sum()
                    # Strain changes G -> (1 - strain) G at fixed phases, giving
                    # d(f(|G|) Y(G))/d(strain_ij) = -G_j d(f Y)/dG_i:
                    radial_prime = Ginterp_prime(coeff[:, :, atoms])
                    T_prime_radial = species.combine(radial_prime, self.Y)
                    for i_dir in range(3):
                        dT_dG = T_prime_radial * Ghat[..., i_dir] + species.combine(
                            radial, (Y_prime[0][i_dir], Y_prime[1][i_dir])
                        )
                        lattice_grad[i_dir] -= torch.einsum(
                            "da..., ...j -> j", (U * dT_dG).real, self.G
                        )
        if self.grid.is_split:
            dist.all_reduce(pos_grad, group=self.grid.group)
            dist.all_reduce(lattice_grad, group=self.grid.group)
        if need_forces:
            assert ions.positions.grad is not None
            ions.positions.grad += pos_grad
        if need_stress:
            lattice.grad += lattice_grad


class Species:
    """Augmentation data of one ultrasoft species."""

    atom_slice: slice  #: Atoms of this species within `Ions`
    proj_slice: slice  #: Projectors of these atoms within `Ions.beta` and `D_all`
    ps: Pseudopotential  #: Corresponding pseudopotential
    augmentation: Augmentation  #: Parent augemntation object for shared quantities
    n_proj: int  #: Number of projectors per atom (including m)
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
        self.ps = ps
        self.augmentation = augmentation
        self.n_proj = ps.n_projectors

        # Split gaunt coefficients into even and odd pieces:
        gaunt = ps.gaunt
        self.n_even_LM = gaunt[0].shape[1]
        self.n_odd_LM = gaunt[1].shape[1]
        self.gaunt = torch.cat(gaunt, dim=1)

    @property
    def n_atoms(self) -> int:
        return self.atom_slice.stop - self.atom_slice.start

    def update_density_matrix(
        self, beta_C: torch.Tensor, wf: torch.Tensor, group_k: dist.ProcessGroup
    ) -> None:
        """Set atomic density matrix from wavefunction projections `beta_C`
        and weighted occupations `wf`."""
        proj = beta_C[:, :, self.proj_slice].unflatten(2, (self.n_atoms, self.n_proj))
        self.rho = torch.einsum("skain, skn, skajn -> saij", proj.conj(), wf, proj)
        dist.all_reduce(self.rho, group=group_k)

    def radial_coefficients(self) -> torch.Tensor:
        """Compute radial spline coefficients (density component, atom, lm)
        for density augmentation from atomic density matrix `rho` (spin, atom, i, j)."""
        # Symmetry of Gaunt coefficients + hermiticity of rho => only rho.real matters:
        spin_map = _spin_map(self.rho.shape[0])
        Q_coeff = self.ps.Q.f_tilde_coeff
        return torch.einsum(
            "cq, qLij, ds, saij -> cdaL", Q_coeff, self.gaunt, spin_map, self.rho.real
        )

    def radial_coefficients_grad(self, coeff_grad: torch.Tensor) -> torch.Tensor:
        """Gradient propagation from spline coefficients to atomic density matrix,
        corresponding to `radial_coefficients`."""
        Q_coeff = self.ps.Q.f_tilde_coeff[: coeff_grad.shape[0]]  # rows used by Ginterp
        spin_map = _spin_map(coeff_grad.shape[1])
        return torch.einsum(
            "ds, qLij, cq, cdaL -> saij", spin_map, self.gaunt, Q_coeff, coeff_grad
        )

    def combine(
        self, radial: torch.Tensor, Y: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Combine radial and angular parts, leveraging purely real and imaginary
        contributions from even and odd L respectively in the angular part."""
        n_even = self.n_even_LM
        n_odd = self.n_odd_LM
        return torch.complex(
            torch.einsum("daL..., L... -> da...", radial[:, :, :n_even], Y[0][:n_even]),
            torch.einsum("daL..., L... -> da...", radial[:, :, n_even:], Y[1][:n_odd]),
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
        return ions.translation_phase(iG, atom_slice_sub).permute(3, 0, 1, 2)

    def atom_chunks(self, n_batch: int) -> list[slice]:
        """Split the atoms of `species` so that per-chunk grid intermediates of
        n_batch x LM real values per atom and grid point fit `ATOM_CHUNK_BYTES`."""
        nG = np.prod(self.augmentation.grid.shapeH_mine)
        bytes_per_atom = 8 * n_batch * max(1, nG * self.gaunt.shape[1])
        n_chunk = max(1, ATOM_CHUNK_BYTES // bytes_per_atom)
        n_atoms = self.n_atoms
        return [slice(i, min(i + n_chunk, n_atoms)) for i in range(0, n_atoms, n_chunk)]


def _spin_map(n_spins: int) -> torch.Tensor:
    """Map from spin channels (up, down) to the density components (n, M_z).
    (Currently, only supports non-spinorial modes.)"""
    if n_spins == 1:
        return torch.ones((1, 1), device=rc.device)
    return torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=rc.device)
