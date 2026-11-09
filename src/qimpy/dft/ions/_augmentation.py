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

ATOM_CHUNK_BYTES: int = 1 << 28  #: Memory target of per-atom grid intermediates


class Augmentation:
    r"""Augmentation charges of ultrasoft pseudopotentials on the density grid.

    Atom a at :math:`R_a` carries the augmentation density
    :math:`\sum_{ij} \rho^\beta_{a,ij} Q_{ij}(r - R_a)`, where
    :math:`Q_{ij}(r) = \sum_{LM} c^{LM}_{ij} Q^L_{ij}(r) Y_{LM}(\hat{r})` (real Gaunt
    coefficients c) and :math:`\rho^\beta_{a,ij} = \sum_{kn} w_k f_{kn}
    \langle\psi_{kn}|\beta_{ai}\rangle \langle\beta_{aj}|\psi_{kn}\rangle` is the
    atomic density matrix in the projector basis (from the cached projections).

    In reciprocal space, `density` is a chain of linear maps:
    rho_beta -> multipoles (Gaunt) -> spline coefficients (sum over Q functions)
    -> radial parts at \|G\| (interpolate) -> n_aug(G) (harmonics and phases).
    Radial functions are combined before interpolating, so that each G evaluates
    one spline per (atom, LM) instead of one per (i, j, L). `D_aug` applies the
    transposed chain to a potential, and `accumulate_geometry_grad` differentiates
    the last two maps with respect to atom positions and strain.

    Projections use the phase convention of `Ions.get_projectors`, i^l instead of
    the (-i)^l of the Fourier transform, i.e. an extra (-1)^l per projector.
    The harmonics here carry i^L likewise, and since L = l_i + l_j (mod 2) in the
    Gaunt coefficients, the extra (-1)^(l_i + l_j + L) = 1 cancels throughout.
    """

    ions: dft.ions.Ions  #: Ions whose augmentation charges these are
    grid: Grid  #: Density grid (local half-space portion used)
    species: list[Species]  #: Data of each ultrasoft species
    iG: torch.Tensor  #: Local reciprocal lattice vectors (integer coordinates)
    G: torch.Tensor  #: Local reciprocal lattice vectors (Cartesian)
    Ginterp: Interpolator  #: Interpolates radial functions at \|G\|
    in_sphere: torch.Tensor  #: Local G within the density cutoff sphere (augmented)
    L_max: int  #: Largest L of all augmentation functions
    LM_order: torch.Tensor  #: Order of (L, M) used here: even L first, then odd L
    n_even: int  #: Number of (L, M) with even L (leading block in `LM_order`)
    Y: tuple[torch.Tensor, torch.Tensor]  #: Even and odd signed harmonics
    Y_prime: tuple[torch.Tensor, torch.Tensor]  #: Spatial derivatives of Y
    q_all: torch.Tensor  #: Overlap augmentation
    D_aug: torch.Tensor  #: Augmentation of `Ions.D_all`

    def __init__(self, ions: dft.ions.Ions, grid: Grid, q_all: torch.Tensor) -> None:
        """Prepare augmentation for the current `ions` positions on `grid`.
        Requires the radial transforms of the augmentation functions,
        which are updated with the local pseudopotential terms."""
        self.ions = ions
        self.grid = grid
        self.iG = grid.get_mesh("H").to(torch.double)
        self.G = self.iG @ grid.lattice.Gbasis.T
        Gmag = self.G.norm(dim=-1)
        self.Ginterp = Interpolator(Gmag, RadialFunction.DG, need_grad=True)

        # Augment only within the density cutoff sphere, |G|^2 <= 2 grid.ke_cutoff
        # (or the sphere inscribed in the FFT box, without its Nyquist planes, if
        # only the grid shape is specified). Symmetry images of these G stay in the
        # sphere, so the augmentation density is exactly symmetric; in the corners
        # of the box, field symmetrization would alias it instead (it wraps images
        # that leave the box back into it). Augmenting the full box, as JDFTx does,
        # makes symmetry-reduced and full k-mesh runs differ (3e-6 Eh for Si at
        # 100 Eh, in both codes), versus 2e-12 Eh within the sphere. Results then
        # depend on the density cutoff alone, not on the FFT box, as in QE.
        R_lengths = grid.lattice.Rbasis.norm(dim=0)
        G_inscribed = np.pi * min(n / float(a) for n, a in zip(grid.shape, R_lengths))
        self.in_sphere = Gmag < G_inscribed
        if grid.ke_cutoff:
            self.in_sphere &= Gmag <= np.sqrt(2.0 * grid.ke_cutoff)

        # Harmonics up to the largest L of any augmentation function. The i^L in
        # the reciprocal-space harmonics (iG)^L Y_LM is real for even L and
        # imaginary for odd L. Ordering (L, M) with even L first makes these two
        # contiguous blocks of real harmonics, so the large contractions on the
        # grid below stay real, and the complex result is assembled only once:
        self.L_max = max(
            int(ps.Q.l.max()) for ps in ions.pseudopotentials if (ps.Q is not None)
        )
        L = torch.arange(self.L_max + 1, device=rc.device)
        self.L_of_LM = L.repeat_interleave(2 * L + 1)
        self.LM_order = torch.argsort(self.L_of_LM % 2, stable=True)
        self.n_even = int((self.L_of_LM % 2 == 0).sum().item())
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
    def density_matrix(
        self, beta_C: torch.Tensor, wf: torch.Tensor, group: dist.ProcessGroup
    ) -> list[torch.Tensor]:
        """Atomic density matrix in the projector basis for each ultrasoft species,
        rho[s, a, i, j] = sum_{k, n} wf[s, k, n] conj(beta_C[s, k, (a, i), n])
        beta_C[s, k, (a, j), n] (n_spins x n_atoms x n_proj x n_proj), from the
        projections `beta_C` = C.proj and weights `wf` = fillings times k-point
        weights (n_spins x nk x n_bands). The k sum is reduced over `group`.
        (Projections are complete in bands and basis on each process already.)"""
        rho_beta = []
        for species in self.species:
            beta_C_cur = beta_C[:, :, species.proj_slice].unflatten(
                2, (species.n_atoms, species.n_proj)
            )  # spin, k, atom, projector, band
            rho = torch.einsum(
                "skain, skajn -> saij",
                beta_C_cur.conj(),
                beta_C_cur * wf[:, :, None, None, :],
            )
            dist.all_reduce(rho, group=group)
            rho_beta.append(rho)
        return rho_beta

    @stopwatch
    def density(self, rho_beta: list[torch.Tensor]) -> FieldH:
        """Augmentation density from the atomic density matrices `rho_beta` (per
        species, from `density_matrix`). The result has the density components of
        `Electrons.n_tilde`: n, and M_z if spin-polarized (collinear)."""
        n_spins = rho_beta[0].shape[0]
        n_aug = FieldH(self.grid, shape_batch=(n_spins,))
        inv_volume = 1.0 / self.grid.lattice.volume
        for species, rho in zip(self.species, rho_beta):
            coeff = species.radial_coefficients(rho)  # spline, density, atom, lm
            for atoms in self._atom_chunks(species, n_spins):
                radial = self.Ginterp(coeff[:, :, atoms])  # density, atom, LM, grid
                T = self._combine(radial)  # density, atom, grid
                phase = self._phase(species, atoms)  # atom, grid
                n_aug.data += inv_volume * torch.einsum("da..., a... -> d...", T, phase)
        return n_aug

    @stopwatch
    def density_grad(self, V: FieldH) -> None:
        """Augmentation part of the nonlocal matrix, from the Kohn-Sham potential V
        (i.e. `Electrons.n_tilde.grad`, with components n and M_z if polarized):
        D_aug[s, (a, i), (a, j)] = int V_s(r) Q_ij(r - R_a) d^3r, which is also
        dE / d rho_beta[s, a, i, j]. Obtained by transposing each map in `density`.
        Returns the block-diagonal matrix on all projectors, n_spins x 1 x
        n_projectors x n_projectors (broadcast over k), to add to `Ions.D_all`."""
        n_spins = V.data.shape[0]
        n_proj_tot = self.ions.n_projectors
        D_aug = torch.zeros(
            (n_spins, 1, n_proj_tot, n_proj_tot), dtype=torch.cdouble, device=rc.device
        )
        V_weighted = V.data.conj() * self.grid.weight2H  # Hermitian-symmetry weights
        for species in self.species:
            coeff_grad = []
            for atoms in self._atom_chunks(species, n_spins):
                U = V_weighted[:, None] * self._phase(species, atoms)  # d, atom, grid
                radial_grad = self._combine_adjoint(U)  # d, atom, LM, grid
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
    def accumulate_geometry_grad(self, V: FieldH, rho_beta: list[torch.Tensor]) -> None:
        """Accumulate force and stress contributions of the augmentation charges
        moving with the ions and the lattice, at fixed `rho_beta`, in the potential
        V (see `D_aug`). Changes of rho_beta itself (through the projectors) are
        accounted for by `Ions.beta.grad`. As in `Ions.accumulate_geometry_grad`,
        each is collected only if the corresponding `requires_grad` is enabled."""
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
            self.Y_prime = self._get_harmonics(prime=True)

        for species, rho in zip(self.species, rho_beta):
            coeff = species.radial_coefficients(rho)
            for atoms in self._atom_chunks(
                species, n_spins * (4 if need_stress else 1)
            ):
                radial = self.Ginterp(coeff[:, :, atoms])  # d, atom, LM, grid
                U = V_weighted[:, None] * self._phase(species, atoms)  # d, atom, grid
                # E = sum Re(U T), with d(phase)/d(position) = -2 pi i iG phase:
                UT = U * self._combine(radial)
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
                    T_prime_radial = self._combine(radial_prime)
                    for i_dir in range(3):
                        dT_dG = T_prime_radial * Ghat[..., i_dir] + self._combine(
                            radial, i_dir
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
            del self.Y_prime  # save memory

    def _combine(self, radial: torch.Tensor, prime_dir: int = -1) -> torch.Tensor:
        """Complex sum over LM of radial[d, a, LM, grid] i^L Y_LM[grid], with signed
        real harmonics `Y`: real part from even L and imaginary part from odd L."""
        n = self.n_even
        if prime_dir >= 0:
            Yeven, Yodd = self.Y_prime[0][prime_dir], self.Y_prime[1][prime_dir]
        else:
            Yeven, Yodd = self.Y
        return torch.complex(
            torch.einsum("daL..., L... -> da...", radial[:, :, :n], Yeven),
            torch.einsum("daL..., L... -> da...", radial[:, :, n:], Yodd),
        )

    def _combine_adjoint(self, U: torch.Tensor) -> torch.Tensor:
        """Transpose of `_combine` for the real inner product sum Re(U T):
        d/d(radial) of sum Re(U[d, a, grid] T[d, a, grid]), which is real."""
        Yeven, Yodd = self.Y
        return torch.cat(
            (U.real[:, :, None] * Yeven, -U.imag[:, :, None] * Yodd), dim=2
        )

    def _phase(self, species: Species, atoms: slice) -> torch.Tensor:
        """Translation phases (atom, grid) of `atoms` (a slice within `species`)."""
        start = species.atom_slice.start
        atom_slice = slice(start + atoms.start, start + atoms.stop)
        return self.ions.translation_phase(self.iG, atom_slice).permute(3, 0, 1, 2)

    def _atom_chunks(self, species: Species, n_batch: int) -> list[slice]:
        """Split the atoms of `species` so that per-chunk grid intermediates of
        n_batch x LM real values per atom and grid point fit `ATOM_CHUNK_BYTES`."""
        bytes_per_atom = 8 * n_batch * max(1, self.Y[0].numel())  # (slab may be empty)
        n_chunk = max(1, ATOM_CHUNK_BYTES // bytes_per_atom)
        n_atoms = species.n_atoms
        return [slice(i, min(i + n_chunk, n_atoms)) for i in range(0, n_atoms, n_chunk)]


class Species:
    """Augmentation data of one ultrasoft species."""

    atom_slice: slice  #: Atoms of this species within `Ions`
    proj_slice: slice  #: Projectors of these atoms within `Ions.beta` and `D_all`
    ps: Pseudopotential  #: Corresponding pseudopotential
    n_proj: int  #: Number of projectors per atom (including m)
    gaunt: torch.Tensor  #: Gaunt tensor (see `_get_gaunt`), in `LM_order`

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
        self.n_proj = ps.n_projectors
        self.gaunt = _get_gaunt(ps, augmentation.L_of_LM)[:, augmentation.LM_order]

    @property
    def n_atoms(self) -> int:
        return self.atom_slice.stop - self.atom_slice.start

    def radial_coefficients(self, rho: torch.Tensor) -> torch.Tensor:
        """Compute radial spline coefficients (density component, atom, lm)
        for density augmentation from atomic density matrix `rho` (spin, atom, i, j)."""
        # Symmetry of Gaunt coefficients + hermiticity of rho => only rho.real matters:
        spin_map = _spin_map(rho.shape[0])
        Q_coeff = self.ps.Q.f_tilde_coeff
        return torch.einsum(
            "cq, qLij, ds, saij -> cdaL", Q_coeff, self.gaunt, spin_map, rho.real
        )

    def radial_coefficients_grad(self, coeff_grad: torch.Tensor) -> torch.Tensor:
        """Gradient propagation from spline coefficients to atomic density matrix,
        corresponding to `radial_coefficients`."""
        Q_coeff = self.ps.Q.f_tilde_coeff[: coeff_grad.shape[0]]  # rows used by Ginterp
        spin_map = _spin_map(coeff_grad.shape[1])
        return torch.einsum(
            "ds, qLij, cq, cdaL -> saij", spin_map, self.gaunt, Q_coeff, coeff_grad
        )


def _get_gaunt(ps: dft.ions.Pseudopotential, L_of_LM: torch.Tensor) -> torch.Tensor:
    """Real Gaunt coefficients of each augmentation function of `ps` on projector
    pairs: gaunt[q, LM, a, b] = c^LM_{(l m)_a, (l m)_b} if projectors a and b
    (including m) belong to the radial pair of Q function q (in either order) and
    the L of LM is that of q, and zero otherwise (n_Q x n_LM x n_proj x n_proj)."""
    assert (ps.Q is not None) and (ps.Q_pair is not None)
    pqn = ps.pqn_beta
    n_LM = len(L_of_LM)
    product = sh.get_product_coefficients(pqn.l_max)  # (lm, lm', LM) for LM of lm lm'
    n_LM_product = min(n_LM, product.shape[-1])
    c = torch.zeros(product.shape[:2] + (n_LM,), device=rc.device)
    c[..., :n_LM_product] = product[..., :n_LM_product]
    c_ab = c[pqn.i_lm][:, pqn.i_lm].permute(2, 0, 1)  # LM, projector a, projector b
    i_rf, (i, j) = pqn.i_rf, ps.Q_pair.T
    in_pair = (i_rf[None, :, None] == i[:, None, None]) & (
        i_rf[None, None, :] == j[:, None, None]
    )  # q, a, b
    in_pair = in_pair | in_pair.transpose(1, 2)  # Q_ij = Q_ji (not in-place: a view)
    L_match = ps.Q.l[:, None] == L_of_LM[None, :]  # q, LM
    return (in_pair[:, None] & L_match[:, :, None, None]) * c_ab[None]


def _spin_map(n_spins: int) -> torch.Tensor:
    """Map from spin channels (up, down) to the density components (n, M_z).
    (Currently, only supports non-spinorial modes.)"""
    if n_spins == 1:
        return torch.ones((1, 1), device=rc.device)
    return torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=rc.device)
