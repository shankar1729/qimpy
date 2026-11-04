"""Ultrasoft augmentation on synthetic data (no pseudopotential files needed):
density against a direct real-space evaluation (phases and normalization), charge
conservation, and D_aug as the exact adjoint of the density (also via autograd)."""

from functools import cache

import numpy as np
import pytest
import torch
import torch.distributed as dist

from qimpy import rc
from qimpy.io import Unit
from qimpy.lattice import Lattice
from qimpy.symmetries import Symmetries
from qimpy.grid import Grid, FieldH
from qimpy.math import RadialFunction, spherical_harmonics as sh
from . import Ions, Pseudopotential, PseudoQuantumNumbers
from ._augmentation import Augmentation

SIGMA = 0.6  #: Gaussian width (bohr) of the synthetic augmentation functions
L_BETA = [0, 0, 1, 2]  #: angular momenta of the synthetic radial projectors
POSITIONS = [[0.1, 0.2, 0.3], [0.6, 0.55, 0.8]]  #: fractional, two atoms


def amplitude(i: int, j: int, L: int) -> float:
    """Amplitude of the synthetic Q^L_ij, symmetric in (i, j)."""
    return 1.0 + 0.3 * (i + j) + 0.1 * i * j + 0.2 * L


def gaussian_Q(r: np.ndarray, i: int, j: int, L: int) -> np.ndarray:
    """Synthetic augmentation function Q^L_ij(r) = amplitude r^L exp(-r^2 / 2 sigma^2)."""
    return amplitude(i, j, L) * (r**L) * np.exp(-0.5 * (r / SIGMA) ** 2)


@cache
def get_augmentation() -> Augmentation:
    """Two atoms of a synthetic ultrasoft species in a triclinic cell."""
    deg = Unit.MAP["deg"]
    lattice = Lattice(
        system=dict(
            name="triclinic",
            a=6.0,
            b=6.5,
            c=7.0,
            alpha=80 * deg,
            beta=85 * deg,
            gamma=75 * deg,
        )
    )
    grid = Grid(
        lattice=lattice,
        symmetries=Symmetries(lattice=lattice),
        shape=(40, 40, 40),
        group=None,
    )

    # Pseudopotential with only the data augmentation needs:
    r_np = np.linspace(1e-4, 12.0, 3001)
    r = torch.tensor(r_np, device=rc.device)
    dr = torch.full_like(r, r_np[1] - r_np[0])
    ps = Pseudopotential.__new__(Pseudopotential)
    ps.r, ps.dr = r, dr
    n_beta = len(L_BETA)
    ps.beta = RadialFunction(r, dr, torch.zeros((n_beta, len(r))), np.array(L_BETA))
    ps.pqn_beta = PseudoQuantumNumbers(ps.beta.l, None)
    r2_Q = np.zeros((n_beta, n_beta, 2 * max(L_BETA) + 1, len(r_np)))
    for i, l_i in enumerate(L_BETA):
        for j, l_j in enumerate(L_BETA):
            for L in range(abs(l_i - l_j), l_i + l_j + 1, 2):
                r2_Q[i, j, L] = (r_np**2) * gaussian_Q(r_np, i, j, L)
    ps.set_augmentation(r2_Q, np.zeros((n_beta, n_beta)))
    assert (ps.Q is not None) and (dist.group.WORLD is not None)
    RadialFunction.transform([ps.Q], grid.get_Gmax(), dist.group.WORLD)

    # Ions with only the attributes augmentation needs:
    ions = Ions.__new__(Ions)
    ions.lattice = lattice
    ions.pseudopotentials = [ps]
    ions.n_ions_type = [len(POSITIONS)]
    ions.slices = [slice(0, len(POSITIONS))]
    ions.positions = torch.tensor(POSITIONS, device=rc.device)
    return Augmentation(ions, grid)


def random_rho(n_spins: int, seed: int) -> torch.Tensor:
    """Random Hermitian density matrices (spin x atom x n_proj x n_proj)."""
    generator = torch.Generator(device=rc.device).manual_seed(seed)
    n_proj = get_augmentation().species[0].n_proj
    shape = (n_spins, len(POSITIONS), n_proj, n_proj)
    rho = torch.randn(shape, generator=generator, dtype=torch.cdouble, device=rc.device)
    return rho + rho.transpose(-2, -1).conj()


def atom_blocks(D_aug: torch.Tensor, n_proj: int) -> torch.Tensor:
    """Per-atom diagonal blocks (spin x atom x i x j) of D_aug from `Augmentation`."""
    D = D_aug[:, 0].unflatten(1, (-1, n_proj)).unflatten(3, (-1, n_proj))
    return D.diagonal(dim1=1, dim2=3).permute(0, 3, 1, 2)


@pytest.mark.mpi_skip
@pytest.mark.parametrize("n_spins", [1, 2])
def test_D_aug_adjoint(n_spins: int) -> None:
    """<V, n_aug(rho)> = sum Re(D_aug(V) rho) for any V and Hermitian rho, and
    D_aug is the autograd gradient of that energy with respect to rho."""
    augmentation = get_augmentation()
    rho = random_rho(n_spins, seed=0).requires_grad_(True)
    V = FieldH(augmentation.grid, shape_batch=(n_spins,))
    V.randomize(seed=1)
    energy = (V ^ augmentation.density([rho])).sum()
    D = atom_blocks(augmentation.D_aug(V), augmentation.species[0].n_proj)
    assert torch.allclose(D.imag, torch.zeros_like(D.real))
    assert torch.allclose(D, D.transpose(-2, -1))  # real symmetric
    energy_from_D = (D.real * rho.real).sum()
    assert abs(energy - energy_from_D) < 1e-12 * abs(energy)
    (rho_grad,) = torch.autograd.grad(energy, rho)
    assert torch.allclose(rho_grad.real, D.real, rtol=1e-10, atol=1e-12)
    assert torch.allclose(rho_grad.imag, torch.zeros_like(D.real))


@pytest.mark.mpi_skip
def test_density_real_space() -> None:
    """Augmentation density on the grid equals sum_ij rho_ij Q_ij(r - R_a) evaluated
    directly in real space (summing periodic images), including odd L, which checks
    the phase convention: projections carry (-1)^l, so rho_ij is used with
    (-1)^(l_i + l_j) relative to the physical density matrix."""
    augmentation = get_augmentation()
    grid = augmentation.grid
    ps = augmentation.ions.pseudopotentials[0]
    pqn = ps.pqn_beta
    rho_physical = random_rho(1, seed=2)
    parity = (-1.0) ** (pqn.l[:, None] + pqn.l[None, :])  # (-1)^(l_i + l_j)
    n_aug = ~augmentation.density([rho_physical * parity])  # real space

    # Direct evaluation of Q_ij(r) = sum_LM c^LM_ij Q^L_ij(|r|) Y_LM(r_hat): with the
    # synthetic Q^L_ij = amplitude r^L exp(-r^2 / 2 sigma^2), each atom contributes
    # exp(-r^2 / 2 sigma^2) sum_LM W_LM r^L Y_LM(r_hat), where W_LM collects rho,
    # the Gaunt coefficients and amplitudes (independent of FFTs and transforms):
    L_max = 2 * max(L_BETA)
    L_of_LM = torch.arange(L_max + 1).repeat_interleave(2 * torch.arange(L_max + 1) + 1)
    c = sh.get_product_coefficients(max(L_BETA))[pqn.i_lm][:, pqn.i_lm]  # a, b, LM
    i_rf = pqn.i_rf.tolist()
    A = torch.tensor(
        [[[amplitude(i, j, int(L)) for L in L_of_LM] for j in i_rf] for i in i_rf]
    )  # amplitude for projectors a, b and each LM
    r_grid = grid.get_mesh("R").to(torch.double) / torch.tensor(grid.shape)
    shifts = torch.stack(
        torch.meshgrid(*([torch.arange(-1, 2)] * 3), indexing="ij"), dim=-1
    ).reshape(
        -1, 3
    )  # neighbouring periodic images suffice for this sigma
    n_direct = torch.zeros(grid.shape, device=rc.device)
    for rho_atom, position in zip(rho_physical[0].real, augmentation.ions.positions):
        W = torch.einsum("ab, abL -> L", rho_atom, c * A)
        for shift in shifts:
            x = (r_grid - position - shift) @ grid.lattice.Rbasis.T  # Cartesian
            gaussian = torch.exp(-0.5 * (x.norm(dim=-1) / SIGMA) ** 2)
            n_direct += gaussian * torch.einsum(
                "L, L... -> ...", W, sh.get_harmonics(L_max, x)
            )
    error = (n_aug.data[0] - n_direct).abs().max() / n_direct.abs().max()
    print(f"Relative error of augmentation density vs real space: {error:.1e}")
    assert error < 1e-6


@pytest.mark.mpi_skip
@pytest.mark.parametrize("n_spins", [1, 2])
def test_charge(n_spins: int) -> None:
    """Augmentation charge is sum_a Tr(rho_a q) (q from the G = 0 limit)."""
    augmentation = get_augmentation()
    ps = augmentation.ions.pseudopotentials[0]
    assert ps.q is not None
    rho = random_rho(n_spins, seed=3)
    n_aug = augmentation.density([rho])
    q = ps.pqn_beta.expand_matrix(ps.q, 1)
    charge = torch.einsum("saij, ij -> s", rho.real, q)  # per spin channel
    expected = charge.sum() if (n_spins == 1) else charge @ torch.tensor([1.0, -1.0])
    got = n_aug.integral()
    assert abs(got[0] - charge.sum()) < 1e-12 * abs(charge).sum()  # total n
    assert abs(got[-1] - expected) < 1e-12 * abs(charge).sum()  # M_z if polarized
