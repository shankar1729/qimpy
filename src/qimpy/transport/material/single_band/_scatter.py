from __future__ import annotations

import torch
import numpy as np

from qimpy import rc, log, TreeNode
from qimpy.io import CheckpointPath, CheckpointContext
from qimpy.math import ortho_matrix
from qimpy.mpi import get_block_slices
from qimpy.transport import material


class Scatter(TreeNode):
    """Explicit k-dependent scattering kernel.
    Currently, e-e scattering only, but can be extended to e-ph.
    """

    single_band: material.single_band.SingleBand
    dE: float
    epsilon_bg: float
    lambda_D: float
    block_size: int
    conserve_energy: bool
    S: torch.Tensor  #: energy-momentum conservation sparse matrix
    ST: torch.Tensor  #: cached transpose of `S`
    Msq: torch.Tensor  #: e-e matrix element squared (including rate prefactors)
    vE: torch.Tensor  #: vector for energy conservation correction

    def __init__(
        self,
        *,
        single_band: material.single_band.SingleBand,
        dE: float,
        epsilon_bg: float,
        lambda_D: float,
        block_size: int = 128,
        conserve_energy: bool = True,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        """
        Initialize ab initio Lindbladian scattering.

        Parameters
        ----------
        dE
            :yaml:`Energy conservation width.`
            Determines resolution of internal frequency grid.
        epsilon_bg
            :yaml:`Background dielectric constant.`
        lambda_D
            :yaml:`Debye screening length of electrons.`
        block_size
            :yaml:`Number of real-space points to calculate together.`
            Performance should generally increase with block_size,
            but so does the memory requirement.
        conserve_energy
            :yaml:`Whether to enforce exact energy conservation of e-e scattering.`
        """
        super().__init__()
        self.single_band = single_band
        self.dE = dE
        self.epsilon_bg = epsilon_bg
        self.lambda_D = lambda_D
        self.block_size = block_size
        self.conserve_energy = conserve_energy

        # Initialize scattering operator
        log.info("\n--- Initializing Scatter operator ---")
        assert single_band.comm.size == 1  # MPI over k not yet supported
        kmesh = torch.tensor(single_band.kmesh, device=rc.device)
        k = single_band.k  # may need to pass in k_all later for MPI support
        E = single_band.E  # may need E_all for MPI

        # Frequency grid
        omega_max = (E.max() - E.min()).item()
        i_omega_max = int(np.ceil(omega_max / dE))
        omega = torch.arange(-i_omega_max, i_omega_max + 1, device=rc.device) * dE
        n_omega = len(omega)
        log.info(
            f"Initialized frequency grid with {n_omega} points and resolution {dE}"
        )

        # Momentum-transfer grid
        ik = torch.round(k * kmesh).to(torch.int)
        iq_pair = (ik[:, None] - ik[None]) % kmesh
        iq, q_index = iq_pair.flatten(0, 1).unique(dim=0, return_inverse=True)
        q = iq / kmesh
        q -= torch.round(q)  # wrap to [-0.5, 0.5) in each dimension
        q_mag = (q @ single_band.lattice.Gbasis.T).norm(dim=-1)
        n_q = len(q_mag)
        log.info(f"Found {n_q} unique momentum transfers")

        # Construct sparse energy-momentum matrix and matrix elements
        ik_pair = torch.arange(len(k) ** 2, device=rc.device)  # flattened
        x_omega = i_omega_max + (E - E.T).flatten() / dE  # since E is len(k) x 1
        i_omega = x_omega.floor().to(torch.int)
        i_omega.clamp_(min=0, max=(n_omega - 1))
        w_omega_plus = x_omega - i_omega  # weight of i_omega + 1 point
        w_omega_minus = 1 - w_omega_plus  # weight of i_omega point
        i_q_omega = q_index * n_omega + i_omega  # flattened q, omega index
        i_q_omega = torch.cat((i_q_omega + 1, i_q_omega))  # plus and minus indices
        i_q_omega_used, i_q_omega_reduced = torch.unique(i_q_omega, return_inverse=True)
        n_q_omega = len(i_q_omega_used)  # number of accessible (q, omega) pairs
        log.info(f"Found {n_q_omega} accessible energy-momentum combinations")
        prefac = 1 / (np.prod(single_band.kmesh) * single_band.lattice.volume)
        S = torch.sparse_coo_tensor(
            torch.stack((i_q_omega_reduced, torch.cat((ik_pair, ik_pair))), dim=0),
            prefac * torch.cat((w_omega_plus, w_omega_minus)),
            size=(n_q_omega, len(ik_pair)),
        )
        Msq = (
            ((4 * np.pi) ** 3 / (dE * epsilon_bg))
            / (q_mag.square() + lambda_D**-2).square()
        )[i_q_omega_used // n_omega]
        log.info(
            f"Initialized {S.shape[0]} x {S.shape[1]} sparse energy-momentum matrix"
            f" with {S._nnz()} non-zero elements"
        )
        self.S = S
        self.ST = S.T
        self.Msq = Msq[:, None]  # dimension added for broadcasting

        if self.conserve_energy:
            # Construct energy conservation correction:
            vE = E.flatten() - single_band.mu
            dims = torch.where(kmesh > 1)[0]  # only use momenta in periodic directions
            constraints = torch.cat((torch.ones_like(vE)[:, None], k[:, dims]), dim=1)
            constraints = constraints @ ortho_matrix(constraints.T @ constraints)
            vE -= constraints @ (constraints.T @ vE)  # orthogonalize w.r.t constraints
            vE *= 1.0 / (vE @ E.flatten())  # normalize to unit energy
            self.vE = vE

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        attrs = cp_path.attrs
        attrs["dE"] = self.dE
        attrs["epsilon_bg"] = self.epsilon_bg
        attrs["lambda_D"] = self.lambda_D
        attrs["conserve_energy"] = self.conserve_energy
        return list(attrs.keys())

    def rho_dot(self, rho: torch.Tensor) -> torch.Tensor:
        rho_shape = rho.shape
        rho = rho.view(-1, rho_shape[-1])  # flatten real space dimensions
        rho_dot = torch.empty_like(rho)
        for sel in get_block_slices(len(rho), self.block_size):
            rho_sel = rho[sel]
            rho_pair = rho_sel[..., None] * (1 - rho_sel[:, None, :])
            rho_pair_dot = (
                self.ST @ (self.Msq * (self.S @ rho_pair.flatten(-2, -1).T))
            ).T.unflatten(1, rho_pair.shape[1:])
            rho_pair_dot *= rho_pair.swapaxes(-1, -2)
            rho_dot[sel] = rho_pair_dot.sum(dim=-1) - rho_pair_dot.sum(dim=-2)
        if self.conserve_energy:
            rho_dot -= (rho_dot @ self.single_band.E.flatten())[:, None] * self.vE
        return rho_dot.reshape(rho_shape)
