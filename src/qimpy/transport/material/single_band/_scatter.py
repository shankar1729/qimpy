from __future__ import annotations

import torch
import numpy as np

from qimpy import rc, log, TreeNode
from qimpy.io import CheckpointPath, CheckpointContext
from qimpy.transport import material


class Scatter(TreeNode):
    """Explicit k-dependent scattering kernel.
    Currently, e-e scattering only, but can be extended to e-ph.
    """

    single_band: material.single_band.SingleBand
    dE: float
    epsilon_bg: float
    lambda_D: float

    def __init__(
        self,
        *,
        single_band: material.single_band.SingleBand,
        dE: float,
        epsilon_bg: float,
        lambda_D: float,
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
        """
        super().__init__()
        self.single_band = single_band
        self.dE = dE
        self.epsilon_bg = epsilon_bg
        self.lambda_D = lambda_D

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
            torch.stack((torch.cat((ik_pair, ik_pair)), i_q_omega_reduced), dim=0),
            prefac * torch.cat((w_omega_plus, w_omega_minus)),
            size=(len(ik_pair), n_q_omega),
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
        self.Msq = Msq
        exit()

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        attrs = cp_path.attrs
        attrs["dE"] = self.dE
        attrs["epsilon_bg"] = self.epsilon_bg
        attrs["lambda_D"] = self.lambda_D
        return list(attrs.keys())

    def rho_dot(self, rho: torch.Tensor) -> torch.Tensor:
        return NotImplemented
