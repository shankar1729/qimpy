from __future__ import annotations

import numpy as np
import torch

from qimpy import TreeNode, log, rc
from qimpy.io import CheckpointPath, CheckpointContext
from qimpy.profiler import stopwatch
from qimpy.transport import material


class RelaxationTime(TreeNode):
    ab_initio: material.ab_initio.AbInitio
    tau_p: dict[int, torch.Tensor]  #: momentum relaxation time
    tau_s: dict[int, torch.Tensor]  #: spin relaxation time
    tau_e: dict[int, torch.Tensor]  #: electron relaxation time
    tau_h: dict[int, torch.Tensor]  #: hole relaxation time
    tau_eh: dict[int, torch.Tensor]  #: electron-hole off-diagonal relaxation
    tau_recomb: dict[int, torch.Tensor]  #: recombination time
    max_dmu: float  #: maximum change of mu in find_mu
    nv: int  #: number of valance bands
    eps: float
    only_diagonal: bool

    constant_params: dict[str, torch.Tensor]  #: constant values of parameters

    def __init__(
        self,
        *,
        ab_initio: material.ab_initio.AbInitio,
        tau_p: float = np.inf,
        tau_s: float = np.inf,
        tau_e: float = np.inf,
        tau_h: float = np.inf,
        tau_eh: float = np.inf,
        max_dmu: float = 1e-3,
        tau_recomb: float = np.inf,
        nv: int = 0,
        eps: float = 1e-12,
        dmu_eps: float = 1e-8,
        only_diagonal: bool = True,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        """
        Initialize relaxation time approximation for scattering.

        Parameters
        ----------
        tau_p
            :yaml:`Momentum relaxation time.`
        tau_s
            :yaml:`Spin relaxation time.`
        tau_e
            :yaml:`Conduction bands relaxation time.`
        tau_h
            :yaml:`Valance bands relaxation time.`
        tau_eh
            :yaml:`Relaxation time of conduction-valance off diagonal terms.`
        max_dmu
            :yaml:`Maximum mu change in Newton-Rhapson method.`
        tau_recomb
            :yaml:`Recombination time.`
        nv
            :yaml:`Number of valance bands.`
        eps
            :yaml:`Precision in the determination of mu_e and mu_h.`
        only_diagonal
            :yaml:`Whether only diagonal terms change.`
        """
        super().__init__()
        self.ab_initio = ab_initio
        self.constant_params = dict(
            tau_p=torch.tensor(tau_p, device=rc.device),
            tau_s=torch.tensor(tau_s, device=rc.device),
            tau_e=torch.tensor(tau_e, device=rc.device),
            tau_h=torch.tensor(tau_h, device=rc.device),
            tau_eh=torch.tensor(tau_eh, device=rc.device),
            tau_recomb=torch.tensor(tau_recomb, device=rc.device),
        )
        self.nv = nv
        self.tau_e = {}
        self.tau_h = {}
        self.tau_eh = {}
        self.tau_recomb = {}
        if np.isfinite(tau_e):
            log.info("Enable RTA for conduction bands.")
            self.exp_betaEe = {}
            self.betamue0 = {}
        if np.isfinite(tau_h):
            log.info("Enable RTA for valance bands.")
            self.exp_betaEh = {}
            self.betamuh0 = {}
        if np.isfinite(tau_eh):
            log.info("Enable RTA for conduction-valance off-diagonal terms.")
        if np.isfinite(tau_recomb):
            log.info("Enable phenomenon recombination.")
            self.exp_betaE = {}
            self.betamuk0 = {}
        self.max_dbetamu = max_dmu / ab_initio.T
        self.max_dmu = max_dmu
        self.eps = float(eps)
        self.dbetamu_eps = float(dmu_eps) / ab_initio.T
        self.nbands = ab_initio.E.shape[-1]
        self.dmu_eps = float(dmu_eps)
        self.only_diagonal = only_diagonal
        self.sum_rules = {
            2: "xykb -> xy", # electron or hole
            3: "xykb -> xyk", # recombination
        }

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        attrs = cp_path.attrs
        attrs["tau_p"] = self.constant_params["tau_p"].item()
        attrs["tau_s"] = self.constant_params["tau_s"].item()
        attrs["tau_e"] = self.constant_params["tau_e"].item()
        attrs["tau_h"] = self.constant_params["tau_h"].item()
        attrs["tau_eh"] = self.constant_params["tau_eh"].item()
        attrs["tau_recomb"] = self.constant_params["tau_recomb"].item()
        attrs["max_dmu"] = self.max_dmu
        attrs["nv"] = self.nv
        attrs["eps"] = self.eps
        attrs["dmu_eps"] = self.dmu_eps
        attrs["only_diagonal"] = self.only_diagonal
        return list(attrs.keys())

    def initialize_fields(self, params: dict[str, torch.Tensor], patch_id: int) -> None:
        self._initialize_fields(patch_id, **params)

    def _initialize_fields(
        self,
        patch_id: int,
        *,
        tau_p: torch.Tensor,
        tau_s: torch.Tensor,
        tau_e: torch.Tensor,
        tau_h: torch.Tensor,
        tau_eh: torch.Tensor,
        tau_recomb: torch.Tensor,
    ) -> None:
        ab_initio = self.ab_initio
        nv = self.nv
        betamu0 = ab_initio.mu / ab_initio.T
        if torch.isfinite(tau_e):
            self.tau_e[patch_id] = tau_e
            self.exp_betaEe[patch_id] = torch.exp(ab_initio.E[:, nv:] / ab_initio.T)
            self.betamue0[patch_id] = torch.ones([1, 1]).to(rc.device) * betamu0
        if torch.isfinite(tau_h):
            self.tau_h[patch_id] = tau_h
            self.exp_betaEh[patch_id] = torch.exp(-ab_initio.E[:, :nv] / ab_initio.T)
            self.betamuh0[patch_id] = -torch.ones([1, 1]).to(rc.device) * betamu0
        if torch.isfinite(tau_eh):
            self.tau_eh[patch_id] = tau_eh
        if torch.isfinite(tau_recomb):
            self.tau_recomb[patch_id] = tau_recomb
            self.exp_betaE[patch_id] = torch.exp(ab_initio.E / ab_initio.T)
            self.betamuk0[patch_id] = (
                torch.ones([1, 1, ab_initio.E.shape[0]]).to(rc.device) * betamu0
            )

    @stopwatch
    def rho_dot(self, rho: torch.Tensor, t: float, patch_id: int) -> torch.Tensor:
        # rho.shape: (x,y,k,b)
        nv = self.nv
        result = torch.zeros_like(rho)

        if patch_id in self.tau_e:
            nbands = self.nbands
            fe = rho[..., range(nv, nbands), range(nv, nbands)].real
            if self.betamue0[patch_id].shape != rho.shape[:2]:
                self.betamue0[patch_id] = torch.tile(
                    self.betamue0[patch_id],
                    rho.shape[:2]
                )
            betamue, fe_eq = self.find_mu(
                fe,
                self.exp_betaEe[patch_id],
                self.betamue0[patch_id],
            )
            if self.only_diagonal:
                result[..., range(nv, nbands), range(nv, nbands)] -= (fe - fe_eq) / (
                    2 * self.tau_e[patch_id]
                )
            else:
                result[..., nv:, nv:] -= (
                    rho[..., nv:, nv:] - torch.diag_embed(fe_eq)
                ) / (2 * self.tau_e[patch_id])
            self.betamue0[patch_id] = betamue

        if patch_id in self.tau_h:
            fh = rho[..., range(nv), range(nv)].real
            if self.betamuh0[patch_id].shape != rho.shape[:2]:
                self.betamuh0[patch_id] = torch.tile(
                    self.betamuh0[patch_id],
                    rho.shape[:2]
                )
            betamuh, fh_eq = self.find_mu(
                1 - fh,
                self.exp_betaEh[patch_id],
                self.betamuh0[patch_id],
            )
            fh_eq = 1 - fh_eq
            if self.only_diagonal:
                result[..., range(nv), range(nv)] -= (fh - fh_eq) / (
                    2 * self.tau_h[patch_id]
                )
            else:
                result[..., :nv, :nv] -= (
                    rho[..., :nv, :nv] - torch.diag_embed(fh_eq)
                ) / (2 * self.tau_h[patch_id])
            self.betamuh0[patch_id] = betamuh

        if patch_id in self.tau_eh:
            result[..., nv:, :nv] -= (
                rho[..., nv:, :nv] / self.tau_eh[patch_id]
            )  # + h.c later

        if patch_id in self.tau_recomb:
            fk = torch.einsum("...bb -> ...b", rho).real
            if self.betamuk0[patch_id].shape[:2] != rho.shape[:2]:
                self.betamuk0[patch_id] = torch.tile(
                    self.betamuk0[patch_id],
                    rho.shape[:2] + (1,)
                )
            betamuk, fk_eq = self.find_mu(
                fk,
                self.exp_betaE[patch_id],
                self.betamuk0[patch_id],
            )
            result -= (rho - torch.diag_embed(fk_eq)) / (2 * self.tau_recomb[patch_id])
            self.betamuk0[patch_id] = betamuk

        return result

    def find_mu(
        self,
        f: torch.Tensor,
        exp_betaE: torch.Tensor,
        betamu0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        betamu = betamu0
        sum_rule = self.sum_rules[betamu.ndim]
        f_total = torch.einsum(sum_rule, f)
        reshape = betamu.shape + (1,) * (4 - betamu.ndim)
        # Fermi-dirac distribution, shape (Nx, Ny, Nk, Nb)
        exp_beta_Emu = exp_betaE[None, None] / torch.exp(betamu).reshape(reshape)
        distribution = 1 / (exp_beta_Emu + 1)
        F = torch.einsum(sum_rule, distribution) - f_total
        while torch.max(torch.abs(F)) > self.eps:
            dF = torch.einsum(sum_rule, exp_beta_Emu / (exp_beta_Emu + 1) ** 2)
            dbetamu = F / dF
            limit_ind = (
                torch.abs(dbetamu) > self.max_dbetamu
            )  # indices that need to be limited by self.max_dbetamu
            dbetamu[limit_ind] = torch.sign(F[limit_ind]) * self.max_dbetamu
            betamu = betamu - dbetamu
            exp_beta_Emu = exp_betaE[None, None] / torch.exp(betamu).reshape(reshape)
            distribution = 1 / (exp_beta_Emu + 1)
            F = torch.einsum(sum_rule, distribution) - f_total
            if torch.max(torch.abs(dbetamu)) < self.dbetamu_eps:
                break
        return betamu, distribution
