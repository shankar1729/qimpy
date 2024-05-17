from __future__ import annotations

import numpy as np
import torch

from qimpy import TreeNode, log, rc
from qimpy.io import CheckpointPath
from qimpy.transport import material


class RelaxationTime(TreeNode):
    ab_initio: material.ab_initio.AbInitio
    tau_p: float  #: momentum relaxation time
    tau_s: float  #: spin relaxation time
    tau_e: float  #: electron relaxation time
    tau_h: float  #: hole relaxation time
    max_dmu: float  #: maximum change of mu in find_mu
    tau_recomb: float  #: recombination time
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
        max_dmu: float = 1e-3,
        tau_recomb: float = np.inf,
        nv: int = 0,
        eps: float = 1e-12,
        only_diagonal: boll = True,
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
        self.constant_params = dict()
        self.tau_p = tau_p
        self.tau_s = tau_s
        self.tau_e = tau_e
        self.tau_h = tau_h
        self.tau_recomb = tau_recomb
        self.nv = nv
        if np.isfinite(tau_e):
            log.info("Enable RTA for conduction bands.")
            self.exp_betaEe = torch.exp(ab_initio.E[:,nv:]/ab_initio.T)
            self.betamue0 = torch.Tensor([ab_initio.mu/ab_initio.T]).to(rc.device)
        if np.isfinite(tau_h):
            log.info("Enable RTA for valance bands.")
            self.exp_betaEh = torch.exp(ab_initio.E[:,:nv]/ab_initio.T)
            self.betamuh0 = torch.Tensor([ab_initio.mu/ab_initio.T]).to(rc.device)
        if np.isfinite(tau_recomb):
            log.info("Enable phenomenon recombination.")
            self.exp_betaE = torch.exp(ab_initio.E/ab_initio.T)
            self.betamuk0 = torch.ones([1,ab_initio.E.shape[0], 1]).to(rc.device) * ab_initio.mu/ab_initio.T
        self.max_dbetamu = max_dmu/ab_initio.T
        self.eps = eps
        self.nbands = ab_initio.E.shape[-1]
        self.only_diagonal = only_diagonal

    def initialize_fields(self, params: dict[str, torch.Tensor], patch_id: int) -> None:
        pass  # TODO

    def rho_dot(self, rho: torch.Tensor, t: float, patch_id: int) -> torch.Tensor:
        result = torch.zeros_like(rho)
        
        if np.isfinite(self.tau_e):
            fe = rho[...,range(self.nv,self.nbands),range(self.nv,self.nbands)].real
            betamue,fe_eq = self.find_mu(
                fe, self.exp_betaEe, self.betamue0, sum_rule="...kb -> ...", reshape=(fe.shape[0],)
            )
            if self.only_diagonal:
                result[...,range(self.nv,self.nbands),range(self.nv,self.nbands)] -= (fe - fe_eq)/(2*self.tau_e) 
            else:
                result[...,self.nv:,self.nv:] -= (rho[...,self.nv:,self.nv:] -  torch.diag_embed(fe_eq))/(2*self.tau_e)
            self.betamue0 = betamue
            
        if np.isfinite(self.tau_h):
            fh = rho[...,range(self.nv),range(self.nv)].real
            betamuh,fh_eq = self.find_mu(
                fh, self.exp_betaEh, self.betamuh0, sum_rule="...kb -> ...", reshape=(fh.shape[0],)
            )
            if self.only_diagonal:
                result[...,range(self.nv),range(self.nv)] -= (fh - fh_eq)/(2*self.tau_h) 
            else:
                result[...,:self.nv,:self.nv] -= (rho[...,:self.nv,:self.nv] -  torch.diag_embed(fh_eq))/(2*self.tau_h)
                if np.isfinite(self.tau_e):
                    tau_comb = np.sqrt(self.tau_e*self.tau_h)
                    result[...,self.nv:,:self.nv] -= rho[...,self.nv:,:self.nv]/(2*tau_comb)
                    result[...,:self.nv,self.nv:] -= rho[...,:self.nv,self.nv:]/(2*tau_comb)
            self.betamuh0 = betamuh
            
        if np.isfinite(self.tau_recomb):
            fk = torch.einsum("...bb -> ...b",rho).real
            betamuk,fk_eq = self.find_mu(
                fk, self.exp_betaE, self.betamuk0, sum_rule="...kb -> ...k", reshape=fk.shape[:-1]+(1,)
            )
            result -= (rho -  torch.diag_embed(fk_eq))/(2*self.tau_recomb)
            self.betamuk0 = betamuk

        return result
    
    def find_mu(self, f: torch.Tensor, exp_betaE: torch.Tensor, betamu0: torch.Tensor, sum_rule: str, reshape: tuple = (1,)) -> tuple[float, torch.Tensor]:
        f_total = torch.einsum(sum_rule, f).reshape(reshape)
        betamu = betamu0
        distribution = self.fermi_dirac(exp_betaE,betamu)
        F = torch.einsum(sum_rule, distribution).reshape(reshape) - f_total
        while torch.max(torch.abs(F)) > self.eps:
            exp_beta_Emu = exp_betaE/torch.exp(betamu)
            dF = torch.einsum(sum_rule, exp_beta_Emu/(exp_beta_Emu + 1)**2).reshape(reshape)
            dbetamu = F / dF
            limit_ind = (torch.abs(dbetamu) > self.max_dbetamu) # indices that need to be limited by self.max_dbetamu
            dbetamu[limit_ind] = torch.sign(F[limit_ind]) * self.max_dbetamu
            print(betamu.shape, dbetamu.shape)
            betamu -= dbetamu
            distribution = self.fermi_dirac(exp_betaE,betamu)
            F = torch.einsum(sum_rule, distribution).reshape(reshape) - f_total
        return betamu, distribution
    
    def fermi_dirac(self, exp_betaE: torch.Tensor, betamu: torch.Tensor) -> torch.Tensor:
        expbetamu = torch.exp(betamu)
        return 1/(exp_betaE/expbetamu + 1)
