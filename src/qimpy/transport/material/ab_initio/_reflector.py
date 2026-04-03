import torch
from qimpy.transport.material import fermi

class Reflector:
    def __init__(self,
                 ab_initio,
                 n):
        self.ab_initio = ab_initio
        self.n = n

    def __call__(self, rho: torch.Tensor) -> torch.Tensor:
        Nk = self.ab_initio.k.shape[0]
        wk = self.ab_initio.wk
        Nb = self.ab_initio.n_bands
        T = self.ab_initio.T
        E = self.ab_initio.E.flatten()
        f_total = torch.einsum("...kbb -> ...", rho.unflatten(-1, (Nk, Nb, Nb))).real
        mu = torch.ones_like(f_total) * self.ab_initio.mu
        TOL = 1E-12
        DMU_MAX = 3*T
        while True:
            f = fermi(E, mu[..., None], T)
            print(f)
            print(rho)
            f_total_err = f.sum(dim=-1) - f_total
            print(f_total_err)
            if f_total_err.abs().max() < TOL:
                break
            df_total_dmu = (f * (1 - f)).sum(dim=-1) / T
            mu += torch.clip(-f_total_err / df_total_dmu, min=-DMU_MAX, max=DMU_MAX)
        return torch.diag_embed(f).flatten(-3, -1)

