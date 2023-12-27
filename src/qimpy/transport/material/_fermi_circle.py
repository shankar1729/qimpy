from __future__ import annotations
from typing import Callable

import numpy as np
import torch

from qimpy import rc, MPI
from qimpy.mpi import ProcessGrid, BufferView, TaskDivision
from qimpy.profiler import stopwatch
from qimpy.io import CheckpointPath
from . import Material


class FermiCircle(Material):
    """Fermi-circle representation suitable for graphene and 2DEGs."""

    kF: float  #: Fermi wave-vector
    vF: float  #: Fermi velocity
    tau_p: float  #: Momentum relaxation time
    tau_ee: float  #: Electron internal scattering time (momentum-conserving)

    def __init__(
        self,
        *,
        kF: float,
        vF: float,
        N_theta: int,
        tau_p: float,
        tau_ee: float,
        theta0: float = 0.0,
        process_grid: ProcessGrid,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ):
        """
        Initialize ab initio material.

        Parameters
        ----------
        kF
            :yaml:`Fermi wave vector in atomic units.`
        vF
            :yaml:`Fermi velocity in atomic units.`
        N_theta
            :yaml:`Number of k along Fermi circle.`
        theta0
            :yaml:`Angle of first k-point.`
        """
        self.kF = kF
        self.vF = vF
        self.tau_p = tau_p
        self.tau_ee = tau_ee
        super().__init__(
            wk=1.0 / N_theta,
            nk=N_theta,
            n_bands=1,
            n_dim=2,
            checkpoint_in=checkpoint_in,
            process_grid=process_grid,
        )

        # Create theta grid on Fermi circle:
        dtheta = 2 * np.pi / N_theta
        theta = theta0 + torch.arange(N_theta, device=rc.device) * dtheta
        k_hat = torch.stack([theta.cos(), theta.sin()], dim=-1)
        self.k[:] = k_hat * kF
        self.v[:] = k_hat[:, None] * vF

        # Cached normalizations for collision intergal
        self.nk_inv = 1.0 / len(self.k)
        self.v_sq_inv = 1.0 / self.v.square().sum()

    def get_contact_distribution(
        self, n: torch.Tensor, *, dmu: float = 0.0, vD: float = 0.0
    ) -> torch.Tensor:
        """Return contact distribution function for specified chemical potential
        shift and drift velocity. Note that positive vD corresponds to current
        flowing into the device (along -n), while negative vD flows out (along +n)."""
        v_hat = self.transport_velocity / self.vF
        return dmu - (n @ v_hat.T) * (vD / self.vF)  # TODO: check

    def get_reflector(self, n: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
        return SpecularReflector(n, self.v, self.comm, self.k_division)

    @stopwatch
    def rho_dot(self, rho: torch.Tensor) -> torch.Tensor:
        # Compute stationary and moving equlibrium carrier densities:
        v = self.transport_velocity
        rho_sum = rho.sum(dim=-1)
        rho_v_sum = torch.einsum("...k, ki -> ...i", rho, v)
        if self.comm.size > 1:
            self.comm.Allreduce(MPI.IN_PLACE, BufferView(rho_sum))
            self.comm.Allreduce(MPI.IN_PLACE, BufferView(rho_v_sum))
        rho_0 = self.nk_inv * rho_sum[..., None]
        rho_v = self.v_sq_inv * torch.einsum("...i, ki -> ...k", rho_v_sum, v)
        return -(rho - rho_0) / self.tau_p - (rho - rho_0 - rho_v) / self.tau_ee


class SpecularReflector:
    """Reflect velocities specularly i.e. with reflection angle = incidence angle."""

    def __init__(
        self, n: torch.Tensor, v: torch.Tensor, comm: MPI.Comm, k_division: TaskDivision
    ) -> None:
        assert v.shape[1] == 1  # only for single-band case
        v_normal = torch.einsum(
            "ri, rk -> rki", n, torch.einsum("ri, ki -> rk", n, v[:, 0])
        )
        v_reflected = v[None, :, 0] - 2 * v_normal
        v_diff = (v_reflected[:, :, None] - v[None, None, :, 0]).norm(dim=-1)
        i_reflected = v_diff.argmin(dim=2)

        # Compute flattened index on Nr x Nk_mine for efficient indexing
        self.k_division = k_division
        self.comm = comm
        nr = len(n)
        if comm.size == 1:
            self.i_reflected_flat = (
                i_reflected + torch.arange(nr, device=rc.device)[:, None] * len(v)
            ).flatten()
        else:
            # Determine process mapping of reflection:
            self.index_pairs = []
            i_reflected_cur = i_reflected[:, k_division.i_start : k_division.i_stop]
            i_reflected_mine = i_reflected_cur % k_division.n_each
            i_proc_reflected = i_reflected_cur // k_division.n_each
            nk_proc = np.diff(k_division.n_prev)
            nk_i = nk_proc[k_division.i_proc]
            for j_proc, nk_j in enumerate(nk_proc):
                sel = torch.where(i_proc_reflected == j_proc)
                sel_i = sel[0] * nk_i + sel[1]  # on process i
                sel_j = sel[0] * nk_j + i_reflected_mine[sel]  # on process j
                self.index_pairs.append((sel_i, sel_j))

    def __call__(self, rho: torch.Tensor) -> torch.Tensor:
        rho_flat = rho.flatten(1, 2)  # flatten r and k indices
        comm = self.comm
        if comm.size == 1:
            out_flat = rho_flat[:, self.i_reflected_flat]
        else:
            out_flat = torch.empty_like(rho_flat)
            buf = rho_flat
            k_division = self.k_division
            n_procs = k_division.n_procs
            j_proc = k_division.i_proc
            i_proc_next = (k_division.i_proc + 1) % n_procs
            i_proc_prev = (k_division.i_proc - 1) % n_procs
            n_ghost, nr, _ = rho.shape  # same on all processes
            nk_proc = np.diff(k_division.n_prev)  # number of k differs
            for i_iter in range(n_procs):
                # Set data from buf which corresponds to slice on j_proc
                sel_i, sel_j = self.index_pairs[j_proc]
                out_flat[:, sel_i] = buf[:, sel_j]
                if i_iter + 1 == n_procs:
                    break  # only need n - 1 communications
                # Communication ring: move buf to next process
                j_next = (j_proc + 1) % n_procs
                buf_next = torch.empty(
                    (n_ghost, nr * nk_proc[j_next]), device=rc.device
                )
                MPI.Request.Waitall(
                    [
                        comm.Irecv(BufferView(buf_next), i_proc_next, tag=j_next),
                        comm.Isend(BufferView(buf), i_proc_prev, tag=j_proc),
                    ]
                )
                j_proc = j_next
                buf = buf_next
        return out_flat.unflatten(1, rho.shape[1:])  # restore r and k
