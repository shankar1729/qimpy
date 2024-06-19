from __future__ import annotations
from typing import Callable
from functools import cache

import numpy as np
import torch

from qimpy import rc, MPI
from qimpy.mpi import ProcessGrid, BufferView, TaskDivision
from qimpy.profiler import stopwatch
from qimpy.io import CheckpointPath, CheckpointContext
from . import Material


class FermiCircle(Material):
    """Fermi-circle representation suitable for graphene and 2DEGs."""

    kF: float  #: Fermi wave-vector
    vF: float  #: Fermi velocity
    tau_inv_p: float  #: Momentum relaxation rate
    tau_inv_ee: float  #: Electron internal scattering rate (momentum-conserving)

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
        self.tau_inv_p = 1.0 / tau_p
        self.tau_inv_ee = 1.0 / tau_ee
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
        self.v_all = k_hat[:, None] * vF
        self.k[:] = k_hat[self.k_mine] * kF
        self.v[:] = self.v_all[self.k_mine]
        self.theta0 = theta0

        # Cached normalizations for collision intergal
        self.nk_inv = 1.0 / N_theta
        self.vv_inv = torch.linalg.inv(
            torch.einsum("...i, ...j -> ij", self.v_all, self.v_all)
        )

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        attrs = cp_path.attrs
        attrs["kF"] = self.kF
        attrs["vF"] = self.vF
        attrs["N_theta"] = self.k_division.n_tot
        attrs["tau_p"] = (1.0 / self.tau_inv_p) if self.tau_inv_p else np.inf
        attrs["tau_ee"] = (1.0 / self.tau_inv_ee) if self.tau_inv_ee else np.inf
        attrs["theta0"] = self.theta0
        return list(attrs.keys())

    def initialize_fields(
        self, rho: torch.Tensor, params: dict[str, torch.Tensor], patch_id: int
    ) -> None:
        pass  # No spatially-varying / parameter sweep fields yet

    def get_contactor(
        self, n: torch.Tensor, *, dmu: float = 0.0, vD: float = 0.0
    ) -> Callable[[float], torch.Tensor]:
        """Return contact distribution function for specified chemical potential
        shift and drift velocity. Note that positive vD corresponds to current
        flowing into the device (along -n), while negative vD flows out (along +n)."""
        v_hat = self.transport_velocity / self.vF
        rho_contact = dmu - (n @ v_hat.T) * (vD / self.vF)  # TODO: check
        return lambda t: rho_contact  # TODO: add time-dependence options

    def get_reflector(self, n: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
        return SpecularReflector(n, self.v_all, self.comm, self.k_division)

    @stopwatch
    def rho_dot(self, rho: torch.Tensor, t: float, patch_id: int) -> torch.Tensor:
        if not (self.tau_inv_p or self.tau_inv_ee):
            return torch.zeros_like(rho)  # no scattering

        # Compute stationary carrier density:
        rho_sum = rho.sum(dim=-1)
        if self.comm.size > 1:
            self.comm.Allreduce(MPI.IN_PLACE, BufferView(rho_sum))
        rho_0 = self.nk_inv * rho_sum[..., None]
        result = (rho_0 - rho) * (self.tau_inv_p + self.tau_inv_ee)

        # Compute moving equlibrium carrier density if needed:
        if self.tau_inv_ee:
            v = self.transport_velocity
            rho_v_sum = torch.einsum("...k, ki -> ...i", rho, v)
            if self.comm.size > 1:
                self.comm.Allreduce(MPI.IN_PLACE, BufferView(rho_v_sum))
            rho_v = torch.einsum("...i, ij, kj -> ...k", rho_v_sum, self.vv_inv, v)
            result += rho_v * self.tau_inv_ee  # combines with rho_0 - rho above

        return result

    def get_observable_names(self) -> list[str]:
        return ["n", "jx", "jy"]  # density and fluxes

    @cache
    def get_observables(self, t: float) -> torch.Tensor:
        return torch.cat(
            (
                torch.ones((1, self.nk_mine), device=rc.device),
                self.transport_velocity.T,
            ),
            dim=0,
        )


class SpecularReflector:
    """Reflect velocities specularly i.e. with reflection angle = incidence angle."""

    def __init__(
        self, n: torch.Tensor, v: torch.Tensor, comm: MPI.Comm, k_division: TaskDivision
    ) -> None:
        assert v.shape[1] == 1  # only for single-band case
        # Find which theta reflects into the first one:
        v0_normal = torch.einsum("ri, rj, j -> ri", n, n, v[0, 0])
        v0_reflected = v[None, 0, 0] - 2 * v0_normal
        v0_diff = (v0_reflected[:, None] - v[None, :, 0]).norm(dim=-1)
        i0_reflected = v0_diff.argmin(dim=1)
        comm.Bcast(BufferView(i0_reflected))  # ensure indices consistent
        # Map the rest correspondingly:
        nk = k_division.n_tot
        i_reflected = (i0_reflected[:, None] - torch.arange(nk, device=rc.device)) % nk

        # Compute flattened index on Nr x Nk_mine for efficient indexing
        self.k_division = k_division
        self.comm = comm
        nr = len(n)
        ir = torch.arange(nr, device=rc.device)[:, None]
        if comm.size == 1:
            self.i_reflected_flat = (ir * nk + i_reflected).flatten()
        else:
            # Determine how to receive data:
            i_reflected_cur = i_reflected[:, k_division.i_start : k_division.i_stop]
            j_proc = i_reflected_cur.flatten() // k_division.n_each
            self.recv_counts, self.recv_offsets = get_counts_offsets(j_proc, comm.size)
            self.recv_index = invert_index(j_proc.argsort(stable=True))
            # Determine how to send data:
            send_sel = torch.where(i_reflected // k_division.n_each == comm.rank)
            send_sel_index_mine = send_sel[0] * k_division.n_mine + (
                i_reflected[send_sel] - k_division.i_start
            )
            j_proc = send_sel[1] // k_division.n_each
            self.send_counts, self.send_offsets = get_counts_offsets(j_proc, comm.size)
            self.send_index = send_sel_index_mine[j_proc.argsort(stable=True)]

    def __call__(self, rho: torch.Tensor) -> torch.Tensor:
        rho_flat = rho.flatten(1, 2)  # flatten r and k indices
        comm = self.comm
        if comm.size == 1:
            out_flat = rho_flat[:, self.i_reflected_flat]
        else:
            n_ghost, nr, _ = rho.shape  # same on all processes
            send_buf = rho_flat.T[self.send_index].contiguous()
            send_counts = self.send_counts * n_ghost
            send_offsets = self.send_offsets * n_ghost
            recv_counts = self.recv_counts * n_ghost
            recv_offsets = self.recv_offsets * n_ghost
            recv_buf = torch.empty_like(send_buf)
            mpi_type = rc.mpi_type[rho.dtype]
            comm.Alltoallv(
                (BufferView(send_buf), send_counts, send_offsets, mpi_type),
                (BufferView(recv_buf), recv_counts, recv_offsets, mpi_type),
            )
            out_flat = recv_buf[self.recv_index].T
        return out_flat.unflatten(1, rho.shape[1:])  # restore r and k


def get_counts_offsets(
    indices: torch.Tensor, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    counts = torch.bincount(indices, minlength=n_bins).to(rc.cpu).numpy()
    offsets = np.concatenate(([0], np.cumsum(counts[:-1])))
    return counts, offsets


def invert_index(indices: torch.Tensor) -> torch.Tensor:
    inv_indices = torch.empty_like(indices)
    inv_indices[indices] = torch.arange(len(indices), device=rc.device)
    return inv_indices
