from __future__ import annotations
from typing import Callable
from functools import cache
import operator

import numpy as np
import torch
import torch.distributed as dist

from qimpy import rc
from qimpy.mpi import ProcessGrid, TaskDivision
from qimpy.profiler import stopwatch
from qimpy.io import CheckpointPath, CheckpointContext
from qimpy.transport.advect import Advect, N_GHOST
from . import Material


class FermiCircle(Material):
    """Fermi-circle model material suitable for graphene and 2DEGs."""

    kF: float  #: Fermi wave-vector
    vF: float  #: Fermi velocity
    tau_inv_p: float  #: Momentum relaxation rate
    tau_inv_ee: float  #: Electron internal scattering rate (momentum-conserving)
    theta0: float  #: Initial angle in fermi circle grid
    specularity: float  #: Specularity of reflection at all boundaries
    r_c: float  #: Cyclotron radius for extenral magnetic field (disabled if infinity)
    F_theta: float  #: Angular force in grid coordinates
    advect: torch.jit.ScriptModule  #: Momentum-space advection logic

    def __init__(
        self,
        *,
        kF: float,
        vF: float,
        n_theta: int,
        n_r: int = 1,
        tau_p: float = np.inf,
        tau_ee: float = np.inf,
        r_c: float = np.inf,
        theta0: float = 0.0,
        specularity: float = 1.0,
        process_grid: ProcessGrid,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ):
        """
        Initialize Fermi-circle model material.

        Parameters
        ----------
        kF
            :yaml:`Fermi wave vector in atomic units.`
        vF
            :yaml:`Fermi velocity in atomic units.`
        n_theta
            :yaml:`Number of k along Fermi circle.`
        theta0
            :yaml:`Angle of first k-point.`
        r_c
            :yaml:`Cyclotron radius, corresponding to external magnetic field`.
            If infinite, disable magentic field (default).
        specularity
            :yaml:`Specularity of reflection at all surfaces.`
            Should be in the range of 0 for fully diffuse scattering,
            to 1 for perfectly specular reflection.
        """
        super().__init__()
        self.kF = kF
        self.vF = vF
        self.r_c = r_c
        self.tau_inv_p = 1.0 / tau_p
        self.tau_inv_ee = 1.0 / tau_ee
        self.initialize(
            wk=torch.tensor([1.0 / n_theta], device=rc.device),
            nk=n_theta,
            n_bands=1,
            n_dim=2,
            process_grid=process_grid,
        )
        assert n_r == 1  # TODO: merge n_r support from FermiSurface

        # Create theta grid on Fermi circle:
        dtheta = 2 * np.pi / n_theta
        theta = theta0 + torch.arange(n_theta, device=rc.device) * dtheta
        k_hat = torch.stack([theta.cos(), theta.sin()], dim=-1)
        self.v_all = k_hat[:, None] * vF
        self.k[:] = k_hat[self.k_mine] * kF
        self.v[:] = self.v_all[self.k_mine]
        self.theta0 = theta0
        self.specularity = specularity

        # Cached normalizations for collision intergal
        self.nk_inv = 1.0 / n_theta
        self.vv_inv = torch.linalg.inv(
            torch.einsum("...i, ...j -> ij", self.v_all, self.v_all)
        )

        # Initialize F*drho/dk calculator, if needed:
        if np.isfinite(r_c):
            self.F_theta = self.vF / (self.r_c * dtheta)
            self.dt_max = 0.5 / abs(self.F_theta)
            self.advect = torch.jit.script(Advect())
            self.riemann_mask = torch.tensor(
                (1.0, 0.0) if self.F_theta > 0 else (0.0, 1.0), device=rc.device
            ).view(
                1, 1, 1, -1
            )  # faster constant-F equivalent to get_riemann_mask
        else:
            self.F_theta = 0.0

    def _save_checkpoint(
        self, cp_path: CheckpointPath, context: CheckpointContext
    ) -> list[str]:
        attrs = cp_path.attrs
        attrs["kF"] = self.kF
        attrs["vF"] = self.vF
        attrs["n_theta"] = self.k_division.n_tot
        attrs["tau_p"] = (1.0 / self.tau_inv_p) if self.tau_inv_p else np.inf
        attrs["tau_ee"] = (1.0 / self.tau_inv_ee) if self.tau_inv_ee else np.inf
        attrs["r_c"] = self.r_c
        attrs["theta0"] = self.theta0
        attrs["specularity"] = self.specularity
        return list(attrs.keys())

    def initialize_fields(
        self, rho: torch.Tensor, params: dict[str, torch.Tensor], patch_id: int
    ) -> None:
        pass  # No spatially-varying / parameter sweep fields yet

    def get_contactor(
        self, n: torch.Tensor, **kwargs
    ) -> Callable[[float], torch.Tensor]:
        return Contactor(self, n, **kwargs)

    def get_reflector(self, n: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
        v = self.v_all[:, 0]
        return SpecularReflector(
            n, v, self.theta0, self.group, self.k_division, self.specularity
        )

    @stopwatch
    def rho_dot(self, rho: torch.Tensor, t: float, patch_id: int) -> torch.Tensor:
        result = self.rho_dot_collisions(rho)
        if self.F_theta:
            # k-space advection due to magnetic fields:
            result += self.accumulate_edges(
                *self.advect(
                    self.pad_ghost(rho) * self.F_theta,
                    self.riemann_mask,
                    -1,
                    True,
                    (True, True),
                )
            )
        return result

    def pad_ghost(self, rho: torch.Tensor) -> torch.Tensor:
        """Pad by ghost zones for monetum-space advection."""
        assert self.nk_mine >= 1
        rhoL, rhoR = ring_exchange(self.group, rho[..., 0], rho[..., -1])
        rho_padded = torch.nn.functional.pad(rho, (N_GHOST, N_GHOST))
        rho_padded[..., N_GHOST - 1] = rhoL
        rho_padded[..., -N_GHOST] = rhoR
        return rho_padded

    def accumulate_edges(
        self, out: torch.Tensor, outL: torch.Tensor, outR: torch.Tensor
    ) -> torch.Tensor:
        """Accumulate momentum-space advection edge contributions `outL` and `outR`
        into `out`, and return `out` for convenience."""
        outL, outR = ring_exchange(self.group, outL, outR)
        out[..., 0] += outL
        out[..., -1] += outR
        return out

    def rho_dot_collisions(self, rho: torch.Tensor) -> torch.Tensor:
        if not (self.tau_inv_p or self.tau_inv_ee):
            return torch.zeros_like(rho)  # no scattering

        # Compute stationary carrier density:
        rho_sum = rho.sum(dim=-1).contiguous()
        if self.group.size() > 1:
            dist.all_reduce(rho_sum, group=self.group)
        rho_0 = self.nk_inv * rho_sum[..., None]
        result = (rho_0 - rho) * (self.tau_inv_p + self.tau_inv_ee)

        # Compute moving equlibrium carrier density if needed:
        if self.tau_inv_ee:
            v = self.transport_velocity
            rho_v_sum = torch.einsum("...k, ki -> ...i", rho, v).contiguous()
            if self.group.size() > 1:
                dist.all_reduce(rho_v_sum, group=self.group)
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

    def get_flux_indices(self) -> list[int]:
        return [0]


class Contactor:
    rho_contact: torch.Tensor  #: Cached constant contact distribution

    def __init__(
        self, fc: FermiCircle, n: torch.Tensor, *, dmu: float = 0.0, vD: float = 0.0
    ) -> None:
        """Return contact distribution function for specified chemical potential
        shift and drift velocity. Note that positive vD corresponds to current
        flowing into the device (along -n), while negative vD flows out (along +n)."""
        v_hat = fc.transport_velocity / fc.vF
        self.rho_contact = dmu - (n @ v_hat.T) * (vD / fc.vF)  # TODO: check

    def __call__(self, t):
        # TODO: add time dependence options
        return self.rho_contact


class SpecularReflector:
    """Reflect velocities specularly i.e. with reflection angle = incidence angle.
    Optionally, account for diffuse scattering by replacing portion of the flux with a
    uniform distribution at all angles (matching total incoming and outgoing flux)."""

    def __init__(
        self,
        n: torch.Tensor,
        v: torch.Tensor,
        theta0: float,
        group: dist.ProcessGroup,
        k_division: TaskDivision,
        specularity: float,
    ) -> None:
        # Find which theta reflects into the first one:
        angle_n = torch.atan2(n[:, 1], n[:, 0])
        dtheta0_reflected = 2 * (angle_n - theta0) + np.pi  # deflection of theta0
        n_theta = k_division.n_tot
        d_theta = 2 * np.pi / n_theta
        i0_reflected = (dtheta0_reflected / d_theta).floor().to(torch.long) % n_theta
        dist.broadcast(i0_reflected, group=group, group_src=0)  # ensure consistency

        # Map the rest correspondingly:
        nk = k_division.n_tot
        i0 = (i0_reflected[:, None] - torch.arange(n_theta, device=rc.device)) % n_theta
        i1 = (i0 + 1) % n_theta
        v_n_0 = (n[:, None] * v[i0]).sum(dim=-1)
        v_n_1 = (n[:, None] * v[i1]).sum(dim=-1)
        v_n_in = (n[:, None] * v).sum(dim=-1)
        sel = torch.where(v_n_in.flatten() > 0)[0]
        w0 = ((v_n_in + v_n_1) / (v_n_1 - v_n_0)).flatten()[sel]
        w1 = 1 - w0
        if len(outwards0 := torch.where(v_n_0.flatten()[sel] > 0)[0]):
            w0[outwards0] = 0.0
            w1[outwards0] = -(v_n_in / v_n_1).flatten()[sel][outwards0]
        if len(outwards1 := torch.where(v_n_1.flatten()[sel] > 0)[0]):
            w0[outwards1] = -(v_n_in / v_n_0).flatten()[sel][outwards1]
            w1[outwards1] = 0.0

        # Compute flattened index on Nr x Nk_mine for efficient indexing
        self.k_division = k_division
        self.group = group
        nr = len(n)
        ir = torch.arange(nr, device=rc.device)[:, None]
        i_proc = group.rank()
        n_procs = group.size()
        if n_procs == 1:
            self.specular_matrix = torch.sparse_coo_tensor(
                indices=torch.stack(
                    (
                        torch.cat(
                            (
                                (ir * nk + i0).flatten()[sel],
                                (ir * nk + i1).flatten()[sel],
                            )
                        ),
                        torch.cat((sel, sel)),
                    )
                ),
                values=torch.cat((w0, w1)),
                size=(nr * n_theta, nr * n_theta),
            ).to_sparse_csr()
            self.i_reflected_flat = (ir * nk + i0).flatten()
        else:
            # Determine how to receive data:
            i_reflected_cur = i0[:, k_division.mine]
            j_proc = i_reflected_cur.flatten() // k_division.n_each
            self.recv_counts, self.recv_offsets = get_counts_offsets(j_proc, n_procs)
            self.recv_index = invert_index(j_proc.argsort(stable=True))
            # Determine how to send data:
            send_sel = torch.where(i0 // k_division.n_each == i_proc)
            send_sel_index_mine = send_sel[0] * k_division.n_mine + (
                i0[send_sel] - k_division.i_start
            )
            j_proc = send_sel[1] // k_division.n_each
            self.send_counts, self.send_offsets = get_counts_offsets(j_proc, n_procs)
            self.send_index = send_sel_index_mine[j_proc.argsort(stable=True)]

        # Prepare input and output projectors for diffuse scattering
        self.specularity = specularity
        if specularity != 1.0:
            # Select outgoing and incoming states at each boundary point:
            flux_weight = torch.einsum("ri, ki -> rk", n, v)
            flux_weight_out = torch.where(flux_weight > 0.0, flux_weight, 0.0)
            flux_weight_in = torch.where(flux_weight < 0.0, -flux_weight, 0.0)
            norm_factor = (1.0 - specularity) / flux_weight_in.sum(dim=-1, keepdim=True)
            self.diffuse_weight = flux_weight_out[:, k_division.mine] * norm_factor

    def __call__(self, rho: torch.Tensor) -> torch.Tensor:
        rho_flat = rho.flatten()
        group = self.group
        if group.size() == 1:
            out_flat = self.specular_matrix @ rho_flat
            # out_flat = rho_flat[self.i_reflected_flat]
        else:
            nr, _ = rho.shape  # same on all processes
            send_buf = rho_flat.T[self.send_index].contiguous()
            recv_buf = torch.empty_like(send_buf)
            dist.all_to_all_single(
                recv_buf.view(-1),
                send_buf.view(-1),
                self.recv_counts,
                self.send_counts,
                group=group,
            )
            out_flat = recv_buf[self.recv_index].T
        out = out_flat.reshape(rho.shape)

        # Optionally account for diffuse contributions:
        if self.specularity != 1.0:
            out *= self.specularity
            # Add rho such that incoming flux matches outgoing flux:
            weighted_flux = (rho * self.diffuse_weight).sum(dim=-1).contiguous()
            if group.size() > 1:
                dist.all_reduce(weighted_flux, group=group)
            out += weighted_flux[:, None]
        return out


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


def ring_exchange(
    group: dist.ProcessGroup, buf_l: torch.Tensor, buf_r: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Send `buf_l` and `buf_r` to left and right neighbors of `group`, taken as a ring.
    Return the results received from the left and right neighbors respectively."""
    n_procs = group.size()
    if n_procs == 1:
        return buf_r, buf_l
    else:
        rank = group.rank()
        rank_l = (rank - 1) % n_procs  # rank to the "left" on ring
        rank_r = (rank + 1) % n_procs  # rank to the "right" on ring
        send_buf_l = buf_l.contiguous()
        send_buf_r = buf_r.contiguous()
        recv_buf_r = torch.zeros_like(send_buf_l)
        recv_buf_l = torch.zeros_like(send_buf_r)
        requests = []
        for direction in (operator.lt, operator.gt):
            # Post all low to hi group-rank send/recvs first, and then all hi to lo ones
            if direction(rank_r, rank):
                requests.append(dist.irecv(recv_buf_r, group=group, group_src=rank_r))
            if direction(rank, rank_l):
                requests.append(dist.isend(send_buf_l, group=group, group_dst=rank_l))
            if direction(rank_l, rank):
                requests.append(dist.irecv(recv_buf_l, group=group, group_src=rank_l))
            if direction(rank, rank_r):
                requests.append(dist.isend(send_buf_r, group=group, group_dst=rank_r))
        for request in requests:
            request.wait()
        return recv_buf_l, recv_buf_r
