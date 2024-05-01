from __future__ import annotations
from typing import Union

import numpy as np
import torch

from qimpy import rc, log, TreeNode
from qimpy.io import Checkpoint, CheckpointPath, InvalidInputException
from qimpy.mpi import BufferView
from qimpy.math import ceildiv
from qimpy.profiler import stopwatch
from qimpy.transport import material
from .. import bose


class Lindblad(TreeNode):
    ab_initio: material.ab_initio.AbInitio
    P: torch.Tensor  #: P and Pbar operators stacked together
    rho_dot0: torch.Tensor  #: rho_dot(rho0) for detailed balance correction

    constant_params: dict[str, torch.Tensor]  #: constant values of parameters
    scale_factor: dict[int, torch.Tensor]  #: scale factors per patch

    @stopwatch
    def __init__(
        self,
        *,
        ab_initio: material.ab_initio.AbInitio,
        data_file: Checkpoint,
        scale_factor: float = 1.0,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ) -> None:
        """
        Initialize ab initio Lindbladian scattering.

        Parameters
        ----------
        scale_factor
            :yaml:`Overall scale factor for scattering rate.`
        """
        super().__init__()
        self.ab_initio = ab_initio
        if not bool(data_file.attrs["ePhEnabled"]):
            raise InvalidInputException("No e-ph scattering available in data file")

        log.info("Constructing P tensor")
        nk = ab_initio.k_division.n_tot
        ik_start = ab_initio.k_division.i_start
        ik_stop = ab_initio.k_division.i_stop
        nk_mine = ab_initio.nk_mine
        n_bands_sq = ab_initio.n_bands**2
        ph = ab_initio.packed_hermitian
        block_shape_flat = (-1, n_bands_sq, n_bands_sq)
        P_shape = (2, nk_mine * nk, n_bands_sq, n_bands_sq)
        P = torch.zeros(P_shape, dtype=torch.double, device=rc.device)
        prefactor = np.pi * ab_initio.wk

        def get_mine(ik) -> Union[torch.Tensor, slice, None]:
            """Utility to fetch efficient slices of relevant k-points."""
            if ab_initio.k_division.n_procs == 1:
                return slice(None)  # no split, so bypass search
            sel = torch.where(torch.logical_and(ik >= ik_start, ik < ik_stop))[0]
            if not len(sel):
                return None
            sel_start = sel[0].item()
            sel_stop = sel[-1].item() + 1
            if sel_stop - sel_start == len(sel):
                return slice(sel_start, sel_stop)  # contiguous
            return sel  # general selection

        def pack_real(einsum_path, G1, G2):
            """Pack Hermitian `einsum_path` combination of G1 and G2 to real"""
            out = torch.einsum(einsum_path, G1, G2).reshape(block_shape_flat)
            return torch.einsum("AB, kBC, CD -> kAD", ph.Rinv, out, ph.R).real

        # Operate in blocks to reduce working memory:
        cp_ikpair = data_file["ikpair"]
        n_pairs = cp_ikpair.shape[0]
        n_blocks = 100  # may want to set this from input later on
        block_size = ceildiv(n_pairs, n_blocks)
        block_lims = np.minimum(
            np.arange(0, n_pairs + block_size - 1, block_size), n_pairs
        )
        cp_omega_ph = data_file["omega_ph"]
        cp_G = data_file["G"]
        for block_start, block_stop in zip(block_lims[:-1], block_lims[1:]):
            # Read current slice of data:
            cur = slice(block_start, block_stop)
            ik, jk = torch.from_numpy(cp_ikpair[cur]).to(rc.device).T
            omega_ph = torch.from_numpy(cp_omega_ph[cur]).to(rc.device)
            G = torch.from_numpy(cp_G[cur]).to(rc.device)
            bose_occ = bose(omega_ph, ab_initio.T)[:, None, None]
            wm = prefactor * bose_occ
            wp = prefactor * (bose_occ + 1.0)

            # Contributions to dynamics of ik:
            if (sel := get_mine(ik)) is not None:
                i_pair = (ik[sel] - ik_start) * nk + jk[sel]
                Gcur = G[sel]
                Gsq = pack_real("kac, kbd -> kabcd", Gcur, Gcur.conj())
                P[0].index_add_(0, i_pair, wm[sel] * Gsq)  # P contribution
                P[1].index_add_(0, i_pair, wp[sel] * Gsq)  # Pbar contribution

            # Contributions to dynamics of jk:
            if (sel := get_mine(jk)) is not None:
                i_pair = (jk[sel] - ik_start) * nk + ik[sel]
                Gcur = G[sel]
                Gsq = pack_real("kca, kdb -> kabcd", Gcur.conj(), Gcur)
                P[0].index_add_(0, i_pair, wp[sel] * Gsq)  # P contribution
                P[1].index_add_(0, i_pair, wm[sel] * Gsq)  # Pbar contribution

        op_shape = (2, nk_mine * n_bands_sq, nk * n_bands_sq)
        self.P = P.unflatten(1, (nk_mine, nk)).swapaxes(2, 3).reshape(op_shape)

        # Finishing up ...
        self.P_eye = apply_batched(
            self.P, torch.tile(ab_initio.eye_bands[None], (nk, 1, 1))[..., None]
        )
        nnzP = ab_initio.comm.allreduce(torch.count_nonzero(self.P))
        ntotP = ab_initio.comm.allreduce(np.prod(self.P.shape))
        fill_percent_P = 100.0 * nnzP / ntotP
        log.info(f"P tensor fill fraction: {fill_percent_P:.1f}%")

        self.rho_dot0 = self._calculate(ph.unpack(ab_initio.rho0))
        self.constant_params = dict(
            scale_factor=torch.tensor(scale_factor, device=rc.device)
        )
        self.scale_factor = dict()

    def initialize_fields(self, params: dict[str, torch.Tensor], patch_id: int) -> None:
        self._initialize_fields(patch_id, **params)

    def _initialize_fields(self, patch_id: int, *, scale_factor: torch.Tensor) -> None:
        self.scale_factor[patch_id] = scale_factor[..., None, None, None]

    @stopwatch
    def rho_dot(self, rho: torch.Tensor, t: float, patch_id: int) -> torch.Tensor:
        """drho/dt due to scattering in Schrodinger picture.
        Input and output rho are in unpacked (complex Hermitian) form."""
        return self.scale_factor[patch_id] * (self._calculate(rho) - self.rho_dot0)

    def _calculate(self, rho: torch.Tensor) -> torch.Tensor:
        """Internal drho/dt calculation without detailed balance / scaling."""
        ab_initio = self.ab_initio
        ph = ab_initio.packed_hermitian
        eye = ab_initio.eye_bands
        rho_all = self._collectT(ph.pack(rho))  # packed, all k
        Prho_packed = apply_batched(self.P, rho_all)
        Prho_packed[1] -= self.P_eye[1]  # convert [1] to Pbar @ (rho - eye)
        Prho, minus_Prhobar = ph.unpack(Prho_packed)
        return (eye - rho) @ Prho + rho @ minus_Prhobar  # unpacked, my k only

    def _collectT(self, rho: torch.Tensor) -> torch.Tensor:
        """Collect rho from all MPI processes and transpose batch dimension.
        Batch dimension is put at end for efficient matrix multiplication."""
        ab_initio = self.ab_initio
        if ab_initio.comm.size == 1:
            return torch.einsum("...kab -> kab...", rho)
        nk = ab_initio.k_division.n_tot
        n_bands = ab_initio.n_bands
        batch_shape = rho.shape[:-3]
        n_batch = np.prod(batch_shape)
        sendbuf = rho.reshape(n_batch, -1).T.contiguous()
        recvbuf = torch.zeros(
            (n_batch, nk * n_bands * n_bands), dtype=rho.dtype, device=rc.device
        )
        mpi_type = rc.mpi_type[rho.dtype]
        recv_prev = ab_initio.k_division.n_prev * n_bands * n_bands * n_batch
        ab_initio.comm.Allgatherv(
            (BufferView(sendbuf), np.prod(rho.shape), 0, mpi_type),
            (BufferView(recvbuf), np.diff(recv_prev), recv_prev[:-1], mpi_type),
        )
        return recvbuf.reshape((nk, n_bands, n_bands) + batch_shape)


def apply_batched(P: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """Apply batched flattened-rho operator P on batched rho.
    Batch dimension is at end of input, and at beginning of output."""
    result = torch.einsum("ikK, K... -> i...k", P, rho.flatten(0, 2))
    return result.unflatten(-1, (-1,) + rho.shape[1:3])
