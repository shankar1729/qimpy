from __future__ import annotations
from typing import Sequence, Callable, Union, Optional
from functools import cache

import torch
import numpy as np

from qimpy import log, rc
from qimpy.mpi import TaskDivision, BufferView
from qimpy.math import ceildiv
from qimpy.profiler import StopWatch, stopwatch
from qimpy.io import Checkpoint, CheckpointPath, Unit
from qimpy.mpi import ProcessGrid
from . import Material
from . import PackedHermitian


def fermi(E, mu, T):
    return torch.special.expit((mu - E) / T)


def bose(omegaPh, T):
    return 1 / torch.expm1(omegaPh / T)


def apply_batched(P: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """Apply batched flattened-rho operator P on batched rho.
    Batch dimension is at end of input, and at beginning of output."""
    result = (P @ rho.flatten(0, 2)).swapaxes(-2, -1)
    return result.unflatten(-1, (-1,) + rho.shape[1:3])


class AbInitio(Material):
    """Ab initio material specification."""

    T: float
    mu: float
    nk: int
    S: torch.Tensor
    L: Optional[torch.Tensor]
    P: torch.Tensor  # P and Pbar operators stacked together
    rho0: torch.Tensor  # Equilibrium density matrix
    rho: torch.Tensor  # Current density matrix

    def __init__(
        self,
        *,
        fname: str,
        mu: float = 0.0,
        eph_scatt: bool = True,
        rotation: Sequence[Sequence[float]] = (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        process_grid: ProcessGrid,
        checkpoint_in: CheckpointPath = CheckpointPath(),
    ):
        """
        Initialize ab initio material.

        Parameters
        ----------
        fname
            :yaml:`File name to load materials data from.`
        rotation
            :yaml:`3 x 3 rotation matrix from material to simulation frame.`
        """
        self.comm = process_grid.get_comm("k")
        self.mu = mu
        self.eph_scatt = eph_scatt
        watch = StopWatch("Dynamics.read_checkpoint")
        with Checkpoint(fname) as checkpoint:
            attrs = checkpoint.attrs
            self.T = float(attrs["Tmax"])
            nk = int(attrs["nk"])
            self.k_division = TaskDivision(
                n_tot=nk, i_proc=self.comm.rank, n_procs=self.comm.size
            )
            wk = 1 / float(attrs["nkTot"])
            k_mine = slice(self.k_division.i_start, self.k_division.i_stop)
            self.k = torch.from_numpy(checkpoint["k"][k_mine]).to(rc.device)
            self.E = torch.from_numpy(checkpoint["E"][k_mine]).to(rc.device)
            P = torch.from_numpy(checkpoint["P"][k_mine]).to(rc.device)
            spinorial = bool(attrs["spinorial"])
            self.S = (
                torch.from_numpy(checkpoint["S"][k_mine]).to(rc.device)
                if spinorial
                else None
            )
            haveL = bool(attrs["haveL"])
            self.L = (
                torch.from_numpy(checkpoint["L"][k_mine]).to(rc.device)
                if haveL
                else None
            )
            ePhEnabled = bool(attrs["ePhEnabled"])
            watch.stop()

        n_bands = self.E.shape[-1]
        self.eye_bands = torch.eye(n_bands, device=rc.device)
        self.packed_hermitian = PackedHermitian(n_bands)

        super().__init__(
            wk=wk,
            nk=nk,
            n_bands=n_bands,
            n_dim=3,
            checkpoint_in=checkpoint_in,
            process_grid=process_grid,
        )

        self.v = torch.einsum("kibb->kbi", P).real
        # Zeroth order Hamiltonian:
        H0 = torch.diag_embed(self.E) + self.zeemanH(
            torch.tensor([[0.0, 0.0, 0.0]]).to(rc.device)
        )
        self.rho0, _, _ = self.rho_fermi(H0, self.mu)
        # Construct P operators from matrix elements in checkpoint:
        if eph_scatt and ePhEnabled:
            self.P = self.constructP(checkpoint)
            self.P_eye = apply_batched(
                self.P, torch.tile(self.eye_bands[None], (nk, 1, 1))[..., None]
            )
            nnzP = self.comm.allreduce(torch.count_nonzero(self.P))
            ntotP = self.comm.allreduce(np.prod(self.P.shape))
            fill_percent_P = 100.0 * nnzP / ntotP
            log.info(f"P tensor fill fraction: {fill_percent_P:.1f}%")
            self.rho_dot_scatter0 = self.rho_dot_scatter(
                self.packed_hermitian.unpack(self.rho0)
            )  # for detailed balance correction

    @stopwatch
    def constructP(self, checkpoint: Checkpoint, n_blocks: int = 100) -> torch.Tensor:
        log.info("Constructing P tensor")
        nk = self.nk
        ik_start = self.k_division.i_start
        ik_stop = self.k_division.i_stop
        nk_mine = ik_stop - ik_start
        n_bands_sq = self.n_bands**2
        ph = self.packed_hermitian
        block_shape_flat = (-1, n_bands_sq, n_bands_sq)
        P_shape = (2, nk_mine * nk, n_bands_sq, n_bands_sq)
        P = torch.zeros(P_shape, dtype=torch.double, device=rc.device)
        prefactor = np.pi * self.wk

        def get_mine(ik) -> Union[torch.Tensor, slice, None]:
            """Utility to fetch efficient slices of relevant k-points."""
            if self.k_division.n_procs == 1:
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
        cp_ikpair = checkpoint["ikpair"]
        n_pairs = cp_ikpair.shape[0]
        block_size = ceildiv(n_pairs, n_blocks)
        block_lims = np.minimum(
            np.arange(0, n_pairs + block_size - 1, block_size), n_pairs
        )
        cp_omega_ph = checkpoint["omega_ph"]
        cp_G = checkpoint["G"]
        for block_start, block_stop in zip(block_lims[:-1], block_lims[1:]):
            # Read current slice of data:
            cur = slice(block_start, block_stop)
            ik, jk = torch.from_numpy(cp_ikpair[cur]).to(rc.device).T
            omega_ph = torch.from_numpy(cp_omega_ph[cur]).to(rc.device)
            G = torch.from_numpy(cp_G[cur]).to(rc.device)
            bose_occ = bose(omega_ph, self.T)[:, None, None]
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
        return P.unflatten(1, (nk_mine, nk)).swapaxes(2, 3).reshape(op_shape)

    def collectT(self, rho: torch.Tensor) -> torch.Tensor:
        """Collect rho from all MPI processes and transpose batch dimension.
        Batch dimension is put at end for efficient matrix multiplication."""
        if self.comm.size == 1:
            return rho.permute(1, 2, 3, 0)
        n_bands = self.n_bands
        n_batch = rho.shape[0]
        sendbuf = rho.reshape(n_batch, -1).T.contiguous()
        recvbuf = torch.zeros(
            (n_batch, self.nk * n_bands * n_bands), dtype=rho.dtype, device=rc.device
        )
        mpi_type = rc.mpi_type[rho.dtype]
        recv_prev = self.k_division.n_prev * n_bands * n_bands * n_batch
        self.comm.Allgatherv(
            (BufferView(sendbuf), np.prod(rho.shape), 0, mpi_type),
            (BufferView(recvbuf), np.diff(recv_prev), recv_prev[:-1], mpi_type),
        )
        return recvbuf.reshape(self.nk, n_bands, n_bands, n_batch)

    def rho_dot_scatter(self, rho: torch.Tensor) -> torch.Tensor:
        """drho/dt due to scattering in Schrodinger picture.
        Input and output rho are in unpacked (complex Hermitian) form."""
        ph = self.packed_hermitian
        eye = self.eye_bands
        rho_all = self.collectT(ph.pack(rho))  # packed, all k
        Prho_packed = apply_batched(self.P, rho_all)
        Prho_packed[1] -= self.P_eye[1]  # convert [1] to Pbar @ (rho - eye)
        Prho, minus_Prhobar = ph.unpack(Prho_packed)
        return (eye - rho) @ Prho + rho @ minus_Prhobar  # unpacked, my k only

    def schrodingerV(self, t: float) -> torch.Tensor:
        """Compute unitary rotations from interaction to Schrodinger picture."""
        phase = torch.exp((-1j * t) * self.E)
        return torch.einsum("ka, kb -> kab", phase, phase.conj())

    def zeemanH(self, B: torch.Tensor) -> torch.Tensor:
        """Get Zeeman Hamiltonian due to specified external magnetic fields."""
        g_e = Unit.MAP["g_e"]  # spin gyromagnetic ratio
        muB_B = (B * Unit.MAP["mu_B"]).to(self.S.dtype)
        H = torch.einsum("...i, kiab -> ...kab", muB_B * g_e * 0.5, self.S)
        if self.L is not None:
            H += torch.einsum("...i, kiab -> ...kab", muB_B, self.L)
        return H

    def rho_fermi(
        self, H: torch.Tensor, mu: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the equilibrium density matrix corresponding to `H`.
        Also return the energies and eigenvectors of `H`."""
        E, V = torch.linalg.eigh(H)
        f = fermi(E, mu, self.T)
        rho = torch.einsum("...ab, ...b, ...cb -> ...ac", V, f, V.conj())
        return self.packed_hermitian.pack(rho), E, V

    def get_contact_distribution(
        self, n: torch.Tensor, **kwargs
    ) -> Callable[[float], torch.Tensor]:
        return Contactor(self, n, **kwargs)

    def get_reflector(
        self, n: torch.Tensor
    ) -> Callable[[torch.Tensor], torch.Tensor]:  # absorbing boundary
        return torch.zeros_like

    @stopwatch
    def rho_dot(self, rho: torch.Tensor, t: float) -> torch.Tensor:
        """Overall drho/dt in interaction picture.
        Input and output rho are in packed (real) form."""
        if not self.eph_scatt:
            return torch.zeros_like(rho)
        ik_start = self.k_division.i_start
        ik_stop = self.k_division.i_stop
        nk_mine = ik_stop - ik_start
        n_spatial_1 = rho.shape[0]
        n_spatial_2 = rho.shape[1]
        n_spatial = np.prod(rho.shape[:2])
        nkbb = np.prod(rho.shape[2:])
        rho = rho.view(n_spatial, nk_mine, self.n_bands, self.n_bands)
        # Compute scattering in Schrodinger picture:
        ph = self.packed_hermitian
        phase = self.schrodingerV(t)
        rho_I = ph.unpack(rho)  # interaction picture, unpacked to complex
        rho_S = rho_I * phase
        rho_dot_S = self.rho_dot_scatter(rho_S)
        rho_dot_I = rho_dot_S * phase.conj()
        return (ph.pack(rho_dot_I + rho_dot_I.conj().swapaxes(-1, -2))).view(
            n_spatial_1, n_spatial_2, nkbb
        )  # + h.c.

    def get_observable_names(self) -> list[str]:
        return ["q", "Sx", "Sy", "Sz"]  # charge, components of spin operator

    @cache
    def get_observables(self, Nkbb: int, t: float) -> torch.Tensor:
        q = torch.ones((1, Nkbb), device=rc.device)  # charge observable
        ph = self.packed_hermitian
        phase = self.schrodingerV(t)
        S_obs = self.S.swapaxes(0, 1)
        assert Nkbb == np.prod(S_obs.shape[1:])
        S_obs = S_obs.conj() * phase[None, :]  # complex conjugate then phase of rho
        S_obs_packed = ph.pack(S_obs)  # packed to real
        weight = torch.ones(self.n_bands, self.n_bands, device=rc.device) * 2.0
        # Multiply weight of 2 to off-diagonal only:
        S_obs_packed *= weight.fill_diagonal_(1.0)[None, None, :]
        S_obs_packed = torch.reshape(S_obs_packed, (3, Nkbb))
        return torch.cat((q, S_obs_packed), dim=0)


class Contactor:
    """Contact with fixed chemical potential and magnetic field."""

    ab_initio: AbInitio  #: Corresponding AbInitio instance
    rho0_S: torch.Tensor  #: Contact distribution fixed in Schrodinger picture

    def __init__(
        self,
        ab_initio: AbInitio,
        n: torch.Tensor,
        *,
        dmu: float = 0.0,
        Bfield: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.ab_initio = ab_initio
        # Zeroth order Hamiltonian including constant ext field:
        H0 = torch.diag_embed(ab_initio.E) + ab_initio.zeemanH(
            torch.tensor([Bfield]).to(rc.device)
        )
        self.rho0_S, _, _ = ab_initio.rho_fermi(H0, ab_initio.mu + dmu)

    def __call__(self, t: float) -> torch.Tensor:
        """Return interaction-picture contact distribution at time `t`."""
        ab_initio = self.ab_initio
        ph = ab_initio.packed_hermitian
        phase = ab_initio.schrodingerV(t)
        rho0_I = ph.pack(ph.unpack(self.rho0_S) * phase.conj)
        return torch.flatten(rho0_I)
