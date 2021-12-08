from __future__ import annotations
import qimpy as qp
import numpy as np
import torch
from dataclasses import dataclass


def _split_bands(
    self: qp.electrons.Wavefunction,
) -> qp.utils.Waitable[qp.electrons.Wavefunction]:
    """Return wavefunction split by bands, bringing all basis coefficients of
    each band together on some process. Note that the result may be a view if
    there is only one process, or if the wavefunction is already split by bands
    """
    if (self.basis.division.n_procs == 1) or self.band_division:
        return qp.utils.Waitless(self)  # already in required configuration
    basis = self.basis

    # Bring band-dimension to outermost (so that send chunks are contiguous):
    # --- after this, dim order is (band, spin, k, spinor, basis)
    send_coeff = self.coeff.permute(2, 0, 1, 3, 4).contiguous()
    n_per_band = np.prod(send_coeff.shape[1:])

    # All-to-all MPI rearrangement:
    band_division = qp.utils.TaskDivision(
        n_tot=self.coeff.shape[2],
        n_procs=basis.division.n_procs,
        i_proc=basis.division.i_proc,
    )
    send_counts = np.diff(band_division.n_prev) * n_per_band
    send_offset = band_division.n_prev[:-1] * n_per_band
    recv_counts = band_division.n_mine * n_per_band
    recv_offset = np.arange(band_division.n_procs) * recv_counts
    mpi_type = basis.rc.mpi_type[send_coeff.dtype]
    recv_coeff = torch.zeros(
        (band_division.n_procs, band_division.n_mine) + send_coeff.shape[1:],
        dtype=send_coeff.dtype,
        device=send_coeff.device,
    )
    basis.rc.current_stream_synchronize()
    request = basis.rc.comm_b.Ialltoallv(
        (qp.utils.BufferView(send_coeff), send_counts, send_offset, mpi_type),
        (qp.utils.BufferView(recv_coeff), recv_counts, recv_offset, mpi_type),
    )
    return SplitBandsWait(request, send_coeff, recv_coeff, basis, band_division)


@dataclass
class SplitBandsWait:
    """`Waitable` object for the result of `Wavefunction.split_bands`."""

    request: qp.MPI.Request
    send_coeff: torch.Tensor
    recv_coeff: torch.Tensor
    basis: qp.electrons.Basis
    band_division: qp.utils.TaskDivision

    def wait(self) -> qp.electrons.Wavefunction:
        """Complete `Wavefunction.split_bands` after waiting on MPI transfers."""
        # Wait for MPI completion:
        self.request.Wait()
        del self.send_coeff
        # Unscramble data to bring all basis for each band together:
        # --- before this data order is (proc, band, spin, k, spinor, basis)
        result = qp.electrons.Wavefunction(
            self.basis,
            coeff=self.recv_coeff.permute(2, 3, 1, 4, 0, 5).flatten(4, 5),
            band_division=self.band_division,
        )
        del self.recv_coeff
        return result


def _split_basis(
    self: qp.electrons.Wavefunction,
) -> qp.utils.Waitable[qp.electrons.Wavefunction]:
    """Return wavefunction split by basis, bringing all bands of each basis
    coefficient together on some process. Note that the result may be a view if
    there is only one process, or if the wavefunction is already split by basis
    """
    if (self.basis.division.n_procs == 1) or (self.band_division is None):
        return qp.utils.Waitless(self)  # already in required configuration
    basis = self.basis

    # Split basis dimension to proc and basis-each, bring proc dimension
    # outermost for contiguous send chunks and band dim right after
    # --- after this, dim order is (proc, band, spin, k, spinor, basis-each)
    send_coeff = (
        self.coeff.view(
            self.coeff.shape[:-1] + (basis.division.n_procs, basis.division.n_each)
        )
        .permute(4, 2, 0, 1, 3, 5)
        .contiguous()
    )
    n_per_band = np.prod(send_coeff.shape[2:])

    # All-to-all MPI rearrangement:
    band_division = self.band_division
    send_counts = band_division.n_mine * n_per_band
    send_offset = np.arange(band_division.n_procs) * send_counts
    recv_counts = np.diff(band_division.n_prev) * n_per_band
    recv_offset = band_division.n_prev[:-1] * n_per_band
    mpi_type = basis.rc.mpi_type[send_coeff.dtype]
    recv_coeff = torch.zeros(
        (band_division.n_tot,) + send_coeff.shape[2:],
        dtype=send_coeff.dtype,
        device=send_coeff.device,
    )
    basis.rc.current_stream_synchronize()
    request = basis.rc.comm_b.Ialltoallv(
        (qp.utils.BufferView(send_coeff), send_counts, send_offset, mpi_type),
        (qp.utils.BufferView(recv_coeff), recv_counts, recv_offset, mpi_type),
    )
    return SplitBasisWait(request, send_coeff, recv_coeff, basis)


@dataclass
class SplitBasisWait:
    """`Waitable` object for the result of `Wavefunction.split_basis`."""

    request: qp.MPI.Request
    send_coeff: torch.Tensor
    recv_coeff: torch.Tensor
    basis: qp.electrons.Basis

    def wait(self) -> qp.electrons.Wavefunction:
        """Complete `Wavefunction.split_basis` after waiting on MPI transfers."""
        # Wait for MPI completion:
        self.request.Wait()
        del self.send_coeff
        # Move band index into correct position (already together):
        # --- before this data order is (band, spin, k, spinor, basis)
        result = qp.electrons.Wavefunction(
            self.basis, coeff=self.recv_coeff.permute(1, 2, 0, 3, 4)
        )
        del self.recv_coeff
        return result


# Test wavefunction splitting and randomization:
if __name__ == "__main__":

    # Create a system to test with:
    qp.utils.log_config()
    rc = qp.utils.RunConfig(process_grid=(1, 1, -1))  # ensure basis-split
    if rc.n_procs == 1:
        raise ValueError("Wavefunction split test requires > 1 processes.")
    system = qp.System(
        rc=rc,
        lattice={"system": "hexagonal", "a": 9.0, "c": 12.0},
        electrons={
            "k-mesh": {"size": [4, 4, 3]},
            "basis": {"real-wavefunctions": False},
        },
    )

    qp.log.info("\n--- Checking Wavefunction.split* ---")
    n_bands = 317  # deliberately prime!
    b_random_start = 8
    b_random_stop = 21  # randomize only a range of bands for testing
    n_repeat = 10  # number of times to repeat split for benchmarking

    # Test random wavefunction created basis-split to band-split and back:
    Cg = qp.electrons.Wavefunction(system.electrons.basis, n_bands=n_bands)
    Cg.randomize(b_start=b_random_start, b_stop=b_random_stop)
    rc.comm_b.Barrier()  # for mroe reliable timing below
    for i_repeat in range(n_repeat):
        Cgb = Cg.split_bands().wait()
        Cgbg = Cgb.split_basis().wait()
    qp.log.info(f"Norm(G): {Cg.norm():.3f}")
    qp.log.info(f"Norm(G - G->B->G): {(Cg - Cgbg).norm():.3e}")

    # Test random wavefunction created band-split to basis-split and back:
    Cb = qp.electrons.Wavefunction(
        system.electrons.basis, band_division=Cgb.band_division
    )
    Cb.randomize(b_start=b_random_start, b_stop=b_random_stop)
    rc.comm_b.Barrier()  # for more reliable timing below
    for i_repeat in range(n_repeat):
        Cbg = Cb.split_basis().wait()
        Cbgb = Cbg.split_bands().wait()
    qp.log.info(f"Norm(B): {Cb.norm():.3f}")
    qp.log.info(f"Norm(B - B->G->B): {(Cb - Cbgb).norm():.3e}")

    # Check equivalence of randomization across the two splits:
    qp.log.info("\n--- Checking Wavefunction.randomize ---")
    qp.log.info(f"Norm(G - B->G): {(Cg - Cbg).norm():.3e}")
    qp.log.info(f"Norm(B - G->B): {(Cb - Cgb).norm():.3e}")

    # Check norm and orthogonalization routines:
    qp.log.info("\n--- Checking Wavefunction.norm and overlaps ---")
    for i_repeat in range(n_repeat):
        Cg.randomize()
        Cg = Cg.orthonormalize()
    Cg_overlap = Cg.dot_O(Cg).wait()
    expected_overlap = torch.eye(Cg_overlap.shape[-1], device=Cg.coeff.device)[
        None, None
    ]
    ortho_err = (Cg_overlap - expected_overlap).norm().item()
    qp.log.info(f"Orthonormality error: {ortho_err:.3e}")
    qp.log.info("Norm(band)[selected]:\n" + rc.fmt(Cg.band_norm()[0, :, :5]))
    qp.log.info("Norm(ke)[selected]:\n" + rc.fmt(Cg.band_ke()[0, :, :5]))

    qp.utils.StopWatch.print_stats()
