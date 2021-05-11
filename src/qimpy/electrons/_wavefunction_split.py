import qimpy as qp
import numpy as np
import torch


def _split_bands(self):
    """Return wavefunction split by bands, bringing all basis coefficients of
    each band together on some process. Note that the result may be a view if
    there is only one process, or if the wavefunction is already split by bands
    """
    if (self.basis.n_procs == 1) or self.band_division:
        return self  # already in required configuration
    watch = qp.utils.StopWatch('Wavefunction.split_bands', self.basis.rc)
    basis = self.basis

    # Bring band-dimension to outermost (so that send chunks are contiguous):
    # --- after this, dim order is (band, spin, k, spinor, basis)
    send_coeff = self.coeff.permute(2, 0, 1, 3, 4).contiguous()
    n_per_band = np.prod(send_coeff.shape[1:])

    # All-to-all MPI rearrangement:
    band_division = qp.utils.TaskDivision(self.coeff.shape[2],
                                          basis.n_procs, basis.i_proc)
    send_counts = np.diff(band_division.n_prev) * n_per_band
    send_offset = band_division.n_prev[:-1] * n_per_band
    recv_counts = band_division.n_mine * n_per_band
    recv_offset = np.arange(basis.n_procs) * recv_counts
    mpi_type = basis.rc.mpi_type[send_coeff.dtype]
    recv_coeff = torch.zeros(
        (basis.n_procs, band_division.n_mine) + send_coeff.shape[1:],
        dtype=send_coeff.dtype, device=send_coeff.device)
    basis.rc.comm_b.Alltoallv(
        (qp.utils.BufferView(send_coeff), send_counts, send_offset, mpi_type),
        (qp.utils.BufferView(recv_coeff), recv_counts, recv_offset, mpi_type))
    del send_coeff

    # Unscramble data to bring all basis for each band together:
    # --- before this data order is (proc, band, spin, k, spinor, basis)
    recv_coeff = recv_coeff.permute(2, 3, 1, 4, 0, 5).flatten(4, 5)
    watch.stop()
    return qp.electrons.Wavefunction(basis, coeff=recv_coeff,
                                     band_division=band_division)


def _split_basis(self):
    """Return wavefunction split by basis, bringing all bands of each basis
    coefficient together on some process. Note that the result may be a view if
    there is only one process, or if the wavefunction is already split by basis
    """
    if (self.basis.n_procs == 1) or (self.band_division is None):
        return self  # already in required configuration
    watch = qp.utils.StopWatch('Wavefunction.split_basis', self.basis.rc)
    basis = self.basis

    # Split basis dimension to proc and basis-each, bring proc dimension
    # outermost for contiguous send chunks and band dim right after
    # --- after this, dim order is (proc, band, spin, k, spinor, basis-each)
    send_coeff = self.coeff.view(self.coeff.shape[:-1]
                                 + (basis.n_procs, basis.n_each)).permute(
                                     4, 2, 0, 1, 3, 5).contiguous()
    n_per_band = np.prod(send_coeff.shape[2:])

    # All-to-all MPI rearrangement:
    band_division = self.band_division
    send_counts = band_division.n_mine * n_per_band
    send_offset = np.arange(basis.n_procs) * send_counts
    recv_counts = np.diff(band_division.n_prev) * n_per_band
    recv_offset = band_division.n_prev[:-1] * n_per_band
    mpi_type = basis.rc.mpi_type[send_coeff.dtype]
    recv_coeff = torch.zeros((band_division.n_tot,) + send_coeff.shape[2:],
                             dtype=send_coeff.dtype, device=send_coeff.device)
    basis.rc.comm_b.Alltoallv(
        (qp.utils.BufferView(send_coeff), send_counts, send_offset, mpi_type),
        (qp.utils.BufferView(recv_coeff), recv_counts, recv_offset, mpi_type))
    del send_coeff

    # Move band index into correct position (already together):
    # --- before this data order is (band, spin, k, spinor, basis)
    recv_coeff = recv_coeff.permute(1, 2, 0, 3, 4)
    watch.stop()
    return qp.electrons.Wavefunction(basis, coeff=recv_coeff)


# Test wavefunction splitting and randomization:
if __name__ == '__main__':

    # Create a system to test with:
    qp.utils.log_config()
    rc = qp.utils.RunConfig(process_grid=(1, 1, -1))  # ensure basis-split
    system = qp.System(
        rc=rc, lattice={'system': 'hexagonal', 'a': 5., 'c': 4.},
        electrons={'k-mesh': {'size': [4, 4, 5]}})

    qp.log.info('\n--- Checking Wavefunction.split* ---')
    n_bands = 17  # deliberately prime!
    b_random_start = 4
    b_random_stop = 11  # randpomize only a range of bands for testing

    # Test random wavefunction created basis-split to band-split and back:
    Cg = qp.electrons.Wavefunction(system.electrons.basis, n_bands=n_bands,
                                   randomize=True)
    Cgb = Cg.split_bands()
    Cgbg = Cgb.split_basis()
    qp.log.info('Norm(G): {:.3f}'.format(Cg.norm()))
    qp.log.info('Norm(G - G->B->G): {:.3e}'.format((Cg - Cgbg).norm()))

    # Test random wavefunction created band-split to basis-split and back:
    Cb = qp.electrons.Wavefunction(system.electrons.basis,
                                   band_division=Cgb.band_division,
                                   randomize=True)
    Cbg = Cb.split_basis()
    Cbgb = Cbg.split_bands()
    qp.log.info('Norm(B): {:.3f}'.format(Cb.norm()))
    qp.log.info('Norm(B - B->G->B): {:.3e}'.format((Cb - Cbgb).norm()))

    # Check equivalence of randomization across the two splits:
    qp.log.info('\n--- Checking Wavefunction.randomize ---')
    qp.log.info('Norm(G - B->G): {:.3e}'.format((Cg - Cbg).norm()))
    qp.log.info('Norm(B - G->B): {:.3e}'.format((Cb - Cgb).norm()))
    qp.utils.StopWatch.print_stats()
