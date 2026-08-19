import torch
import torch.distributed as dist
import numpy as np

from qimpy import rc, log
from qimpy.profiler import StopWatch, stopwatch
from qimpy.dft.electrons import Wavefunction
from qimpy.grid import Grid, FieldC
from qimpy.io import Checkpoint, CheckpointPath, log_config
from qimpy.dft import System
from qimpy.mpi import get_comm


def band_ifft(grid: Grid, coeff: torch.Tensor, fftq_index: torch.Tensor):
    n_spin, n_spinor, _ = coeff.shape
    Cb = torch.zeros(
        (int(np.prod(grid.shape)), n_spin, n_spinor),
        dtype=coeff.dtype,
        device=coeff.device,
    )
    Cb[fftq_index] = coeff.permute(2, 0, 1)
    fft_shape = (n_spin, n_spinor) + grid.shape
    return grid.ifft(Cb.permute(1, 2, 0).view(fft_shape))


@stopwatch
def ExactExchangeEval(system: System):
    EXX = 0.0
    e = system.electrons
    comm = get_comm(e.group)  # MPI communicator for scalar exchanges
    log.info("\nSetting up exact exchange:")
    kpoints = e.basis.kpoints
    k_div = kpoints.division
    Nk = kpoints.k.shape[0]
    f = e.fillings.f
    grid = e.basis.grid
    V = grid.lattice.volume
    G = grid.get_mesh("G")
    fft_index = e.basis.fft_index
    Rc = (3 * V * Nk / (4 * np.pi)) ** (1 / 3)
    prefac = -1 / (2 * Nk**2) * e.w_spin
    HC = preApplyHamiltonian(system).split_bands().wait()
    C = e.C.split_bands().wait()
    b_start = 0 if C.band_division is None else C.band_division.i_start
    n_spin, Nk_mine, n_bands_mine, n_spinor, n_basis = C.coeff.shape
    for j_process in range(e.group.size()):
        k_other_start = comm.bcast(k_div.i_start, j_process)
        k_other_stop = comm.bcast(k_div.i_stop, j_process)
        nk_other = k_other_stop - k_other_start
        b_other_start = comm.bcast(b_start, j_process)
        n_bands_other = comm.bcast(n_bands_mine, j_process)
        n_basis_other = comm.bcast(n_basis, j_process)
        if e.group.rank() == j_process:
            fft_index_other = fft_index
            C_other = C.coeff
            f2 = f
        else:
            fft_index_other = torch.empty(
                (nk_other, n_basis_other), dtype=fft_index.dtype, device=rc.device
            )
            C_other = torch.empty(
                (n_spin, nk_other, n_bands_other, n_spinor, n_basis_other),
                dtype=C.dtype,
                device=rc.device,
            )
            f2 = torch.empty(
                (n_spin, nk_other, n_bands_other), dtype=f.dtype, device=rc.device
            )
        dist.broadcast(fft_index_other, group=e.group, group_src=j_process)
        dist.broadcast(C_other, group=e.group, group_src=j_process)
        dist.broadcast(f2, group=e.group, group_src=j_process)
        for ik1 in range(k_div.i_start, k_div.i_stop):
            k1 = kpoints.k[ik1]
            ik1_mine = ik1 - k_div.i_start
            fft_k1_mine = fft_index[ik1_mine]
            for bk1 in range(n_bands_mine):
                Ipsi_k1 = band_ifft(grid, C.coeff[:, ik1_mine, bk1, :], fft_k1_mine)
                fbk1 = f[:, ik1_mine, bk1 + b_start]
                for ik2 in range(k_other_start, k_other_stop):
                    k2 = kpoints.k[ik2]
                    ik2_other = ik2 - k_other_start
                    kernel = ExchangeSpherical_calc(
                        Rc, (-k2 + k1 + G), grid.lattice.Gbasis
                    ).type(torch.complex128)
                    fft_k2_other = fft_index_other[ik2_other]
                    for bk2 in range(n_bands_other):
                        bk2_other = bk2 + b_other_start
                        Ipsi_k2 = band_ifft(
                            grid, C_other[:, ik2_other, bk2, :], fft_k2_other
                        )
                        In = (
                            torch.einsum("sSxyz, sSxyz->sxyz", Ipsi_k1, Ipsi_k2.conj())
                            / V
                        )
                        n_pair = FieldC(grid, data=In)
                        Kn = n_pair.convolve(kernel)
                        EXX += (
                            prefac * fbk1 * f2[:, ik2_other, bk2_other] * (n_pair ^ Kn)
                        ).item()
                        grad_Ipsi_k = (
                            ~FieldC(
                                grid,
                                data=prefac
                                * Nk
                                * Ipsi_k2
                                * f2[:, ik2_other, bk2_other]
                                * Kn.data[:, None],
                            )
                        ).data.flatten(-3)
                        HC.coeff[:, ik1_mine, bk1] += grad_Ipsi_k[
                            :, :, fft_index[ik1_mine]
                        ]
    EXX = comm.allreduce(EXX)
    HC = HC.split_basis().wait()
    print(f"Exchange energy: {EXX}, {torch.real(EXX[0])}")
    exx_jdftx = -2.1883797529475943  # Si
    # exx_jdftx = -3.8845413033404732 # Water
    print(f"EXX Ratio: {torch.real(EXX[0]) / exx_jdftx}")
    cp = Checkpoint(filename="eig-check.h5", writable=True)
    cp_path = CheckpointPath(cp)
    eig, _ = torch.linalg.eigh((e.C ^ HC).wait())
    e.fillings.write_band_scalars(cp_path.relative("eigs"), eig)
    cp.close()
    # np.savetxt("kpoints", kpoints.k)
    # np.savetxt("eigs", eig[0])


def preApplyHamiltonian(system: System) -> Wavefunction:
    system.electrons.xc._functionals.clear()
    system.electrons.update(system)
    HC = system.electrons.hamiltonian(system.electrons.C)
    return HC


def ExchangeSpherical_calc(Rc: float, k: torch.Tensor, Gbasis: torch.Tensor):
    kSq = ((k @ Gbasis.T) ** 2).sum(dim=-1)
    return 2 * np.pi * Rc**2 * (kSq.sqrt() * Rc / (2 * np.pi)).sinc() ** 2


def main():
    log_config()
    input_dict = dict(
        checkpoint="/home/mkelley/RPI/Software/QimPy/tests/exchange/timing/Si.h5"
    )
    # input_dict = dict(checkpoint="/home/mkelley/RPI/Software/QimPy/tests/exchange/water.h5")
    sys = System(process_grid_shape=[-1, -1, -1], **input_dict)
    # sys = System(process_grid_shape=[1, 2, 3], **input_dict)
    # pg = sys.process_grid
    # print(pg.n_procs, pg.dim_names, pg.shape)
    # exit()

    ExactExchangeEval(sys)
    StopWatch.print_stats()


if __name__ == "__main__":
    main()
