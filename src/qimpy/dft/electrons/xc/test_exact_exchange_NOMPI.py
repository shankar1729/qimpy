import torch
import numpy as np

from h5py import File

from qimpy import log
from qimpy.dft.electrons import Electrons, Wavefunction
from qimpy.grid import Grid, FieldC, FieldG, FieldR
from qimpy.io import Checkpoint
from qimpy.dft import System
from qimpy.rc import MPI
from qimpy.mpi import ProcessGrid

def band_ifft(grid: Grid, coeff: torch.Tensor, fftq_index: torch.Tensor):
    n_spin, n_spinor, _ = coeff.shape
    Cb = torch.zeros((int(np.prod(grid.shape)), n_spin, n_spinor),
                     dtype=coeff.dtype,
                     device=coeff.device
                     )
    Cb[fftq_index] = coeff.permute(2, 0, 1)
    fft_shape = (n_spin, n_spinor) + grid.shape
    return grid.ifft(Cb.permute(1, 2, 0).view(fft_shape))


def ExactExchangeEval(system: System):
    EXX = 0.0
    e = system.electrons
    sym_tol = system.symmetries.tolerance
    log.info("\nSetting up exact exchange:")
    kpoints = e.basis.kpoints
    Nk = kpoints.k.shape[0]
    f = e.fillings.f
    grid = e.basis.grid
    V = grid.lattice.volume
    G = grid.get_mesh("G")
    n_bands = e.C.coeff.shape[2]
    fft_index = e.basis.fft_index
    Rc = (3 * V * Nk / (4 * np.pi))**(1/3)
    prefac = -1 / (2 * Nk**2) * e.w_spin
    HC = preApplyHamiltonian(system)
    for ik, k in enumerate(kpoints.k):
        for bk in range(n_bands):
            Ipsik = band_ifft(grid, e.C.coeff[:, ik, bk,:], fft_index[ik])
            fbk = f[:, ik, bk]
            for iq, q in enumerate(kpoints.k):
                kernel = ExchangeSpherical_calc(Rc, (q - k + G), grid.lattice.Gbasis,
                                                sym_tol).type(torch.complex128)
                for bq in range(n_bands):
                    Ipsiq = band_ifft(grid, e.C.coeff[:, iq, bq,:], fft_index[iq])
                    In = torch.einsum("sSxyz, sSxyz->sxyz", Ipsik.conj(), Ipsiq) / V
                    n_pair = FieldC(grid, data=In)
                    Kn = n_pair.convolve(kernel)
                    EXX += prefac * fbk * f[:,iq, bq] * (n_pair ^ Kn)
                    grad_Ipsiq = ~FieldC(grid,data=prefac * fbk * Kn.data[:, None] * Ipsik *Nk) # *Nk ??
                    HC.coeff[:, iq, bq] += grad_Ipsiq.data.flatten(-3)[:, :, fft_index[iq]]
    print(f"Exchange energy: {EXX}, {torch.real(EXX[0])}")
    exx_jdftx = -2.1883797529475943 # Si
    #exx_jdftx = -3.8845413033404732 # Water
    print(f"EXX Ratio: {torch.real(EXX[0]) / exx_jdftx}")
    #eig, _ = torch.linalg.eigh((e.C ^ HC).wait())
    #print("EIGS:", eig)

def preApplyHamiltonian(system: System) -> Wavefunction:
    system.electrons.xc._functionals.clear()
    system.electrons.update(system)
    HC = system.electrons.hamiltonian(system.electrons.C)
    return HC

def ExchangeSpherical_calc(Rc: float, k: torch.Tensor, Gbasis: torch.Tensor, tol: float):
    kSq = ((k @ Gbasis.T)**2).sum(dim=-1)
    return 2 * np.pi * Rc**2 * (kSq.sqrt() * Rc / (2 * np.pi)).sinc() ** 2

input_dict = dict(checkpoint="/home/mkelley/RPI/Software/QimPy/tests/exchange/Si.h5")
#input_dict = dict(checkpoint="/home/mkelley/RPI/Software/QimPy/tests/exchange/water.h5")
sys = System(process_grid_shape=[-1, -1, -1], **input_dict)
ExactExchangeEval(sys)
