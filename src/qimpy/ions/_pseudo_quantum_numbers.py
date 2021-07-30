import torch
import numpy as np
from functools import lru_cache
from typing import Optional


class PseudoQuantumNumbers:
    """Quantum numbers of pseudo-atom in projector or orbital order."""
    __slots__ = ('n_tot', 'l_max', 'i_rf', 'n', 'l', 'm', 'i_lm')
    n_tot: int  #: total number of projectors / orbitals (accounting for m)
    l_max: int  #: maximum l of projectors / orbitals
    i_rf: torch.Tensor  #: index of radial function (across all l)
    n: torch.Tensor  #: pseudo-principal quantum number (for each l, 1-based)
    l: torch.Tensor  #: orbital angular momentum
    m: torch.Tensor  #: azimuthal quantum number
    i_lm: torch.Tensor  #: combined (l, m) index into solid harmonics array

    def __init__(self, l: torch.Tensor) -> None:
        """Initialize given `l` of each projector / orbital of a
        pseudopotential. The input corresponds to the distinct radial functions
        read from a pseudoptential, while the members of this class are for
        each projector function including spherical harmonics."""
        self.n_tot = int((2*l + 1).sum().item())
        self.l_max = l.max().item()
        self.n = torch.zeros(self.n_tot, dtype=torch.long, device=l.device)
        self.l = torch.zeros_like(self.n)
        self.m = torch.zeros_like(self.n)
        self.i_rf = torch.zeros_like(self.n)
        # Initialize one l at a time:
        i_start = 0
        n_l = [0] * int(self.l_max + 1)
        for i_rf, l_iter in enumerate(l):
            # Size of shell at current m:
            l_i = l_iter.item()
            n_l[l_i] += 1  # starts at 1 for first of each l
            m_count = 2*l_i + 1
            i_stop = i_start + m_count
            # Assign properties for this shell:
            self.i_rf[i_start:i_stop] = i_rf
            self.n[i_start:i_stop] = n_l[l_i]
            self.l[i_start:i_stop] = l_i
            self.m[i_start:i_stop] = torch.arange(-l_i, l_i+1, device=l.device)
            # To next shell:
            i_start = i_stop
        self.i_lm = self.l*(self.l + 1) + self.m

    def expand_matrix(self, D: torch.Tensor, n_spinor: int,
                      j: Optional[torch.Tensor]) -> torch.Tensor:
        """Expand matrix `D` from (n, l) basis to include m and s.
        Spin s is included explicitly only of n_spinor is 2.
        Additionally, if j is specified (relativistic pseudopotential),
        the expanded matrix includes spin-angle overlap factors;
        n_spinor must be 2 for this mode."""
        if j is None:
            # Non-relativistic pseuodopotential:
            D_nlm = D[self.i_rf][:, self.i_rf].contiguous()
            D_nlm[self.i_lm[None] != self.i_lm[:, None]] = 0.  # delta_{lm,lm'}
            if n_spinor == 1:
                return D_nlm
            else:  # n_spinor == 2
                # Repeat for spinor component:
                n_nlms = D_nlm.shape[0] * n_spinor
                D_nlms = torch.zeros((n_nlms, n_nlms), dtype=D_nlm.dtype,
                                     device=D_nlm.device)
                for i_spinor in range(n_spinor):
                    D_nlms[i_spinor::n_spinor, i_spinor::n_spinor] = D_nlm
                return D_nlms
        else:
            # Relativistic pseudopotential:
            # Compute indices into (l, j, m_l) basis:
            i_lms = 2*self.i_lm[:, None].tile((1, 2))
            i_lms[:, 1] += 1  # Index is now 2*(l(l+1) + m) + s
            l = self.l[:, None]
            i_ljm = i_lms + (2*(l ** 2) - 2)  # offset from previous l
            sel_plus = torch.where(j[self.i_rf] > self.l)[0]
            i_ljm[sel_plus] += 2 * (2*l[sel_plus]+1)  # offset from j = l-1/2
            # Collect matrix:
            i_rf = self.i_rf[:, None].tile((1, 2)).flatten()
            D_nljm = D[i_rf][:, i_rf].contiguous()
            # Apply spin-angle overlaps (akin to delta_{lm,lm'} above):
            i_ljm = i_ljm.flatten()
            f_lj_all = torch.from_numpy(get_Ylm_overlaps(self.l_max)
                                        ).to(D.device)  # spin-angle overlaps
            return D_nljm * f_lj_all[i_ljm][:, i_ljm]


@lru_cache
def get_Ylm_to_spin_angle(l: int) -> np.ndarray:
    """Create 2(2l+1) x 2(2l+1) matrix transforming from (m, s) to (j, mj).
    The (m, s) indices are along the first dimension, with inner s index.
    The (j, mj) indices along the second dimension have an inner mj index for
    j = l - 1/2 first (2l entries) and then for j = l + 1/2 (2l + 2 entries).
    """
    n_m = 2*l + 1
    n_minus = n_m - 1  # number of mj at j = l - 1/2
    n_plus = n_m + 1  # number of mj at j = l + 1/2

    # Initialize Clebsch Gordon coefficients:
    C = np.zeros((2, n_m, n_minus + n_plus))  # s, m, (j,mj)
    C_entries = np.sqrt(1. - np.arange(n_m)/n_m)  # unique entries at this l
    if n_minus:
        np.fill_diagonal(C[0, 0:, :n_minus], C_entries[1:])
        np.fill_diagonal(C[1, 1:, :n_minus], -C_entries[:0:-1])
    if n_plus:
        np.fill_diagonal(C[0, :, n_minus+1:], C_entries[::-1])
        np.fill_diagonal(C[1, :, n_minus+0:], C_entries)

    # Account for real to complex harmonics transformation:
    Y = np.zeros((n_m, n_m), dtype=np.complex128)
    Y[l, l] = 1.  # m = 0 not transformed
    m = np.arange(1, l+1)
    parity = (-1) ** m
    sqrt_half = np.sqrt(0.5)
    Y[l+m, l+m] = sqrt_half * parity
    Y[l+m, l-m] = sqrt_half
    Y[l-m, l+m] = sqrt_half * parity * 1j
    Y[l-m, l-m] = sqrt_half * (-1j)
    return (Y @ C  # transform Clebsh-Gordon to apply for real harmonics
            ).swapaxes(0, 1).reshape(n_m*2, n_m*2)  # s, m -> combined (m, s)


@lru_cache
def get_Ylm_overlaps(l_max: int) -> np.ndarray:
    """Compute spin-angle overlap matrix of Ylm up to `l_max`.
    Contains overlaps for (l=0, j=1/2), (l=1, j=1/2), (l=1, j=3/2) etc.,
    i.e. s, p-, p+, d-, d+ etc. in order in a block-diagonal matrix.
    Each square block dimension is 2(2l+1), for a total square matrix
    size of 4(l_max+1)^2 - 2."""
    C = get_Ylm_to_spin_angle(l_max)
    if l_max == 0:
        return C @ C.conj().T  # Only j = 1/2 for l = 0
    else:
        # Compute overlaps for j = l_max - 1/2 and l_max + 1/2 at l = l_max:
        C_minus = C[:, :2*l_max]
        C_plus = C[:, 2*l_max:]
        O_minus = C_minus @ C_minus.conj().T
        O_plus = C_plus @ C_plus.conj().T
        # Put together with results for previous l:
        C_prev = get_Ylm_overlaps(l_max - 1)
        i_minus = C_prev.shape[0]  # start index of j = l_max - 1/2
        i_plus = i_minus + O_minus.shape[0]  # start index of j = l_max + 1/2
        n_tot = i_plus + O_plus.shape[0]  # total entries including l_max
        C = np.zeros((n_tot, n_tot), dtype=np.complex128)
        C[:i_minus, :i_minus] = C_prev
        C[i_minus:i_plus, i_minus:i_plus] = O_minus
        C[i_plus:, i_plus:] = O_plus
        return C
