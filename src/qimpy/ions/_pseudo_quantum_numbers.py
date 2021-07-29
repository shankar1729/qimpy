import torch
import numpy as np
from typing import Optional


class PseudoQuantumNumbers:
    """Quantum numbers of pseudo-atom in projector or orbital order."""
    __slots__ = ('n_tot', 'i_rf', 'n', 'l', 'm', 'i_lm')
    n_tot: int  #: total number of projectors / orbitals (accounting for m)
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
        self.n = torch.zeros(self.n_tot, dtype=torch.long, device=l.device)
        self.l = torch.zeros_like(self.n)
        self.m = torch.zeros_like(self.n)
        self.i_rf = torch.zeros_like(self.n)
        # Initialize one l at a time:
        i_start = 0
        n_l = [0] * int(l.max().item() + 1)
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
        # Repeat for all m components:
        D_nlm = D[self.i_rf][:, self.i_rf].contiguous()
        # Handle spinor components:
        if j is None:
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
            # Spin-angle transformations:
            for l in range(3):
                Cl = get_Ylm_to_spin_angle(l)
                print(l)
                print(Cl[:, :2*l])
                print(Cl[:, 2*l:])
                print()

            exit()
            raise NotImplementedError('Spin-angle transformations')


def get_Ylm_to_spin_angle(l: int) -> np.ndarray:
    """Create 2(2l+1) x 2(2l+1) matrix transforming from (m, s) to (j, mj).
    The (m, s) indices are along the first dimension, with inner s index.
    The (j, mj) indices along the second dimension have an inner mj index.
    """
    n_m = 2*l + 1
    n_minus = n_m - 1  # number of mj at j = l - 1/2
    n_plus = n_m + 1  # number of mj at j = l + 1/2
    # Initialize Clebsch Gordon coefficients:
    C = np.zeros((2, n_m, n_minus + n_plus))  # s, m, (j,mj)
    if n_minus:
        m = np.arange(-l+1, l+1)
        i_mj = (l-1) + m
        C[0, (l-1)+m, i_mj] = np.sqrt((l+1)-m)
        C[1, l+m, i_mj] = -np.sqrt(l+m)
    if n_plus:
        m = np.arange(-l, l+1)
        i_mj = n_minus + (l+1) + m
        C[0, l+m, i_mj] = np.sqrt((l+1)+m)
        C[1, l+m, i_mj-1] = np.sqrt((l+1)-m)
    C *= np.sqrt(1./n_m)
    return C.swapaxes(0, 1).reshape(n_m*2, n_m*2)
