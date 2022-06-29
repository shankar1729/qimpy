import torch
import numpy as np
from functools import lru_cache
from typing import Optional


class PseudoQuantumNumbers:
    """Quantum numbers of pseudo-atom in projector or orbital order."""

    n_tot: int  #: total number of projectors / orbitals (accounting for m)
    l_max: int  #: maximum l of projectors / orbitals
    i_rf: torch.Tensor  #: index of radial function (across all l)
    n: torch.Tensor  #: pseudo-principal quantum number (for each l,j, 1-based)
    l: torch.Tensor  #: orbital angular momentum
    m: torch.Tensor  #: azimuthal quantum number
    i_lm: torch.Tensor  #: combined (l, m) index into solid harmonics array

    # Extra properties for relativistic pseudopotentials:
    is_relativistic: bool  #: whether this pseudopotential is relativistic
    n_tot_s: int  #: total number of spinorial states
    ns: torch.Tensor  #: pseudo-principal quantum number of spinorial states
    ls: torch.Tensor  #: orbital angular momentum in spinor-expanded set
    j: torch.Tensor  #: total angular momentum of each spinorial orbital
    mj: torch.Tensor  #: total azimuthal angular momentum
    i_ljm: torch.Tensor  #: combined (l, j, m_j) index used for orbitals
    i_ljms: torch.Tensor  #: combined (l, j, m_l, s) index used for projectors

    def __init__(self, l: torch.Tensor, j: Optional[torch.Tensor]) -> None:
        """Initialize given `l` of each projector / orbital of a
        pseudopotential. The input corresponds to the distinct radial functions
        read from a pseudoptential, while the members of this class are for
        each projector function including spherical harmonics."""
        self.n_tot = int((2 * l + 1).sum().item())
        self.l_max = int(l.max().item())
        self.n = torch.zeros(self.n_tot, dtype=torch.long, device=l.device)
        self.l = torch.zeros_like(self.n)
        self.m = torch.zeros_like(self.n)
        self.i_rf = torch.zeros_like(self.n)
        # Initialize one l at a time:
        i_start = 0
        n_l = [0] * (self.l_max + 1)
        for i_rf, l_iter in enumerate(l):
            # Size of shell at current m:
            l_i = l_iter.item()
            n_l[l_i] += 1  # starts at 1 for first of each l
            m_count = 2 * l_i + 1
            i_stop = i_start + m_count
            # Assign properties for this shell:
            self.i_rf[i_start:i_stop] = i_rf
            self.n[i_start:i_stop] = n_l[l_i]
            self.l[i_start:i_stop] = l_i
            self.m[i_start:i_stop] = torch.arange(-l_i, l_i + 1, device=l.device)
            # To next shell:
            i_start = i_stop
        self.i_lm = self.l * (self.l + 1) + self.m

        if j is None:
            self.is_relativistic = False
        else:
            self.is_relativistic = True
            self._initialize_relativistic(l, j)

    def _initialize_relativistic(self, l: torch.Tensor, j: torch.Tensor) -> None:
        """Initialize the relativistic quantum numbers and indices."""
        assert ((j - l).abs() == 0.5).all()
        self.n_tot_s = int((2 * j + 1).sum().item())
        self.ns = torch.zeros(self.n_tot_s, dtype=torch.long, device=l.device)
        self.ls = torch.zeros_like(self.ns)
        self.j = torch.zeros_like(self.ns, dtype=j.dtype)  # half-integral
        self.mj = torch.zeros_like(self.j)
        # Initialize one l, j set at a time:
        i_start = 0
        n_lj = [0] * (2 * self.l_max + 1)
        for i_rf, l_iter in enumerate(l):
            # Size of shell at current m:
            l_i = l_iter.item()
            j_i = j[i_rf].item()
            lj_i = (2 * l_i - 1) if (j_i < l_i) else (2 * l_i)
            n_lj[lj_i] += 1  # starts at 1 for first of each l, j
            mj_count = int(2 * j_i + 1)
            i_stop = i_start + mj_count
            # Assign properties for this shell:
            self.n[self.i_rf == i_rf] = n_lj[lj_i]  # fix n computed without j
            self.ns[i_start:i_stop] = n_lj[lj_i]
            self.ls[i_start:i_stop] = l_i
            self.j[i_start:i_stop] = j_i
            self.mj[i_start:i_stop] = torch.arange(mj_count, device=l.device) - j_i
            # To next shell:
            i_start = i_stop
        self.i_ljm = (
            self.ls * (2 * self.j + 1) + self.j + self.mj  # from previous l,j
        ).to(
            torch.long
        )  # within current l,j
        self.i_ljms = self._get_i_ljms(j)

    def _get_i_ljms(self, j: torch.Tensor) -> torch.Tensor:
        """Calculate index from projectors into Ylm overlaps and transforms.
        Note that this is an index from the spinor-repeated (l, j, m, s) basis,
        and not from the (l, j, mj) basis used for spinorial orbitals,
        and hence is separated into its own function for clarity."""
        # Set up indices with spinor repetition as n_tot x 2 tensors:
        i_lms = 2 * self.i_lm[:, None].tile((1, 2))
        i_lms[:, 1] += 1  # Index is now 2*(l(l+1) + m) + s
        l = self.l[:, None]
        # Accumulate offsets into the Ylm overlaps array
        # These are sorted by l, j (=l+/-1/2), m_l (2l+1 entries) and s
        i_ljms = i_lms + (2 * (l ** 2) - 2)  # offset from previous l
        sel_plus = torch.where(j[self.i_rf] > self.l)[0]
        i_ljms[sel_plus] += 2 * (2 * l[sel_plus] + 1)  # offset from j = l-1/2
        return i_ljms.flatten()

    def expand_matrix(self, D: torch.Tensor, n_spinor: int) -> torch.Tensor:
        """Expand matrix `D` from (n, l) basis to include m and s.
        Spin s is included explicitly only if `n_spinor` is 2.
        Additionally, if pseudopotential is relativistic, the expanded matrix
        includes spin-angle overlap factors; n_spinor must be 2 for this mode.
        """
        if self.is_relativistic:
            if n_spinor != 2:
                raise ValueError(
                    "Relativistic pseudopotentials require" " spinorial calculation"
                )
            # Collect matrix with spinor repetition:
            i_rf = self.i_rf[:, None].tile((1, 2)).flatten()  # spinor repeat
            D_nljms = D[i_rf][:, i_rf].contiguous()
            # Apply spin-angle overlaps (akin to delta_{lm,lm'} above):
            f_lj = torch.from_numpy(get_Ylm_overlaps(self.l_max)).to(D.device)
            return D_nljms * f_lj[self.i_ljms][:, self.i_ljms]
        else:
            # Non-relativistic pseuodopotential:
            D_nlm = D[self.i_rf][:, self.i_rf].contiguous()
            D_nlm[self.i_lm[None] != self.i_lm[:, None]] = 0.0  # delta_{lm,lm'}
            if n_spinor == 1:
                return D_nlm
            else:  # n_spinor == 2
                # Repeat for spinor component:
                n_nlms = D_nlm.shape[0] * n_spinor
                D_nlms = torch.zeros(
                    (n_nlms, n_nlms), dtype=D_nlm.dtype, device=D_nlm.device
                )
                for i_spinor in range(n_spinor):
                    D_nlms[i_spinor::n_spinor, i_spinor::n_spinor] = D_nlm
                return D_nlms

    def get_spin_angle_transform(self) -> torch.Tensor:
        """Return transformation matrix from (l, j, m_l), s to (l, j, m_j).
        Only for relativistic cases, with output shape n_tot x 2 x n_tot_s.
        """
        C = torch.from_numpy(get_Ylm_to_spin_angle_all(self.l_max)).to(self.l.device)
        C_out = C[self.i_ljms][:, self.i_ljm].reshape((self.n_tot, 2, self.n_tot_s))
        # Project out contributions between different shells:
        sel = torch.where(self.n[:, None] != self.ns[None, :])
        C_out[sel[0], slice(None), sel[1]] = 0.0
        return C_out


@lru_cache
def get_Ylm_to_spin_angle(l: int) -> np.ndarray:
    """Create 2(2l+1) x 2(2l+1) matrix transforming from (m, s) to (j, mj).
    The (m, s) indices are along the first dimension, with inner s index.
    The (j, mj) indices along the second dimension have an inner mj index for
    j = l - 1/2 first (2l entries) and then for j = l + 1/2 (2l + 2 entries).
    """
    n_m = 2 * l + 1
    n_minus = n_m - 1  # number of mj at j = l - 1/2
    n_plus = n_m + 1  # number of mj at j = l + 1/2

    # Initialize Clebsch Gordon coefficients:
    C = np.zeros((2, n_m, n_minus + n_plus))  # s, m, (j,mj)
    C_entries = np.sqrt(1.0 - np.arange(n_m) / n_m)  # unique entries at this l
    if n_minus:
        np.fill_diagonal(C[0, 0:, :n_minus], C_entries[1:])
        np.fill_diagonal(C[1, 1:, :n_minus], -C_entries[:0:-1])
    if n_plus:
        np.fill_diagonal(C[0, :, n_minus + 1 :], C_entries[::-1])
        np.fill_diagonal(C[1, :, n_minus + 0 :], C_entries)

    # Account for real to complex harmonics transformation:
    Y = np.zeros((n_m, n_m), dtype=np.complex128)
    Y[l, l] = 1.0  # m = 0 not transformed
    m = np.arange(1, l + 1)
    parity = (-1) ** m
    sqrt_half = np.sqrt(0.5)
    Y[l + m, l + m] = sqrt_half * parity
    Y[l + m, l - m] = sqrt_half
    Y[l - m, l + m] = sqrt_half * parity * 1j
    Y[l - m, l - m] = sqrt_half * (-1j)
    return (
        (Y @ C)  # transform Clebsh-Gordon to apply for real harmonics
        .swapaxes(0, 1)
        .reshape(n_m * 2, n_m * 2)
    )  # s, m -> combined (m, s)


@lru_cache
def get_Ylm_to_spin_angle_all(l_max: int) -> np.ndarray:
    """Cumulative spin angle transformation matrix up to l = `l_max`.
    Result is block-diagonal with dimension (4(l_max+1)^2 - 2) x 2(l_max+1)^2,
    with 2(2l+1) x (2j+1) blocks for each l, j."""
    C = get_Ylm_to_spin_angle(l_max)
    if l_max == 0:
        return C
    else:
        C_prev = get_Ylm_to_spin_angle_all(l_max - 1)
        i_minus_0, i_minus_1 = C_prev.shape
        C_minus = C[:, : 2 * l_max]
        C_plus = C[:, 2 * l_max :]
        # First index covers m, s for each l, j
        i_plus_0 = i_minus_0 + C.shape[0]
        n_tot_0 = i_plus_0 + C.shape[0]
        # Second index covers mj for each l, j
        i_plus_1 = i_minus_1 + C_minus.shape[1]
        n_tot_1 = i_plus_1 + C_plus.shape[1]
        # Combine previous l with j = l -/+ 1/2 at this l:
        C_tot = np.zeros((n_tot_0, n_tot_1), dtype=np.complex128)
        C_tot[:i_minus_0, :i_minus_1] = C_prev
        C_tot[i_minus_0:i_plus_0, i_minus_1:i_plus_1] = C_minus
        C_tot[i_plus_0:, i_plus_1:] = C_plus
        return C_tot


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
        C_minus = C[:, : 2 * l_max]
        C_plus = C[:, 2 * l_max :]
        O_minus = C_minus @ C_minus.conj().T
        O_plus = C_plus @ C_plus.conj().T
        # Put together with results for previous l:
        C_prev = get_Ylm_overlaps(l_max - 1)
        i_minus = C_prev.shape[0]  # start index of j = l_max - 1/2
        i_plus = i_minus + O_minus.shape[0]  # start index of j = l_max + 1/2
        n_tot = i_plus + O_plus.shape[0]  # total entries including l_max
        C_tot = np.zeros((n_tot, n_tot), dtype=np.complex128)
        C_tot[:i_minus, :i_minus] = C_prev
        C_tot[i_minus:i_plus, i_minus:i_plus] = O_minus
        C_tot[i_plus:, i_plus:] = O_plus
        return C_tot
