import torch


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
