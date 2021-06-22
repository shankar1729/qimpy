import qimpy as qp
import numpy as np
import torch
from typing import Optional, Union


class RadialFunction:
    """Set of radial functions in real and reciprocal space.
    For l > 0, the convention is to remove a factor of r^l in real space (f)
    and G^l in reciprocal space (f_t). This makes it convenient to work with
    solid harmonics that already contain these factors of r^l or G^l.
    """
    __slots__ = ('r', 'dr', 'f', 'l')
    r: torch.Tensor  #: radial grid
    dr: torch.Tensor  #: radial grid integration weights (dr in 4 pi r^2 dr)
    f: torch.Tensor  #: real-space values corresponding to r (n x len(r))
    l: torch.Tensor  #: angular momentum for each function in f

    def __init__(self, r: torch.Tensor, dr: torch.Tensor,
                 f: Union[np.ndarray, torch.Tensor],
                 l: Optional[Union[np.ndarray, torch.Tensor]] = None) -> None:
        """Initialize real-space portion of radial function.
        Note that f should have a factor of r^l removed for correct
        subsequent behavior with transforms and solid harmonics."""
        self.r = r
        self.dr = dr
        f = (f if isinstance(f, torch.Tensor)
             else torch.tensor(f, device=r.device))
        self.f = (f if (len(f.shape) == 2) else f[None])
        self.l = (torch.zeros(1, dtype=torch.int, device=r.device)
                  if l is None
                  else (l if isinstance(l, torch.Tensor)
                        else torch.tensor(l, device=r.device)))
        assert(dr.shape == r.shape)
        assert(r.shape[0] == self.f.shape[-1])
        assert(self.l.shape == self.f.shape[:-1])
