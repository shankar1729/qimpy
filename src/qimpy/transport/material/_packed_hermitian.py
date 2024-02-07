import torch
import numpy as np

from qimpy import rc


class PackedHermitian:
    """Packed real representation of Hermitian matrices."""

    def __init__(self, N: int) -> None:
        """Initialize representation for N x N hermitian matrices."""
        i = np.arange(N)
        i, j = (x.flatten() for x in np.meshgrid(i, i, indexing="ij"))
        i1 = i * N + j  # direct entry
        i2 = j * N + i  # swapped entry

        # Separate real / imaginary part from complex:
        self.pack_index = torch.from_numpy(+np.where(i >= j, 2 * i1, 2 * i2 + 1)).to(
            device=rc.device
        )

        # Reconstruct complex hermitian version:
        unpack_real = np.where(i >= j, i1, i2)
        unpack_imag = np.where(i >= j, i2, i1)
        unpack_imag_sign = np.sign(i - j)
        self.unpack_index = torch.from_numpy(
            np.stack((unpack_real, unpack_imag), axis=-1).flatten()
        ).to(device=rc.device)
        self.unpack_sign = torch.from_numpy(
            np.stack((np.ones_like(i1), unpack_imag_sign), axis=-1).flatten()
        ).to(device=rc.device)

        # Construct matrices for transforming super-operators to packed form:
        R = np.zeros((N * N, N * N), dtype=complex)
        Rinv = np.zeros_like(R)
        # --- diagonal terms:
        diag = np.where(i == j)[0]
        d1 = d2 = i1[diag]
        R[d1, d2] = 1.0
        Rinv[d1, d2] = 1.0
        # --- off-diagonal terms
        offdiag = np.where(i > j)[0]
        o1 = i1[offdiag]
        o2 = i2[offdiag]
        R[o1, o1] = 1.0
        R[o1, o2] = 1.0j
        R[o2, o1] = 1.0
        R[o2, o2] = -1.0j
        Rinv[o1, o1] = +0.5
        Rinv[o1, o2] = +0.5
        Rinv[o2, o1] = -0.5j
        Rinv[o2, o2] = +0.5j
        self.R = torch.from_numpy(R).to(rc.device)
        self.Rinv = torch.from_numpy(Rinv).to(rc.device)
        self.N = N
        self.w_overlap = torch.from_numpy(
            np.where(i == j, 1.0, 2.0).reshape((N, N))
        ).to(rc.device)

    def pack(self, m: torch.Tensor) -> torch.Tensor:
        """Pack ... x N x N complex hermitian tensor to real version."""
        buf = torch.view_as_real(m).flatten(-3, -1)
        return buf[..., self.pack_index].view(m.shape)

    def unpack(self, m: torch.Tensor) -> torch.Tensor:
        """Unpack ... x N x N real tensor to complex hermitian version."""
        result = m.flatten(-2, -1)[..., self.unpack_index] * self.unpack_sign
        return torch.view_as_complex(result.view(m.shape + (2,)))

    def apply_packed(self, op: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        vec_packed = self.pack(vec).flatten()
        return self.unpack((op @ vec_packed).view(vec.shape))
