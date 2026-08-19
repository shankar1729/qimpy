import numpy as np
import torch
import torch.distributed as dist


def all_gather_padded(
    send: torch.Tensor, sizes: np.ndarray, group: dist.ProcessGroup
) -> torch.Tensor:
    """Gather `send` with possibly uneven dimension 0 `sizes` along `group`.
    Return the result as a single tensor with concatenated dimension 0,
    automatically handling uneven sizes when not supported by the backend."""
    size_max = sizes.max()
    size_tot = sizes.sum()
    sizes_equal = sizes.min() == size_max
    if sizes_equal or (dist.get_backend(group) == "nccl"):
        # Directly gather with equal sizes, or leveraging support for unequal sizes:
        recv_shape = (size_tot,) + send.shape[1:]
        recv = torch.empty(recv_shape, dtype=send.dtype, device=send.device)
        recv_views = list(recv.split(sizes.tolist(), dim=0))
        dist.all_gather(recv_views, send.contiguous(), group=group)
    else:
        # Pad inputs to constant size:
        shape = (size_max,) + send.shape[1:]
        size_mine = send.shape[0]
        if size_mine != size_max:
            send_padded = torch.empty(shape, dtype=send.dtype, device=send.device)
            send_padded[:size_mine] = send
            send = send_padded
        recvs = [
            torch.empty(shape, dtype=send.dtype, device=send.device) for _ in sizes
        ]
        dist.all_gather(recvs, send.contiguous(), group=group)
        # Remove padding from outputs:
        recv = torch.cat([buf[:size] for buf, size in zip(recvs, sizes)], dim=0)
    return recv
