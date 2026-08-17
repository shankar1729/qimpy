"""Reduction of distributed tensors to scalars.
The functions of this module correctly handle zero-sized pieces of
distributed tensors on certain processes, which is a frequently
encountered and cumbersome corner case in such global reductions."""

__all__ = ["sum", "prod", "min", "max", "all", "any"]

import torch
import torch.distributed as dist


def sum(v: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Global sum of tensor `v` distributed over `group`."""
    result = v.sum() if v.numel() else torch.zeros(1, dtype=v.dtype, device=v.device)
    dist.all_reduce(result, op=dist.ReduceOp.SUM, group=group)
    return result


def prod(v: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Global product of tensor `v` distributed over `group`."""
    result = v.prod() if v.numel() else torch.ones(1, dtype=v.dtype, device=v.device)
    dist.all_reduce(result, op=dist.ReduceOp.PRODUCT, group=group)
    return result


def min(v: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Global minimum of tensor `v` distributed over `group`."""
    if v.numel():
        result = v.min()
    else:
        info = torch.finfo if v.dtype.is_floating_point else torch.iinfo
        result = torch.full(1, info(v.dtype).max, dtype=v.dtype, device=v.device)
    dist.all_reduce(result, op=dist.ReduceOp.MIN, group=group)
    return result


def max(v: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Global maximum of tensor `v` distributed over `group`."""
    if v.numel():
        result = v.max()
    else:
        info = torch.finfo if v.dtype.is_floating_point else torch.iinfo
        result = torch.full(1, info(v.dtype).min, dtype=v.dtype, device=v.device)
    dist.all_reduce(result, op=dist.ReduceOp.MAX, group=group)
    return result
