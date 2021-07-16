"""Reduction of MPI-distributed tensors to scalars.
The functions of this module correctly handle zero-sized pieces of
distributed tensors on certain processes, whic is a frequently
encountered and cumbersome corner case in such global reductions. """
__all__ = ['sum', 'prod', 'min', 'max', 'all', 'any']

import qimpy as qp
import torch
from typing import Any


def sum(v: torch.Tensor, comm: qp.MPI.Comm) -> Any:
    """Global sum of tensor `v` distributed over `comm`."""
    return comm.allreduce(torch.sum(v).item() if v.numel()
                          else torch.zeros(1, dtype=v.dtype).item(),
                          qp.MPI.SUM)


def prod(v: torch.Tensor, comm: qp.MPI.Comm) -> Any:
    """Global product of tensor `v` distributed over `comm`."""
    return comm.allreduce(torch.prod(v).item() if v.numel()
                          else torch.ones(1, dtype=v.dtype).item(),
                          qp.MPI.PROD)


def min(v: torch.Tensor, comm: qp.MPI.Comm) -> Any:
    """Global minimum of tensor `v` distributed over `comm`."""
    return comm.allreduce(torch.min(v).item() if v.numel()
                          else (torch.finfo(v.dtype).max
                                if v.dtype.is_floating_point
                                else torch.iinfo(v.dtype).max),
                          qp.MPI.MIN)


def max(v: torch.Tensor, comm: qp.MPI.Comm) -> Any:
    """Global maximum of tensor `v` distributed over `comm`."""
    return comm.allreduce(torch.max(v).item() if v.numel()
                          else (torch.finfo(v.dtype).min
                                if v.dtype.is_floating_point
                                else torch.iinfo(v.dtype).min),
                          qp.MPI.MAX)


def all(v: torch.Tensor, comm: qp.MPI.Comm) -> Any:
    """Global minimum of tensor `v` distributed over `comm`."""
    return comm.allreduce(torch.all(v).item() if v.numel() else True,
                          qp.MPI.LAND)


def any(v: torch.Tensor, comm: qp.MPI.Comm) -> Any:
    """Global maximum of tensor `v` distributed over `comm`."""
    return comm.allreduce(torch.any(v).item() if v.numel() else False,
                          qp.MPI.LOR)
