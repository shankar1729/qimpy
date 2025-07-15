import matplotlib.pyplot as plt
import numpy as np
import torch

from qimpy import rc
from qimpy.io import log_config
from qimpy.mpi._sparse_matrices import SparseMatrixRight


def main():
    log_config()
    rc.init()
    torch.manual_seed(0)
    indices = torch.stack([torch.arange(100), torch.arange(2, 102)])
    values = torch.randn(indices.shape[1])
    sm = SparseMatrixRight(indices, values, comm=rc.comm)
    Md = sm.getM().to_dense()
    randvec1 = torch.randn(sm.size[0])
    res1d = randvec1 @ Md
    res1 = sm.vecTimesMatrix(randvec1)
    indicesT = torch.stack([indices[1], indices[0]])
    sm = SparseMatrixRight(indicesT, values, comm=rc.comm)
    Md = sm.getM().to_dense()
    randvec2 = torch.randn(sm.size[0])
    res2d = randvec2 @ Md
    res2 = sm.vecTimesMatrix(randvec2)
    if rc.is_head:
        print(res1 - res1d)
        print(res2 - res2d)


if __name__ == "__main__":
    main()
