import torch


def _get_lattice_point_group(Rbasis, tolerance):
    '''Return point group (n_sym x 3 x 3 tensor in lattice coordinates),
    given lattice vectors Rbasis (3 x 3 tensor).'''

    # Reduce lattice vectors:
    T = _reduce_matrix33(Rbasis, tolerance)
    Rreduced = Rbasis @ T

    # Construct all possible matrices with entries from (-1, 0, 1):
    entries = torch.tensor([-1, 0, 1], device=T.device, dtype=float)
    matrices = torch.stack(torch.meshgrid([entries]*9))
    matrices = matrices.reshape((9, -1)).T.reshape((-1, 3, 3))

    # Find matrices that preserve reduced metric:
    metric = Rreduced.T @ Rreduced
    metric_new = matrices.transpose(-2, -1) @ (metric @ matrices)
    metric_err = ((metric_new - metric) ** 2).sum(dim=(1, 2))
    sym = matrices[torch.where(metric_err < tolerance*(metric**2).sum())[0]]

    # Transform to original (unreduced) coordinates:
    return (T @ sym) @ torch.linalg.inv(T)


def _reduce_matrix33(M, tolerance):
    '''Return integer transformation matrix T that minimizes norm(M * T)
    with accuracy set by tolerance. M must be a 3 x 3 tensor.'''
    assert(M.shape == (3, 3))

    # Prepare a list of transformations based on +/-1 offsets:
    direction_combinations = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    offset_combinations = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                           (0, 1), (1, -1), (1, 0), (1, 1))
    D = []
    for k0, k1, k2 in direction_combinations:
        for offset1, offset2 in offset_combinations:
            # Propose transformation with +/-1 offsets:
            Dcur = torch.eye(3)
            Dcur[k1, k0] = offset1
            Dcur[k2, k0] = offset2
            D.append(Dcur)
    D = torch.stack(D).to(M.device)

    # Repeatedly transform till norm no longer reduces:
    T = torch.eye(3, device=M.device)
    MT = M.clone().detach()
    norm = (M ** 2).sum()
    while True:
        MT_new = MT @ D
        norm_new = (MT_new ** 2).sum(dim=(1, 2))
        i_min = norm_new.argmin()
        if norm_new[i_min] < norm * (1. - tolerance):
            T = T @ D[i_min]
            MT = MT_new[i_min]
            norm = norm_new[i_min]
        else:
            return T  # converged


if __name__ == '__main__':
    tol = 1e-6
    torch.set_default_tensor_type(torch.DoubleTensor)
    M = torch.tensor([[5, 1, 0], [4, 2, 0], [0, 3, 1]], dtype=float)
    T = _reduce_matrix33(M, tol)
    invT = torch.linalg.inv(T)
    Mnew = M @ T
    print(T, '\n', invT, '\n', invT @ T)
