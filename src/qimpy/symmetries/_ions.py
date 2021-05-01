import torch


def _get_space_group(lattice_sym, ions):
    '''Given lattice point group and ions, return space group as rot, trans:
        Rotations rot is an n_sym x 3 x 3 int tensor in lattice coordinates.
        Translations trans is an n_sym x 3 tensor in lattice coordinates.'''
    pass
