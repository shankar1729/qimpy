from __future__ import annotations
import torch
import qimpy as qp
from qimpy.rc import MPI


class History(qp.TreeNode):
    """Helper to save history along trajectory."""

    comm: MPI.Comm
    save_map: dict[str, torch.Tensor]  # Names and data for quantities to save

    def __init__(
        self, *, comm: MPI.Comm, checkpoint_in: qp.utils.CpPath = qp.utils.CpPath()
    ) -> None:
        super().__init__()
        self.comm = comm
        self.save_map = {}

    def _save_checkpoint(
        self, cp_path: qp.utils.CpPath, context: qp.utils.CpContext
    ) -> list[str]:
        stage, i_iter = context
        saved_list = []
        if stage == "geometry":
            for name, data in self.save_map.items():
                self.save(cp_path.relative(name), data, i_iter)
                saved_list.append(name)
        return saved_list

    def save(self, cp_path: qp.utils.CpPath, data: torch.Tensor, i_iter: int) -> None:
        """Save history of `data` for `i_iter`'th iteration to `cp_path`.
        When the dataset doesn't exist, it will be created with size
        `(i_iter + 1,) + data.shape`, resizable along the first dimension.
        Data is assumed consistent across `comm`, and is written from head."""
        checkpoint, path = cp_path
        assert checkpoint is not None
        if path in checkpoint:
            dset = checkpoint[path]
            assert dset.shape[1:] == data.shape
            dset.resize(i_iter + 1, axis=0)
        else:
            dset = checkpoint.create_dataset(
                path,
                shape=(i_iter + 1,) + data.shape,
                dtype=qp.rc.np_type[data.dtype],
                maxshape=(None,) + data.shape,  # resizable infinitely along first axis
            )
        if self.comm.rank == 0:
            dset[-1, ...] = data.to(qp.rc.cpu).detach().numpy()
