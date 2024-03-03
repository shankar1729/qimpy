from __future__ import annotations
import torch


class TensorList(list[torch.Tensor]):
    def __mul__(self, scale: float) -> TensorList:
        return TensorList(scale * ti for ti in self)

    def __rmul__(self, scale: float) -> TensorList:
        return self * scale

    def __add__(self, other: TensorList) -> TensorList:
        return TensorList(ti + tj for ti, tj in zip(self, other))
