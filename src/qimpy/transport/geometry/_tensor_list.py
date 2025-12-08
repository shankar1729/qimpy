from __future__ import annotations
from typing import Iterator

import torch


class TensorList:
    data: list[torch.Tensor]

    def __init__(self, data: Iterator[torch.Tensor]):
        self.data = list(data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.data[index]

    def __mul__(self, scale: float) -> TensorList:
        return TensorList(scale * ti for ti in self)

    def __rmul__(self, scale: float) -> TensorList:
        return self * scale

    def __add__(self, other: TensorList) -> TensorList:
        return TensorList(ti + tj for ti, tj in zip(self, other))
