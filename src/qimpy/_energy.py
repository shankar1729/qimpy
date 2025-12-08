from typing import Union, Optional

import torch


class Energy(dict[str, Union[float, torch.Tensor]]):
    """Energy of system with access to components"""

    name: str  #: standard label for (free) energy type, e.g., E, A, Phi etc.

    def __init__(self, name: str = "E") -> None:
        super().__init__()
        self.name = name

    def __float__(self) -> float:
        """Compute total energy from energy components"""
        return float(sum(self.values()))

    def __repr__(self) -> str:
        terms: list[list[str]] = [[], []]  # collect terms with +/- separately
        for name, value in sorted(self.items()):
            term_index = 1 if (name[0] in "+-") else 0
            terms[term_index].append(f"{name:>9s} = {value:25.16f}")
        terms[0].extend(terms[1])
        terms[0].append("-" * 37)  # separator
        terms[0].append(f"{self.name:>9s} = {float(self):25.16f}")  # total
        return "\n".join(terms[0])

    def sum_tensor(self) -> Optional[torch.Tensor]:
        result = None
        for value in self.values():
            assert isinstance(value, torch.Tensor)
            result = value if (result is None) else (result + value)
        return result
